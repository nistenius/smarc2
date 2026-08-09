#!/usr/bin/python3
"""
Cascaded-PID dive controller with cross-track guidance — Session C part 3, 2026-08-09.

Third member of the controller family (stock PID / BlendPID / CascadePID / MPC), same
DiveControllerInterface, selectable per launch (SAM_DIVE_LAUNCH=cascade_pid_wp_following)
or side by side via the dive_action_name parameter. Tuning lives in
config/cascade_pid.yaml — its own package of settings, returnable to independently.

Structure (proposal §1–§3, plus what the 2026-08-09 velocity sweep taught):

  LATERAL   cross-track e (ILOS on the leg line)  ->  course chi_d
                                                  ->  heading ref (sideslip-compensated)
            heading error (wrapped)               ->  yaw-rate ref r_d   [P, capped]
            r error                               ->  rudder             [PI]

  VERTICAL  depth error                           ->  pitch ref          [LOS lookahead
                                                      scaled with speed, + slow integral
                                                      to kill the 0.2 m standing bias]
            pitch error                           ->  pitch-rate ref q_d [P, capped]
            q error                               ->  stern plane        [PI]

  ALLOCATION (speed-nested, per Ivan): VBS and LCG carry the P-effort weighted by
  (1 - w) so they dominate near hover and fade at cruise; their INTEGRALS (trim) stay
  active at all speeds. The stern-plane cascade authority is weighted by w. w is the
  same speed blend as BlendPID: sat((u - u_lo)/(u_hi - u_lo)).

  SPEED     u_d = min(u_cruise, braking profile); RPM = kff*u_d + PI trim.
            (carried over from BlendPID; blend_u_cruise remains live-settable)

The leg line for cross-track error is anchored at the vehicle position when the active
waypoint changes (the action interface only exposes the current waypoint). On each new
leg the leg bearing is checked against DiveSub's pure-pursuit heading setpoint; if the
conventions ever disagree by more than ~30 deg the controller falls back to pure
pursuit for that leg and says so — measured self-check instead of a silent wrong frame.

Signs (measured this session, run 133216): positive pitch = nose DOWN; dive/depth error
positive when too shallow -> positive pitch ref. Stern actuator negation kept from
stock. Yaw positive CCW (ENU), rudder direct.
"""
import numpy as np

from sam_diving_controller.controllers.PIDControl import PIDControl
from sam_diving_controller.IDivePub import MissionStates, ActuatorStates
from sam_diving_controller.controllers.DiveControllerInterface import DiveControllerInterface
from smarc_control_msgs.msg import ControlError, ControlInput
from nav_msgs.msg import Odometry


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class DiveControllerCascadePID(DiveControllerInterface):

    def __init__(self, node, dive_pub, dive_sub, param, rate=0.1):
        self._node = node
        self._dive_sub = dive_sub
        self._dive_pub = dive_pub
        self._dt = rate
        self.param = param

        self.traj_index = 0
        self._control_ref = None
        self._current_state_in_dr = None

        super().__init__(self._node, self._dive_pub, self._dive_sub, self.param, self._dt)

        # --- cascade parameters: this controller's own package of tuning ---------
        d = node.declare_parameter
        d('casc_delta_h', 5.0)      # m, horizontal lookahead (ILOS)
        d('casc_kappa', 0.03)       # ILOS integral gain
        d('casc_e_int_max', 5.0)    # m, ILOS integrator clamp
        d('casc_delta_z', 3.0)      # m, vertical lookahead at/below casc_uref
        d('casc_uref', 0.5)         # m/s, vertical lookahead speed scaling
        d('casc_z_ki', 0.02)        # rad/(m s), slow outer depth integral (bias killer)
        d('casc_pitch_kp', 1.5)     # (rad/s)/rad
        d('casc_q_max', 0.30)       # rad/s
        d('casc_q_kp', 0.30)        # stern rad per rad/s
        d('casc_q_ki', 0.05)
        d('casc_yaw_kp', 0.80)      # (rad/s)/rad
        d('casc_r_max', 0.26)       # rad/s (15 deg/s, DVL bottom-lock envelope)
        d('casc_r_kp', 0.40)        # rudder rad per rad/s
        d('casc_r_ki', 0.05)
        g = lambda n: node.get_parameter(n).get_parameter_value().double_value
        self.c = {n: g(n) for n in ('casc_delta_h', 'casc_kappa', 'casc_e_int_max',
                                    'casc_delta_z', 'casc_uref', 'casc_z_ki',
                                    'casc_pitch_kp', 'casc_q_max', 'casc_q_kp',
                                    'casc_q_ki', 'casc_yaw_kp', 'casc_r_max',
                                    'casc_r_kp', 'casc_r_ki')}

        # --- inner-loop PIDs ------------------------------------------------------
        tv_lim = self.param['tv_u_max']
        self._q_pid = PIDControl(Kp=self.c['casc_q_kp'], Ki=self.c['casc_q_ki'], Kd=0.0,
                                 Kaw=1.0, u_neutral=0.0, u_min=-tv_lim, u_max=tv_lim)
        self._r_pid = PIDControl(Kp=self.c['casc_r_kp'], Ki=self.c['casc_r_ki'], Kd=0.0,
                                 Kaw=1.0, u_neutral=0.0, u_min=-tv_lim, u_max=tv_lim)

        # --- speed-nested VBS / LCG (transparent form, per-design) ---------------
        # integral = trim, always active; P-effort weighted by (1 - w) at speed.
        self._vbs_int = 0.0
        self._lcg_int = 0.0
        self._u_vbs_prev = None
        self._vbs_slew = self.param['blend_vbs_slew']

        # --- surge (carried over from BlendPID) -----------------------------------
        self._surge_rpm_pid = PIDControl(Kp=100.0, Ki=5.25, Kd=1.5, Kaw=1.0,
                                         u_neutral=0, u_min=-300, u_max=300)

        # --- guidance state -------------------------------------------------------
        self._leg_anchor = None      # (x, y) odom at leg start
        self._leg_wp = None          # (x, y) of the active waypoint when anchored
        self._leg_alpha = None       # leg bearing
        self._leg_pure_pursuit = False
        self._e_int = 0.0            # ILOS integrator
        self._beta_lpf = 0.0

        self._loginfo("Cascade Dive Controller created")

    # ------------------------------------------------------------------ helpers --
    def _u_cruise_live(self):
        try:
            return float(self._node.get_parameter('blend_u_cruise')
                         .get_parameter_value().double_value)
        except Exception:
            return self.param['blend_u_cruise']

    def _blend_weight(self, u):
        lo, hi = self.param['blend_u_lo'], self.param['blend_u_hi']
        if hi <= lo:
            return 0.0
        return float(np.clip((abs(u) - lo) / (hi - lo), 0.0, 1.0))

    def _update_leg(self, px, py, wx, wy, heading_setpoint):
        """(Re)anchor the leg line when the active waypoint moves."""
        if self._leg_wp is None or np.hypot(wx - self._leg_wp[0], wy - self._leg_wp[1]) > 0.5:
            self._leg_anchor = (px, py)
            self._leg_wp = (wx, wy)
            alpha = float(np.arctan2(wy - py, wx - px))
            self._leg_alpha = alpha
            self._e_int = 0.0
            # convention self-check against DiveSub's pure-pursuit bearing
            mismatch = abs(_wrap(alpha - heading_setpoint))
            self._leg_pure_pursuit = mismatch > 0.52  # ~30 deg
            if self._leg_pure_pursuit:
                self._logwarn(
                    f"leg bearing {np.degrees(alpha):.0f} deg vs pure-pursuit "
                    f"{np.degrees(heading_setpoint):.0f} deg (mismatch "
                    f"{np.degrees(mismatch):.0f}) — falling back to pure pursuit for this leg")
            else:
                self._loginfo(
                    f"new leg: anchor ({px:.1f},{py:.1f}) -> wp ({wx:.1f},{wy:.1f}), "
                    f"bearing {np.degrees(alpha):.0f} deg")

    # ------------------------------------------------------------------- update --
    def update(self):
        mission_state = self._dive_sub.get_mission_state()
        self._loginfo_once(f"DC(cascade): {mission_state}")

        if mission_state in (MissionStates.RECEIVED,
                             MissionStates.COMPLETED,
                             MissionStates.CANCELLED):
            self._loginfo_once(f"Mission {mission_state} — actuators neutral")
            self._set_actuators_neutral()
            self._leg_wp = None
            return

        self._dive_pub.set_actuator_states(ActuatorStates.ENGAGED, "DP")

        depth_setpoint = self._dive_sub.get_depth_setpoint()
        pitch_setpoint = self._dive_sub.get_pitch_setpoint()
        heading_setpoint = self._dive_sub.get_heading_setpoint()
        waypoint_odom = self._dive_sub.get_waypoint_in_odom()
        waypoint_global = self._dive_sub.get_waypoint()

        self._current_state = self._dive_sub.get_states()
        self._current_state_in_dr = self._dive_sub.get_states_in_dr()
        current_depth = self._dive_sub.get_depth()
        current_pitch = self._dive_sub.get_pitch()
        current_heading = self._dive_sub.get_heading()
        current_distance = self._dive_sub.get_distance()

        if not self._dive_sub.has_waypoint():
            self._loginfo("No waypoint yet")
            return
        if depth_setpoint is None:
            self._loginfo("No depth setpoint yet")
            return

        current_depth = float(np.abs(current_depth))
        depth_setpoint = float(np.abs(depth_setpoint))

        px = self._current_state.pose.pose.position.x
        py = self._current_state.pose.pose.position.y
        u_meas = self._current_state.twist.twist.linear.x
        v_meas = self._current_state.twist.twist.linear.y
        q_meas = self._current_state.twist.twist.angular.y
        r_meas = self._current_state.twist.twist.angular.z

        w = self._blend_weight(u_meas)

        # ================= SPEED =================================================
        u_cruise = self._u_cruise_live()
        a_brake = self.param['blend_a_brake']
        surge_ref = min(u_cruise, float(np.sqrt(2.0 * a_brake * (current_distance + 0.5))))
        pi_trim, surge_error, _ = self._surge_rpm_pid.get_control(u_meas, surge_ref, self._dt)
        u_rpm = float(np.clip(self.param['blend_rpm_kff'] * surge_ref + pi_trim,
                              self.param['rpm_u_min'], self.param['rpm_u_max']))

        # ================= LATERAL: cross-track -> heading -> r -> rudder ========
        self._update_leg(px, py, waypoint_odom.position.x, waypoint_odom.position.y,
                         heading_setpoint)

        if not self._leg_pure_pursuit and self._leg_anchor is not None:
            ax, ay = self._leg_anchor
            alpha = self._leg_alpha
            e_ct = -(px - ax) * np.sin(alpha) + (py - ay) * np.cos(alpha)
            # ILOS integrator: freeze when far off the line (anti-windup)
            if abs(e_ct) < 3.0:
                self._e_int = float(np.clip(self._e_int + e_ct * self._dt,
                                            -self.c['casc_e_int_max'],
                                            self.c['casc_e_int_max']))
            chi_d = alpha + float(np.arctan(-(e_ct + self.c['casc_kappa'] * self._e_int)
                                            / self.c['casc_delta_h']))
        else:
            e_ct = 0.0
            chi_d = heading_setpoint

        # sideslip compensation: command heading = course - beta
        if abs(u_meas) > 0.15:
            beta = float(np.clip(np.arctan2(v_meas, u_meas), -0.35, 0.35))
            self._beta_lpf += (beta - self._beta_lpf) * min(1.0, self._dt / 2.0)
        yaw_ref = chi_d - self._beta_lpf

        yaw_error = _wrap(yaw_ref - current_heading)
        r_ref = float(np.clip(self.c['casc_yaw_kp'] * yaw_error,
                              -self.c['casc_r_max'], self.c['casc_r_max']))
        u_tv_rudder, _, _ = self._r_pid.get_control(r_meas, r_ref, self._dt)

        # ================= VERTICAL: depth -> pitch -> q -> stern ================
        depth_error = depth_setpoint - current_depth
        dz_eff = self.c['casc_delta_z'] * max(1.0, abs(u_meas) / max(self.c['casc_uref'], 1e-3))
        # slow outer integral kills the standing bias seen in the blend sweep
        self._z_int = getattr(self, '_z_int', 0.0)
        if abs(depth_error) < 1.0:  # only integrate near the setpoint
            self._z_int = float(np.clip(self._z_int + self.c['casc_z_ki'] * depth_error * self._dt,
                                        -0.15, 0.15))
        pitch_ref_dyn = float(np.clip(np.arctan2(depth_error, dz_eff) + self._z_int,
                                      -self.param['max_dive_pitch'],
                                      self.param['max_dive_pitch']))
        # blend the attitude reference: level trim near hover, dive pitch at speed
        pitch_ref = (1.0 - w) * pitch_setpoint + w * pitch_ref_dyn

        pitch_error = pitch_ref - current_pitch
        q_ref = float(np.clip(self.c['casc_pitch_kp'] * pitch_error,
                              -self.c['casc_q_max'], self.c['casc_q_max']))
        u_q, _, _ = self._q_pid.get_control(q_meas, q_ref, self._dt)
        u_tv_stern = -float(u_q) * w  # stern authority ~ u^2; gate by w. Sign per stock.

        # ================= ALLOCATION: speed-nested VBS / LCG ====================
        # integral (trim) always active; P-effort fades with speed.
        kp_v, ki_v = self.param['blend_vbs_kp'], self.param['blend_vbs_ki']
        self._vbs_int += ki_v * depth_error * self._dt
        self._vbs_int = float(np.clip(self._vbs_int, -100.0, 100.0))
        u_vbs = float(np.clip(self.param['vbs_u_neutral'] + self._vbs_int
                              + (1.0 - w) * kp_v * depth_error,
                              self.param['vbs_u_min'], self.param['vbs_u_max']))
        if self._u_vbs_prev is not None:
            step = self._vbs_slew * self._dt
            u_vbs = float(np.clip(u_vbs, self._u_vbs_prev - step, self._u_vbs_prev + step))
        self._u_vbs_prev = u_vbs

        kp_l, ki_l = self.param['lcg_pid_kp'], self.param['lcg_pid_ki']
        lcg_err = pitch_ref - current_pitch
        self._lcg_int += ki_l * lcg_err * self._dt
        self._lcg_int = float(np.clip(self._lcg_int, -50.0, 50.0))
        u_lcg = float(np.clip(self.param['lcg_u_neutral'] + self._lcg_int
                              + (1.0 - w) * kp_l * lcg_err,
                              self.param['lcg_u_min'], self.param['lcg_u_max']))

        mode = "Hover (cascade)" if w < 0.2 else ("Flight (cascade)" if w > 0.8 else "Cascade")
        self._dive_mode = mode

        self._dive_pub.set_vbs(u_vbs)
        self._dive_pub.set_lcg(u_lcg)
        self._dive_pub.set_thrust_vector(float(u_tv_rudder), u_tv_stern)
        self._dive_pub.set_rpm(u_rpm, u_rpm)

        # ================= convenience topics ====================================
        self._ref = Odometry()
        self._ref.pose.pose.position.x = waypoint_odom.position.x
        self._ref.pose.pose.position.y = waypoint_odom.position.y
        self._ref.pose.pose.position.z = waypoint_odom.position.z
        self._ref.pose.pose.orientation = waypoint_odom.orientation

        self._error = ControlError()
        self._error.z = depth_error
        self._error.pitch = pitch_error
        self._error.yaw = yaw_error
        self._error.heading = current_heading
        self._error.distance = current_distance

        self._input = ControlInput()
        self._input.vbs = u_vbs
        self._input.lcg = u_lcg
        self._input.thrustervertical = u_tv_stern
        self._input.thrusterhorizontal = float(u_tv_rudder)
        self._input.thrusterrpm1 = float(u_rpm)
        self._input.thrusterrpm2 = float(u_rpm)

        if current_distance < 0.75:
            self.traj_index += 1
        self._dive_sub.set_current_idx(self.traj_index)

        s = f'\nCASCADE PID INFO idx {self.traj_index}  mode: {mode}  w: {w:.2f}\n'
        s += f'xtrack: {e_ct:+.2f} m (int {self._e_int:+.2f})  chi_d: {np.degrees(chi_d):.1f}  beta: {np.degrees(self._beta_lpf):+.1f} deg\n'
        s += f'yaw err: {np.degrees(yaw_error):+.1f} deg  r: {np.degrees(r_meas):+.1f}->{np.degrees(r_ref):+.1f} deg/s  rudder: {u_tv_rudder:+.3f}\n'
        s += f'depth: {current_depth:.2f}/{depth_setpoint:.2f}  z_int: {np.degrees(self._z_int):+.1f} deg  pitch: {np.degrees(current_pitch):+.1f}->{np.degrees(pitch_ref):+.1f} deg\n'
        s += f'q: {np.degrees(q_meas):+.1f}->{np.degrees(q_ref):+.1f} deg/s  stern: {u_tv_stern:+.3f}  vbs: {u_vbs:.1f} (int {self._vbs_int:+.1f})  lcg: {u_lcg:.1f}\n'
        s += f'surge: {u_meas:.2f}/{surge_ref:.2f} (cruise {u_cruise:.2f})  rpm: {u_rpm:.0f}  dist: {current_distance:.1f}\n'
        self._loginfo(s)
        return
