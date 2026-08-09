#!/usr/bin/python3
"""
Blend-allocation dive controller — Session C, 2026-08-09.

A separate controller, NOT a modification of DiveControllerPID, so the BT (and later
Mission Control, or the AUV itself mid-mission) can switch between the two and A/B them
with nothing but launch/action selection. Same DiveControllerInterface, same DiveSub/
DivePub, same PID gains; only the structure differs.

What it fixes (docs/dive-controller-limit-cycle.md, measured 2026-08-07 and reproduced
in both 2026-08-09 estimator A/B runs):

  The stock controller switches static/active on a bare |depth_error| <= 0.5 threshold
  and, in active mode, dumps VBS/LCG to neutral. A positively buoyant vehicle therefore
  floats out of the band the moment it enters it, and parks ON the switching boundary
  (1.3-1.8 m band against a 2.0 m setpoint = setpoint minus threshold).

The design (docs/waypoint-control-design-proposal.md §3): there are no modes. Static
(VBS/LCG) and dynamic (stern plane + speed) depth control are complementary bandwidths,
blended on measured surge speed:

    w = sat((u - u_lo) / (u_hi - u_lo))            # 0 = hover, 1 = flight

    VBS         = depth PI output                  # ALWAYS running; its integral IS the
                                                   # buoyancy trim and never gets dumped
    stern plane = w * pitch PID output             # authority ~ u^2, so gate on speed
    LCG         = pitch trim PI                    # always running, never dumped
    props       = surge PI on the braking profile  # always running

There is no threshold to chatter across: w moves with speed, which moves slowly. At
u -> 0 the vehicle degrades into a hovering VBS regulator instead of losing depth
control. A discrete label is derived from w for logging only ("Hover"/"Blended"/
"Flight") — a label, not a controller input.
"""
from sam_diving_controller.controllers.PIDControl import PIDControl
from sam_diving_controller.IDivePub import MissionStates, ActuatorStates
from sam_diving_controller.controllers.DiveControllerInterface import DiveControllerInterface
from smarc_control_msgs.msg import ControlError, ControlInput
from nav_msgs.msg import Odometry

import numpy as np


class DiveControllerBlendPID(DiveControllerInterface):

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

        # Blend-specific depth gains (measured 2026-08-09, run 120402): the stock
        # Ki=5 (integral time 4 s) against the ~2-minute heave mode produced a
        # ±1 m, 122 s oscillation. Depth loop bandwidth must sit well below the
        # pitch/heave dynamics: small Ki, and a pump-realistic slew limit on VBS.
        self._depth_vbs_pid = PIDControl(Kp=self.param['blend_vbs_kp'],
                                         Ki=self.param['blend_vbs_ki'],
                                         Kd=self.param['blend_vbs_kd'],
                                         Kaw=self.param['vbs_pid_kaw'],
                                         u_neutral=self.param['vbs_u_neutral'],
                                         u_min=self.param['vbs_u_min'],
                                         u_max=self.param['vbs_u_max'])
        self._vbs_slew = self.param['blend_vbs_slew']  # %/s, pump-limited
        self._u_vbs_prev = None
        self._pitch_lcg_pid = PIDControl(Kp=self.param['lcg_pid_kp'],
                                         Ki=self.param['lcg_pid_ki'],
                                         Kd=self.param['lcg_pid_kd'],
                                         Kaw=self.param['lcg_pid_kaw'],
                                         u_neutral=self.param['lcg_u_neutral'],
                                         u_min=self.param['lcg_u_min'],
                                         u_max=self.param['lcg_u_max'])
        self._pitch_tv_pid = PIDControl(Kp=self.param['tv_pid_kp'],
                                        Ki=self.param['tv_pid_ki'],
                                        Kd=self.param['tv_pid_kd'],
                                        Kaw=self.param['tv_pid_kaw'],
                                        u_neutral=self.param['tv_u_neutral'],
                                        u_min=self.param['tv_u_min'],
                                        u_max=self.param['tv_u_max'])
        self._yaw_pid = PIDControl(Kp=self.param['yaw_pid_kp'],
                                   Ki=self.param['yaw_pid_ki'],
                                   Kd=self.param['yaw_pid_kd'],
                                   Kaw=self.param['yaw_pid_kaw'],
                                   u_neutral=self.param['yaw_u_neutral'],
                                   u_min=self.param['yaw_u_min'],
                                   u_max=self.param['yaw_u_max'])
        # Velocity controller (2026-08-09, after Ivan's reference run plateaued at
        # 0.34 m/s): RPM = kff*u_d + PI trim. The PI only trims around the
        # feedforward, so its authority is bounded; the final command is clamped to
        # the vehicle's rpm limits (the old hardcoded 450 cap was the plateau).
        self._surge_rpm_pid = PIDControl(Kp=100.0, Ki=5.25, Kd=1.5, Kaw=1.0,
                                         u_neutral=0, u_min=-300, u_max=300)

        # Integral hold-off, v2 (v1 zeroed the integral until arrival and the
        # vehicle never sank — the integral IS what finds the sink trim, measured
        # run 131212_dive_vel_0p5 stuck at 0.1 m). v2: the integrator RUNS while the
        # vehicle is not making progress (so it can find the trim), HOLDS its value
        # while depth is actually moving toward the setpoint (that transit is where
        # the windup came from), and runs normally once within blend_int_holdoff.
        self._int_released = self.param['blend_int_holdoff'] <= 0.0

        # Depth-rate estimate for the hold-off progress test.
        self._depth_prev = None
        self._depth_rate_lpf = 0.0

        # Low-passed sideslip estimate for course compensation.
        self._beta_lpf = 0.0

        # Protective stop (2026-08-10): subscribe the obstacle detector's stop flag.
        # Launch-gated (blend_obstacle_stop, default False) so hardware and every
        # existing launch are bit-for-bit unchanged. On a fresh True flag the surge
        # reference is forced to zero (the surge PI actively brakes); depth, pitch
        # and heading loops keep running, so the vehicle holds depth where it is
        # instead of pinning on the wall with props turning (bags 143736, 165932).
        self._obstacle_stop_flag = False
        self._obstacle_stop_stamp = None
        self._obstacle_stop_logged = False
        if self.param['blend_obstacle_stop']:
            from std_msgs.msg import Bool as _Bool
            self._node.create_subscription(
                _Bool, self.param['blend_obstacle_stop_topic'],
                self._obstacle_stop_cb, 1)
            self._loginfo("Protective stop ENABLED, listening on "
                          f"{self.param['blend_obstacle_stop_topic']}")

        self._loginfo("Blend Dive Controller created")

    def _obstacle_stop_cb(self, msg):
        self._obstacle_stop_flag = bool(msg.data)
        self._obstacle_stop_stamp = self._node.get_clock().now()

    def _obstacle_hold(self):
        """True while a FRESH stop flag is raised. A stale flag (detector dead >2 s)
        does not hold the vehicle — same fail-open policy as the detector's own
        watchdog; revisit for hardware (session doc)."""
        if not self.param['blend_obstacle_stop'] or not self._obstacle_stop_flag:
            return False
        if self._obstacle_stop_stamp is None:
            return False
        age = (self._node.get_clock().now() - self._obstacle_stop_stamp).nanoseconds * 1e-9
        return age < 2.0

    def _u_cruise_live(self):
        """blend_u_cruise is read live so runs can be set up with `ros2 param set`
        without restarting the bringup."""
        try:
            return float(self._node.get_parameter('blend_u_cruise')
                         .get_parameter_value().double_value)
        except Exception:
            return self.param['blend_u_cruise']

    def _blend_weight(self, current_surge):
        """0 = pure static (VBS), 1 = pure dynamic (stern plane). Proposal §3."""
        u_lo = self.param['blend_u_lo']
        u_hi = self.param['blend_u_hi']
        if u_hi <= u_lo:
            return 0.0
        return float(np.clip((np.abs(current_surge) - u_lo) / (u_hi - u_lo), 0.0, 1.0))

    def update(self):
        mission_state = self._dive_sub.get_mission_state()

        self._loginfo_once(f"DC(blend): {mission_state}")

        if mission_state in (MissionStates.RECEIVED,
                             MissionStates.COMPLETED,
                             MissionStates.CANCELLED):
            self._loginfo_once(f"Mission {mission_state} — actuators neutral")
            self._set_actuators_neutral()
            return

        self._dive_pub.set_actuator_states(ActuatorStates.ENGAGED, "DP")

        # --- references -----------------------------------------------------------
        depth_setpoint = self._dive_sub.get_depth_setpoint()
        pitch_setpoint = self._dive_sub.get_pitch_setpoint()
        dive_pitch_setpoint = self._dive_sub.get_dive_pitch()
        heading_setpoint = self._dive_sub.get_heading_setpoint()
        waypoint_odom = self._dive_sub.get_waypoint_in_odom()
        waypoint_global = self._dive_sub.get_waypoint()

        # --- state ----------------------------------------------------------------
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

        current_depth = np.abs(current_depth)
        depth_setpoint = np.abs(depth_setpoint)

        # Velocity reference (proposal §4): cruise setpoint bounded by the braking
        # profile. a_brake is a parameter now — the old 0.005 capped a 59 m leg at
        # 0.77 m/s and made any cruise setpoint above that unreachable.
        u_cruise = self._u_cruise_live()
        a_brake = self.param['blend_a_brake']
        d_eps = 0.5  # m
        surge_ref = min(u_cruise,
                        float(np.sqrt(2.0 * a_brake * (current_distance + d_eps))))
        current_surge = self._current_state.twist.twist.linear.x

        # Protective stop: obstacle inside the detector's envelope -> surge ref 0.
        # The surge PI drives rpm through zero (active braking); VBS/LCG/rudder
        # loops keep running so depth is held during and after the stop.
        obstacle_hold = self._obstacle_hold()
        if obstacle_hold:
            surge_ref = 0.0
            if not self._obstacle_stop_logged:
                self._node.get_logger().warn(
                    "OBSTACLE PROTECTIVE STOP — surge ref forced to 0, holding depth "
                    f"(distance to wp {current_distance:.1f} m)")
                self._obstacle_stop_logged = True
        elif self._obstacle_stop_logged:
            self._node.get_logger().info("Obstacle stop cleared — resuming mission")
            self._obstacle_stop_logged = False

        # --- allocation: blend, don't switch --------------------------------------
        w = self._blend_weight(current_surge)

        # Label for logging/BT only — never a controller input.
        if w < 0.2:
            self._dive_mode = "Hover (blend)"
        elif w > 0.8:
            self._dive_mode = "Flight (blend)"
        else:
            self._dive_mode = "Blended"

        # Speed-scaled dive pitch: get_dive_pitch() uses a fixed 3 m vertical
        # lookahead, which is 3x more aggressive at 1.5 m/s than at 0.5. Scaling the
        # angle by min(1, uref/u) keeps the commanded heave rate (u*theta) roughly
        # speed-invariant. Equivalent to lookahead ∝ speed (proposal §1 vertical).
        uref = self.param['blend_dive_pitch_uref']
        dp_scale = min(1.0, uref / max(abs(current_surge), 1e-3))
        # Cap at max_dive_pitch — get_dive_pitch() itself does not (measured 0.543 rad
        # against a 0.349 limit in run 133216).
        eff_dive_pitch = float(np.clip(dive_pitch_setpoint * dp_scale,
                                       -self.param['max_dive_pitch'],
                                       self.param['max_dive_pitch']))

        # One pitch reference for both pitch actuators, so LCG and stern plane never
        # fight: level/trim attitude when slow (pitching down without speed produces
        # no heave), guidance dive pitch when at speed.
        #
        # SIGN (measured, run 133216_dive_vel_1p0): dive_pitch is POSITIVE when too
        # shallow and positive pitch is nose-DOWN, so the reference takes dive_pitch
        # directly. The stock controller's `-1.0 *` inversion put the vehicle
        # nose-up while VBS sat at 100% and the hull planed at 0.19 m — stock never
        # saw this because its active mode only runs near the setpoint where
        # dive_pitch is ~0. (The actuator-side sign flip on u_tv_stern is separate
        # and kept as in stock.)
        pitch_ref = (1.0 - w) * pitch_setpoint + w * eff_dive_pitch

        # Depth-rate estimate (LPF tau ~1 s) for the hold-off progress test.
        if self._depth_prev is not None and self._dt > 0:
            raw_rate = (current_depth - self._depth_prev) / self._dt
            self._depth_rate_lpf += (raw_rate - self._depth_rate_lpf) * min(1.0, self._dt / 1.0)
        self._depth_prev = current_depth

        # Depth PI on VBS: always running. The integral is the buoyancy trim.
        i_prev = self._depth_vbs_pid._integral
        aw_prev = self._depth_vbs_pid._anti_windup
        u_vbs, depth_error, u_vbs_raw = self._depth_vbs_pid.get_control(
            current_depth, depth_setpoint, self._dt)

        # Integral hold-off v2: HOLD (not zero) the integral while depth is already
        # moving toward the setpoint at a real rate — progress means no more
        # integral is needed, and accumulating through the whole descent is exactly
        # the windup that caused the first-dive overshoot. Integrate normally when
        # stuck (to find the trim) and after first arrival (released).
        if not self._int_released:
            if np.abs(depth_error) < self.param['blend_int_holdoff']:
                self._int_released = True
                self._loginfo("Depth integral released (error %.2f m)" % depth_error)
            elif depth_error * self._depth_rate_lpf > 0.0 and np.abs(self._depth_rate_lpf) > 0.03:
                # moving toward the setpoint -> hold the integrator at its value
                self._depth_vbs_pid._integral = i_prev
                self._depth_vbs_pid._anti_windup = aw_prev

        # Pump-limited slew (proposal §5: ~5 %/s stops bang-bang and overshoot).
        if self._u_vbs_prev is not None:
            step = self._vbs_slew * self._dt
            u_vbs = float(np.clip(u_vbs, self._u_vbs_prev - step, self._u_vbs_prev + step))
        self._u_vbs_prev = u_vbs

        # Pitch trim on LCG: always running, never dumped to neutral.
        u_lcg, pitch_error, u_lcg_raw = self._pitch_lcg_pid.get_control(
            current_pitch, pitch_ref, self._dt)

        # Stern plane: dynamic depth/pitch authority, gated by speed.
        u_tv_stern_full, _, u_tv_stern_raw = self._pitch_tv_pid.get_control(
            current_pitch, pitch_ref, self._dt)
        u_tv_stern = w * u_tv_stern_full

        # Course compensation (proposal §1: "command course, not heading"). The
        # vehicle travels along chi = psi + beta; with pure pursuit aiming the
        # heading at the waypoint, a steady sideslip shows up as the constant few
        # degrees of bearing offset seen in every run. Command psi = bearing - beta,
        # with beta measured from the DVL/estimator body velocity, low-passed
        # (tau 2 s), gated on real forward speed, clamped to +/-20 deg.
        yaw_ref = heading_setpoint
        if self.param['blend_course_comp']:
            vy = self._current_state.twist.twist.linear.y
            if np.abs(current_surge) > 0.15:
                beta = float(np.arctan2(vy, current_surge))
                beta = float(np.clip(beta, -0.35, 0.35))
                self._beta_lpf += (beta - self._beta_lpf) * min(1.0, self._dt / 2.0)
            yaw_ref = heading_setpoint - self._beta_lpf

        # Heading on rudder: always running.
        u_tv_rudder, yaw_error, u_tv_rudder_raw = self._yaw_pid.get_control(
            current_heading, yaw_ref, self._dt)

        # Surge on props: feedforward + PI trim, clamped to the vehicle rpm limits.
        pi_trim, surge_error, u_rpm_raw = self._surge_rpm_pid.get_control(
            current_surge, surge_ref, self._dt)
        u_rpm = float(np.clip(self.param['blend_rpm_kff'] * surge_ref + pi_trim,
                              self.param['rpm_u_min'], self.param['rpm_u_max']))

        # Sketchy minus sign kept from stock: positive steering angle compensates
        # a negative pitch.
        u_tv_stern = -u_tv_stern

        self._dive_pub.set_vbs(u_vbs)
        self._dive_pub.set_lcg(u_lcg)
        self._dive_pub.set_thrust_vector(u_tv_rudder, u_tv_stern)
        self._dive_pub.set_rpm(u_rpm, u_rpm)

        # --- convenience topics ---------------------------------------------------
        self._ref = Odometry()
        self._ref.pose.pose.position.x = waypoint_odom.position.x
        self._ref.pose.pose.position.y = waypoint_odom.position.y
        self._ref.pose.pose.position.z = waypoint_odom.position.z
        self._ref.pose.pose.orientation.w = waypoint_odom.orientation.w
        self._ref.pose.pose.orientation.x = waypoint_odom.orientation.x
        self._ref.pose.pose.orientation.y = waypoint_odom.orientation.y
        self._ref.pose.pose.orientation.z = waypoint_odom.orientation.z

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
        self._input.thrusterhorizontal = u_tv_rudder
        self._input.thrusterrpm1 = float(u_rpm)
        self._input.thrusterrpm2 = float(u_rpm)

        if current_distance < 0.75:
            self.traj_index += 1

        self._dive_sub.set_current_idx(self.traj_index)

        # Debug — w is the number that makes "it wobbles" tunable (proposal §7).
        s = f'\nBLEND PID INFO for index {self.traj_index}\n'
        s += f'mode: {self._dive_mode}  w: {w:.2f}  surge: {current_surge:.3f}  surge_ref: {surge_ref:.3f}  u_cruise: {u_cruise:.2f}\n'
        if obstacle_hold:
            s += 'OBSTACLE HOLD ACTIVE\n'
        s += f'beta: {np.degrees(self._beta_lpf):.1f} deg  int_released: {self._int_released}  dp_scale: {dp_scale:.2f}\n'
        s += f'depth: {current_depth:.3f}  setpoint: {depth_setpoint:.3f}  error: {depth_error:.3f}\n'
        s += f'pitch: {current_pitch:.3f}  pitch_ref: {pitch_ref:.3f}  dive_pitch: {dive_pitch_setpoint:.3f}\n'
        s += f'heading: {current_heading:.3f}  setpoint: {heading_setpoint:.3f}  yaw error: {yaw_error:.3f}\n'
        s += f'vbs: {u_vbs:.3f} (raw {u_vbs_raw:.3f})  lcg: {u_lcg:.3f}\n'
        s += f'tv stern: {u_tv_stern:.3f}  tv rudder: {u_tv_rudder:.3f}  rpm: {u_rpm:.1f}\n'
        s += f'distance: {current_distance:.3f}\n'
        self._loginfo(s)

        return
