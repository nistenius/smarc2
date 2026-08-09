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
        # Same surge gains the stock controller hardcodes; kept identical on purpose
        # so the A/B difference is structure, not tuning.
        self._surge_rpm_pid = PIDControl(Kp=100.0, Ki=5.25, Kd=1.5, Kaw=1.0,
                                         u_neutral=0, u_min=-400, u_max=450)

        self._loginfo("Blend Dive Controller created")

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

        # Braking profile, same constants as stock (tightened 2026-08-07).
        a_brake = 0.005  # m/s^2
        d_eps = 0.5      # m
        surge_ref = np.sqrt(2.0 * a_brake * (current_distance + d_eps))
        current_surge = self._current_state.twist.twist.linear.x

        # --- allocation: blend, don't switch --------------------------------------
        w = self._blend_weight(current_surge)

        # Label for logging/BT only — never a controller input.
        if w < 0.2:
            self._dive_mode = "Hover (blend)"
        elif w > 0.8:
            self._dive_mode = "Flight (blend)"
        else:
            self._dive_mode = "Blended"

        # One pitch reference for both pitch actuators, so LCG and stern plane never
        # fight: level/trim attitude when slow (pitching down without speed produces
        # no heave), guidance dive pitch when at speed.
        pitch_ref = (1.0 - w) * pitch_setpoint + w * (-1.0 * dive_pitch_setpoint)

        # Depth PI on VBS: always running. The integral is the buoyancy trim.
        u_vbs, depth_error, u_vbs_raw = self._depth_vbs_pid.get_control(
            current_depth, depth_setpoint, self._dt)

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

        # Heading on rudder: always running.
        u_tv_rudder, yaw_error, u_tv_rudder_raw = self._yaw_pid.get_control(
            current_heading, heading_setpoint, self._dt)

        # Surge on props: always running, braking profile toward the waypoint.
        u_rpm, surge_error, u_rpm_raw = self._surge_rpm_pid.get_control(
            current_surge, surge_ref, self._dt)

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
        s += f'mode: {self._dive_mode}  w: {w:.2f}  surge: {current_surge:.3f}  surge_ref: {surge_ref:.3f}\n'
        s += f'depth: {current_depth:.3f}  setpoint: {depth_setpoint:.3f}  error: {depth_error:.3f}\n'
        s += f'pitch: {current_pitch:.3f}  pitch_ref: {pitch_ref:.3f}  dive_pitch: {dive_pitch_setpoint:.3f}\n'
        s += f'heading: {current_heading:.3f}  setpoint: {heading_setpoint:.3f}  yaw error: {yaw_error:.3f}\n'
        s += f'vbs: {u_vbs:.3f} (raw {u_vbs_raw:.3f})  lcg: {u_lcg:.3f}\n'
        s += f'tv stern: {u_tv_stern:.3f}  tv rudder: {u_tv_rudder:.3f}  rpm: {u_rpm:.1f}\n'
        s += f'distance: {current_distance:.3f}\n'
        self._loginfo(s)

        return
