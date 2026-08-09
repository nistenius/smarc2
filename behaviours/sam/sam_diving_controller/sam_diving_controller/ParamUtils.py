#!/usr/bin/python3

import rclpy
import sys

class DivingModelParam():

    def __init__(self, node):

        self._node = node

        self._node.declare_parameter('vbs_pid_kp', 40.0) 
        self._node.declare_parameter('vbs_pid_ki', 5.0)
        self._node.declare_parameter('vbs_pid_kd', 1.0)
        self._node.declare_parameter('vbs_pid_kaw', 1.0)
        self._node.declare_parameter('vbs_u_neutral', 0.0)
        self._node.declare_parameter('vbs_u_min', 0.0)
        self._node.declare_parameter('vbs_u_max', 100.0)
        self._node.declare_parameter('vbs_u_emergency', 0.0)

        self._node.declare_parameter('lcg_pid_kp', 40.0)
        self._node.declare_parameter('lcg_pid_ki', 5.0)
        self._node.declare_parameter('lcg_pid_kd', 1.0)
        self._node.declare_parameter('lcg_pid_kaw', 1.0)
        self._node.declare_parameter('lcg_u_neutral', 50.0)
        self._node.declare_parameter('lcg_u_min', 0.0)
        self._node.declare_parameter('lcg_u_max', 100.0)
        self._node.declare_parameter('lcg_u_emergency', 50.0)

        self._node.declare_parameter('tv_pid_kp', 1.0)
        self._node.declare_parameter('tv_pid_ki', 0.25)
        self._node.declare_parameter('tv_pid_kd', 0.5)
        self._node.declare_parameter('tv_pid_kaw', 1.0)
        self._node.declare_parameter('tv_u_neutral', 0.0)
        self._node.declare_parameter('tv_u_min', -0.122173)
        self._node.declare_parameter('tv_u_max', 0.122173)
        self._node.declare_parameter('tv_u_emergency', 0.0)

        self._node.declare_parameter('yaw_pid_kp', 2.5)
        self._node.declare_parameter('yaw_pid_ki', 0.25)
        self._node.declare_parameter('yaw_pid_kd', 0.5)
        self._node.declare_parameter('yaw_pid_kaw', 1.0)
        self._node.declare_parameter('yaw_u_neutral', 0.0)
        self._node.declare_parameter('yaw_u_min', -0.122173)
        self._node.declare_parameter('yaw_u_max', 0.122173)

        self._node.declare_parameter('rpm_u_neutral', 0)
        self._node.declare_parameter('rpm_u_min', -400)
        self._node.declare_parameter('rpm_u_max', 800)
        self._node.declare_parameter('rpm_u_emergency', 0)

        self._node.declare_parameter('max_dive_pitch', 0.349)

        # Blend allocation (DiveControllerBlendPID only; ignored by the stock PID).
        # Speed band over which depth authority hands over from VBS (static) to
        # stern plane + pitch (dynamic). See docs/waypoint-control-design-proposal.md §3.
        self._node.declare_parameter('blend_u_lo', 0.2)
        self._node.declare_parameter('blend_u_hi', 0.6)
        # Blend depth loop: slow integral (Ti = Kp/Ki = 40 s) + pump-limited slew.
        self._node.declare_parameter('blend_vbs_kp', 20.0)
        self._node.declare_parameter('blend_vbs_ki', 0.5)
        self._node.declare_parameter('blend_vbs_kd', 0.0)
        self._node.declare_parameter('blend_vbs_slew', 5.0)
        # Velocity control + tightened setpoint following (2026-08-09 pm).
        self._node.declare_parameter('blend_u_cruise', 0.5)        # m/s, read LIVE
        self._node.declare_parameter('blend_a_brake', 0.05)        # m/s^2
        self._node.declare_parameter('blend_rpm_kff', 1300.0)      # rpm per m/s (measured: 450 rpm -> 0.34 m/s)
        self._node.declare_parameter('blend_int_holdoff', 0.3)     # m; 0 disables
        self._node.declare_parameter('blend_course_comp', True)    # sideslip comp
        self._node.declare_parameter('blend_dive_pitch_uref', 0.5) # m/s
        # Protective stop (2026-08-10 obstacle-avoidance session): obey the
        # sam_perception obstacle detector's stop flag. Default OFF -> hardware and
        # every existing launch unchanged; blend_sim.yaml turns it on in sim.
        self._node.declare_parameter('blend_obstacle_stop', False)
        self._node.declare_parameter('blend_obstacle_stop_topic', 'perception/obstacle/stop')

    def get_param(self):

        param = {}

        param['vbs_pid_kp'] = self._node.get_parameter('vbs_pid_kp').get_parameter_value().double_value
        param['vbs_pid_ki'] = self._node.get_parameter('vbs_pid_ki').get_parameter_value().double_value
        param['vbs_pid_kd'] = self._node.get_parameter('vbs_pid_kd').get_parameter_value().double_value
        param['vbs_pid_kaw'] = self._node.get_parameter('vbs_pid_kaw').get_parameter_value().double_value
        param['vbs_u_neutral'] = self._node.get_parameter('vbs_u_neutral').get_parameter_value().double_value
        param['vbs_u_min'] = self._node.get_parameter('vbs_u_min').get_parameter_value().double_value
        param['vbs_u_max'] = self._node.get_parameter('vbs_u_max').get_parameter_value().double_value
        param['vbs_u_emergency'] = self._node.get_parameter('vbs_u_emergency').get_parameter_value().double_value

        param['lcg_pid_kp'] = self._node.get_parameter('lcg_pid_kp').get_parameter_value().double_value
        param['lcg_pid_ki'] = self._node.get_parameter('lcg_pid_ki').get_parameter_value().double_value
        param['lcg_pid_kd'] = self._node.get_parameter('lcg_pid_kd').get_parameter_value().double_value
        param['lcg_pid_kaw'] = self._node.get_parameter('lcg_pid_kaw').get_parameter_value().double_value
        param['lcg_u_neutral'] = self._node.get_parameter('lcg_u_neutral').get_parameter_value().double_value
        param['lcg_u_min'] = self._node.get_parameter('lcg_u_min').get_parameter_value().double_value
        param['lcg_u_max'] = self._node.get_parameter('lcg_u_max').get_parameter_value().double_value
        param['lcg_u_emergency'] = self._node.get_parameter('lcg_u_emergency').get_parameter_value().double_value

        param['tv_pid_kp'] = self._node.get_parameter('tv_pid_kp').get_parameter_value().double_value
        param['tv_pid_ki'] = self._node.get_parameter('tv_pid_ki').get_parameter_value().double_value
        param['tv_pid_kd'] = self._node.get_parameter('tv_pid_kd').get_parameter_value().double_value
        param['tv_pid_kaw'] = self._node.get_parameter('tv_pid_kaw').get_parameter_value().double_value
        param['tv_u_neutral'] = self._node.get_parameter('tv_u_neutral').get_parameter_value().double_value
        param['tv_u_min'] = self._node.get_parameter('tv_u_min').get_parameter_value().double_value
        param['tv_u_max'] = self._node.get_parameter('tv_u_max').get_parameter_value().double_value
        param['tv_u_emergency'] = self._node.get_parameter('tv_u_emergency').get_parameter_value().double_value

        param['yaw_pid_kp'] = self._node.get_parameter('yaw_pid_kp').get_parameter_value().double_value
        param['yaw_pid_ki'] = self._node.get_parameter('yaw_pid_ki').get_parameter_value().double_value
        param['yaw_pid_kd'] = self._node.get_parameter('yaw_pid_kd').get_parameter_value().double_value
        param['yaw_pid_kaw'] = self._node.get_parameter('yaw_pid_kaw').get_parameter_value().double_value
        param['yaw_u_neutral'] = self._node.get_parameter('yaw_u_neutral').get_parameter_value().double_value
        param['yaw_u_min'] = self._node.get_parameter('yaw_u_min').get_parameter_value().double_value
        param['yaw_u_max'] = self._node.get_parameter('yaw_u_max').get_parameter_value().double_value

        param['rpm_u_neutral'] = self._node.get_parameter('rpm_u_neutral').get_parameter_value().integer_value
        param['rpm_u_min'] = self._node.get_parameter('rpm_u_min').get_parameter_value().integer_value
        param['rpm_u_max'] = self._node.get_parameter('rpm_u_max').get_parameter_value().integer_value
        param['rpm_u_emergency'] = self._node.get_parameter('rpm_u_emergency').get_parameter_value().integer_value

        param['max_dive_pitch'] = self._node.get_parameter('max_dive_pitch').get_parameter_value().double_value

        param['blend_u_lo'] = self._node.get_parameter('blend_u_lo').get_parameter_value().double_value
        param['blend_u_hi'] = self._node.get_parameter('blend_u_hi').get_parameter_value().double_value
        param['blend_vbs_kp'] = self._node.get_parameter('blend_vbs_kp').get_parameter_value().double_value
        param['blend_vbs_ki'] = self._node.get_parameter('blend_vbs_ki').get_parameter_value().double_value
        param['blend_vbs_kd'] = self._node.get_parameter('blend_vbs_kd').get_parameter_value().double_value
        param['blend_vbs_slew'] = self._node.get_parameter('blend_vbs_slew').get_parameter_value().double_value
        param['blend_u_cruise'] = self._node.get_parameter('blend_u_cruise').get_parameter_value().double_value
        param['blend_a_brake'] = self._node.get_parameter('blend_a_brake').get_parameter_value().double_value
        param['blend_rpm_kff'] = self._node.get_parameter('blend_rpm_kff').get_parameter_value().double_value
        param['blend_int_holdoff'] = self._node.get_parameter('blend_int_holdoff').get_parameter_value().double_value
        param['blend_course_comp'] = self._node.get_parameter('blend_course_comp').get_parameter_value().bool_value
        param['blend_dive_pitch_uref'] = self._node.get_parameter('blend_dive_pitch_uref').get_parameter_value().double_value
        param['blend_obstacle_stop'] = self._node.get_parameter('blend_obstacle_stop').get_parameter_value().bool_value
        param['blend_obstacle_stop_topic'] = self._node.get_parameter('blend_obstacle_stop_topic').get_parameter_value().string_value

        return param
