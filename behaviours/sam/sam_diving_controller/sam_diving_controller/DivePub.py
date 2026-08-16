#!/usr/bin/python3
import sys
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Float64, Bool
from smarc_msgs.msg import ThrusterRPM, PercentStamped
from smarc_control_msgs.msg import Topics as ControlTopics
from smarc_control_msgs.msg import MissionEvent 
from sam_msgs.msg import Topics as SamTopics
from sam_msgs.msg import ThrusterAngles, ThrusterRPMs

from sam_diving_controller.IDivePub import ActuatorStates
from sam_diving_controller.neutral_handoff import neutral_handoff

from .ParamUtils import DivingModelParam

try:
    from .IDivePub import IDivePub, MissionStates
except:
    from IDivePub import IDivePub, MissionStates

class DivePub(IDivePub):
    """
    Implements the simple interface we defined in IDiveView for the SAM AUV.
    """
    def __init__(self, node: Node, dive_sub, param) -> None:

        self._node = node
        self._dive_sub = dive_sub
        self.param = param

        self._actuator_state = ActuatorStates.NEUTRAL
        self.neutral_pub_count = 0

        # Publishers
        self._vbs_pub = node.create_publisher(PercentStamped, SamTopics.VBS_CMD_TOPIC, 10)
        self._lcg_pub = node.create_publisher(PercentStamped, SamTopics.LCG_CMD_TOPIC, 10)
        self._rpm1_pub = node.create_publisher(ThrusterRPM, SamTopics.THRUSTER1_CMD_TOPIC, 10)
        self._rpm2_pub = node.create_publisher(ThrusterRPM, SamTopics.THRUSTER2_CMD_TOPIC, 10)
        self.thrust_rpms_pub = node.create_publisher(ThrusterRPMs, "core/thruster_rpms_cmd", qos_profile=10)
        self._thrust_vector_pub = node.create_publisher(ThrusterAngles, SamTopics.THRUST_VECTOR_CMD_TOPIC, 10)
        self._joy_thrust_vector_pub = node.create_publisher(Float64, ControlTopics.ELEVATOR_PID_CTRL, 10)
        self._joy_assisted_driving_pub = node.create_publisher(Bool, ControlTopics.ASSIST_ENABLE, qos_profile=10)
        self._mission_event_pub = node.create_publisher(MissionEvent, "ctrl/conv/mission_event", qos_profile=10)

        # Messages
        self._vbs_msg = PercentStamped()
        self._lcg_msg = PercentStamped()
        self._t1_msg = ThrusterRPM()
        self._t2_msg = ThrusterRPM()
        self.rpm_msg = ThrusterRPMs()
        self._thrust_vector_msg = ThrusterAngles()
        self._joy_tv_msg = Float64()

        self._vbs_msg.value = self.param['vbs_u_neutral']
        self._loginfo(f"{self._vbs_msg.value}")
        self._lcg_msg.value = self.param['lcg_u_neutral']
        self._thrust_vector_msg.thruster_horizontal_radians = self.param['tv_u_neutral']
        self._thrust_vector_msg.thruster_vertical_radians = self.param['tv_u_neutral']
        self._t1_msg.rpm = self.param['rpm_u_neutral']
        self._t2_msg.rpm = self.param['rpm_u_neutral']

    def _loginfo(self, s):
        self._node.get_logger().info(s)


    def set_vbs(self, vbs: float) -> None:
        """
        Set vbs
        """
        self._vbs_msg.value = float(vbs)
        now = self._node.get_clock().now()
        self._vbs_msg.header.stamp = now.to_msg()


    def set_lcg(self, lcg: float) -> None:
        """
        Set LCG
        """
        self._lcg_msg.value = float(lcg)
        now = self._node.get_clock().now()
        self._lcg_msg.header.stamp = now.to_msg()


    def set_rpm(self, rpm1: float, rpm2: float) -> None:
        """
        Set RPMs
        """
        self._t1_msg.rpm = int(rpm1)
        self._t2_msg.rpm = int(rpm2)
        self.rpm_msg.thruster_1_rpm = int(rpm1)
        self.rpm_msg.thruster_2_rpm = int(rpm2)

        now = self._node.get_clock().now()
        self.rpm_msg.header.stamp = now.to_msg()


    def set_thrust_vector(self, horizontal_tv: float, vertical_tv: float) -> None:
        """
        Set thrust vector
        """
        self._thrust_vector_msg.thruster_horizontal_radians = float(horizontal_tv)
        self._thrust_vector_msg.thruster_vertical_radians = float(vertical_tv)

        now = self._node.get_clock().now()
        self._thrust_vector_msg.header.stamp = now.to_msg()


    def set_stern(self, u_tv_ver):
        self._joy_tv_msg.data = float(u_tv_ver)


    def set_actuator_states(self, actuator_state, node_name):

        old_state = self._actuator_state
        self._actuator_state = actuator_state

        if self._actuator_state != old_state:
            self._loginfo(f"DiveController state: from {node_name}: {old_state} --> {self._actuator_state}")

    def get_actuator_states(self):
        return self._actuator_state

    def publish_mission_event(self, commanding, mission_state: MissionStates):
        
        if commanding:
            details = "Start commanding"
        else:
            details = "Stop commanding"

        event_msg = MissionEvent()
        now = self._node.get_clock().now()
        event_msg.header.stamp = now.to_msg()
        event_msg.commanding = commanding
        event_msg.details = details
        event_msg.mission_state = f"{mission_state}"

        self._mission_event_pub.publish(event_msg)




    def _surface_depth_m(self):
        """The vehicle's own depth, positive down, or None if it has not said.

        Wrapped and defensive on purpose: this runs on the actuator publish path, and a controller
        that throws here stops publishing entirely -- which is the one failure worse than the one
        being fixed. An exception degrades to "not reported", which the hand-off treats as
        unconfirmed rather than as arrival.
        """
        try:
            d = self._dive_sub.get_depth()
        except Exception:
            return None
        return float(d) if d is not None else None

    def _vbs_feedback_pct(self):
        """VBS as the VEHICLE reports it (SamTopics.VBS_FB_TOPIC), never as we commanded it.

        Reading back our own command would make the check vacuous -- it would confirm that we said
        zero, which was never in doubt. The whole question is whether the tank got there. Same
        rule as spec invariant 11: a probe must read something its own node does not write.
        """
        try:
            v = self._dive_sub.get_control_input()['vbs']
        except Exception:
            return None
        return float(v) if v is not None else None

    def update(self) -> None:
        """
        Publish all actuator values
        """
        if self._actuator_state == ActuatorStates.DISENGAGED:
            return
        
        if self._actuator_state == ActuatorStates.NEUTRAL:
            self._vbs_pub.publish(self._vbs_msg)
            self._lcg_pub.publish(self._lcg_msg)
            self._rpm1_pub.publish(self._t1_msg)
            self._rpm2_pub.publish(self._t2_msg)
            self.thrust_rpms_pub.publish(self.rpm_msg)
            self._thrust_vector_pub.publish(self._thrust_vector_msg)

            self.neutral_pub_count += 1

            # LETTING GO IS A CLAIM THAT THE VEHICLE IS SAFE, AND ONLY THE VEHICLE CAN SUPPORT IT.
            #
            # This used to be `if self.neutral_pub_count > 20: DISENGAGED` -- twenty publishes,
            # one or two seconds, and then silence. A VBS tank does not empty in two seconds, so
            # the command to empty was withdrawn mid-purge and the tank stopped wherever it had
            # got to. That is the ~45% Ivan has been seeing on the HUD after every mission: not a
            # held setpoint, an interrupted one. See neutral_handoff.py's header for the full
            # chain and why a tick count can never answer this question.
            verdict = neutral_handoff(
                ticks=self.neutral_pub_count,
                depth_m=self._surface_depth_m(),
                vbs_pct=self._vbs_feedback_pct(),
                vbs_target_pct=self.param['vbs_u_neutral'],
                min_ticks=self.param.get('neutral_min_ticks', 20),
                max_ticks=self.param.get('neutral_max_ticks', 600),
                surface_depth_m=self.param.get('neutral_surface_depth_m', 0.35),
                vbs_tol_pct=self.param.get('neutral_vbs_tol_pct', 5.0),
            )
            self._loginfo(f"NEUTRAL hand-off: {verdict.reason}")
            if verdict.release:
                if not verdict.confirmed:
                    # Loud, because an unconfirmed release means the vehicle is being left with
                    # nobody commanding it while it may still be under. That is a thing to find in
                    # a log, not a thing to infer later from a screenshot of the HUD.
                    self._node.get_logger().warn(
                        f"Releasing actuators WITHOUT confirming the vehicle surfaced: "
                        f"{verdict.reason}")
                self.neutral_pub_count = 0
                self.set_actuator_states(ActuatorStates.DISENGAGED, "DP")

        else:
            self._vbs_pub.publish(self._vbs_msg)
            self._lcg_pub.publish(self._lcg_msg)
            self._rpm1_pub.publish(self._t1_msg)
            self._rpm2_pub.publish(self._t2_msg)
            self.thrust_rpms_pub.publish(self.rpm_msg)
            self._thrust_vector_pub.publish(self._thrust_vector_msg)

    def joy_update(self):
        """
        Publish all actuator values
        """
        self._joy_assisted_driving_msg = Bool()
        self._joy_assisted_driving_msg.data = True
        self._vbs_pub.publish(self._vbs_msg)
        self._lcg_pub.publish(self._lcg_msg)
        self._joy_thrust_vector_pub.publish(self._joy_tv_msg)
        self._joy_assisted_driving_pub.publish(self._joy_assisted_driving_msg)

