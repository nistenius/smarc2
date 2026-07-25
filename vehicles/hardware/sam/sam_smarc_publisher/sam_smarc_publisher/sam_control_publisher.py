import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from smarc_msgs.msg import PercentStamped, ThrusterRPM, ThrusterFeedback, Leak
from sam_msgs.msg import ThrusterAngles
from sam_msgs.msg import Topics as SamTopics

class SAMStandardControlPublisher(Node):
    """
    This node listens to the custom smarc control messages from SAM and republishes them
    to standard control messages for the simulation. It also listens to the feedback messages
    from the simulation and republishes them to the custom SAM feedback topics.

    MANUAL OVERRIDE GATE (Data Cube Vehicle Control, 2026-07-25):
    This node is the single funnel from the SAM stack onto the sim's `core/standard/*_cmd`
    topics, which makes it the correct place to arbitrate manual operator control. The
    Data Cube unity_bridge streams the operator's manual actuator setpoints directly to the
    standard topics while ARMed; without arbitration the stack's idle output (e.g. the diving
    controller's neutral commands) interleaves with them and the actuators jitter -- two
    publishers fighting on the same topic.

    While a fresh `True` has been seen on `core/standard/manual_override` (std_msgs/Bool,
    published at ~5 Hz by unity_bridge while it streams), this node DROPS the stack->sim
    command forwarding, so exactly one command source is live at a time. A `False` releases
    the gate immediately, and a staleness timeout (MANUAL_OVERRIDE_TIMEOUT_S) releases it
    automatically if the bridge dies mid-override -- the stack can never be locked out by a
    crashed peer. Feedback (sim -> SAM custom topics) is NEVER gated: the stack keeps seeing
    true actuator state during manual moves.
    """

    MANUAL_OVERRIDE_TIMEOUT_S = 2.0

    def __init__(self):
        super().__init__('sam_standard_control_publisher')
        self.get_logger().info('SAM Standard Control Publisher Node has been started.')
        self.declare_parameter('robot_name', 'sam')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        # Manual-override gate state -- see class docstring.
        self._manual_override = False
        self._manual_override_stamp = 0.0
        self._override_was_active = False  # for edge logging only
        self.manual_override_sub = self.create_subscription(
            Bool,
            'core/standard/manual_override',
            self.manual_override_callback,
            10
        )

        ##########################################################################
        # Subscribe to control topics from SAM custom topics and republishes to sim standard topics
        ##########################################################################
        self.vbs_sub = self.create_subscription(
            PercentStamped,
            SamTopics.VBS_CMD_TOPIC,
            self.vbs_callback,
            10
        )

        self.lcg_sub = self.create_subscription(
            PercentStamped,
            SamTopics.LCG_CMD_TOPIC,
            self.lcg_callback,
            10
        )

        self.thruster1_sub = self.create_subscription(
            ThrusterRPM,
            SamTopics.THRUSTER1_CMD_TOPIC,
            self.thruster1_callback,
            10
        )

        self.thruster2_sub = self.create_subscription(
            ThrusterRPM,
            SamTopics.THRUSTER2_CMD_TOPIC,
            self.thruster2_callback,
            10
        )

        self.thruster_vector_sub = self.create_subscription(
            ThrusterAngles,
            SamTopics.THRUST_VECTOR_CMD_TOPIC,
            self.thrust_vector_callback,
            10
        )
        
        self.vbs_pub = self.create_publisher(Float32, SamTopics.STD_VBS_CMD_TOPIC, 10)
        self.lcg_pub = self.create_publisher(Float32, SamTopics.STD_LCG_CMD_TOPIC, 10)
        self.thruster1_pub = self.create_publisher(Float32, SamTopics.STD_THRUSTER1_CMD_TOPIC, 10)
        self.thruster2_pub = self.create_publisher(Float32, SamTopics.STD_THRUSTER2_CMD_TOPIC, 10)
        self.thrust_vector_yaw_pub = self.create_publisher(Float32, SamTopics.STD_THRUST_VECTOR_YAW_CMD_TOPIC, 10)
        self.thrust_vector_pitch_pub = self.create_publisher(Float32, SamTopics.STD_THRUST_VECTOR_PITCH_CMD_TOPIC, 10)

        #############################################################################
        # Subscribe to Feedback topics from SIM and republishes to SAM custom topics
        #############################################################################
        self.vbs_feedback_sub = self.create_subscription(
            Float32,
            SamTopics.STD_VBS_FB_TOPIC,
            self.vbs_feedback_callback,
            10
        )

        self.lcg_feedback_sub = self.create_subscription(
            Float32,
            SamTopics.STD_LCG_FB_TOPIC,
            self.lcg_feedback_callback,
            10
        )

        self.rmp1_feedback_sub = self.create_subscription(
            Float32,
            SamTopics.STD_THRUSTER1_FB_TOPIC,
            self.thruster1_feedback_callback,
            10
        )

        self.rmp2_feedback_sub = self.create_subscription(
            Float32,
            SamTopics.STD_THRUSTER2_FB_TOPIC,
            self.thruster2_feedback_callback,
            10
        )

        self.vbs_feedback_pub = self.create_publisher(PercentStamped, SamTopics.VBS_FB_TOPIC, 10)
        self.lcg_feedback_pub = self.create_publisher(PercentStamped, SamTopics.LCG_FB_TOPIC, 10)
        self.thruster1_feedback_pub = self.create_publisher(ThrusterFeedback, SamTopics.THRUSTER1_FB_TOPIC, 10)
        self.thruster2_feedback_pub = self.create_publisher(ThrusterFeedback, SamTopics.THRUSTER2_FB_TOPIC, 10)

        ##############################################################################
        # Other SAM topic translations
        ##############################################################################
        self.leak_sub = self.create_subscription(
            Leak,
            SamTopics.LEAK_TOPIC,
            self.leak_callback,
            10
        )
        self.leak_pub = self.create_publisher(Bool, SamTopics.STD_LEAK_TOPIC, 10)

    def manual_override_callback(self, msg):
        """Latch the manual-override flag with a timestamp -- see class docstring."""
        self._manual_override = bool(msg.data)
        self._manual_override_stamp = self.get_clock().now().nanoseconds * 1e-9

    def _manual_override_active(self) -> bool:
        """True while a FRESH override=True is in effect. Stale True (bridge died) releases
        after MANUAL_OVERRIDE_TIMEOUT_S; a False releases immediately."""
        active = False
        if self._manual_override:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            active = (now_s - self._manual_override_stamp) < self.MANUAL_OVERRIDE_TIMEOUT_S
        if active != self._override_was_active:
            self._override_was_active = active
            self.get_logger().info(
                'manual override ENGAGED -- suppressing stack->sim command forwarding'
                if active else
                'manual override RELEASED -- stack->sim command forwarding restored'
            )
        return active

    def leak_callback(self, msg):
        """
        Callback for Leak messages.
        Converts Leak to Bool and publishes to standard topic.
        """
        std_msg = Bool()
        std_msg.data = msg.value
        self.leak_pub.publish(std_msg)


    def vbs_callback(self, msg):
        """
        Callback for VBS command messages.
        Converts PercentStamped to Float32 and publishes to standard topic.
        """
        if self._manual_override_active():
            return
        std_msg = Float32()
        std_msg.data = msg.value
        self.vbs_pub.publish(std_msg)

    def lcg_callback(self, msg):
        """
        Callback for LCG command messages.
        Converts PercentStamped to Float32 and publishes to standard topic.
        """
        if self._manual_override_active():
            return
        std_msg = Float32()
        std_msg.data = msg.value
        self.lcg_pub.publish(std_msg)

    def thruster1_callback(self, msg):
        """
        Callback for Thruster 1 command messages.
        Converts ThrusterRPM to Float32 and publishes to standard topic.
        """
        if self._manual_override_active():
            return
        std_msg = Float32()
        std_msg.data = float(msg.rpm)
        self.thruster1_pub.publish(std_msg)

    def thruster2_callback(self, msg):
        """
        Callback for Thruster 2 command messages.
        Converts ThrusterRPM to Float32 and publishes to standard topic.
        """
        if self._manual_override_active():
            return
        std_msg = Float32()
        std_msg.data = float(msg.rpm)
        self.thruster2_pub.publish(std_msg)

    def thrust_vector_callback(self, msg):
        """
        Callback for Thruster Angles command messages.
        Publishes yaw and pitch angles to standard topics.
        """
        if self._manual_override_active():
            return
        yaw_msg = Float32()
        yaw_msg.data = msg.thruster_horizontal_radians
        self.thrust_vector_yaw_pub.publish(yaw_msg)

        pitch_msg = Float32()
        pitch_msg.data = msg.thruster_vertical_radians
        self.thrust_vector_pitch_pub.publish(pitch_msg)

    def vbs_feedback_callback(self, msg):
        """
        Callback for VBS feedback messages.
        Converts Float32 to PercentStamped and publishes to SAM custom topic.
        """
        sam_msg = PercentStamped()
        sam_msg.value = float(msg.data)
        self.vbs_feedback_pub.publish(sam_msg)

    def lcg_feedback_callback(self, msg):
        """
        Callback for LCG feedback messages.
        Converts Float32 to PercentStamped and publishes to SAM custom topic.
        """
        sam_msg = PercentStamped()
        sam_msg.value = float(msg.data)
        self.lcg_feedback_pub.publish(sam_msg)

    def standard_thruster_feedback_to_custom_msg(self, msg):
        """
        Converts standard thruster feedback message to SAM custom ThrusterFeedback message.
        Assumes msg.data is a Float32 representing RPM.
        All other fields are set to zero.
        """
        sam_msg = ThrusterFeedback()
        sam_msg.rpm.rpm = int(msg.data)
        sam_msg.dc.dc = 0.
        sam_msg.current = 0.
        sam_msg.torque = 0.
        return sam_msg

    def thruster1_feedback_callback(self, msg):
        """
        Callback for Thruster 1 feedback messages.
        Converts Float32 to custom ThrusterFeedback and publishes to SAM custom topic.
        """
        sam_msg = self.standard_thruster_feedback_to_custom_msg(msg)
        self.thruster1_feedback_pub.publish(sam_msg)

    def thruster2_feedback_callback(self, msg):
        """
        Callback for Thruster 2 feedback messages.
        Converts Float32 to custom ThrusterFeedback and publishes to SAM custom topic.
        """
        sam_msg = self.standard_thruster_feedback_to_custom_msg(msg)
        self.thruster2_feedback_pub.publish(sam_msg)


    
def main(args=None):
    rclpy.init(args=args)
    node = SAMStandardControlPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt, shutting down...')
    finally:
        node.get_logger().info('Shutting down SAM Standard Control Publisher Node.')
        node.destroy_node()
        rclpy.shutdown()
