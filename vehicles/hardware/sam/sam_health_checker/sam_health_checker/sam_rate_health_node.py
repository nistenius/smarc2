import os
import yaml
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from ament_index_python import get_package_share_directory

# General messages
from sensor_msgs.msg import Imu
from std_msgs.msg import Int8, Float32, String
from sensor_msgs.msg import BatteryState, Imu

# SMaRC messages
from smarc_msgs.msg import Topics as SmarcTopics
from smarc_msgs.msg import Leak, DVL

# Vehicle specific messages -- Consider how to handle this
from sam_msgs.msg import Topics as SamTopics
from diagnostic_msgs.msg import DiagnosticArray
from std_srvs.srv import Trigger

try:
    from .helpers.health_helpers import TopicRateMonitor
except ImportError:
    from helpers.health_helpers import TopicRateMonitor


@dataclass
class StatusReport:
    ready: bool = field(default=False)
    fault: bool = field(default=False)
    # Why this check is unhappy, for the report topic and the log. Empty when healthy.
    reason: str = field(default="")
    # Consecutive healthy evaluations, used to debounce recovery when faults don't latch.
    healthy_streak: int = field(default=0)


class MonitorNode(Node):
    """
    This is a basic example of how health checks will be performed
    """

    def __init__(self, namespace=None):
        super().__init__('Rate_monitor_node', namespace=namespace)
        self.namespace = namespace

        # === Load Parameters ===

        # Limits
        self.declare_parameter("limits_filename", "sam_health_limits.yaml")
        self.limits_filename = self.get_parameter("limits_filename").value
        self.limits = self.read_limits()

        self.declare_parameter("output_rate", 1.0)
        self.output_rate = self.get_parameter("output_rate").value

        # Leak and battery parameters
        self.declare_parameter("leak_topic", SamTopics.LEAK_TOPIC)
        self.leak_topic = self.get_parameter("leak_topic").value

        self.declare_parameter("battery_topic", SamTopics.BATTERY_STATUS_TOPIC)
        self.battery_topic = self.get_parameter("battery_topic").value

        self.declare_parameter("battery_min_voltage", 20.0)
        self.battery_min_voltage = self.get_parameter("battery_min_voltage").value

        self.declare_parameter("battery_min_capacity", 0.25)
        self.battery_min_capacity = self.get_parameter("battery_min_capacity").value

        # Depth and Altitude parameters
        # The values from these topics are compared to the values read from limits
        self.declare_parameter("depth_topic", SmarcTopics.DEPTH_TOPIC)
        self.depth_topic = self.get_parameter("depth_topic").value

        self.declare_parameter("altitude_topic", SmarcTopics.ALTITUDE_TOPIC)
        self.altitude_topic = self.get_parameter("altitude_topic").value

        # Topic rate monitor parameters
        # time since start if no message is received
        self.declare_parameter("initial_timeout_time_sec", 30.0)
        self.initial_timeout_time_sec = self.get_parameter("initial_timeout_time_sec").value

        # time since last message was received
        self.declare_parameter("timeout_time_sec", 5.0)
        self.timeout_time_sec = self.get_parameter("timeout_time_sec").value

        self.declare_parameter("report_topic", SmarcTopics.VEHICLE_HEALTH_TOPIC)
        self.report_topic = self.get_parameter("report_topic").value

        # === Fault latching / recovery ===
        # Historically every fault in this node was permanent: each check starts with
        # `if status.fault: return status`, and nothing ever set it back to False. On a real
        # dive that is the right call -- a sensor dropout means surface and investigate, not
        # carry on because it came back. In simulation it is actively harmful: stopping and
        # re-playing the Unity editor drops every topic at once, the node latches ERROR, goes
        # quiet (see fault_logged), and from then on wasp_bt rejects every start-tst with
        # "vehicle health status is not ok" until someone notices and restarts this process.
        #
        # latch_faults=False makes a fault clear itself once the underlying condition has been
        # good for recover_cycles consecutive evaluations. Default stays True so hardware
        # behaviour is untouched; sim bringups pass False explicitly.
        self.declare_parameter("latch_faults", True)
        self.latch_faults = self.get_parameter("latch_faults").value

        self.declare_parameter("recover_cycles", 3)
        self.recover_cycles = max(1, int(self.get_parameter("recover_cycles").value))

        # A topic is rate-faulted below desired_rate * rate_tolerance rather than below
        # desired_rate exactly -- the old comparison had zero headroom, so any jitter under
        # nominal was a fault, which the 1.0 Hz DVL could not survive.
        self.declare_parameter("rate_tolerance", 0.8)
        self.rate_tolerance = float(self.get_parameter("rate_tolerance").value)

        self.declare_parameter('verbose', False)
        self.verbose = self.get_parameter("verbose").value
        if self.verbose:
            self.log_all_parameters()

        self.declare_parameter('testing', False)
        self.testing = self.get_parameter('testing').value

        # === Hardcoded topics, message types, and rates ===
        # Essential topics - Will trigger fault if no detected within initial timeout time
        self.essential_topics = {
            SamTopics.STIM_IMU_TOPIC: [Imu, 20.0]
        }

        # Optional topics - Will NOT trigger fault based on initial timeout time
        self.optional_topics = {
            SamTopics.DVL_TOPIC: [DVL, 1.0],
        }

        self.start_time = self.get_clock().now().nanoseconds/1e9

        self.current_leak = None
        self.current_leak_time = None
        self.current_leak_status = StatusReport(ready=True,fault=False)  # No waiting, messages only when a fault is detected
        self.current_battery = None
        self.current_battery_time = None
        self.current_battery_status = StatusReport()

        self.current_depth = None
        self.current_depth_time = None
        self.current_depth_status = StatusReport()
        self.current_altitude = None
        self.current_altitude_time = None
        self.current_altitude_status = StatusReport()

        self.fault_report = None
        self.fault_logged = False

        # === Set up subs, pubs, and timers ===
        self.leak_sub = self.create_subscription(msg_type=Leak,
                                                 topic=self.leak_topic,
                                                 callback=self.leak_callback,
                                                 qos_profile=10)

        self.battery_sub = self.create_subscription(msg_type=BatteryState,
                                                    topic=self.battery_topic,
                                                    callback=self.battery_callback,
                                                    qos_profile=10)

        self.depth_sub = self.create_subscription(msg_type=Float32,
                                                  topic=self.depth_topic,
                                                  callback=self.depth_callback,
                                                  qos_profile=10)

        self.altitude_sub = self.create_subscription(msg_type=Float32,
                                                     topic=self.altitude_topic,
                                                     callback=self.altitude_callback,
                                                     qos_profile=10)

        self.get_logger().info(f"topic rate monitor(s) instantiated")
        self.essential_monitor = TopicRateMonitor(self, self.essential_topics, timeout_time_sec=self.timeout_time_sec,
                                                  verbose=self.verbose, latch_faults=self.latch_faults,
                                                  rate_tolerance=self.rate_tolerance,
                                                  recover_cycles=self.recover_cycles)

        self.optional_monitor = TopicRateMonitor(self, self.optional_topics, timeout_time_sec=self.timeout_time_sec,
                                                 verbose=self.verbose, latch_faults=self.latch_faults,
                                                 rate_tolerance=self.rate_tolerance,
                                                 recover_cycles=self.recover_cycles)

        # Lets an operator re-arm the vehicle after a transient fault without restarting the
        # node. Mirrors wasp_bt's own `reset_emergency` service -- clearing that one alone was
        # never enough, because _handle_tst_command gates start-tst on emergency_flag AND on
        # health_status separately, and this node owns the second gate.
        self._reset_faults_srv = self.create_service(Trigger, "reset_faults", self._reset_faults_cb)

        if self.testing:
            self.report_pub = self.create_publisher(msg_type=Int8, topic='health_testing', qos_profile=10)
        else:
            self.report_pub = self.create_publisher(msg_type=Int8, topic=self.report_topic, qos_profile=10)
        
        self.report_string_pub = self.create_publisher(msg_type=String, topic=f"{self.report_topic}/report", qos_profile=10)
        
        self.report_timer = self.create_timer(timer_period_sec=float(1.0 / self.output_rate),
                                              callback=self.output_callback)
        

    def raise_fault(self, status: StatusReport, report: str):
        """Mark a check faulted, logging only on the healthy -> faulted edge.

        Replaces the old inline pattern of `if self.fault_report is None: self.fault_report = ...`
        followed by an unconditional warn. That kept the FIRST fault forever and re-logged it on
        every cycle until fault_logged silenced the node entirely, so the console could never
        tell you what was wrong *now*.
        """
        status.healthy_streak = 0
        if not status.fault:
            status.fault = True
            self.get_logger().warn(report)
        status.reason = report

    def clear_fault(self, status: StatusReport, label: str):
        """Called on every evaluation where a check looks healthy.

        Under latch_faults this does nothing -- a fault is terminal. Otherwise the fault clears
        after recover_cycles consecutive healthy evaluations, so a value hovering on its limit
        doesn't flap the vehicle in and out of READY.
        """
        if not status.fault:
            status.healthy_streak = 0
            return
        if self.latch_faults:
            return
        status.healthy_streak += 1
        if status.healthy_streak >= self.recover_cycles:
            status.fault = False
            status.reason = ""
            status.healthy_streak = 0
            self.get_logger().info(f"Recovered: {label} is healthy again")

    def _reset_faults_cb(self, request, response):
        """Clear every latched fault in this node, including inside the rate monitors."""
        for status in (self.current_leak_status, self.current_battery_status,
                       self.current_depth_status, self.current_altitude_status):
            status.fault = False
            status.reason = ""
            status.healthy_streak = 0
        self.essential_monitor.reset_faults()
        self.optional_monitor.reset_faults()
        self.fault_report = None
        self.fault_logged = False
        response.success = True
        response.message = "All health faults cleared."
        self.get_logger().info("All health faults cleared by service call.")
        return response

    def active_fault_reports(self):
        """Every fault that is true right now, most useful first."""
        reports = []
        for label, status in (("leak", self.current_leak_status),
                              ("battery", self.current_battery_status),
                              ("depth", self.current_depth_status),
                              ("altitude", self.current_altitude_status)):
            if status.fault:
                reports.append(status.reason or f"Fault detected: {label}")
        reports.extend(self.essential_monitor.fault_reasons())
        reports.extend(self.optional_monitor.fault_reasons())
        return reports

    def leak_callback(self, msg):
        self.current_leak = msg
        self.current_leak_time = self.get_clock().now().nanoseconds/1e9

    def battery_callback(self, msg):
        self.current_battery = msg
        self.current_battery_time = self.get_clock().now().nanoseconds/1e9

    def depth_callback(self, msg):
        self.current_depth = msg
        self.current_depth_time = self.get_clock().now().nanoseconds / 1e9

    def altitude_callback(self, msg):
        self.current_altitude = msg
        self.current_altitude_time = self.get_clock().now().nanoseconds / 1e9

    def leak_check(self):
        # NOTE: unlike the other checks, a leak never auto-recovers even when latch_faults is
        # False. Water inside the pressure vessel does not become fine again because the sensor
        # stopped reporting it -- that is exactly the case where a stale True is the safe read.
        if self.current_leak_status.fault:
            return self.current_leak_status

        if self.current_leak is None:
            return self.current_leak_status

        self.current_leak_status.ready = True

        if self.current_leak.value:
            self.raise_fault(self.current_leak_status, "Fault detected: Leak!")

        return self.current_leak_status

    def battery_check(self):
        
        # TODO the batter is not required to move into ready
        if self.current_battery_status.fault and self.latch_faults:
            return self.current_battery_status

        # check_time = self.get_clock().now().nanoseconds/1e9

        if self.current_battery is None:
            self.current_battery_status.ready = True
            return self.current_battery_status

        # Battery status received
        self.current_battery_status.ready = True

        if self.current_battery.voltage < self.battery_min_voltage:
            self.raise_fault(
                self.current_battery_status,
                f"Fault detected: Battery low voltage! Current voltage: {self.current_battery.voltage}, Min voltage: {self.battery_min_voltage}")
        elif self.current_battery.percentage < self.battery_min_capacity:
            self.raise_fault(
                self.current_battery_status,
                f"Fault detected: Battery low capacity! Current capacity: {self.current_battery.percentage}, Min capacity: {self.battery_min_capacity}")
        else:
            self.clear_fault(self.current_battery_status, "battery")

        # if (check_time - self.current_battery_time) > self.timeout_time_sec:
        #     self.get_logger().info(f"Fault detected: Leak time out!")
        #     self.current_battery_status.fault = True

        return self.current_battery_status

    def depth_check(self):

        if self.current_depth_status.fault and self.latch_faults:
            return self.current_depth_status

        check_time = self.get_clock().now().nanoseconds/1e9

        if self.current_depth is None:
            if self.check_initial_timeout(check_time):
                self.raise_fault(self.current_depth_status,
                                 "Fault detected: depth initial time out!")
        else:
            self.current_depth_status.ready = True

            # Check that 'max_depth' is defined in limits
            if 'max_depth' not in self.limits:
                return self.current_depth_status

            time_diff = check_time - self.current_depth_time
            if self.current_depth.data > self.limits['max_depth']:
                self.raise_fault(
                    self.current_depth_status,
                    f"Fault detected: Max depth exceeded! Current depth: {self.current_depth.data:.2f}, Max depth: {self.limits['max_depth']}")
            elif time_diff > self.timeout_time_sec:
                self.raise_fault(
                    self.current_depth_status,
                    f"Fault detected: depth time out! Time diff = {time_diff:.2f} s > {self.timeout_time_sec:.2f} s")
            else:
                self.clear_fault(self.current_depth_status, "depth")

        return self.current_depth_status

    def altitude_check(self):
        """
        REMOVED INITIAL
        """

        if self.current_altitude_status.fault and self.latch_faults:
            return self.current_altitude_status

        check_time = self.get_clock().now().nanoseconds/1e9

        if self.current_altitude is None:
            return self.current_altitude_status

        self.current_altitude_status.ready = True

        # Check that 'max_depth' is defined in limits
        if 'min_altitude' not in self.limits:
            return self.current_altitude_status

        if self.current_altitude.data == -1:
            # -1 is the "no bottom lock" sentinel, not a reading of zero altitude. Treat it as
            # no information: don't fault on it, but don't count it as a healthy sample towards
            # recovery either.
            return self.current_altitude_status

        time_diff = check_time - self.current_altitude_time
        if self.current_altitude.data < self.limits['min_altitude']:
            self.raise_fault(
                self.current_altitude_status,
                f"Fault detected: low altitude! Current altitude: {self.current_altitude.data:.2f}, Min altitude: {self.limits['min_altitude']}")
        elif time_diff > self.timeout_time_sec:
            self.raise_fault(
                self.current_altitude_status,
                f"Fault detected: altitude time out! Time diff = {time_diff:.2f} s > {self.timeout_time_sec:.2f} s")
        else:
            self.clear_fault(self.current_altitude_status, "altitude")

        return self.current_altitude_status

    def output_callback(self):
        """
        Perform
        """

        self.leak_check()
        self.battery_check()
        self.altitude_check()
        self.depth_check()

        if self.verbose and not self.fault_logged:
            self.get_logger().info(f"Leak: {self.current_leak_status}")
            self.get_logger().info(f"Battery: {self.current_battery_status}")
            self.get_logger().info(f"Altitude: {self.current_altitude_status}")
            self.get_logger().info(f"Depth: {self.current_depth_status}")


        # Self.monitor updates on it's own

        faults = [
            self.essential_monitor.fault, self.optional_monitor.fault,
            self.current_leak_status.fault, self.current_battery_status.fault,
            self.current_altitude_status.fault, self.current_depth_status.fault
        ]

        # These are subject to the initial timeout
        readys = [
            self.essential_monitor.ready,
            self.current_leak_status.ready, self.current_battery_status.ready,
            # self.current_altitude_status.ready, self.current_depth_status.ready
        ]

        # These are not subject to the initial timeout
        optional_readys = [
            self.optional_monitor.ready,
            self.current_altitude_status.ready, self.current_depth_status.ready
        ]

        if True in faults:
            # fault_report tracks what is wrong RIGHT NOW rather than whichever fault happened
            # to land first, so the string published on <report_topic>/report stays useful as
            # conditions change (and as they clear, when latch_faults is False).
            active = self.active_fault_reports()
            self.fault_report = "; ".join(active) if active else self.fault_report
            if not self.fault_logged:
                self.get_logger().warn(f"Fault detected [essential, optional, leak, battery, altitude, depth]: {faults}")
                for report in active:
                    self.get_logger().warn(f"  - {report}")
                if not self.latch_faults:
                    self.get_logger().warn(
                        "Faults are recoverable (latch_faults=False): this will clear itself "
                        f"after {self.recover_cycles} healthy cycles once the cause goes away.")
                self.fault_logged = True
            self.publish_fault()
            return


        elif all(readys) and all(optional_readys):
            if self.fault_logged:
                # Re-arm the one-shot logging so the NEXT fault is announced too. Without this
                # the node would recover and then fail again in silence.
                self.get_logger().info("All health faults cleared -- vehicle is READY again.")
                self.fault_logged = False
                self.fault_report = None
            ready_msg = Int8()
            ready_msg.data = SmarcTopics.VEHICLE_HEALTH_READY
            self.report_pub.publish(ready_msg)
        else:
            # Another check for initial timeing out
            current_time = self.get_clock().now().nanoseconds/1e9
            elapsed_time = current_time - self.start_time

            if self.verbose:
                self.get_logger().info(f"Waiting, Elapsed time: {elapsed_time}")

            if self.check_initial_timeout(current_time) and False in readys:
                self.publish_fault()
                return

            else:
                waiting_msg = Int8()
                waiting_msg.data = SmarcTopics.VEHICLE_HEALTH_WAITING
                self.report_pub.publish(waiting_msg)

    def publish_fault(self):

            fault_msg = Int8()
            fault_msg.data = SmarcTopics.VEHICLE_HEALTH_ERROR
            self.report_pub.publish(fault_msg)

            if self.fault_report is not None:
                fault_report_msg = String()
                fault_report_msg.data = str(self.fault_report)
                self.report_string_pub.publish(fault_report_msg)

    def read_limits(self):
        """
        Read YAML file with Lolo's limits.
        Returns a dictionary with the values.
        """
        if not self.limits_filename:
            self.limits_filename = "sam_health_limits.yaml"

        path_to_pkg = get_package_share_directory('sam_health_checker')
        yaml_path = os.path.join(path_to_pkg, "config", self.limits_filename)

        with open(yaml_path, 'r') as file:
            limits = yaml.safe_load(file)
        self.get_logger().info(f"SAM limits have been configured with filename {self.limits_filename}")
        [self.get_logger().info(f"{key}:{value}") for key, value in limits.items()]

        return limits

    def check_initial_timeout(self, time):
        if self.initial_timeout_time_sec < 0:
            return False

        if (time - self.start_time) > self.initial_timeout_time_sec:
            return True
        else:
            return False


    def log_all_parameters(self):
        param_names = self._parameters.keys()
        self.get_logger().info("Declared Parameters and their values:")
        for name in param_names:
            value = self.get_parameter(name).value
            self.get_logger().info(f"  {name}: {value}")

def main(args=None, namespace=None):
    rclpy.init(args=args)
    lolo_health_node = MonitorNode(namespace=namespace)
    try:
        rclpy.spin(lolo_health_node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    default_namespace = "lolo"
    main(namespace=default_namespace)
