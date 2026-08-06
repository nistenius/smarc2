
# General
import time
from collections import deque
import importlib

# ROS
import rclpy
from rclpy.node import Node


class TopicRateMonitor:
    def __init__(self, node: Node,
                 topics_dict: dict,
                 timeout_time_sec: float=5.0,
                 window_size: int = 5,
                 report_interval: float = 1.0,
                 verbose: bool = False,
                 latch_faults: bool = True,
                 rate_tolerance: float = 0.8,
                 recover_cycles: int = 3):
        """
        Will check that the topics are maintained at the desired rate
        Args:
            node: The ROS 2 node using this monitor.
            topics: A dict {topic_name: msg_type}
            window_size: Sliding window size for rate calculation.
            report_interval: Seconds between each rate log per topic.
            latch_faults: When True (default, and the behaviour this class has always had),
                a fault is permanent for the lifetime of the node -- correct for a real dive,
                where a dropout means "surface and investigate", not "carry on if it comes
                back". When False, a topic's fault clears once it has looked healthy for
                `recover_cycles` consecutive evaluations. Sim bringups want False: stopping and
                re-playing the Unity editor drops every topic, and under latching that
                permanently poisons mission acceptance until someone restarts this node.
            rate_tolerance: A topic is rate-faulted below `desired_rate * rate_tolerance`
                rather than below `desired_rate` exactly. The old zero-tolerance comparison
                meant any jitter under nominal was a permanent fault -- unusable for the DVL,
                whose nominal is 1.0 Hz.
            recover_cycles: Consecutive healthy evaluations required to clear a fault when
                `latch_faults` is False. Debounces a topic that is flapping around the limit.
        """
        #
        self.node = node
        self.topics_dict = topics_dict  # { topic_name: [message_type, desired_rate]}
        self.window_size = window_size
        self.timeout_time_sec = timeout_time_sec
        self.report_interval = report_interval

        self.verbose = verbose
        self.latch_faults = latch_faults
        self.rate_tolerance = rate_tolerance
        self.recover_cycles = max(1, int(recover_cycles))
        # self.report_timer_rate =  # IGNORE FOR NOW

        self.timers = {}
        self.timestamps = {}
        self.timer_output = {}

        # Per-topic fault state. `self.fault` stays as the aggregate boolean every existing
        # caller reads; these add the detail needed to report *which* topic is unhappy and to
        # clear topics independently of each other.
        self.topic_faults = {}      # topic -> bool
        self.topic_reasons = {}     # topic -> str (empty when healthy)
        self._healthy_streak = {}   # topic -> consecutive healthy evaluations

        self.ready = False
        self.fault = False

        for topic, msg_info in self.topics_dict.items():
            msg_type, msg_rate = msg_info
            if self.verbose:
                self._log(f"Monitoring '{topic}' at {report_interval}s interval")
            self.timestamps[topic] = deque(maxlen=window_size)
            self.timer_output[topic] = False
            self.topic_faults[topic] = False
            self.topic_reasons[topic] = ""
            self._healthy_streak[topic] = 0
            node.create_subscription(msg_type, topic, self._make_callback(topic), 10)
            # TODO - for now ignoring the individual timers
            # self.timers[topic] = node.create_timer(report_interval, self._make_reporter(topic))

        # Timer for checking if topics have been received at least once
        self.report_timer = self.node.create_timer(timer_period_sec=float(self.report_interval),
                                                   callback=self.report_callback)

    def _log(self, message):
        self.node.get_logger().info(message)

    def _make_callback(self, topic_name):
        def subscriber_callback(msg):
            # if self.verbose:
            #     self.node.get_logger().info(f"Subscription callback: {topic_name}")
            self.timestamps[topic_name].append(self.node.get_clock().now().nanoseconds/1e9)
        return subscriber_callback

    def _make_timer(self, topic_name):
        def timer_callback():
            self.node.get_logger().info(f"Timer callback: {topic_name}")
            self.timer_output[topic_name] = True
            self.timestamps[topic_name].append(self.node.get_clock().now().nanoseconds/1e9)
        return timer_callback

    # Use this if it is desired that each topic has a timer
    # For now I will just check at a given rate
    # def _make_reporter(self, topic_name):
    #     def report():
    #         topic_names = self.topics_dict.keys()
    #         for topic_name in topic_names:
    #
    #             times = self.timestamps[topic_name]
    #             if len(times) < 2:
    #                 self.node.get_logger().info(f"[{topic_name}] Waiting for data...")
    #                 return
    #
    #             intervals = [t2 - t1 for t1, t2 in zip(times, list(times)[1:])]
    #             if intervals:
    #                 avg_rate = 1.0 / (sum(intervals) / len(intervals))
    #                 self.node.get_logger().info(f"[{topic_name}] Rate: {avg_rate:.2f} Hz")
    #             else:
    #                 self.node.get_logger().info(f"[{topic_name}] Insufficient data.")
    #     return report

    def report_callback(self):
        self.determine_ready()
        self.determine_fault()

    def determine_ready(self):
        if self.ready:
            return True
        for topic_name in self.topics_dict.keys():
            if len(self.timestamps[topic_name]) == 0:
                return False

        # Set to ready
        self.ready = True
        return True

    def reset_faults(self):
        """Clear every latched topic fault. Used by the node's `reset_faults` service so an
        operator can re-arm the vehicle without restarting the process."""
        for topic_name in self.topics_dict.keys():
            self.topic_faults[topic_name] = False
            self.topic_reasons[topic_name] = ""
            self._healthy_streak[topic_name] = 0
            # Drop the stale window too -- otherwise the timestamps left over from before the
            # dropout immediately re-trigger the staleness check on the very next evaluation.
            self.timestamps[topic_name].clear()
        self.fault = False

    def fault_reasons(self):
        """Every currently-faulted topic as 'topic: reason' strings."""
        return [f"{t}: {r}" for t, r in self.topic_reasons.items() if self.topic_faults.get(t)]

    def _evaluate_topic(self, topic_name, msg_rate):
        """Judge one topic against the current window. Returns '' if healthy, else the reason.

        Deliberately has no memory: latching and recovery are applied by the caller, so this
        always reports what is true *right now*.
        """
        times = self.timestamps[topic_name]
        if len(times) < 2:
            if self.verbose:
                self.node.get_logger().info(f"[{topic_name}] Waiting for data...")
            return ""  # not enough data to judge either way

        # Check for timeout.
        # NOTE: this measures against times[-1], the NEWEST sample. It used to read times[0],
        # the oldest entry in a `window_size`-deep deque, which conflates "the topic is stale"
        # with "the window is simply long". At 20 Hz the window spans 0.2 s and the difference
        # is invisible; at the DVL's 1.0 Hz nominal the oldest sample is ~4 s old during
        # perfectly healthy operation, leaving under 4 s of headroom against a 7.5 s timeout
        # and producing spurious permanent faults on any small hiccup.
        now = self.node.get_clock().now().nanoseconds/1e9
        time_diff = now - times[-1]
        if time_diff > self.timeout_time_sec:
            return (f"timeout, last message {time_diff:.2f} s ago "
                    f"> {self.timeout_time_sec:.2f} s")

        # Check frequency
        intervals = [t2 - t1 for t1, t2 in zip(times, list(times)[1:])]
        avg_interval = (sum(intervals) / len(intervals)) if intervals else 0.0
        # A zero average interval means every sample in the window carries the SAME timestamp.
        # That is routine under use_sim_time: stamps come from /clock, several messages can arrive
        # inside one tick, and they are all stamped with it. The old guard was `if intervals:`,
        # which catches the EMPTY window but not the all-zero one -- so this divided by zero and
        # killed the process outright, on both SAM VMs within a second of the simulator starting
        # to publish (2026-08-05).
        #
        # A dead health checker is far worse than a faulting one. Faulting, it still publishes
        # smarc/vehicle_health and an operator can read the reason. Dead, the topic has no
        # publisher at all, wasp_bt keeps the VEHICLE_HEALTH_ERROR it initialises to, and every
        # start-tst is refused with nothing anywhere explaining it.
        #
        # It is not a fault on the merits either: samples sharing a timestamp arrived at least as
        # fast as the clock can distinguish, which is the opposite of "below nominal rate".
        if avg_interval > 0:
            avg_rate = 1.0 / avg_interval

            if self.verbose:
                self.node.get_logger().info(f"[{topic_name}] Rate: {avg_rate:.2f} Hz / {msg_rate:.2f} Hz")

            min_rate = msg_rate * self.rate_tolerance
            if avg_rate < min_rate:
                return (f"rate {avg_rate:.2f} Hz < {min_rate:.2f} Hz "
                        f"({msg_rate:.2f} Hz x {self.rate_tolerance:.2f})")

        return ""

    def determine_fault(self):
        # Under latching, one fault anywhere is terminal and there is nothing left to decide.
        if self.fault and self.latch_faults:
            return True

        for topic_name, msg_info in self.topics_dict.items():
            msg_type, msg_rate = msg_info
            reason = self._evaluate_topic(topic_name, msg_rate)

            if reason:
                self._healthy_streak[topic_name] = 0
                if not self.topic_faults[topic_name]:
                    self.topic_faults[topic_name] = True
                    self.topic_reasons[topic_name] = reason
                    self.node.get_logger().warn(f"Fault: {topic_name}: {reason}")
                else:
                    self.topic_reasons[topic_name] = reason
                if self.latch_faults:
                    self.fault = True
                    return True
            elif self.topic_faults[topic_name]:
                # Healthy again. Require a few consecutive good evaluations before believing it,
                # so a topic sitting right on the limit doesn't flap the vehicle in and out of
                # READY (which wasp_bt gates mission acceptance on).
                self._healthy_streak[topic_name] += 1
                if self._healthy_streak[topic_name] >= self.recover_cycles:
                    self.topic_faults[topic_name] = False
                    self.topic_reasons[topic_name] = ""
                    self._healthy_streak[topic_name] = 0
                    self.node.get_logger().info(f"Recovered: {topic_name} is healthy again")

        self.fault = any(self.topic_faults.values())
        return self.fault


class DynamicSubscriberManager:
    def __init__(self, node, target_topics, desired_values, min_rates, fault_callback):
        """
        Args:
            node: The rclpy node instance.
            target_topics: List of topic names to subscribe to.
            desired_values: Dict of topic -> desired value to check against.
            min_rates: Dict of topic -> minimum expected message rate (Hz).
            fault_callback: Function to call when a fault is detected.
                            Should accept (manager_name, fault_message).
        """
        self.node = node
        self.manager_name = self.__class__.__name__
        self.target_topics = target_topics
        self.desired_values = desired_values
        self.min_rates = min_rates
        self.fault_callback = fault_callback

        self.subscriptions = {}
        self.msg_counters = {topic: 0 for topic in target_topics}
        self.last_check_time = self.node.get_clock().now()

        # Set up timers
        self.node.create_timer(1.0, self.check_and_subscribe)  # every 1 sec
        self.node.create_timer(5.0, self.check_message_rates)  # every 5 sec

    def check_and_subscribe(self):
        topics = dict(self.node.get_topic_names_and_types())

        for topic_name in self.target_topics:
            if topic_name in topics and topic_name not in self.subscriptions:
                type_name = topics[topic_name][0]
                self.add_subscription(topic_name, type_name)
                self.node.get_logger().info(f"✅ Subscribed to {topic_name} [{type_name}]")

    def add_subscription(self, topic_name, type_name):
        pkg, _, msg = type_name.partition('/')
        try:
            module = importlib.import_module(f'{pkg}.msg')
            msg_type = getattr(module, msg)
        except ModuleNotFoundError as e:
            self.node.get_logger().error(
                f"Failed to import module '{pkg}.msg' for topic '{topic_name}': {e}"
            )
            return
        except AttributeError as e:
            self.node.get_logger().error(
                f"Message type '{msg}' not found in '{pkg}.msg' for topic '{topic_name}': {e}"
            )
            return
        except Exception as e:
            self.node.get_logger().error(
                f"Unexpected error during import for topic '{topic_name}': {e}"
            )
            return

        desired_value = self.desired_values.get(topic_name)

        def callback(msg_obj, topic=topic_name, desired=desired_value):
            try:
                self.msg_counters[topic] += 1
                if desired is not None:
                    if not self.check_message_match(msg_obj, desired):
                        self.fault_callback(
                            self.manager_name,
                            f"Message mismatch on {topic}. Got: {msg_obj}, Expected: {desired}"
                        )
            except Exception as e:
                self.node.get_logger().error(
                    f"Error in callback for topic '{topic}': {e}"
                )

        try:
            sub = self.node.create_subscription(msg_type, topic_name, callback, 10)
            self.subscriptions[topic_name] = sub
        except Exception as e:
            self.node.get_logger().error(
                f"Failed to create subscription for topic '{topic_name}': {e}"
            )

    def check_message_match(self, msg, desired):
        """
        Compare the received message with the desired value.
        You can customize this per message type if needed.
        """
        try:
            return str(msg) == str(desired)
        except Exception as e:
            self.node.get_logger().error(
                f"Error comparing message on topic: {e}"
            )
            return False

    def check_message_rates(self):
        now = self.node.get_clock().now()
        elapsed_duration = now - self.last_check_time
        elapsed_sec = elapsed_duration.nanoseconds / 1e9 if elapsed_duration.nanoseconds > 0 else 1e-6
        self.last_check_time = now

        for topic, count in self.msg_counters.items():
            actual_rate = count / elapsed_sec
            min_rate = self.min_rates.get(topic, 0.0)
            self.node.get_logger().info(
                f"[{self.manager_name}] {topic}: {actual_rate:.2f} Hz (min {min_rate:.2f} Hz)"
            )

            if actual_rate < min_rate:
                self.fault_callback(
                    self.manager_name,
                    f"Message rate too low on {topic}: {actual_rate:.2f} Hz < {min_rate:.2f} Hz"
                )

            self.msg_counters[topic] = 0  # Reset counter for next check







