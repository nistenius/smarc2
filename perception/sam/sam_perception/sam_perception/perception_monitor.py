#!/usr/bin/env python3
"""Rate monitor for the SAM 2.2 perception sensors.

Runs in the bringup's perception pane in SIMULATION, where the sensor topics come
from Unity over the ros_tcp bridge. Gives one glanceable line every few seconds
per topic, and an explicit warning when a topic is silent — so "is sim data
actually flowing into the VM?" is answered by the bringup itself, not by manual
`ros2 topic hz` archaeology.

Subscribes with sensor-data QoS (best effort) which matches both the bridge's
reliable publishers and real drivers.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image, CompressedImage, CameraInfo


# topic (relative to robot namespace) -> msg type
WATCHED = {
    'payload/sonar3d/points': PointCloud2,
    'payload/realsense/left/image_raw/compressed': CompressedImage,
    'payload/realsense/right/image_raw/compressed': CompressedImage,
    'payload/realsense/left/camera_info': CameraInfo,
    'payload/realsense/depth/image_raw': Image,
}


class PerceptionMonitor(Node):
    def __init__(self):
        super().__init__('perception_monitor')
        self.declare_parameter('robot_name', 'sam21')
        self.declare_parameter('report_period_s', 5.0)
        robot = self.get_parameter('robot_name').value
        period = float(self.get_parameter('report_period_s').value)

        self.counts = {}
        self.ever = set()
        for rel_topic, msg_type in WATCHED.items():
            topic = f'/{robot}/{rel_topic}'
            self.counts[topic] = 0
            self.create_subscription(
                msg_type, topic,
                lambda msg, t=topic: self._on_msg(t),
                qos_profile_sensor_data)

        self.period = period
        self.create_timer(period, self._report)
        self.get_logger().info(
            f'Watching {len(self.counts)} perception topics for {robot}, '
            f'reporting every {period:.0f} s.')

    def _on_msg(self, topic):
        self.counts[topic] += 1
        self.ever.add(topic)

    def _report(self):
        lines = []
        for topic, n in self.counts.items():
            hz = n / self.period
            short = topic.split('/', 2)[-1]  # strip /robot/
            if n == 0:
                tag = 'SILENT (never seen)' if topic not in self.ever else 'SILENT'
                lines.append(f'{short}: {tag}')
            else:
                lines.append(f'{short}: {hz:.1f} Hz')
            self.counts[topic] = 0
        silent = sum(1 for l in lines if 'SILENT' in l)
        # DO NOT collapse these into `log = ... if ... else ...; log(msg)`.
        # rclpy caches logging state per CALL SITE, so a single line that logs at
        # two severities raises
        #     ValueError: Logger severity cannot be changed between calls
        # and kills the node. This monitor therefore died the FIRST time a topic
        # went silent and came back — i.e. on every Unity Stop/Play and every ROS
        # reconnect — which is what a rig session is made of. Found 2026-08-12
        # when rig_doctor dumped the perception pane; it had been dying silently
        # since 2026-08-09, so "is sim data flowing?" had no answer for days.
        # Separate call sites, always.
        msg = ' | '.join(lines)
        if silent:
            self.get_logger().warning(msg)
        else:
            self.get_logger().info(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
