#!/usr/bin/env python3
"""`fls_target_detector` — proud objects on the seabed from the Water Linked 3D-15's cloud.

Strategy §4b. A SIBLING of `sam_perception/obstacle_detector.py`, never a change to it: that
node is a safety layer (spec invariants 3 and 8) and stays exactly as flown. This one keeps
precisely what it discards — the returns near the seabed.

WHAT IT PUBLISHES
  <ns>/perception/target/fls_candidates      std_msgs/String, one JSON object per CANDIDATE
  <ns>/perception/target/fls_detector/health std_msgs/String, the obstacle detector's format

THE MODE TOPIC IS AN INPUT, NOT A DECISION. `payload/sonar3d/mode` announces the sonar's range
and rate; this node BOUNDS ITSELF by the announced range and says so on its health line. It
never commands a mode — the mission does that, at CLOSE-OPS entry and exit (the 2026-08-12
decision: the mission knows when it is inspecting, the sonar does not).

Run: ros2 run sam_target_inspection fls_target_detector --ros-args -p robot_name:=sam21
"""
import json
import math
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Float32, String
    from nav_msgs.msg import Odometry
    import tf2_ros
    from rclpy.time import Time
    from rclpy.duration import Duration
    _HAVE_ROS = True
except ImportError:                                            # pragma: no cover
    Node = object
    _HAVE_ROS = False

from sam_target_inspection._health import HealthLine
from sam_target_inspection.fls_target_core import (FlsConfig, FlsTargetTracker, candidate_from_track,
                                                   detect_cloud, parse_cloud, quat_to_rot)


class FlsTargetDetector(Node):

    def __init__(self):
        super().__init__("fls_target_detector")
        self.declare_parameter("robot_name", "sam21")
        self.declare_parameter("cloud_topic", "payload/sonar3d/points")
        # SAME NAME AND SAME REASON as the obstacle detector's: a surfaced hull sees itself
        # (SETTLED §3u). Declared separately rather than read from that node, because reading a
        # live parameter of another node at startup is a dependency this must not have.
        self.declare_parameter("range_min", 0.8)
        self.declare_parameter("intensity_min", 1)
        self.declare_parameter("body_frame", "base_link_gt")
        self.declare_parameter("footprint_min_m", 0.8)
        self.declare_parameter("footprint_max_m", 8.0)
        self.declare_parameter("height_min_m", 0.4)
        self.declare_parameter("height_max_m", 2.5)
        self.declare_parameter("persistence_pings", 0)         # 0 = the derived default
        self.declare_parameter("stale_timeout_s", 2.0)
        self.declare_parameter("sigma_m", 0.5)

        gp = lambda n: self.get_parameter(n).value             # noqa: E731
        self.robot_name = gp("robot_name")
        self.body_frame = gp("body_frame")
        self.intensity_min = int(gp("intensity_min"))
        self.range_min = float(gp("range_min"))
        self.sigma_m = float(gp("sigma_m"))
        self.bands = (float(gp("footprint_min_m")), float(gp("footprint_max_m")),
                      float(gp("height_min_m")), float(gp("height_max_m")))

        ns = f"/{self.robot_name}"
        self.sub = self.create_subscription(
            PointCloud2, f'{ns}/{gp("cloud_topic")}', self.cloud_cb, qos_profile_sensor_data)
        self.create_subscription(Float32, f"{ns}/smarc/altitude", self._alt_cb, 10)
        self.create_subscription(String, f"{ns}/payload/sonar3d/mode", self._mode_cb, 10)
        self.create_subscription(Odometry, f"{ns}/dr/odom", self._odom_cb, 10)
        self.pub_cand = self.create_publisher(
            String, f"{ns}/perception/target/fls_candidates", 5)
        self.pub_health = self.create_publisher(
            String, f"{ns}/perception/target/fls_detector/health", 1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        n_persist = int(gp("persistence_pings"))
        self.tracker = (FlsTargetTracker(persistence_pings=n_persist) if n_persist > 0
                        else FlsTargetTracker())
        self.health = HealthLine(stale_timeout_s=float(gp("stale_timeout_s")),
                                 publisher_count=self._publisher_count)
        self._altitude = None
        self._mode = None
        self._announced_range = None
        self._xy = None
        self._course = None
        self._pitch = 0.0
        self._roll = 0.0
        self._i = 0
        self._ms = 0.0
        self._n_candidates = 0
        self._tf_warned = False

        self.create_timer(0.5, self.publish_health)
        self.get_logger().info(
            f"fls_target_detector up: {ns}/{gp('cloud_topic')} -> "
            f"{ns}/perception/target/fls_candidates; footprint band "
            f"{self.bands[0]}-{self.bands[1]} m, height band {self.bands[2]}-{self.bands[3]} m. "
            f"It reads the same cloud as the obstacle detector and CHANGES NOTHING there.")

    # ------------------------------------------------------------------ inputs
    def _publisher_count(self):
        try:
            return self.sub.get_publisher_count()
        except Exception:
            return -1

    def _alt_cb(self, msg):
        self._altitude = float(msg.data)

    def _mode_cb(self, msg):
        """`NAME|range|rate` — the sonar's own announcement. An input, never a decision."""
        try:
            parts = str(msg.data).split("|")
            self._mode = parts[0]
            self._announced_range = float(parts[1])
        except Exception:
            self.get_logger().warn(f"unparsable sonar mode {msg.data!r}",
                                   throttle_duration_sec=30.0)

    def _odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._xy = (float(p.x), float(p.y))
        # COURSE from the pose's own yaw here, because `dr/odom` is the estimator's own frame
        # and the world transform below is taken from the same message -- mixing a course from
        # one source with a position from another is how a swath ends up rotated.
        self._course = math.degrees(math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        self._pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        self._roll = math.atan2(R[2, 1], R[2, 2])

    # ------------------------------------------------------------------ the cloud
    def _to_body(self, xyz, cloud_frame):
        """World -> body, through TF, exactly as `obstacle_detector.to_body` does it.

        In simulation the cloud arrives in a world frame and is ground truth, so this transform
        is what reconstructs the sensor-relative ranges the real driver outputs natively. On the
        real unit `cloud_frame == body_frame` and this is the identity.
        """
        if cloud_frame == self.body_frame:
            return xyz
        try:
            t = self.tf_buffer.lookup_transform(self.body_frame, cloud_frame, Time(),
                                                timeout=Duration(seconds=0.1))
        except Exception as e:
            if not self._tf_warned:
                self.get_logger().warn(
                    f"TF {cloud_frame} -> {self.body_frame} not available yet: {e}")
                self._tf_warned = True
            return None
        self._tf_warned = False
        q = t.transform.rotation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        p = np.array([t.transform.translation.x, t.transform.translation.y,
                      t.transform.translation.z])
        return xyz @ R.T + p

    def cloud_cb(self, msg):
        t0 = time.perf_counter()
        now = self._now()
        self.health.note_input(now)
        self._i += 1
        if self._altitude is None:
            self.health.blind_streak += 1
            self.health.last_reason = ("no altitude yet on smarc/altitude; the seabed plane fit "
                                       "needs the vehicle's own altimeter as its prior")
            return
        try:
            xyz = parse_cloud(msg, self.intensity_min)
        except Exception as e:
            self.health.blind_streak += 1
            self.health.last_reason = str(e)
            return
        body = self._to_body(xyz, msg.header.frame_id)
        if body is None:
            self.health.blind_streak += 1
            self.health.last_reason = f"no TF {msg.header.frame_id} -> {self.body_frame}"
            return

        # BOUND BY THE ANNOUNCED RANGE. In INSPECTION mode the horizon is 4 m, and searching to
        # 15 m would be searching a range the sensor is not reporting.
        range_max = self._announced_range if self._announced_range else 15.0
        cfg = FlsConfig(altitude_m=self._altitude, pitch_rad=self._pitch, roll_rad=self._roll,
                        footprint_min_m=self.bands[0], footprint_max_m=self.bands[1],
                        height_min_m=self.bands[2], height_max_m=self.bands[3],
                        range_min_m=self.range_min, range_max_m=range_max)
        rep = detect_cloud(body, cfg)
        if not rep.ok:
            self.health.blind_streak += 1
            self.health.last_reason = rep.reason
            self._ms = 1000.0 * (time.perf_counter() - t0)
            return
        self.health.blind_streak = 0
        self.health.last_reason = rep.reason
        if self._xy is not None and self._course is not None:
            for track in self.tracker.update(self._i, rep, self._xy, self._course):
                self._n_candidates += 1
                rec = candidate_from_track(track, cid=f"F{self._i}", t=now,
                                           sigma_m=self.sigma_m)
                rec["sonar_mode"] = self._mode
                rec["sonar_range_m"] = self._announced_range
                self.health.note_output()
                self.pub_cand.publish(String(data=json.dumps(rec, separators=(",", ":"))))
                self.get_logger().info(
                    f"fls candidate {rec['id']}: {rec['n_pings']} pings, box "
                    f"{rec['bbox_m']['length']:.2f} x {rec['bbox_m']['width']:.2f} m, height "
                    f"{rec['height_m']:.2f} m, {rec['n_points']} points, score "
                    f"{rec['score']:.2f}")
        self._ms = 1000.0 * (time.perf_counter() - t0)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ health
    def publish_health(self):
        mode = self._mode or "UNANNOUNCED"
        rng = self._announced_range if self._announced_range else float("nan")
        ok = (f"sonar mode {mode} at {rng:.0f} m (announced), altitude "
              f"{self._altitude if self._altitude is not None else float('nan'):.2f} m, "
              f"{self._n_candidates} candidate(s) in {self._i} clouds, {self._ms:.1f} ms/cloud")
        line = self.health.compute(self._now(), ok_detail=ok)
        state = self.health.state
        self.pub_health.publish(String(data=line))
        if state != getattr(self, "_last_state", None):
            msg = (f"fls detector health: {getattr(self, '_last_state', None)} -> {state} "
                   f"({self.health.detail})")
            if state == "OK":
                self.get_logger().info(msg)
            else:
                self.get_logger().error(msg)
            self._last_state = state


def main(args=None):
    if not _HAVE_ROS:                                          # pragma: no cover
        raise SystemExit("fls_target_detector needs rclpy; the pure core in "
                         "sam_target_inspection.fls_target_core runs without it.")
    rclpy.init(args=args)
    node = FlsTargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
