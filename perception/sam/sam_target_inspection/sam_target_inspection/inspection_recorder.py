#!/usr/bin/env python3
"""`inspection_recorder` — capture, gate, count, and nothing else. Strategy §6.1, ADR-010.

IT IS A SENSOR IN ADR-010'S SENSE. It runs only while a station burst is commanded, it measures
its own cost and prints it, and it does NO reconstruction, NO feature matching and NO learned
detector on the flight computer. The 30 W cap (`SAM_VEHICLE_SPECS.md` §7, written after
camera-induced reboots) is the reason.

  in   <ns>/perception/target/capture           std_msgs/String  {"station": n, "candidate": id}
       <ns>/payload/realsense/left/image_raw    sensor_msgs/Image
       <ns>/payload/sonar3d/points              sensor_msgs/PointCloud2
       <ns>/dr/odom                             nav_msgs/Odometry
  out  <ns>/perception/target/coverage          std_msgs/String  {"station", "done", "why"}
       <ns>/perception/target/recorder/health   std_msgs/String

The coverage line is what the behaviour tree reads to decide a station is DONE — a count of
ACCEPTED evidence, never a stopwatch (SETTLED §3e). Frames and scans are written under the bag
directory as `inspection/<candidate_id>/`, with `frames.jsonl` and `scans.jsonl` manifests that
record REJECTED frames too, with their reason: a rejected frame is a measurement of the water,
and dropping it silently turns a coverage figure into a claim about the camera.

Run: ros2 run sam_target_inspection inspection_recorder --ros-args -p robot_name:=sam21
"""
import json
import os
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    _HAVE_ROS = True
except ImportError:                                            # pragma: no cover
    Node = object
    _HAVE_ROS = False

from sam_target_inspection._health import HealthLine
from sam_target_inspection.fls_target_core import parse_cloud
from sam_target_inspection.inspection_recorder_core import (FrameGate, InspectionRecorderCore,
                                                            ScanGate)


class InspectionRecorder(Node):

    def __init__(self):
        super().__init__("inspection_recorder")
        self.declare_parameter("robot_name", "sam21")
        self.declare_parameter("camera_topic", "payload/realsense/left/image_raw")
        self.declare_parameter("cloud_topic", "payload/sonar3d/points")
        self.declare_parameter("output_root", "")
        self.declare_parameter("min_frames_per_station", 6)
        self.declare_parameter("min_scans_per_station", 3)
        self.declare_parameter("stations_planned", 24)
        self.declare_parameter("stale_timeout_s", 5.0)

        gp = lambda n: self.get_parameter(n).value             # noqa: E731
        self.robot_name = gp("robot_name")
        self.output_root = str(gp("output_root"))
        self.stations_planned = int(gp("stations_planned"))
        self.min_frames = int(gp("min_frames_per_station"))
        self.min_scans = int(gp("min_scans_per_station"))

        ns = f"/{self.robot_name}"
        self.create_subscription(String, f"{ns}/perception/target/capture",
                                 self._capture_cb, 5)
        self.create_subscription(Image, f'{ns}/{gp("camera_topic")}', self._image_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, f'{ns}/{gp("cloud_topic")}', self._cloud_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, f"{ns}/dr/odom", self._odom_cb, 10)
        self.pub_cov = self.create_publisher(String, f"{ns}/perception/target/coverage", 5)
        self.pub_health = self.create_publisher(
            String, f"{ns}/perception/target/recorder/health", 1)

        self.core = None
        self.candidate = None
        self.health = HealthLine(stale_timeout_s=float(gp("stale_timeout_s")))
        self._pose = None
        self._pose_at = None
        self._ms = 0.0
        self._dir = None
        self.create_timer(1.0, self.publish_health)
        self.get_logger().info(
            f"inspection_recorder up on {ns}: IDLE until a capture burst is commanded "
            f"(ADR-010). A station is done on {self.min_frames} accepted frames OR "
            f"{self.min_scans} accepted sonar scans — a count of evidence, not a stopwatch.")

    # ------------------------------------------------------------------ inputs
    def _odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._pose = {"x": float(p.x), "y": float(p.y), "z": float(p.z),
                      "qx": float(q.x), "qy": float(q.y), "qz": float(q.z), "qw": float(q.w)}
        self._pose_at = self._now()

    def _pose_age(self):
        return None if self._pose_at is None else self._now() - self._pose_at

    def _capture_cb(self, msg):
        """`{"station": n, "candidate": id}` opens a burst; a station id of null closes it."""
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        self.health.note_input(self._now())
        cid = d.get("candidate") or "unknown_candidate"
        station = d.get("station")
        if self.core is None or self.candidate != cid:
            self.candidate = cid
            self.core = InspectionRecorderCore(
                cid, min_frames=self.min_frames, min_scans=self.min_scans,
                frame_gate=FrameGate(), scan_gate=ScanGate())
            self._dir = self._make_dir(cid)
        if station is None:
            if self.core.busy:
                self._close_burst()
            return
        if self.core.busy:
            self._close_burst()
        self.core.start_burst(int(station))
        self.get_logger().info(f"capture burst OPEN at station {station} for {cid}")

    def _close_burst(self):
        cov = self.core.end_burst()
        done, why = self.core.station_done(cov.station)
        self.pub_cov.publish(String(data=json.dumps(
            {"station": cov.station, "done": bool(done), "why": why,
             "candidate": self.candidate}, separators=(",", ":"))))
        self._write_manifests()
        self.get_logger().info(f"capture burst CLOSED at station {cov.station}: {why}")

    # ------------------------------------------------------------------ the burst
    def _image_cb(self, msg):
        if self.core is None or not self.core.busy:
            return                                             # IDLE: ADR-010
        t0 = time.perf_counter()
        arr = self._image_array(msg)
        if arr is None:
            return
        ok, why = self.core.offer_frame(arr, t=self._now(), pose=self._pose)
        self.health.note_output(1 if ok else 0)
        self._ms = 1000.0 * (time.perf_counter() - t0)
        self._maybe_report()

    @staticmethod
    def _image_array(msg):
        """A 2-D luma array from the message's own encoding, or None with nothing guessed.

        Only the encodings the sim and the real driver actually publish are handled. An unknown
        encoding returns None rather than being reinterpreted as bytes, because a frame decoded
        under the wrong encoding still produces a Laplacian variance and would be gated on it.
        """
        enc = str(getattr(msg, "encoding", "")).lower()
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        h, w = int(msg.height), int(msg.width)
        if h <= 0 or w <= 0:
            return None
        if enc in ("mono8", "8uc1"):
            return buf[: h * w].reshape(h, w).astype(np.float64)
        if enc in ("rgb8", "bgr8"):
            a = buf[: h * w * 3].reshape(h, w, 3).astype(np.float64)
            return a if enc == "rgb8" else a[..., ::-1]
        if enc in ("rgba8", "bgra8"):
            a = buf[: h * w * 4].reshape(h, w, 4).astype(np.float64)[..., :3]
            return a if enc == "rgba8" else a[..., ::-1]
        return None

    def _cloud_cb(self, msg):
        if self.core is None or not self.core.busy:
            return                                             # IDLE: ADR-010
        t0 = time.perf_counter()
        try:
            xyz = parse_cloud(msg)
        except Exception as e:
            self.get_logger().warn(f"scan not parsed: {e}", throttle_duration_sec=10.0)
            return
        ok, why, thinned = self.core.offer_scan(xyz, t=self._now(), pose=self._pose,
                                                pose_age_s=self._pose_age())
        if ok and thinned is not None and self._dir:
            np.save(os.path.join(self._dir, "sonar3d",
                                 f"scan_{len(self.core.scans):05d}.npy"), thinned)
        self.health.note_output(1 if ok else 0)
        self._ms = 1000.0 * (time.perf_counter() - t0)
        self._maybe_report()

    def _maybe_report(self):
        """Report the moment a station is done, so the tree does not wait a whole timer period."""
        st = self.core._station                                 # noqa: SLF001 - same module family
        if st is None:
            return
        done, why = self.core.station_done(st)
        if done:
            self.pub_cov.publish(String(data=json.dumps(
                {"station": st, "done": True, "why": why, "candidate": self.candidate},
                separators=(",", ":"))))

    # ------------------------------------------------------------------ writing
    def _make_dir(self, cid):
        if not self.output_root:
            self.get_logger().warn(
                "output_root is empty, so frames and scans are GATED AND COUNTED but not "
                "written. The coverage line is still true; there will be nothing for the "
                "station to reconstruct from.")
            return None
        d = os.path.join(self.output_root, "inspection", str(cid))
        try:
            os.makedirs(os.path.join(d, "sonar3d"), exist_ok=True)
        except OSError as e:
            self.get_logger().error(f"cannot create {d}: {e}; nothing will be written")
            return None
        return d

    def _write_manifests(self):
        if not self._dir or self.core is None:
            return
        try:
            with open(os.path.join(self._dir, "frames.jsonl"), "w") as f:
                f.write(self.core.frames_jsonl())
            with open(os.path.join(self._dir, "scans.jsonl"), "w") as f:
                f.write(self.core.scans_jsonl())
        except OSError as e:
            self.get_logger().error(f"manifest not written: {e}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ health
    def publish_health(self):
        if self.core is None:
            ok = "idle; no capture has been commanded this mission"
        else:
            ok = (f"{self.core.coverage_line(self.stations_planned)} · "
                  f"{'BURST OPEN' if self.core.busy else 'idle between stations'} · "
                  f"{self._ms:.1f} ms/frame · writing to "
                  f"{self._dir or 'NOWHERE (output_root unset)'}")
        line = self.health.compute(self._now(), ok_detail=ok)
        self.pub_health.publish(String(data=line))


def main(args=None):
    if not _HAVE_ROS:                                          # pragma: no cover
        raise SystemExit("inspection_recorder needs rclpy; the pure core in "
                         "sam_target_inspection.inspection_recorder_core runs without it.")
    rclpy.init(args=args)
    node = InspectionRecorder()
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
