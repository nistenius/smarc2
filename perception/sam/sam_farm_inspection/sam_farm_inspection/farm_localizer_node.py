#!/usr/bin/env python3
"""`farm_localizer` — sensor-relative detections in, a verified farm map out.

The ROS half of sensors-22-05064 §6. All of the mathematics is in `localizer.py` and is
tested without a graph; this file does three jobs and nothing else:

  1. PLACE each detection in the world, which needs a pose, a target depth and the
     slant-range correction (IROS 2025 Eq. 1);
  2. ACCUMULATE detections over the encircle;
  3. RUN the localizer periodically and publish the report and a health line.

THE POSE SOURCE IS A PARAMETER AND IT IS NAMED IN EVERY REPORT. In simulation Unity
publishes `smarc/odom` and `core/odom_gt` from ground truth (SETTLED §4), so a localizer
reading either would produce a map that is correct because it was told the answer — the
exact failure invariant 11 exists to prevent, one level up from a health probe. The
default is the estimator's own `dr/odom`. If a rig session needs to run against ground
truth to isolate a problem, that is a legitimate experiment and the parameter allows it;
what is not allowed is the report failing to say which it was, so `pose_topic` travels
in the report and on the health line.

WHAT IT PUBLISHES (both solely owned by this node — invariant 11):
  <ns>/perception/farm/report            std_msgs/String, JSON: transform, per-buoy
                                         verdicts, line fits, refusal reasons, caveats
  <ns>/perception/farm/localizer/health  std_msgs/String, STATE|rate|age|dets|clusters|detail

WHAT IS NOT HERE YET (P5): the behaviour tree's trigger. Today the report is recomputed
on a timer once enough detections have arrived, and `perception/farm/command` accepts
"reset". When the inspection task exists, T2 will finalize explicitly and T3 will consume
the finalized map. The timer is a development convenience and is labelled as one.
"""
import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from sam_farm_inspection.farm_prior import PriorRefusal, load_farm_prior
from sam_farm_inspection.localizer import WorldDetection, localize_farm
from sam_farm_inspection.sss_geometry import slant_to_ground_range


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class FarmLocalizer(Node):

    def __init__(self):
        super().__init__("farm_localizer")

        self.declare_parameter("robot_name", "sam21")
        self.declare_parameter("detections_topic", "perception/farm/detections")
        # See the module docstring. `smarc/odom` and `core/odom_gt` are Unity's ground
        # truth in sim; using them makes the map correct by construction and proves
        # nothing about the detector or the fit.
        self.declare_parameter("pose_topic", "dr/odom")
        self.declare_parameter("farm_prior_path", "")
        # Detection accuracy, for the mixture's fixed observation covariance. IROS 2025
        # Table I puts the achievable buoy RMSE around 1 m for the MVP method.
        self.declare_parameter("obs_sigma_m", 1.0)
        self.declare_parameter("confirm_radius_m", 2.0)
        self.declare_parameter("min_buoy_detections", 12)
        self.declare_parameter("report_period_s", 5.0)
        self.declare_parameter("pose_timeout_s", 3.0)
        self.declare_parameter("require_coverage", True)
        # Sonar mounting: which way the port beam looks relative to the hull's heading.
        # -90 deg = port is to the left of the bow, which is the SSS convention in
        # Sonar.cs (beamNum 0 is the -1 side).
        self.declare_parameter("port_bearing_offset_deg", -90.0)

        gp = lambda n: self.get_parameter(n).value            # noqa: E731
        self.robot_name = gp("robot_name")
        self.pose_topic = gp("pose_topic")
        self.obs_sigma_m = float(gp("obs_sigma_m"))
        self.confirm_m = float(gp("confirm_radius_m"))
        self.min_buoy_dets = int(gp("min_buoy_detections"))
        self.pose_timeout_s = float(gp("pose_timeout_s"))
        self.require_coverage = bool(gp("require_coverage"))
        self.port_offset_deg = float(gp("port_bearing_offset_deg"))

        # The prior is loaded ONCE, at startup, and a failure is fatal. A localizer with
        # no prior has nothing to verify against, and the honest response is to refuse to
        # come up rather than to publish reports about a farm it invented.
        path = gp("farm_prior_path") or None
        try:
            self.prior = load_farm_prior(path)
        except PriorRefusal as e:
            self.get_logger().fatal(str(e))
            raise

        ns = f"/{self.robot_name}"
        self.create_subscription(String, f'{ns}/{gp("detections_topic")}',
                                 self.det_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, f"{ns}/{self.pose_topic}",
                                 self.pose_cb, qos_profile_sensor_data)
        self.create_subscription(String, f"{ns}/perception/farm/command",
                                 self.cmd_cb, 5)
        self.pub_report = self.create_publisher(String, f"{ns}/perception/farm/report", 1)
        self.pub_health = self.create_publisher(String, f"{ns}/perception/farm/localizer/health", 1)

        self.dets = []
        self.track = []
        self.pose = None            # (x, y, depth_m, yaw_rad)
        self.pose_at = None
        self._n_pings = 0
        self._n_dropped = 0
        self._drop_reason = ""
        self._last_fix = None
        self._health_state = None

        self.create_timer(float(gp("report_period_s")), self.recompute)
        self.create_timer(1.0, self.publish_health)
        self.get_logger().info(
            f"farm_localizer up: prior {self.prior.path} ({self.prior.n_buoys} buoys, "
            f"ropes at {self.prior.rope_depth_m:.1f} m), pose from {ns}/{self.pose_topic}, "
            f"reports on {ns}/perception/farm/report")
        for c in self.prior.caveats:
            self.get_logger().warn("prior caveat: " + c)

    # ------------------------------------------------------------------ inputs
    def pose_cb(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        # Depth positive-down. `dr/odom` is ENU-ish with z up, so depth = -z.
        self.pose = (float(p.x), float(p.y), -float(p.z),
                     yaw_from_quat(q.x, q.y, q.z, q.w))
        self.pose_at = self.get_clock().now()
        self.track.append((self.pose[0], self.pose[1]))
        if len(self.track) > 20000:
            self.track = self.track[-20000:]

    def cmd_cb(self, msg: String):
        if msg.data.strip().lower() == "reset":
            n = len(self.dets)
            self.dets, self.track, self._last_fix = [], [], None
            self.get_logger().info(f"reset: dropped {n} accumulated detections")

    def det_cb(self, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception as e:
            self._note_drop(f"unparsable detections message: {e}")
            return
        self._n_pings += 1

        now = self.get_clock().now()
        if self.pose is None:
            self._note_drop("no pose yet — detections cannot be placed in the world")
            return
        age = (now - self.pose_at).nanoseconds * 1e-9
        if age > self.pose_timeout_s:
            # A stale pose does not make a detection approximately right; it makes it
            # wrong by however far the vehicle moved, and the map absorbs that as farm
            # displacement — the finding the mission exists to report.
            self._note_drop(f"pose is {age:.1f} s old (> {self.pose_timeout_s} s)")
            return

        vx, vy, vdepth, yaw = self.pose
        for side, block in (d.get("channels") or {}).items():
            if not block.get("ok"):
                continue
            sign = 1.0 if side == "port" else -1.0
            bearing = yaw + math.radians(self.port_offset_deg) * sign
            for det in block.get("detections", []):
                target_depth = self.prior.rope_depth_m if det["target"] == "rope" else 0.0
                dz = target_depth - vdepth
                ground = slant_to_ground_range(float(det["slant_range_m"]), abs(dz))
                if ground is None:
                    # Geometrically impossible: the slant range is shorter than the depth
                    # difference. Dropping it is right; clamping it to zero would put a
                    # phantom detection directly under the vehicle.
                    self._note_drop("slant range shorter than the depth difference")
                    continue
                self.dets.append(WorldDetection(
                    x=vx + ground * math.cos(bearing),
                    y=vy + ground * math.sin(bearing),
                    target=det["target"],
                    confidence=float(det.get("confidence", 0.0)),
                    look_bearing_deg=math.degrees(bearing) % 360.0))

    def _note_drop(self, why):
        self._n_dropped += 1
        self._drop_reason = why
        self.get_logger().warn(f"dropped detections: {why}", throttle_duration_sec=10.0)

    # ------------------------------------------------------------------ the fit
    def recompute(self):
        n_buoy = sum(1 for d in self.dets if d.target == "buoy")
        if n_buoy < self.min_buoy_dets:
            return
        fix = localize_farm(self.dets, self.prior.buoys, self.prior.line_endpoints(),
                            covered_points=self.track, obs_sigma_m=self.obs_sigma_m,
                            confirm_m=self.confirm_m,
                            require_coverage=self.require_coverage)
        self._last_fix = fix
        report = {
            "ok": bool(fix.ok),
            "reason": fix.reason,
            # Provenance travels with the answer, always. A map is only as good as the
            # pose that placed it and the sonar model that produced it, and neither is
            # visible in the numbers.
            "pose_topic": self.pose_topic,
            "prior_path": self.prior.path,
            "n_detections": len(self.dets),
            "n_buoy_detections": n_buoy,
            "n_dropped": self._n_dropped,
            "rotation_deg": round(fix.rotation_deg, 3),
            "translation_m": [round(v, 3) for v in fix.translation_m],
            "rms_m": round(fix.rms_m, 3),
            "n_clusters": fix.n_clusters,
            "n_matched": fix.n_matched,
            "buoys": [{"name": v.name, "status": v.status,
                       "residual_m": (round(v.residual_m, 2) if v.residual_m is not None else None),
                       "predicted_xz": [round(c, 2) for c in v.predicted_xy],
                       "observed_xz": ([round(c, 2) for c in v.observed_xy]
                                       if v.observed_xy else None)}
                      for v in fix.verdicts],
            "lines": [{"name": ln.name, "ok": ln.ok, "reason": ln.reason,
                       "bearing_deg": round(ln.bearing_deg, 2),
                       "offset_m": round(ln.offset_m, 3), "rms_m": round(ln.rms_m, 3),
                       "n_points": ln.n_points}
                      for ln in fix.lines],
            "caveats": self.prior.caveats,
        }
        self.pub_report.publish(String(data=json.dumps(report, separators=(",", ":"))))
        if fix.ok:
            counts = {}
            for v in fix.verdicts:
                counts[v.status] = counts.get(v.status, 0) + 1
            self.get_logger().info(
                f"farm fix: rot {fix.rotation_deg:+.1f} deg, rms {fix.rms_m:.2f} m, "
                f"{fix.n_matched}/{self.prior.n_buoys} matched, " +
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        else:
            self.get_logger().warn(f"farm fix refused: {fix.reason}")

    # ------------------------------------------------------------------ health
    def publish_health(self):
        n_buoy = sum(1 for d in self.dets if d.target == "buoy")
        if self.pose is None:
            state, detail = "NO_POSE", f"nothing on {self.pose_topic} yet"
        elif self._n_pings == 0:
            state, detail = "NO_INPUT", "no detection messages received"
        elif n_buoy < self.min_buoy_dets:
            state = "GATHERING"
            detail = (f"{n_buoy}/{self.min_buoy_dets} buoy detections; "
                      f"{self._n_dropped} dropped"
                      + (f" (last: {self._drop_reason})" if self._n_dropped else ""))
        elif self._last_fix is None:
            state, detail = "GATHERING", "enough detections, first fit not run yet"
        elif not self._last_fix.ok:
            state, detail = "REFUSED", self._last_fix.reason
        else:
            f = self._last_fix
            state = "OK"
            detail = (f"rms {f.rms_m:.2f} m, {f.n_matched}/{self.prior.n_buoys} matched, "
                      f"pose from {self.pose_topic}")

        self.pub_health.publish(String(data=(
            f"{state}|{self._n_pings}|{len(self.dets)}|{n_buoy}|"
            f"{self._last_fix.n_clusters if self._last_fix else 0}|{detail}")))
        if state != self._health_state:
            msg = f"localizer health: {self._health_state} -> {state} ({detail})"
            if state in ("OK", "GATHERING"):
                self.get_logger().info(msg)
            else:
                self.get_logger().error(msg)
            self._health_state = state


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FarmLocalizer()
    except PriorRefusal:
        # Already logged as FATAL with the fix named. Exit non-zero: a localizer that
        # cannot load its prior has not "started with a warning", it has not started.
        if rclpy.ok():
            rclpy.shutdown()
        raise SystemExit(2)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # A clean stop must exit 0 — see SETTLED §1c, the duplicate-bringup chain.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
