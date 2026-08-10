#!/usr/bin/env python3
"""Obstacle detector on the forward 3D sonar — collision-avoidance session, 2026-08-10.

Layer 1 of the avoidance stack (see data-cube/docs/2026-08-10-obstacle-avoidance-session.md):
the simplest representation that converts the two wall-strike bags (143736 cascade,
165932 full-sensor) into information a controller can act on.

    PointCloud2 (payload/sonar3d/points, world frame in sim)
      -> TF into a body frame
      -> gate: no-hit points, near-field, max range, vertical band
      -> polar sector grid (azimuth x elevation), robust min range per sector
      -> nearest range/bearing + a stop flag from a speed-dependent envelope

The stop envelope (Layer 2a, "protective stop"):

    R_stop(u) = margin + u * t_react + u^2 / (2 * a_stop)

with margin defaulting to 2.5 m = the measured nav+turn dispersion (SAFE_zigzag DR
0.94 m max + ~1-2 m of corner-cut spread, 2026-08-09 late session). Only obstacles
inside a forward cone (default +/-35 deg azimuth) trigger the stop — the dry-dock
walls run parallel to the legs a few metres abeam and must not halt a transit.
Hysteresis (release at trigger + 1 m, minimum hold time) prevents chattering on the
noisy sonar range.

FRAMES — read this before changing body_frame:
  In sim, Unity publishes the cloud in `unity_origin` (world). Transforming those
  GT world points by the *estimated* base_link would corrupt every range by the nav
  error — which the physical sensor does not do: a real sonar measures range relative
  to itself. So in sim the launch passes body_frame:=<robot>/base_link_gt, which
  reconstructs exactly what the sensor physically measures. This is NOT navigation
  cheating: no world-frame information leaves this node; only body-relative ranges
  and bearings, the same thing the hardware driver will output natively (the real
  WL Sonar 3D-15 driver publishes sensor-frame points, for which body_frame is just
  the static sonar3d_link->base_link mount transform).

Stale-input policy: if no cloud arrives for cloud_timeout seconds the stop flag is
released and a warning is logged. That is the sim-friendly choice; on hardware a
silent obstacle sensor in confined water should arguably fail CLOSED — revisit
before any real-vehicle deployment (flagged in the session doc).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


def quat_to_rot(x, y, z, w):
    """Quaternion -> 3x3 rotation matrix (numpy, no external deps)."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


class ObstacleDetector(Node):

    def __init__(self):
        super().__init__("obstacle_detector")

        self.declare_parameter("robot_name", "sam_auv_v1")
        # Body frame the sectors are computed in. See FRAMES note in the module
        # docstring: base_link_gt in sim (physically-correct ranges), base_link on
        # hardware (where the cloud is already sensor-frame).
        self.declare_parameter("body_frame", "base_link_gt")
        self.declare_parameter("cloud_topic", "payload/sonar3d/points")
        self.declare_parameter("odom_topic", "dr/odom")  # speed source for the envelope

        # Gating
        self.declare_parameter("range_min", 0.8)     # m; self/near-field rejection
        self.declare_parameter("range_max", 14.5)    # m; sonar is 15 m
        self.declare_parameter("z_band", 1.5)        # m; keep |z_body| <= z_band (walls, not seabed at depth)
        self.declare_parameter("intensity_min", 1)   # no-hit rays encode intensity 0 at world origin

        # Sector grid (matches the WL 3D-15 fan: 90 deg H x 40 deg V)
        self.declare_parameter("n_az", 5)
        self.declare_parameter("n_el", 3)
        self.declare_parameter("az_half_deg", 45.0)
        self.declare_parameter("el_half_deg", 20.0)
        self.declare_parameter("k_smallest", 3)      # robust "min": k-th smallest range in a sector

        # Stop envelope
        self.declare_parameter("stop_enable", True)
        self.declare_parameter("stop_cone_deg", 35.0)   # forward cone that can trigger a stop
        self.declare_parameter("stop_margin", 2.5)      # m; measured dispersion 2026-08-09
        self.declare_parameter("stop_t_react", 1.0)     # s; detector 5 Hz + controller 10 Hz + slew
        self.declare_parameter("stop_a_stop", 0.1)      # m/s^2; conservative coast/brake decel
        self.declare_parameter("stop_hysteresis", 1.0)  # m; release at R_stop + this
        self.declare_parameter("stop_min_hold", 3.0)    # s
        # Timed retry (Ivan, 2026-08-10): after this long stopped with the obstacle
        # still there, release deliberately — the obstacle may have been a small
        # moving object or noise, and re-approaching re-triggers cleanly on the
        # latched envelope. The CONTROLLER counts re-triggers and aborts past its
        # retry budget (blend_obstacle_retries), so trials end in abort + surface.
        # 0.0 = OFF (default since 2026-08-10, Ivan): a timed release drives the
        # vehicle FORWARD into an obstacle that is still inside the envelope — the
        # closest approach crept 4.1 -> 2.7 m across cycles in run 20260810_093534.
        # The stop now clears only when the obstacle genuinely clears (latched
        # envelope + hysteresis); persistence is handled by the controller's hold
        # timeout, which aborts instead of nudging closer.
        self.declare_parameter("stop_retry_wait", 0.0)  # s; >0 re-enables timed release
        self.declare_parameter("cloud_timeout", 2.0)    # s; stale-input policy above

        self.declare_parameter("publish_markers", True)

        gp = lambda n: self.get_parameter(n).value
        self.robot_name = gp("robot_name")
        self.body_frame = f'{self.robot_name}/{gp("body_frame")}'
        self.range_min = float(gp("range_min"))
        self.range_max = float(gp("range_max"))
        self.z_band = float(gp("z_band"))
        self.intensity_min = int(gp("intensity_min"))
        self.n_az = int(gp("n_az"))
        self.n_el = int(gp("n_el"))
        self.az_half = np.radians(float(gp("az_half_deg")))
        self.el_half = np.radians(float(gp("el_half_deg")))
        self.k_smallest = int(gp("k_smallest"))
        self.stop_enable = bool(gp("stop_enable"))
        self.stop_cone = np.radians(float(gp("stop_cone_deg")))
        self.stop_margin = float(gp("stop_margin"))
        self.stop_t_react = float(gp("stop_t_react"))
        self.stop_a_stop = float(gp("stop_a_stop"))
        self.stop_hyst = float(gp("stop_hysteresis"))
        self.stop_min_hold = float(gp("stop_min_hold"))
        self.stop_retry_wait = float(gp("stop_retry_wait"))
        self.cloud_timeout = float(gp("cloud_timeout"))
        self.publish_markers = bool(gp("publish_markers"))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        ns = f"/{self.robot_name}"
        self.sub_cloud = self.create_subscription(
            PointCloud2, f'{ns}/{gp("cloud_topic")}', self.cloud_cb, qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(
            Odometry, f'{ns}/{gp("odom_topic")}', self.odom_cb, qos_profile_sensor_data)

        self.pub_sectors = self.create_publisher(Float32MultiArray, f"{ns}/perception/obstacle/sectors", 1)
        self.pub_nearest = self.create_publisher(PointStamped, f"{ns}/perception/obstacle/nearest", 1)
        self.pub_range = self.create_publisher(Float32, f"{ns}/perception/obstacle/nearest_range", 1)
        self.pub_stop = self.create_publisher(Bool, f"{ns}/perception/obstacle/stop", 1)
        self.pub_markers = self.create_publisher(MarkerArray, f"{ns}/perception/obstacle/markers", 1)

        self.speed = 0.0
        self.stop_active = False
        self.stop_since = None
        # Envelope latch (added after run 20260810_010725: the envelope breathes with
        # speed — stopped -> u drops -> R_stop shrinks -> premature release -> vehicle
        # accelerates -> R_stop grows -> re-trigger, a ~15 s limit cycle measured at
        # t+0:27-0:44. Freeze R_stop at its trigger-time value while active; release
        # only against the frozen value + hysteresis.)
        self.r_stop_latched = None
        self.last_cloud_time = None
        self._tf_warned = False
        self._n_clouds = 0

        # Watchdog: publish stop (and staleness warnings) even when clouds stop coming.
        self.create_timer(0.5, self.watchdog)

        self.get_logger().info(
            f"Obstacle detector up: cloud={ns}/{gp('cloud_topic')} -> body {self.body_frame}, "
            f"grid {self.n_az}x{self.n_el} over ±{np.degrees(self.az_half):.0f}°/±{np.degrees(self.el_half):.0f}°, "
            f"stop margin {self.stop_margin} m + envelope, cone ±{np.degrees(self.stop_cone):.0f}°")

    def refresh_live_params(self):
        """Envelope/retry params are read LIVE (2026-08-10) so they can be tuned with
        `ros2 param set` between runs — the stopping-distance sweep changes them every
        run and a node restart per point would be painful. Geometry/gating params stay
        startup-only (they size preallocated work)."""
        try:
            self.stop_margin = float(self.get_parameter("stop_margin").value)
            self.stop_t_react = float(self.get_parameter("stop_t_react").value)
            self.stop_a_stop = float(self.get_parameter("stop_a_stop").value)
            self.stop_hyst = float(self.get_parameter("stop_hysteresis").value)
            self.stop_min_hold = float(self.get_parameter("stop_min_hold").value)
            self.stop_retry_wait = float(self.get_parameter("stop_retry_wait").value)
            self.stop_cone = np.radians(float(self.get_parameter("stop_cone_deg").value))
            self.stop_enable = bool(self.get_parameter("stop_enable").value)
        except Exception:
            pass

    # ------------------------------------------------------------------ inputs
    def odom_cb(self, msg: Odometry):
        self.speed = abs(msg.twist.twist.linear.x)

    def cloud_cb(self, msg: PointCloud2):
        self.last_cloud_time = self.get_clock().now()
        self.refresh_live_params()
        pts = self.parse_cloud(msg)
        if pts is None:
            return
        body = self.to_body(pts, msg.header.frame_id)
        if body is None:
            return
        self.process(body, msg)

    # ------------------------------------------------------------------ pipeline
    def parse_cloud(self, msg):
        """Unity's SonarPointCloud_Pub: x,y,z float32 + intensity uint8, 13-byte step.
        Falls back to a generic parse if the layout ever changes."""
        try:
            if msg.point_step == 13:
                dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "u1")])
                arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
                xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
                inten = arr["i"].astype(np.int32)
            else:
                from sensor_msgs_py import point_cloud2
                arr = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
                xyz = arr.astype(np.float64)
                inten = np.full(len(xyz), 255, dtype=np.int32)
        except Exception as e:
            self.get_logger().warn(f"cloud parse failed: {e}", throttle_duration_sec=10.0)
            return None
        # No-hit rays: intensity 0, point at the world origin. Drop them first.
        keep = inten >= self.intensity_min
        return xyz[keep]

    def to_body(self, xyz, cloud_frame):
        if cloud_frame == self.body_frame:
            return xyz
        try:
            # Latest available transform: sim-time stamps + the bridge make exact-stamp
            # lookups flaky, and at 5 Hz / <1 m/s the pose moves <0.2 m per cloud.
            t = self.tf_buffer.lookup_transform(self.body_frame, cloud_frame, Time(),
                                                timeout=Duration(seconds=0.1))
        except Exception as e:
            if not self._tf_warned:
                self.get_logger().warn(f"TF {cloud_frame} -> {self.body_frame} not available yet: {e}")
                self._tf_warned = True
            return None
        self._tf_warned = False
        q = t.transform.rotation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        return xyz @ R.T + p

    def process(self, body, msg):
        x, y, z = body[:, 0], body[:, 1], body[:, 2]
        rng = np.linalg.norm(body, axis=1)
        az = np.arctan2(y, x)
        el = np.arctan2(z, np.hypot(x, y))

        keep = ((rng >= self.range_min) & (rng <= self.range_max)
                & (np.abs(z) <= self.z_band)
                & (np.abs(az) <= self.az_half) & (np.abs(el) <= self.el_half))
        body, rng, az, el = body[keep], rng[keep], az[keep], el[keep]

        sectors = np.full((self.n_az, self.n_el), np.inf)
        if len(rng) > 0:
            ia = np.clip(((az + self.az_half) / (2 * self.az_half) * self.n_az).astype(int), 0, self.n_az - 1)
            ie = np.clip(((el + self.el_half) / (2 * self.el_half) * self.n_el).astype(int), 0, self.n_el - 1)
            for a in range(self.n_az):
                for e in range(self.n_el):
                    r = rng[(ia == a) & (ie == e)]
                    if len(r) >= self.k_smallest:
                        sectors[a, e] = np.partition(r, self.k_smallest - 1)[self.k_smallest - 1]
                    elif len(r) > 0:
                        sectors[a, e] = float(np.min(r))

        # nearest point overall (after gating)
        if len(rng) > 0:
            i_min = int(np.argmin(rng))
            nearest_range = float(rng[i_min])
            nearest_az = float(az[i_min])
            nearest_pt = body[i_min]
        else:
            nearest_range, nearest_az, nearest_pt = np.inf, 0.0, None

        # ---- stop envelope (forward cone only) ----
        in_cone = np.abs(az) <= self.stop_cone
        r_cone = float(np.min(rng[in_cone])) if np.any(in_cone) else np.inf
        u = self.speed
        r_stop = self.stop_margin + u * self.stop_t_react + u * u / (2.0 * self.stop_a_stop)
        now = self.get_clock().now()
        if self.stop_enable:
            if not self.stop_active and r_cone <= r_stop:
                self.stop_active = True
                self.stop_since = now
                self.r_stop_latched = r_stop
                self.get_logger().warn(
                    f"PROTECTIVE STOP: obstacle {r_cone:.1f} m in forward cone "
                    f"(R_stop {r_stop:.1f} m at u={u:.2f} m/s, latched)")
            elif self.stop_active:
                held = (now - self.stop_since).nanoseconds * 1e-9 if self.stop_since else 1e9
                r_release = (self.r_stop_latched if self.r_stop_latched is not None else r_stop)
                if r_cone > r_release + self.stop_hyst and held >= self.stop_min_hold:
                    self.stop_active = False
                    self.r_stop_latched = None
                    self.get_logger().info(f"Protective stop released (clear to {r_cone:.1f} m)")
                elif held >= self.stop_retry_wait > 0.0:
                    # Timed retry: obstacle still there after the wait — release and
                    # let the vehicle try again. Re-trigger lands on the latched
                    # envelope; the controller's retry budget turns persistent
                    # obstacles into an abort.
                    self.stop_active = False
                    self.r_stop_latched = None
                    self.get_logger().warn(
                        f"Protective stop RETRY release after {held:.0f} s "
                        f"(obstacle still at {r_cone:.1f} m)")

        # ---- publish ----
        m = Float32MultiArray()
        m.layout.dim = [MultiArrayDimension(label="az", size=self.n_az, stride=self.n_az * self.n_el),
                        MultiArrayDimension(label="el", size=self.n_el, stride=self.n_el)]
        m.data = [float(v) if np.isfinite(v) else -1.0 for v in sectors.flatten()]
        self.pub_sectors.publish(m)

        self.pub_range.publish(Float32(data=float(nearest_range if np.isfinite(nearest_range) else -1.0)))
        if nearest_pt is not None:
            ps = PointStamped()
            ps.header.stamp = msg.header.stamp
            ps.header.frame_id = self.body_frame
            ps.point.x, ps.point.y, ps.point.z = (float(v) for v in nearest_pt)
            self.pub_nearest.publish(ps)
        self.pub_stop.publish(Bool(data=self.stop_active))

        if self.publish_markers:
            self.publish_sector_markers(sectors, msg)

        self._n_clouds += 1
        if self._n_clouds % 50 == 1:
            self.get_logger().info(
                f"clouds {self._n_clouds}: {len(rng)} gated pts, nearest "
                f"{nearest_range if np.isfinite(nearest_range) else float('nan'):.1f} m "
                f"@ {np.degrees(nearest_az):.0f}°, cone {r_cone if np.isfinite(r_cone) else float('nan'):.1f} m, "
                f"R_stop {r_stop:.1f} m, stop={self.stop_active}")

    def watchdog(self):
        if self.last_cloud_time is None:
            return
        age = (self.get_clock().now() - self.last_cloud_time).nanoseconds * 1e-9
        if age > self.cloud_timeout:
            if self.stop_active:
                self.get_logger().warn(
                    f"Sonar silent {age:.1f} s — releasing protective stop (sim policy; "
                    "hardware should fail closed, see session doc)")
                self.stop_active = False
            self.pub_stop.publish(Bool(data=False))
            self.get_logger().warn(f"no sonar cloud for {age:.1f} s", throttle_duration_sec=5.0)

    def publish_sector_markers(self, sectors, msg):
        ma = MarkerArray()
        mid = 0
        for a in range(self.n_az):
            for e in range(self.n_el):
                r = sectors[a, e]
                if not np.isfinite(r):
                    continue
                az_c = -self.az_half + (a + 0.5) * 2 * self.az_half / self.n_az
                el_c = -self.el_half + (e + 0.5) * 2 * self.el_half / self.n_el
                mk = Marker()
                mk.header.frame_id = self.body_frame
                mk.header.stamp = msg.header.stamp
                mk.ns = "obstacle_sectors"
                mk.id = mid
                mid += 1
                mk.type = Marker.SPHERE
                mk.action = Marker.ADD
                mk.pose.position.x = float(r * np.cos(el_c) * np.cos(az_c))
                mk.pose.position.y = float(r * np.cos(el_c) * np.sin(az_c))
                mk.pose.position.z = float(r * np.sin(el_c))
                mk.pose.orientation.w = 1.0
                mk.scale.x = mk.scale.y = mk.scale.z = 0.5
                mk.color.a = 0.8
                mk.color.r = 1.0 if r < 5.0 else 0.2
                mk.color.g = 0.2 if r < 5.0 else 1.0
                mk.color.b = 0.1
                mk.lifetime = Duration(seconds=1.0).to_msg()
                ma.markers.append(mk)
        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
