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

HEALTH (2026-08-12): every output of this node is ambiguous read alone — a clear
leg, a dead input, a failing TF and a gate that discards 100 % of every cloud all
publish `nearest_range = -1` and `stop = False`. `perception/obstacle/health`
resolves the ambiguity (see publish_health), and the cockpit refuses to paint the
obstacle field green without it. Do not remove it to "simplify the topic list":
the four cases above cost a full day of flights on 2026-08-12 precisely because
nothing distinguished them.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayDimension, String
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
        # smarc/odom, NOT dr/odom. Measured on the rig 2026-08-14:
        #   /sam_auv_v1/dr/odom   Publisher count: 0   Subscription count: 5
        # Five nodes, this one included, waiting on a topic nobody publishes. The
        # estimator's output is smarc/odom -- that is the topic bringup_nodes.yaml probes
        # for state_estimator's health, and it carries a sane twist.
        self.declare_parameter("odom_topic", "smarc/odom")  # speed source for the envelope
        # How long the speed may go unheard before it counts as UNKNOWN rather than zero.
        self.declare_parameter("stop_u_timeout", 3.0)   # s

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
        # Ivan, 2026-08-14: 0.5 m. Beckholmen's Vastra dockan is 20 m wide, and the previous
        # trio (2.5 / 1.0 / 0.1) put the trigger at
        #     R_stop = 2.5 + 0.5*1.0 + 0.5^2/(2*0.1) = 4.25 m at 0.5 m/s
        # -- measured live, and the reason every dock run ended in a protective stop.
        #
        # a_stop 0.1 m/s^2 is a COASTING assumption; the file's own note says to raise it
        # once the crash-stop distance is measured. 0.35 assumes active braking and is
        # still a guess, but a less pessimistic one. The trio now gives
        #     0.0 m/s -> 0.50 m    0.3 -> 0.93 m    0.5 -> 1.36 m    1.0 -> 2.93 m
        # i.e. tight at dock speed and still growing with speed, which is the entire
        # purpose of having reaction and braking terms at all. A FLAT 0.5 m is what put SAM
        # into the wall on 2026-08-13; this is not that.
        #
        # NOT MEASURED on this hull. See docs/hardware-affecting-changes.md item 7.
        self.declare_parameter("stop_margin", 0.5)      # m; dock margin, Ivan 2026-08-14
        self.declare_parameter("stop_t_react", 1.0)     # s; detector 5 Hz + controller 10 Hz + slew
        self.declare_parameter("stop_a_stop", 0.35)     # m/s^2; assumes ACTIVE braking
        # Physical ceiling on the surge speed the envelope will believe. SAM does not do
        # 2 m/s; anything above this is an estimator fault, not a fast vehicle.
        self.declare_parameter("stop_u_max", 2.0)       # m/s
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
        self.stop_u_max = float(gp("stop_u_max"))
        self.stop_u_timeout = float(gp("stop_u_timeout"))
        self._speed_at = None
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
        # The sensor's LIVE horizon ("Mode|range|hz"). The stop envelope is
        # computed in metres ahead; if the sonar's horizon is SHORTER than
        # R_stop(u), the vehicle is committed to a stop whose trigger it cannot
        # observe. R_stop(0.5 m/s) = 4.25 m against a 4 m inspection horizon, so
        # this is reachable, not theoretical.
        self.sonar_range = None
        self.sonar_mode = "?"
        self.create_subscription(String, f"{ns}/payload/sonar3d/mode",
                                 self.mode_cb, 5)

        self.pub_sectors = self.create_publisher(Float32MultiArray, f"{ns}/perception/obstacle/sectors", 1)
        self.pub_nearest = self.create_publisher(PointStamped, f"{ns}/perception/obstacle/nearest", 1)
        self.pub_range = self.create_publisher(Float32, f"{ns}/perception/obstacle/nearest_range", 1)
        self.pub_stop = self.create_publisher(Bool, f"{ns}/perception/obstacle/stop", 1)
        self.pub_markers = self.create_publisher(MarkerArray, f"{ns}/perception/obstacle/markers", 1)
        # Health line (2026-08-12, after the day of five silent failures).
        # WHY THIS EXISTS: every output below this node is ambiguous on its own.
        # `nearest_range = -1` means "nothing in the gated volume" — which is
        # what a clear leg looks like AND what a detector whose input is dead,
        # whose TF never resolves, or whose gate discards 100 % of every cloud
        # looks like. The HUD painted all four green. This topic is the node
        # saying which one it is; the cockpit refuses to paint green without it.
        # A plain String: no message generation on either side (same pattern as
        # ctrl/setpoints), and it must never be the thing that fails to build.
        self.pub_health = self.create_publisher(String, f"{ns}/perception/obstacle/health", 1)

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
        # Health bookkeeping (see pub_health). Counts are per-stage so the
        # ambiguity in nearest_range can be resolved from the outside:
        #   n_raw   points in the message
        #   n_hits  after the no-hit/intensity gate   <- sensor actually returned
        #   n_gated after range/z/cone gating         <- what the sectors see
        # hits > 0 with gated == 0 for several clouds running is the range-gate
        # failure; hits == 0 is honest open water.
        self._n_raw = self._n_hits = self._n_gated = self._n_inrange = 0
        self._gate_counts = (0, 0, 0, 0)
        self._blind_streak = 0
        self._tf_ok = True
        self._rate_hz = 0.0
        self._rate_t0 = None
        self._rate_n0 = 0
        self._health_state = None

        # Watchdog: publish stop (and staleness warnings) even when clouds stop coming.
        self.create_timer(0.5, self.watchdog)
        self.create_timer(0.5, self.publish_health)

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
            self.stop_u_max = float(self.get_parameter("stop_u_max").value)
            self.stop_hyst = float(self.get_parameter("stop_hysteresis").value)
            self.stop_min_hold = float(self.get_parameter("stop_min_hold").value)
            self.stop_retry_wait = float(self.get_parameter("stop_retry_wait").value)
            self.stop_cone = np.radians(float(self.get_parameter("stop_cone_deg").value))
            self.stop_enable = bool(self.get_parameter("stop_enable").value)
        except Exception:
            pass

    # ------------------------------------------------------------------ inputs
    def odom_cb(self, msg: Odometry):
        """Surge speed for the stop envelope -- VALIDATED, because R_stop is unbounded in it.

        R_stop = margin + u*t_react + u^2/(2*a_stop). That is quadratic in u, so a bad
        speed does not degrade the envelope, it detonates it. Measured on the rig
        2026-08-14: the estimator published u = 2.1e5 m/s ("DR no data" on the HUD at the
        same moment) and R_stop came out at 2.3e13 m. Everything in the world is inside
        that radius, so the detector latched a protective stop it could never release, the
        BT aborted, and the emergency flag blocked every subsequent mission.

        This was invisible until 2026-08-14 only because the envelope had been running
        FLAT (t_react = 0, a_stop = 1e6), where u contributes nothing. Switching to the
        speed-dependent envelope did not introduce the bug; it revealed one that had been
        sitting under the protective stop the whole time.

        An implausible reading is clamped UP, not discarded: not knowing your speed is not
        a reason to assume you are stopped. It is logged every time, because a stop layer
        quietly running on a fallback is exactly the kind of thing that should be loud.
        """
        u = msg.twist.twist.linear.x
        if not math.isfinite(u):
            self._note_bad_speed("non-finite", u)
            self.speed = self.stop_u_max
            return
        u = abs(float(u))
        if u > self.stop_u_max:
            self._note_bad_speed("implausible", u)
            u = self.stop_u_max
        self.speed = u
        self._speed_at = self.get_clock().now()

    def _envelope_speed(self, now):
        """The speed the envelope may believe. UNKNOWN is not zero.

        `self.speed` initialises to 0.0 and is only ever written by odom_cb. So when the
        speed source is absent -- exactly the state found on the rig 2026-08-14, where
        dr/odom had zero publishers and five subscribers -- u stays 0.0 forever and
        R_stop silently collapses to the bare margin. The reaction and braking terms
        vanish, and the protective stop degrades into the flat envelope that put SAM into
        the dry-dock wall, with nothing anywhere saying so.

        A safety layer whose input is missing must fail LOUD and CONSERVATIVE, not quiet
        and permissive. No speed for stop_u_timeout means "I do not know how fast I am
        going", and the honest answer to that is the worst case.
        """
        if self._speed_at is None:
            age = None
        else:
            age = (now - self._speed_at).nanoseconds / 1e9
        if age is None or age > self.stop_u_timeout:
            self._note_bad_speed(
                "absent" if age is None else f"stale by {age:.1f} s", self.speed)
            return self.stop_u_max
        return self.speed

    def _note_bad_speed(self, why: str, u) -> None:
        """Rate-limited, because a broken estimator publishes at 20 Hz."""
        now = self.get_clock().now()
        last = getattr(self, "_bad_speed_at", None)
        if last is not None and (now - last).nanoseconds < 5e9:
            return
        self._bad_speed_at = now
        self.get_logger().error(
            f"odom surge speed is {why} ({u}) — clamping to stop_u_max "
            f"{self.stop_u_max} m/s for the stop envelope. R_stop is QUADRATIC in u, so an "
            f"unvalidated speed makes the protective stop meaningless. Check the estimator.")

    def mode_cb(self, msg):
        try:
            parts = msg.data.split("|")
            self.sonar_mode, self.sonar_range = parts[0], float(parts[1])
        except Exception:
            self.get_logger().warn(f"unparsable sonar mode '{msg.data}'",
                                   throttle_duration_sec=30.0)

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
        self._n_raw = len(xyz)
        self._n_hits = int(keep.sum())
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
            self._tf_ok = False
            return None
        self._tf_warned = False
        self._tf_ok = True
        q = t.transform.rotation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        return xyz @ R.T + p

    def process(self, body, msg):
        x, y, z = body[:, 0], body[:, 1], body[:, 2]
        rng = np.linalg.norm(body, axis=1)
        az = np.arctan2(y, x)
        el = np.arctan2(z, np.hypot(x, y))

        rng_all = rng          # ranges of every RETURN, before any gate (health)
        # Per-gate survivor counts. "Nothing survived gating" is a question, not an
        # answer — the four gates mean four different things and only one of them
        # is a defect. Naming the gate turns a BLIND from a puzzle into a
        # diagnosis, and it costs four boolean sums per cloud.
        g_rng = (rng >= self.range_min) & (rng <= self.range_max)
        g_z   = np.abs(z) <= self.z_band
        g_az  = np.abs(az) <= self.az_half
        g_el  = np.abs(el) <= self.el_half
        self._gate_counts = (int(g_rng.sum()), int((g_rng & g_z).sum()),
                             int((g_rng & g_z & g_az).sum()),
                             int((g_rng & g_z & g_az & g_el).sum()))
        keep = g_rng & g_z & g_az & g_el
        body, rng, az, el = body[keep], rng[keep], az[keep], el[keep]
        self._n_gated = len(rng)
        # "Returns came back and the gate ate all of them" — the 2026-08-12
        # range-gate-before-TF signature. One cloud proves nothing; a streak does.
        #
        # BUT the test must be against returns that were IN RANGE. A vehicle
        # facing open water gets hits from the far wall at 20 m; the range gate
        # drops them, leaving hits > 0 and gated == 0 — identical arithmetic to
        # the defect, opposite meaning. Measured on the first clean baseline
        # (run 15:15:22, t+70 s, just after the wp1->wp2 turn): a 1 s BLIND while
        # the sonar was working perfectly and simply looking down an open leg.
        #
        # That false positive is not cosmetic. BLIND withholds the rose's
        # free-range claim, which drives u_max to 0 — i.e. it would command the
        # governor to HOLD THE VEHICLE IN OPEN WATER, the exact failure this
        # session's review caught in the rose and fixed there.
        n_inrange = int(((rng_all >= self.range_min) & (rng_all <= self.range_max)).sum())
        self._n_inrange = n_inrange

        # WHICH GATE emptied the cloud decides whether this is blindness at all.
        # `_gate_killer()` already says so in prose — range is "not a fault", z-band
        # means "the fan is looking at the seabed or the surface, not at anything the
        # hull can hit", and only az/el is "unambiguously wrong" — but the streak used
        # to count all three the same, so it contradicted the very string it printed.
        #
        # This is the SAME BUG, ONE GATE OVER, as the 2026-08-12 fix above. That one
        # stopped the RANGE gate manufacturing BLINDs by testing n_inrange instead of
        # n_hits. The Z gate has the identical shape: a surfaced vehicle over a seabed
        # 4.7 m down with z_band 1.5 m gets n_inrange in the hundreds and n_gated == 0
        # on every single cloud, forever. Measured at Kristineberg 2026-08-16:
        # "6144 clouds, 684 in range, ALL outside z-band" — a permanent BLIND on a
        # sonar that was working perfectly and looking at open water.
        #
        # And it is not cosmetic, for the reason already written above: BLIND withholds
        # the rose's free-range claim, the forward cone goes UNKNOWN, u_max(NaN) = 0,
        # and the governor holds the vehicle IN OPEN WATER. A state that means "my
        # answer is not evidence" must not fire in the one condition where the answer
        # is trivially true and safe, or an operator learns to ignore it.
        #
        # So: the cloud counts toward a blind streak only when returns got as far as
        # the SECTOR GRID and were still lost — i.e. they were in range AND within the
        # z-band, and az/el discarded them. The grid is the sensor's own FOV, so that
        # is a mount/frame defect and nothing else.
        n_rng, n_z, _n_az, _n_el = self._gate_counts
        killed_by_sector_grid = (n_z > 0 and self._n_gated == 0)
        self._blind_streak = (self._blind_streak + 1
                              if (n_inrange > 0 and killed_by_sector_grid) else 0)

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
        now = self.get_clock().now()
        u = self._envelope_speed(now)
        r_stop = self.stop_margin + u * self.stop_t_react + u * u / (2.0 * self.stop_a_stop)
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

    def _gate_killer(self):
        """Which gate emptied the cloud, and what it means.

        The four gates are not interchangeable:
          range  — everything is beyond range_max: OPEN WATER, or the sonar has
                   switched to a shorter horizon. Not a fault.
          z-band — returns exist but all outside +-z_band: the fan is looking at
                   the seabed or the surface, not at anything the hull can hit.
          az/el  — returns exist inside the band but outside the sector grid:
                   a mount/frame problem, because the grid is the sensor's own FOV.
        Only the last is unambiguously wrong, and it is the one nobody would have
        guessed."""
        n_rng, n_z, n_az, n_el = self._gate_counts
        if n_rng == 0:
            return f"all {self._n_hits} returns out of range (open water or short horizon)"
        if n_z == 0:
            return f"{n_rng} in range, ALL outside z-band +-{self.z_band} m (floor/surface?)"
        if n_az == 0:
            return f"{n_z} in band, ALL outside azimuth +-{np.degrees(self.az_half):.0f}deg"
        if n_el == 0:
            return f"{n_az} in azimuth, ALL outside elevation +-{np.degrees(self.el_half):.0f}deg"
        return f"gates passed {n_rng}/{n_z}/{n_az}/{n_el} but 0 kept — unexplained"

    def publish_health(self):
        """Publish what only this node knows: whether its answer means anything.

        Format (String, '|' separated so a HUD can split it without a parser):
            STATE|rate_hz|age_s|n_hits|n_gated|detail

        STATE:
            OK        clouds fresh and conclusive
            NO_INPUT  not one cloud since start. `detail` names WHICH side by counting
                      matched publishers: 0 = nothing is producing (Unity side);
                      >0 = advertised but undelivered (QoS / discovery / stale
                      registration). Do not guess between those two — they have
                      different fixes and the count is free.
            STALE     had clouds, stopped (age > cloud_timeout)
            NO_TF     clouds arrive, the body-frame lookup fails, nothing is
                      processed at all
            BLIND     returns present, gating discards every one of them for
                      blind_streak_min clouds running
        Anything that is not OK means the range/stop outputs are NOT evidence,
        and a run gated on them does not count (runbook rule 3).
        """
        now = self.get_clock().now()
        if self._rate_t0 is None:
            self._rate_t0, self._rate_n0 = now, self._n_clouds
        dt = (now - self._rate_t0).nanoseconds * 1e-9
        if dt >= 2.0:
            self._rate_hz = (self._n_clouds - self._rate_n0) / dt
            self._rate_t0, self._rate_n0 = now, self._n_clouds

        if self.last_cloud_time is None:
            age = float("inf")
        else:
            age = (now - self.last_cloud_time).nanoseconds * 1e-9

        if self.last_cloud_time is None:
            # MEASURE, do not speculate. This used to read "(endpoint restart?)" — a
            # guess baked into a status string, which on 2026-08-14 was quoted back as
            # if it were evidence that an endpoint restart had happened. The node can
            # simply ask its own subscription how many publishers it is matched to,
            # which separates the two causes that need completely different fixes:
            #   0 matched  -> nothing is producing (Unity side: sensor disabled, object
            #                 inactive, or the publisher never registered)
            #   >0 matched -> something advertises and delivers nothing (subscriber side:
            #                 QoS mismatch, discovery, or stale registration after an
            #                 endpoint restart)
            # Note the count is of MATCHED publishers, so it already excludes the
            # registered-but-silent case that `ros2 topic info` cannot see.
            try:
                npub = self.sub_cloud.get_publisher_count()
            except Exception:      # older rclpy — degrade to the honest non-answer
                npub = -1
            if npub == 0:
                detail = "no cloud since start; 0 publishers matched — nothing is producing"
            elif npub > 0:
                detail = (f"no cloud since start; {npub} publisher(s) matched but nothing "
                          f"delivered — subscriber-side (QoS / discovery / stale registration)")
            else:
                detail = "no cloud since start; publisher count unavailable"
            state = "NO_INPUT"
        elif age > self.cloud_timeout:
            state, detail = "STALE", f"no cloud for {age:.0f} s"
        elif not self._tf_ok:
            state, detail = "NO_TF", f"TF -> {self.body_frame} failing"
        elif self._blind_streak >= 5:
            state, detail = "BLIND", (f"{self._blind_streak} clouds, {self._gate_killer()} "
                                      f"[{self._n_inrange} in range]")
        else:
            state, detail = "OK", f"{self._n_gated}/{self._n_inrange}/{self._n_hits} of {self._n_raw} pts"

        # Can we see far enough to stop? Independent of everything above: the
        # detector can be perfectly healthy AND unable to observe its own trigger.
        if self.sonar_range is not None and state == "OK":
            # Same guarded speed the stop itself uses. Reading self.speed raw here while
            # the stop read the guarded value would let the HUD and the trigger disagree
            # about the envelope -- two numbers for one question, which is how this whole
            # class of bug keeps starting.
            u = self._envelope_speed(self.get_clock().now())
            r_stop = self.stop_margin + u * self.stop_t_react + u * u / (2.0 * self.stop_a_stop)
            if r_stop > self.sonar_range:
                state = "SHORT_HORIZON"
                detail = (f"R_stop {r_stop:.1f} m at u={u:.2f} EXCEEDS the {self.sonar_mode} "
                          f"horizon {self.sonar_range:.1f} m — slow down or go to navigation mode")

        self.pub_health.publish(String(data=(
            f"{state}|{self._rate_hz:.1f}|{age if np.isfinite(age) else -1.0:.1f}|"
            f"{self._n_hits}|{self._n_gated}|{detail}")))
        if state != self._health_state:
            # DO NOT collapse these into `log = ... if ... else ...; log(msg)`.
            # rclpy caches logging state per CALL SITE, so one line that logs at
            # two different severities raises
            #     ValueError: Logger severity cannot be changed between calls
            # and kills the node. That is exactly what happened the first time
            # this health line ran on the rig (2026-08-12): the detector died at
            # its first OK -> not-OK transition, and the node meant to make
            # perception failures visible became one. Separate call sites, always.
            msg = f"detector health: {self._health_state} -> {state} ({detail})"
            if state == "OK":
                self.get_logger().info(msg)
            else:
                self.get_logger().error(msg)
            self._health_state = state

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
