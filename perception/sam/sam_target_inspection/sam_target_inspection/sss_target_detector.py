#!/usr/bin/env python3
"""`sss_target_detector` — car-like target candidates from side-scan intensities.

The ROS half of strategy §4. All of the arithmetic lives in `sss_target_core.py` and is tested
off-vehicle; this file subscribes, converts, publishes, and reports its own health. Keeping it
thin is deliberate: everything that can be tested without a running graph should be, because the
graph is the expensive place to find a bug.

WHAT IT PUBLISHES
  <ns>/perception/target/candidates        std_msgs/String, one JSON object per CANDIDATE
                                           (a persisted track), not per ping
  <ns>/perception/target/detector/health   std_msgs/String, the obstacle detector's format

WHAT IT DELIBERATELY DOES NOT DO
  * it does not look at the simulator's material labels (SETTLED §3k). Its input is a byte
    array of intensities;
  * it does not divert anything. It publishes candidates; the ledger associates them and the
    behaviour tree decides. A detector that asked for a manoeuvre would be a second decider;
  * it does not invent a range scale: `max_duration` is the publisher's own statement of the
    ping's extent and the parameter fallback says out loud when it is being used;
  * it does NOTHING while `payload/sidescan` is silent (ADR-010), and it measures its own
    per-ping cost and prints it on the health line.

Run: ros2 run sam_target_inspection sss_target_detector --ros-args -p robot_name:=sam_auv_v1
"""
import json
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float32, String
    from smarc_msgs.msg import Sidescan
    from geographic_msgs.msg import GeoPoint
    _HAVE_ROS = True
except ImportError:                                            # pragma: no cover
    # IMPORT-GUARDED SO THE CORE CAN BE TESTED WITHOUT ROS. `Node = object` keeps the class
    # definition below legal; `main()` refuses by name rather than raising an ImportError from
    # somewhere confusing.
    Node = object
    _HAVE_ROS = False

from sam_target_inspection._health import HealthLine
from sam_target_inspection.sss_target_core import (SssConfig, SssTargetTracker,
                                                   candidate_from_track, detect_ping)

SOUND_SPEED_MS = 1500.0


class SssTargetDetector(Node):

    def __init__(self):
        super().__init__("sss_target_detector")
        self.declare_parameter("robot_name", "sam_auv_v1")
        self.declare_parameter("sidescan_topic", "payload/sidescan")
        # Fallback only. 0.0 means "insist on the message's own max_duration".
        self.declare_parameter("max_range_m", 40.0)
        # The false-alarm budget the threshold is DERIVED from, per ping, through the
        # chi-square tail. Raising it makes the detector more talkative in a way that is
        # quantified rather than felt (SETTLED §3k).
        self.declare_parameter("false_alarms_per_ping", 0.05)
        # The object being looked for, in metres. Never in bins: the two side-scan modes differ
        # 4x in bin size and a bin count right for one is meaningless for the other.
        self.declare_parameter("target_height_m", 1.278)        # the Mini, SETTLED §3f0d
        self.declare_parameter("across_min_m", 1.0)
        self.declare_parameter("across_max_m", 2.0)
        self.declare_parameter("highlight_db_min", 6.0)
        self.declare_parameter("persistence_pings", 0)          # 0 = the derived default
        self.declare_parameter("stale_timeout_s", 2.0)
        # Positional sigma inputs that this node does not own and must not invent.
        self.declare_parameter("dr_sigma_m", 2.0)
        self.declare_parameter("along_track_sigma_m", 0.5)

        gp = lambda n: self.get_parameter(n).value              # noqa: E731
        self.robot_name = gp("robot_name")
        self.param_max_range_m = float(gp("max_range_m"))
        self.fa_per_ping = float(gp("false_alarms_per_ping"))
        self.target_height_m = float(gp("target_height_m"))
        self.across_min_m = float(gp("across_min_m"))
        self.across_max_m = float(gp("across_max_m"))
        self.highlight_db_min = float(gp("highlight_db_min"))
        self.dr_sigma_m = float(gp("dr_sigma_m"))
        self.along_track_sigma_m = float(gp("along_track_sigma_m"))

        ns = f"/{self.robot_name}"
        self.sub = self.create_subscription(
            Sidescan, f'{ns}/{gp("sidescan_topic")}', self.ping_cb, qos_profile_sensor_data)
        self.create_subscription(Float32, f"{ns}/smarc/altitude", self._alt_cb, 10)
        self.create_subscription(Float32, f"{ns}/smarc/course", self._course_cb, 10)
        self.create_subscription(GeoPoint, f"{ns}/dr/lat_lon", self._latlon_cb, 10)
        self.pub_cand = self.create_publisher(String, f"{ns}/perception/target/candidates", 5)
        self.pub_health = self.create_publisher(
            String, f"{ns}/perception/target/detector/health", 1)

        n_persist = int(gp("persistence_pings"))
        self.trackers = {
            side: (SssTargetTracker(persistence_pings=n_persist) if n_persist > 0
                   else SssTargetTracker())
            for side in ("port", "starboard")}
        self.health = HealthLine(stale_timeout_s=float(gp("stale_timeout_s")),
                                 publisher_count=self._publisher_count)
        self._altitude = None
        self._course = None
        self._latlon = None
        self._ping_i = 0
        self._range_src = "unknown"
        self._range_m = 0.0
        self._ms = 0.0
        self._n_candidates = 0
        self._range_warned = False

        self.create_timer(0.5, self.publish_health)
        self.get_logger().info(
            f"sss_target_detector up: {ns}/{gp('sidescan_topic')} -> "
            f"{ns}/perception/target/candidates; across-track band "
            f"{self.across_min_m}-{self.across_max_m} m, highlight >= "
            f"{self.highlight_db_min} dB over the LOCAL BACKGROUND, shadow length from "
            f"geometry (H*r/h with H = {self.target_height_m} m), false-alarm budget "
            f"{self.fa_per_ping}/ping")

    # ------------------------------------------------------------------ inputs
    def _publisher_count(self):
        try:
            return self.sub.get_publisher_count()
        except Exception:
            return -1

    def _alt_cb(self, msg):
        self._altitude = float(msg.data)

    def _course_cb(self, msg):
        self._course = float(msg.data)

    def _latlon_cb(self, msg):
        self._latlon = (float(msg.latitude), float(msg.longitude))

    # ------------------------------------------------------------------ the ping
    def _range_res(self, msg, n_bins):
        """Metres per bin, and where the number came from.

        The message's own `max_duration` first. A detector running on an assumed range scale
        produces plausible numbers that are all wrong by one factor, so the source is on every
        health line and the fallback announces itself exactly once.
        """
        md = float(getattr(msg, "max_duration", 0.0) or 0.0)
        if md > 0.0 and n_bins > 0:
            return (md * SOUND_SPEED_MS / 2.0) / n_bins, "message"
        if self.param_max_range_m > 0.0 and n_bins > 0:
            if not self._range_warned:
                self._range_warned = True
                self.get_logger().warn(
                    f"the Sidescan message carries max_duration = 0, so the range scale comes "
                    f"from the max_range_m parameter ({self.param_max_range_m:.1f} m). Every "
                    f"detection range depends on it.")
            return self.param_max_range_m / n_bins, "parameter"
        return None, "no range scale: max_duration is 0 and max_range_m is not positive"

    def ping_cb(self, msg):
        t0 = time.perf_counter()
        now = self._now()
        self.health.note_input(now)
        self._ping_i += 1

        if self._altitude is None:
            # THE ALTIMETER IS NOT OPTIONAL. Slant range becomes ground range through it
            # (SETTLED §3f0h), the seabed's start is where it says, and the shadow length is
            # H*r/h. Without it there is no geometry, and a guessed altitude would produce a
            # detector that fires in the wrong place rather than one that says nothing.
            self.health.blind_streak += 1
            self.health.last_reason = ("no altitude yet on smarc/altitude; without the "
                                       "vehicle's own altimeter there is no ground range, no "
                                       "seabed start and no predicted shadow length")
            return
        port = np.frombuffer(bytes(msg.port_channel), dtype=np.uint8).astype(np.float64)
        stbd = np.frombuffer(bytes(msg.starboard_channel), dtype=np.uint8).astype(np.float64)
        n_bins = max(port.size, stbd.size)
        res, src = self._range_res(msg, n_bins)
        self._range_src = src
        if res is None:
            self.health.blind_streak += 1
            self.health.last_reason = src
            return
        self._range_m = res * n_bins

        cfg = SssConfig(range_res_m=res, altitude_m=self._altitude,
                        target_height_m=self.target_height_m,
                        across_min_m=self.across_min_m, across_max_m=self.across_max_m,
                        highlight_db_min=self.highlight_db_min,
                        false_alarms_per_ping=self.fa_per_ping)
        blind = 0
        for side, chan in (("port", port), ("starboard", stbd)):
            if chan.size == 0:
                continue
            rep = detect_ping(chan, cfg)
            if not rep.ok:
                blind += 1
                self.health.last_reason = f"{side}: {rep.reason}"
                continue
            for track in self.trackers[side].update(self._ping_i, side, rep):
                cand = candidate_from_track(
                    track, cfg, cid=f"S{self._ping_i}_{side[0]}", t=now,
                    lat=self._latlon[0] if self._latlon else None,
                    lon=self._latlon[1] if self._latlon else None,
                    course_deg=self._course,
                    dr_since_fix_m=self.dr_sigma_m,
                    along_track_m=self.along_track_sigma_m)
                self._n_candidates += 1
                self.health.note_output()
                self.pub_cand.publish(String(data=json.dumps(cand.as_dict(),
                                                             separators=(",", ":"))))
                self.get_logger().info(
                    f"target candidate {cand.id}: {cand.n_pings} pings, "
                    f"{cand.highlight_db:.1f} dB highlight, {cand.shadow_db:.1f} dB shadow, "
                    f"across {cand.extent_m['across']:.2f} m, sigma {cand.sigma_m:.2f} m, "
                    f"score {cand.score:.2f}")
        # BOTH channels unable to find their seabed is the state that stops this node meaning
        # anything. One channel is normal -- a side scan off a slope loses one side all the time.
        self.health.blind_streak = self.health.blind_streak + 1 if blind == 2 else 0
        self._ms = 1000.0 * (time.perf_counter() - t0)

    def _now(self):
        n = self.get_clock().now()
        return n.nanoseconds * 1e-9

    # ------------------------------------------------------------------ health
    def publish_health(self):
        ok = (f"range {self._range_m:.0f} m from the {self._range_src}, altitude "
              f"{self._altitude if self._altitude is not None else float('nan'):.2f} m, "
              f"{self._n_candidates} candidate(s) in {self._ping_i} pings, "
              f"{self._ms:.2f} ms/ping")
        line = self.health.compute(self._now(), ok_detail=ok)
        state = self.health.state
        self.pub_health.publish(String(data=line))
        if state != getattr(self, "_last_state", None):
            # SEPARATE CALL SITES PER SEVERITY: rclpy caches logging state per call site and
            # raises if one line logs at two severities (the obstacle detector, 2026-08-12).
            msg = f"detector health: {getattr(self, '_last_state', None)} -> {state} " \
                  f"({self.health.detail})"
            if state == "OK":
                self.get_logger().info(msg)
            else:
                self.get_logger().error(msg)
            self._last_state = state


def main(args=None):
    if not _HAVE_ROS:                                          # pragma: no cover
        raise SystemExit("sss_target_detector needs rclpy; the pure core in "
                         "sam_target_inspection.sss_target_core runs without it and is what "
                         "the tests drive.")
    rclpy.init(args=args)
    node = SssTargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # NO bare rclpy.shutdown(): rclpy's own SIGTERM handler may already have shut the
        # context down, and a second call raises RCLError, exits 1, and a systemd unit with
        # Restart=on-failure then reads a CLEAN STOP AS A CRASH and relaunches. That is the
        # duplicate-bringup chain (SETTLED §1c) -- a clean stop must exit 0.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
