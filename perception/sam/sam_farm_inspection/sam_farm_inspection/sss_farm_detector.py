#!/usr/bin/env python3
"""`sss_farm_detector` — rope and buoy detections from side-scan intensities.

The ROS half of sensors-22-05064 §5. All of the arithmetic lives in `change_point.py`
and is tested off-vehicle; this file subscribes, converts, publishes, and reports its
own health. Keeping it thin is deliberate: everything that can be tested without a
running graph should be, because the graph is the expensive place to find a bug.

WHAT IT PUBLISHES, AND WHY IT IS JSON ON A STRING
  <ns>/perception/farm/detections        std_msgs/String, one JSON object per ping
  <ns>/perception/farm/detector/health   std_msgs/String, the same '|' format the
                                         obstacle detector uses

A String avoids adding a message package that both the vehicle and the station would
have to build before anything could run — the same reason `ctrl/setpoints`,
`perception/obstacle/health` and `comms/link_state` are Strings. This node is the SOLE
publisher of both topics, which is what makes a health probe on them meaningful
(spec invariant 11: a probe must read a topic its node solely owns; Unity publishes
`smarc/*` and a probe there measures Unity).

WHAT IT DELIBERATELY DOES NOT DO
  * It does not look at the simulator's material labels. `Sonar.cs` labels every hit
    Rope / Buoy / Algae, and a detector reading those would be perfect in sim and
    useless on the first real ping (mission design decision D4). Its input is a byte
    array of intensities.
  * It does not know where the vehicle is. Detections are sensor-relative — slant
    range and channel — exactly what the hardware produces. Turning them into positions
    needs a pose and a target depth, and that is `farm_localizer`'s job. Keeping nav out
    of the detector is the same rule that keeps the obstacle detector on body-relative
    ranges.
  * It does not decide the range scale on its own if it can avoid it: `max_duration` in
    the message is the publisher's own statement of the ping's extent, and the parameter
    is a named fallback that says out loud when it is being used.

Run:  ros2 run sam_farm_inspection sss_farm_detector --ros-args -p robot_name:=sam_auv_v1
"""
import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from smarc_msgs.msg import Sidescan

from sam_farm_inspection.change_point import detect_ping, targets_from_metres
from sam_farm_inspection.sss_geometry import SOUND_SPEED_MS, range_per_bin  # noqa: F401


class SssFarmDetector(Node):

    def __init__(self):
        super().__init__("sss_farm_detector")

        self.declare_parameter("robot_name", "sam_auv_v1")
        self.declare_parameter("sidescan_topic", "payload/sidescan")
        # Near-field blanking, in metres of slant range. Transmit ring-down and the
        # vehicle's own hull produce an enormous change point at bin 0 that is not the
        # seabed and not a rope.
        self.declare_parameter("blank_range_m", 0.5)
        # Physical extents the two window sizes are built from. Metres, never bins —
        # HF680 and LF340 differ 4x in bin size and a bin count right for one is
        # meaningless for the other.
        self.declare_parameter("rope_extent_m", 0.10)
        self.declare_parameter("buoy_extent_m", 0.40)
        # An object at least this wide is a buoy. Geometry, not brightness: the
        # simulator's reflectivities are documented wild guesses.
        self.declare_parameter("buoy_min_extent_m", 0.25)
        # The detection threshold is DERIVED from this, per ping, via the chi-square
        # tail — see change_point.change_ratio_for_false_alarm. Raising it makes the
        # detector more talkative in a way that is quantified rather than felt.
        self.declare_parameter("false_alarms_per_ping", 0.05)
        # Fallback only. 0.0 means "insist on the message's own max_duration".
        self.declare_parameter("max_range_m", 100.0)
        self.declare_parameter("nadir_window_m", 1.0)
        self.declare_parameter("stale_timeout_s", 2.0)

        gp = lambda n: self.get_parameter(n).value          # noqa: E731
        self.robot_name = gp("robot_name")
        self.blank_range_m = float(gp("blank_range_m"))
        self.rope_extent_m = float(gp("rope_extent_m"))
        self.buoy_extent_m = float(gp("buoy_extent_m"))
        self.buoy_min_extent_m = float(gp("buoy_min_extent_m"))
        self.fa_per_ping = float(gp("false_alarms_per_ping"))
        self.param_max_range_m = float(gp("max_range_m"))
        self.nadir_window_m = float(gp("nadir_window_m"))
        self.stale_timeout_s = float(gp("stale_timeout_s"))

        ns = f"/{self.robot_name}"
        self.sub = self.create_subscription(
            Sidescan, f'{ns}/{gp("sidescan_topic")}', self.ping_cb, qos_profile_sensor_data)
        self.pub_det = self.create_publisher(String, f"{ns}/perception/farm/detections", 5)
        self.pub_health = self.create_publisher(String, f"{ns}/perception/farm/detector/health", 1)

        self.last_ping_at = None
        self._n_pings = 0
        self._n_det = 0
        self._no_bottom_streak = 0
        self._last_reason = ""
        self._range_src = "unknown"
        self._range_m = 0.0
        self._rate_hz = 0.0
        self._rate_t0 = None
        self._rate_n0 = 0
        self._health_state = None
        self._range_warned = False

        self.create_timer(0.5, self.publish_health)
        self.get_logger().info(
            f"sss_farm_detector up: {ns}/{gp('sidescan_topic')} -> "
            f"{ns}/perception/farm/detections; rope window {self.rope_extent_m} m, "
            f"buoy window {self.buoy_extent_m} m, buoy at >= {self.buoy_min_extent_m} m extent, "
            f"false-alarm budget {self.fa_per_ping}/ping")

    # ------------------------------------------------------------------ range scale
    def _range_for(self, msg, n_bins):
        """Metres per bin and its source, via `sss_geometry.range_per_bin`.

        The arithmetic lives there so it can be tested without a graph; this wrapper
        exists only to turn the returned complaint into a log line and to make the
        parameter fallback loud exactly once.
        """
        res, src, complaint = range_per_bin(
            getattr(msg, "max_duration", 0.0), n_bins, self.param_max_range_m)
        if complaint:
            self.get_logger().error(complaint + " Fix the parameter or the publisher.",
                                    throttle_duration_sec=30.0)
        if src == "parameter" and not self._range_warned:
            self._range_warned = True
            self.get_logger().warn(
                f"the Sidescan message carries max_duration = 0, so the range scale is "
                f"coming from the max_range_m parameter ({self.param_max_range_m:.1f} m). "
                f"Every detection range depends on it. Unity's SSS_Pub fills this field "
                f"as of 2026-08-16 — an unfilled one means the sim has not been rebuilt.")
        return res, src

    # ------------------------------------------------------------------ the ping
    def ping_cb(self, msg: Sidescan):
        self.last_ping_at = self.get_clock().now()
        self._n_pings += 1

        port = np.frombuffer(bytes(msg.port_channel), dtype=np.uint8).astype(np.float64)
        stbd = np.frombuffer(bytes(msg.starboard_channel), dtype=np.uint8).astype(np.float64)
        n_bins = max(port.size, stbd.size)
        res, src = self._range_for(msg, n_bins)
        self._range_src = src
        if res is None:
            self._last_reason = src
            return
        self._range_m = res * n_bins

        targets = targets_from_metres(
            self.rope_extent_m, self.buoy_extent_m, res,
            # Candidate count for the threshold: the water column, not the whole ping.
            # At the farm the seabed is 8-10 m down and the ping is 100 m long, so
            # thresholding as if the whole ping were searched would be ~15 % too strict.
            n_candidates=max(20, int(round(20.0 / res))),
            false_alarms_per_ping=self.fa_per_ping)
        blank_bins = int(round(self.blank_range_m / res))
        nadir_bins = max(4, int(round(self.nadir_window_m / res)))

        out = {
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "frame_id": msg.header.frame_id,
            "range_res_m": round(res, 6),
            "range_src": src,
            "n_bins": int(n_bins),
            "channels": {},
        }
        n_here = 0
        reasons = []
        for name, chan in (("port", port), ("starboard", stbd)):
            if chan.size == 0:
                out["channels"][name] = {"ok": False, "reason": "channel empty", "detections": []}
                reasons.append(f"{name}: empty")
                continue
            r = detect_ping(chan, res, targets,
                            nadir_window_bins=nadir_bins,
                            blank_bins=blank_bins,
                            buoy_min_extent_m=self.buoy_min_extent_m,
                            false_alarms_per_ping=self.fa_per_ping)
            out["channels"][name] = {
                "ok": bool(r.ok),
                "reason": r.reason,
                "nadir_slant_m": (round(r.nadir_slant_m, 3) if r.nadir_slant_m is not None else None),
                "detections": [
                    {"target": d.target,
                     "slant_range_m": round(d.slant_range_m, 3),
                     "extent_m": round(d.extent_m, 3),
                     "snr": round(d.snr, 2),
                     "confidence": round(d.confidence, 3),
                     "ambiguous": bool(d.ambiguous)}
                    for d in r.detections],
            }
            n_here += len(r.detections)
            if not r.ok:
                reasons.append(f"{name}: {r.reason}")

        # "No bottom return" on BOTH channels is the state that stops this node meaning
        # anything, so it is counted rather than logged and forgotten. One channel is
        # normal — a side scan looking off a slope loses one side all the time.
        both_blind = all(not c.get("ok") for c in out["channels"].values())
        self._no_bottom_streak = self._no_bottom_streak + 1 if both_blind else 0
        if reasons:
            self._last_reason = "; ".join(reasons)

        self._n_det += n_here
        self.pub_det.publish(String(data=json.dumps(out, separators=(",", ":"))))

        if self._n_pings % 100 == 1:
            self.get_logger().info(
                f"pings {self._n_pings}: {self._n_det} detections total, range scale "
                f"{res * 100:.1f} cm/bin from the {src}, "
                f"nadir port {out['channels'].get('port', {}).get('nadir_slant_m')} m")

    # ------------------------------------------------------------------ health
    def publish_health(self):
        """STATE|rate_hz|age_s|pings|detections|detail — the obstacle detector's format.

        The states an operator has to be able to tell apart:
            NO_INPUT   not one ping since start; the detail names WHICH side by counting
                       matched publishers (0 = nothing producing, >0 = advertised and
                       undelivered). Never guess between those two.
            STALE      pings arrived and stopped.
            NO_BOTTOM  pings arrive and neither channel can find its bottom return, so
                       the water column has no end and no detection means anything.
            OK         pings fresh and conclusive. Zero detections in this state is a
                       real answer: there is nothing there.
        `range_src` is on every line because a detector running on an assumed range scale
        is producing plausible numbers that are all wrong by one factor.
        """
        now = self.get_clock().now()
        if self._rate_t0 is None:
            self._rate_t0, self._rate_n0 = now, self._n_pings
        dt = (now - self._rate_t0).nanoseconds * 1e-9
        if dt >= 2.0:
            self._rate_hz = (self._n_pings - self._rate_n0) / dt
            self._rate_t0, self._rate_n0 = now, self._n_pings

        age = float("inf") if self.last_ping_at is None else \
            (now - self.last_ping_at).nanoseconds * 1e-9

        if self.last_ping_at is None:
            try:
                npub = self.sub.get_publisher_count()
            except Exception:
                npub = -1
            if npub == 0:
                detail = "no ping since start; 0 publishers matched — nothing is producing"
            elif npub > 0:
                detail = (f"no ping since start; {npub} publisher(s) matched but nothing "
                          f"delivered — subscriber-side (QoS / discovery / stale registration)")
            else:
                detail = "no ping since start; publisher count unavailable"
            state = "NO_INPUT"
        elif age > self.stale_timeout_s:
            state, detail = "STALE", f"no ping for {age:.0f} s"
        elif self._no_bottom_streak >= 5:
            state = "NO_BOTTOM"
            detail = f"{self._no_bottom_streak} pings, both channels: {self._last_reason}"
        else:
            state = "OK"
            detail = (f"range {self._range_m:.0f} m from the {self._range_src}, "
                      f"{self._n_det} detections in {self._n_pings} pings")

        self.pub_health.publish(String(data=(
            f"{state}|{self._rate_hz:.1f}|{age if math.isfinite(age) else -1.0:.1f}|"
            f"{self._n_pings}|{self._n_det}|{detail}")))

        if state != self._health_state:
            # Separate call sites per severity. rclpy caches logging state per CALL SITE
            # and raises "Logger severity cannot be changed between calls" if one line
            # logs at two severities — which killed the obstacle detector's health line
            # the first time it ran on the rig (2026-08-12).
            msg = f"detector health: {self._health_state} -> {state} ({detail})"
            if state == "OK":
                self.get_logger().info(msg)
            else:
                self.get_logger().error(msg)
            self._health_state = state


def main(args=None):
    rclpy.init(args=args)
    node = SssFarmDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # NO bare rclpy.shutdown() here: rclpy's own SIGTERM handler may already have
        # shut the context down, and a second call raises RCLError, exits 1, and a
        # systemd unit with Restart=on-failure then reads a CLEAN STOP AS A CRASH and
        # relaunches. That is the duplicate-bringup chain (SETTLED §1c) — a clean stop
        # must exit 0.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
