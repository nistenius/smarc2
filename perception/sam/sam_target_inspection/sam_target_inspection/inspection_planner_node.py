#!/usr/bin/env python3
"""`inspection_planner` — the goal SOURCE on the wire. Strategy §5.2, §5.3, ADR-004.

THE REQUEST/ANSWER PATTERN IS `farm_inspection.PlannerLink`'s, and not a service: a service call
blocks a behaviour-tree tick and this node may legitimately take a moment. Two topics,
sequence-numbered, and an answer that does not match the question is discarded by the tree.

  <ns>/perception/target/plan/request   std_msgs/String, from the tree
  <ns>/perception/target/plan/answer    std_msgs/String, back
  <ns>/perception/target/divert_request std_msgs/String, this node ASKING for a diversion
  <ns>/perception/target/planner/health std_msgs/String

IT HOLDS NO ACTION CLIENT AND WRITES NO ACTUATOR (invariant 12). It answers "what next?" and the
behaviour tree streams that answer through the ONE existing `auv_depth_move_to` client. The
`divert_request` topic is an ASK, not a command: the tree's own condition decides, and refuses
when the mission carries no policy.

Run: ros2 run sam_target_inspection inspection_planner --ros-args -p robot_name:=sam_auv_v1
"""
import json
import math

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String
    from geographic_msgs.msg import GeoPoint
    _HAVE_ROS = True
except ImportError:                                            # pragma: no cover
    Node = object
    _HAVE_ROS = False

from sam_target_inspection._health import HealthLine
from sam_target_inspection.inspection_planner import (DiversionPoint, InspectionPlanner, Policy,
                                                      TargetSite, VehicleState, distance_m)
from sam_target_inspection.target_ledger import Observation, TargetLedger


class InspectionPlannerNode(Node):

    def __init__(self):
        super().__init__("inspection_planner")
        self.declare_parameter("robot_name", "sam_auv_v1")
        self.declare_parameter("policy_yaml", "")
        # The seabed depth under a candidate. The curated Askö terrain is 29 % gap-filled
        # (SETTLED §3f0), so a depth read from it is not a measurement everywhere; the vehicle's
        # own depth + altitude is, and is what is used when both are fresh. The parameter is the
        # LAST resort and announces itself.
        self.declare_parameter("fallback_seabed_depth_m", 0.0)
        self.declare_parameter("stale_timeout_s", 5.0)

        gp = lambda n: self.get_parameter(n).value             # noqa: E731
        self.robot_name = gp("robot_name")
        self.fallback_depth = float(gp("fallback_seabed_depth_m"))
        self.policy = self._load_policy(str(gp("policy_yaml")))

        ns = f"/{self.robot_name}"
        self.create_subscription(String, f"{ns}/perception/target/candidates",
                                 self._sss_cb, 10)
        self.create_subscription(String, f"{ns}/perception/target/fls_candidates",
                                 self._fls_cb, 10)
        self.create_subscription(String, f"{ns}/perception/target/plan/request",
                                 self._request_cb, 10)
        self.create_subscription(Float32, f"{ns}/smarc/altitude", self._alt_cb, 10)
        self.create_subscription(Float32, f"{ns}/smarc/depth", self._depth_cb, 10)
        self.create_subscription(Float32, f"{ns}/smarc/course", self._course_cb, 10)
        self.create_subscription(GeoPoint, f"{ns}/dr/lat_lon", self._latlon_cb, 10)
        self.create_subscription(String, f"{ns}/ctrl/mission_timer", self._timer_cb, 10)
        self.pub_answer = self.create_publisher(String, f"{ns}/perception/target/plan/answer", 5)
        self.pub_divert = self.create_publisher(String,
                                                f"{ns}/perception/target/divert_request", 5)
        self.pub_ledger = self.create_publisher(String, f"{ns}/perception/target/ledger", 5)
        self.pub_health = self.create_publisher(String,
                                                f"{ns}/perception/target/planner/health", 1)

        self.ledger = TargetLedger()
        self.planner = None
        self.health = HealthLine(stale_timeout_s=float(gp("stale_timeout_s")))
        self._latlon = None
        self._alt = None
        self._depth = None
        self._course = None
        self._pose_at = None
        self._remaining_s = None
        self._diversions_used = 0
        self._asked_for = None
        self.create_timer(1.0, self.publish_health)
        self.get_logger().info(
            f"inspection_planner up on {ns}: candidates in, plan answers out. It holds no "
            f"action client and writes no actuator (ADR-004, invariant 12).")

    # ------------------------------------------------------------------ policy
    def _load_policy(self, path):
        """The mission's own `adaptive` block arrives with each request; this is the FILE's
        default for anything the mission does not state. Absent file => the dataclass defaults,
        which are the generated `config/inspection_policy_defaults.yaml`'s values."""
        if not path:
            return {}
        try:
            import yaml
            with open(path) as f:
                d = yaml.safe_load(f) or {}
            out = d.get("inspection_policy", d)
            self.get_logger().info(f"inspection policy defaults read from {path}")
            return out if isinstance(out, dict) else {}
        except Exception as e:
            self.get_logger().error(
                f"could not read the inspection policy from {path}: {e}. Falling back to the "
                f"built-in defaults, which are the ones config/inspection_policy_defaults.yaml "
                f"was generated with — say so rather than pretend the file was read.")
            return {}

    # ------------------------------------------------------------------ inputs
    def _alt_cb(self, msg):
        self._alt = float(msg.data)

    def _depth_cb(self, msg):
        self._depth = float(msg.data)

    def _course_cb(self, msg):
        self._course = float(msg.data)

    def _latlon_cb(self, msg):
        self._latlon = (float(msg.latitude), float(msg.longitude))
        self._pose_at = self._now()

    def _timer_cb(self, msg):
        try:
            d = json.loads(msg.data)
            self._remaining_s = d.get("remaining_s")
        except Exception:
            pass

    def _seabed_depth(self):
        """Depth of the seabed under the vehicle: its own depth plus its own altitude.

        Two instruments the vehicle carries, not a terrain tile. Returns (value, source) so the
        answer can say where the number came from — and None when neither is available, because
        a ring planned at an invented depth is a ring flown into the bottom.
        """
        if self._depth is not None and self._alt is not None:
            return abs(self._depth) + self._alt, "depth + altitude (the vehicle's own)"
        if self.fallback_seabed_depth() is not None:
            return self.fallback_seabed_depth(), "the fallback_seabed_depth_m PARAMETER"
        return None, "neither depth nor altitude has arrived"

    def fallback_seabed_depth(self):
        return self.fallback_depth if self.fallback_depth > 0 else None

    # ------------------------------------------------------------------ candidates
    def _observe(self, data, sensor):
        """One detector's candidate into the ledger, then ask for a diversion if it earns one."""
        self.health.note_input(self._now())
        if self._latlon is None:
            self.get_logger().warn(
                "a candidate arrived before any position did; it cannot be placed in the site "
                "frame and is dropped rather than placed at the origin",
                throttle_duration_sec=30.0)
            return
        n, e = self._site_xy(data)
        if n is None:
            return
        obs = Observation(sensor=sensor, t=self._now(), xy=(n, e),
                          sigma_m=max(float(data.get("sigma_m", 3.0)), 1e-3),
                          extent_m=self._extent(data, sensor),
                          aspect_deg=self._course,
                          detail=json.dumps(data, separators=(",", ":"))[:400],
                          score=data.get("score"))
        target, status = self.ledger.observe(obs)
        self.pub_ledger.publish(String(data=json.dumps(
            {"target": target.id, "status": status, "verdict": target.verdict(),
             "n_obs": len(target.observations), "sensors": list(target.sensors())},
            separators=(",", ":"))))
        if status == "ambiguous":
            # An association is a SET until something kills the alternatives (farm_ledger rule
            # 2). Asking for a diversion to a position we cannot commit to would be flying to
            # the average of two hypotheses, which is neither.
            self.get_logger().info(
                f"{target.id}: ambiguous association; no diversion is requested until one "
                f"candidate survives")
            return
        dup = self.ledger.duplicate_of((n, e), obs.sigma_m, self._now())
        if dup is not None and dup.id != target.id:
            self.ledger.note_not_inspected(
                target.id, f"duplicate of {dup.id}, already inspected (strategy §8)")
            return
        self.pub_divert.publish(String(data=json.dumps(
            {"id": target.id, "lat": data.get("lat"), "lon": data.get("lon"),
             "north": n, "east": e, "sigma_m": obs.sigma_m,
             "verdict": target.verdict(), "sensors": list(target.sensors())},
            separators=(",", ":"))))

    def _site_xy(self, data):
        """Candidate position in local metres about the vehicle. None with a log if it has none."""
        if data.get("lat") is not None and data.get("lon") is not None:
            from sam_target_inspection.inspection_planner import ne_between
            return ne_between(self._latlon[0], self._latlon[1],
                              float(data["lat"]), float(data["lon"]))
        if data.get("north") is not None and data.get("east") is not None:
            return float(data["north"]), float(data["east"])
        self.get_logger().warn("a candidate arrived with no position at all; dropped",
                               throttle_duration_sec=30.0)
        return None, None

    @staticmethod
    def _extent(data, sensor):
        if sensor == "sss":
            return {k: float(v) for k, v in (data.get("extent_m") or {}).items()}
        out = {}
        for k, v in (data.get("bbox_m") or {}).items():
            out[k] = float(v)
        if data.get("height_m") is not None:
            out["height"] = float(data["height_m"])
        return out

    def _sss_cb(self, msg):
        try:
            self._observe(json.loads(msg.data), "sss")
        except Exception as e:
            self.get_logger().error(f"unusable SSS candidate: {e}")

    def _fls_cb(self, msg):
        try:
            self._observe(json.loads(msg.data), "fls")
        except Exception as e:
            self.get_logger().error(f"unusable FLS candidate: {e}")

    # ------------------------------------------------------------------ the plan
    def _request_cb(self, msg):
        try:
            req = json.loads(msg.data)
        except Exception:
            return
        if req.get("event") == "start" or self.planner is None:
            self.planner = self._build(req)
        if self.planner is None:
            return
        ans = self.planner.what_next(req, now=self._now())
        self.health.note_output()
        self.pub_answer.publish(String(data=json.dumps(ans, separators=(",", ":"))))

    def _build(self, req):
        """Build a planner for the candidate the tree is asking about.

        Refuses BY ANSWER rather than by exception: a tree that gets no answer times out and
        blames the planner, which is the right refusal for the wrong reason.
        """
        target = self._pick_target()
        if target is None:
            self.pub_answer.publish(String(data=json.dumps(
                {"kind": "refused", "phase": "divert", "seq": req.get("seq"),
                 "reason": "no live target in the ledger to plan around",
                 "response": "the candidate that triggered the diversion has been ruled out or "
                             "was never recorded here; the mission resumes unchanged"})))
            return None
        depth, src = self._seabed_depth()
        if depth is None:
            self.pub_answer.publish(String(data=json.dumps(
                {"kind": "refused", "phase": "divert", "seq": req.get("seq"),
                 "reason": f"the seabed depth under the candidate is unknown ({src})",
                 "response": "a ring planned at an invented depth is a ring flown into the "
                             "bottom; the mission resumes unchanged"})))
            return None
        n, e = target.position()
        from sam_target_inspection.inspection_planner import ne_offset
        lat, lon = ne_offset(self._latlon[0], self._latlon[1], n, e)
        site = TargetSite(lat=lat, lon=lon, sigma_m=target.sigma_at(self._now()),
                          seabed_depth_m=depth, id=target.id)
        p0 = DiversionPoint(lat=self._latlon[0], lon=self._latlon[1],
                            leg_id=int(req.get("leg_id", 0) or 0),
                            leg_fraction=float(req.get("leg_fraction", 0.0) or 0.0),
                            leg_heading_deg=self._course if self._course is not None else 0.0,
                            depth_m=abs(self._depth) if self._depth is not None else 0.0)
        age = 1e9 if self._pose_at is None else self._now() - self._pose_at
        veh = VehicleState(lat=self._latlon[0], lon=self._latlon[1], estimator_age_s=age,
                           remaining_mission_s=self._remaining_s,
                           diversions_used=self._diversions_used)
        policy = Policy.from_dict({**self.policy, **(req.get("policy") or {})})
        self.get_logger().info(
            f"planning a diversion to {target.id}: seabed {depth:.1f} m from {src}, sigma "
            f"{site.sigma_m:.2f} m, ring R = {policy.ring_radius_m:.1f} m, "
            f"{policy.n_stations} stations x {len(policy.altitudes_m)} altitudes")
        self._diversions_used += 1
        return InspectionPlanner(policy, site, p0, veh, now=self._now())

    def _pick_target(self):
        live = [t for t in self.ledger.targets.values()
                if t.alive and t.not_inspected_reason is None and not t.inspected]
        if not live:
            return None
        # The one with the most evidence, then the tightest sigma. NOT "the best score": a score
        # is a distance from an expected signature and ranking by it alone would prefer a
        # textbook single look over two sensors agreeing.
        return sorted(live, key=lambda t: (-len(t.sensors()), -len(t.observations),
                                           t.sigma_at(self._now())))[0]

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ health
    def publish_health(self):
        depth, src = self._seabed_depth()
        verdicts = self.ledger.verdicts()
        ok = (f"{len(self.ledger.targets)} target(s) "
              f"[{', '.join(f'{k}:{v}' for k, v in sorted(verdicts.items())) or 'none'}], "
              f"seabed depth {('%.1f m from %s' % (depth, src)) if depth is not None else src}, "
              f"remaining mission "
              f"{self._remaining_s if self._remaining_s is not None else 'unknown'}")
        line = self.health.compute(self._now(), ok_detail=ok)
        self.pub_health.publish(String(data=line))


def main(args=None):
    if not _HAVE_ROS:                                          # pragma: no cover
        raise SystemExit("inspection_planner needs rclpy; the pure planner in "
                         "sam_target_inspection.inspection_planner runs without it.")
    rclpy.init(args=args)
    node = InspectionPlannerNode()
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
