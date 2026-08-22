#!/usr/bin/env python3
"""`farm_planner` — the ROS half of the farm mission plan. It answers questions; it commands
nothing.

Three answers and nothing else, exactly as the behaviour tree's `A_FarmInspection` expects:
**here is the next sub-goal**, **this phase is done**, **I refuse and here is why**. It holds
no action client, publishes no actuator command and owns no phase transition — the tree owns
those (instructions §3 D2, invariant 12, and the same goal-source shape as ADR-004's bottom
following).

All the geometry is in `farm_mission.py` and is tested without a graph. This file:

  1. loads the prior (FATAL if it cannot — a planner with no prior would invent a farm);
  2. keeps the latest `perception/farm/report` from the localizer;
  3. turns each request on `perception/farm/plan/request` into one answer on
     `perception/farm/plan/answer`, echoing the request's `seq` so a stale answer cannot be
     mistaken for a fresh one;
  4. publishes a health line and its own view of the mission for the status field.

WHY REQUEST/ANSWER RATHER THAN A TIMER. The tree asks when it is ready for the next goal, so
there is exactly one outstanding question at a time and the planner never runs ahead of the
vehicle. It also means the planner's state advances on the TREE's word that a goal was
reached, which is the only place that fact exists.
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sam_farm_inspection.farm_mission import (GOAL, MISSION_DONE, PHASE_DONE, REFUSED,
                                              FarmMissionPlanner, MissionConfig)
from sam_farm_inspection.farm_prior import PriorRefusal, load_farm_prior


class FarmPlanner(Node):

    def __init__(self):
        super().__init__("farm_planner")
        self.declare_parameter("robot_name", "sam_auv_v1")
        self.declare_parameter("farm_prior_path", "")
        self.declare_parameter("report_topic", "perception/farm/report")
        self.declare_parameter("rpm", 500.0)
        self.declare_parameter("goal_timeout_s", 900.0)
        # An explicit encircle depth OVERRIDE, in metres positive-down. Negative means "use
        # the prior's derived depth", which is what should normally happen: the prior works
        # out the depth from which the buoys are actually visible. An override is re-checked
        # against the same geometry and REFUSED by name if it would see nothing — that is
        # the whole reason it is allowed to exist rather than being silently obeyed.
        self.declare_parameter("encircle_depth_override_m", -1.0)

        gp = lambda n: self.get_parameter(n).value            # noqa: E731
        self.robot_name = gp("robot_name")
        try:
            self.prior = load_farm_prior(gp("farm_prior_path") or None)
        except PriorRefusal as e:
            self.get_logger().fatal(str(e))
            raise

        override = float(gp("encircle_depth_override_m"))
        self.planner = FarmMissionPlanner(
            self.prior,
            MissionConfig(goal_timeout_s=float(gp("goal_timeout_s"))),
            encircle_depth_override_m=(None if override < 0 else override))
        self.rpm = float(gp("rpm"))
        self.goal_timeout_s = float(gp("goal_timeout_s"))

        ns = f"/{self.robot_name}"
        self.create_subscription(String, f'{ns}/{gp("report_topic")}', self.report_cb, 5)
        self.create_subscription(String, f"{ns}/perception/farm/plan/request",
                                 self.request_cb, 5)
        self.pub_answer = self.create_publisher(String, f"{ns}/perception/farm/plan/answer", 5)
        self.pub_mission = self.create_publisher(String, f"{ns}/perception/farm/mission", 1)
        self.pub_health = self.create_publisher(String, f"{ns}/perception/farm/planner/health", 1)
        self.create_timer(1.0, self.publish_health)

        self._n_requests = 0
        self._n_reports = 0
        self._health_state = None
        self.get_logger().info(
            f"farm_planner up: prior {self.prior.path}, encircle at "
            f"{self.planner._encircle_depth():.2f} m, lanes at "
            f"{self.prior.lane.get('scan_depth_m')} m, answering on "
            f"{ns}/perception/farm/plan/answer")
        for c in self.prior.caveats:
            self.get_logger().warn("prior caveat: " + c)

    # ------------------------------------------------------------------ inputs
    def report_cb(self, msg: String):
        try:
            report = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"unparsable farm report: {e}", throttle_duration_sec=10.0)
            return
        self._n_reports += 1
        self.planner.map_update(report)

    def request_cb(self, msg: String):
        try:
            req = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"unparsable plan request: {e}")
            return
        if not isinstance(req, dict):
            return
        self._n_requests += 1
        if req.get("event") == "reached":
            self.planner.goal_reached()

        ans = self.planner.next()
        out = {"seq": req.get("seq"), "kind": ans.kind, "phase": ans.phase,
               "reason": ans.reason, "response": ans.response}
        if ans.kind == GOAL and ans.goal is not None:
            out["params"] = ans.goal.to_waypoint_params(self.rpm, self.goal_timeout_s)
            out["unity_xz"] = list(ans.goal.unity_xz)
        self.pub_answer.publish(String(data=json.dumps(out, separators=(",", ":"))))
        self.pub_mission.publish(String(data=json.dumps(self.planner.report(),
                                                        separators=(",", ":"))))
        if ans.kind == REFUSED:
            self.get_logger().error(f"farm plan REFUSED in {ans.phase}: {ans.reason} "
                                    f"| response: {ans.response}")
        elif ans.kind in (PHASE_DONE, MISSION_DONE):
            self.get_logger().info(f"farm plan: {ans.reason}")

    # ------------------------------------------------------------------ health
    def publish_health(self):
        p = self.planner
        if p._refusal is not None:
            state, detail = "REFUSED", p._refusal.reason
        elif p.finished:
            state, detail = "DONE", "all four phases planned and flown"
        elif self._n_requests == 0:
            state, detail = "IDLE", "no plan request yet — nothing is flying this mission"
        else:
            state = "PLANNING"
            detail = (f"{p.phase} {p._i}/{len(p._goals)}, "
                      f"map={'yes' if (p.map and p.map.ok) else 'no'}")
        self.pub_health.publish(String(data=(
            f"{state}|{self._n_requests}|{self._n_reports}|{p.phase}|"
            f"{p._i}/{len(p._goals)}|{detail}")))
        if state != self._health_state:
            msg = f"planner health: {self._health_state} -> {state} ({detail})"
            if state == "REFUSED":
                self.get_logger().error(msg)
            else:
                self.get_logger().info(msg)
            self._health_state = state


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FarmPlanner()
    except PriorRefusal:
        if rclpy.ok():
            rclpy.shutdown()
        raise SystemExit(2)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # A clean stop must exit 0 — SETTLED §1c, the duplicate-bringup chain.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
