#!/usr/bin/python3
"""Behaviour-tree shells for the adaptive close inspection.

EVERY DECISION IN HERE LIVES IN `target_inspection_core.py`, which is pure python and is tested
without a graph. These classes translate ticks into core calls, hand the core's goal dicts to
the ONE existing action client, publish the sonar-mode request, and report feedback. If you find
yourself adding an `if` to this file, it belongs in the core.

`A_CloseInspection` takes the SAME `BTActionClient` instance the ordinary `auv-depth-move-to`
task uses — `ros_bt` passes it the cached one — because two clients (or, worse, two servers) on
`auv_depth_move_to` is the defect that stopped every mission at waypoint 1 for a week
(SETTLED §1c).

**NOTHING IN THIS FILE CANCELS A GOAL FROM THE SIDE.** The preemption is py_trees' own: this
subtree sits above `F_Tasks` in the task-handler Fallback, so when it starts running py_trees
terminates the ordinary `A_ActionClient` with `Status.INVALID` and
`bt_action_client_action.py:110-136` does the cancelling and the `get_ready()`. A `cancel_goal`
call from here would put the client into `CANCELLED`, which `A_ActionClient` lists as a FAILURE
state, and the failure branch calls `clear_current_task()` — deleting the interrupted waypoint.
Measured 2026-09-09, SETTLED §3ad.
"""
import json

from py_trees.behaviour import Behaviour
from py_trees.common import Status

from smarc_action_base.smarc_action_base import ActionClientState
from smarc_msgs.action import BaseAction
from std_msgs.msg import String

from wasp_bt.bt.client import BTActionClient
from wasp_bt.bt.target_inspection_core import (FAILURE, RUNNING, SEND_GOAL, SET_SONAR_MODE,
                                               START_BURST, SUCCESS, CloseInspectionRunner,
                                               DiversionLatch, ResumeCore, get_or_create)
from wasp_bt.waraps.waraps_task_handler import WaraPSTaskHandler

_RUNNING_STATES = (ActionClientState.SENT, ActionClientState.ACCEPTED,
                   ActionClientState.RUNNING, ActionClientState.CANCELLING)
_FAILED_STATES = (ActionClientState.DISCONNECTED, ActionClientState.ERROR,
                  ActionClientState.REJECTED, ActionClientState.CANCELLED)


class TargetPlannerLink:
    """Request/answer with the inspection planner node over two topics.

    Not a service, for the same reason `farm_inspection.PlannerLink` is not one: a service call
    blocks a tick and the planner may legitimately take a moment. Requests carry a sequence
    number and the core discards any answer that does not match.
    """

    def __init__(self, node, robot_name: str):
        ns = f"/{robot_name}"
        self._pub = node.create_publisher(String, f"{ns}/perception/target/plan/request", 5)
        node.create_subscription(String, f"{ns}/perception/target/plan/answer",
                                 self._answer_cb, 5)
        self._latest = None

    def _answer_cb(self, msg):
        try:
            self._latest = json.loads(msg.data)
        except Exception:
            # A malformed answer is NOT an answer, and must not overwrite a good one with None:
            # the core would then read "the planner has not replied" and time out with a
            # refusal that names the planner — the right refusal for the wrong reason.
            self._latest = self._latest

    def request(self, payload: dict) -> None:
        self._pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def latest(self):
        return self._latest


class _ClientDriver:
    """Adapts a `BTActionClient` to the core's `send` / `state` pair. Identical in shape to
    `farm_inspection._ClientDriver`, and deliberately a separate object rather than an import:
    the two tasks must be able to hold different send state on the same client."""

    def __init__(self, client: BTActionClient):
        self._client = client
        self._sent = False

    def send(self, params: dict) -> None:
        msg = BaseAction.Goal()
        msg.goal.data = json.dumps(params)
        self._client.send_goal(msg)
        self._sent = True

    def state(self) -> str:
        s = self._client.state
        if not self._sent:
            return "idle"
        if s == ActionClientState.DONE:
            self._sent = False
            self._client.get_ready()
            return "done"
        if s in _FAILED_STATES:
            self._sent = False
            self._client.get_ready()
            return "failed"
        return "running"


class C_TargetCandidatePending(Behaviour):
    """SUCCESS while a candidate is latched on the TASK HANDLER and the policy allows it.

    THREE THINGS IT WILL NOT DO, each one a rule:
      * it does not latch on itself. The latch is on the task handler
        (`target_inspection_core.get_or_create`), because `_update_task_handler_tree` rebuilds
        this subtree whenever an action server's heartbeat changes and a flag on a node is
        erased by that rebuild (SETTLED §3f0p);
      * it does not divert without a policy. A mission with no `adaptive` block runs a plain
        lawnmower — the ABSENCE is the answer, never a default policy;
      * it does not touch `emergency_flag`. A diversion is a mission phase.
    """

    def __init__(self, task_handler: WaraPSTaskHandler, node, robot_name: str):
        super().__init__("C_TargetCandidatePending")
        self._task_handler = task_handler
        self._node = node
        self._said_no_policy = False
        node.create_subscription(String, f"/{robot_name}/perception/target/divert_request",
                                 self._request_cb, 5)

    def _request_cb(self, msg):
        """A detector (or the ledger) asks for a diversion. Latched HERE, on the handler."""
        try:
            req = json.loads(msg.data)
        except Exception:
            return
        latch = get_or_create(self._task_handler)
        policy = self._task_handler.get_adaptive_policy()
        if not policy:
            if not self._said_no_policy:
                self._said_no_policy = True
                self._node.get_logger().info(
                    "a target candidate arrived but this mission carries no `adaptive` block, "
                    "so no diversion policy exists and the lawnmower continues unchanged. Said "
                    "once; the candidate is still recorded by the ledger.")
            return
        p0 = self._task_handler.diversion_point()
        if p0 is None:
            self._node.get_logger().error(
                "a target candidate arrived but the vehicle has no position to record as the "
                "diversion point, so there would be nowhere to resume to. Refusing the "
                "diversion; the lawnmower continues.")
            latch.refuse("no position estimate for the diversion point")
            return
        latch.offer(req, p0, self._task_handler.current_leg_id())

    def update(self) -> Status:
        latch = get_or_create(self._task_handler)
        return Status.SUCCESS if latch.active else Status.FAILURE


class A_CloseInspection(Behaviour):
    """Stream the inspection planner's sub-goals through the one cached client."""

    def __init__(self, client: BTActionClient, bt, task_handler: WaraPSTaskHandler,
                 node, robot_name: str):
        super().__init__(f"A_CloseInspection({client.get_action_name()})")
        self._client = client
        self._bt = bt
        self._task_handler = task_handler
        self._node = node
        self._driver = _ClientDriver(client)
        self._link = TargetPlannerLink(node, robot_name)
        self._runner = None
        self._last_feedback_at = 0.0
        self._sonar_mode = None
        self._coverage = {}

        ns = f"/{robot_name}"
        self._mode_pub = node.create_publisher(String, f"{ns}/payload/sonar3d/set_mode", 5)
        self._burst_pub = node.create_publisher(String, f"{ns}/perception/target/capture", 5)
        node.create_subscription(String, f"{ns}/payload/sonar3d/mode", self._mode_cb, 5)
        node.create_subscription(String, f"{ns}/perception/target/coverage",
                                 self._coverage_cb, 5)

    # ---------------------------------------------------------------- inputs
    def _mode_cb(self, msg):
        self._sonar_mode = str(msg.data)

    def _coverage_cb(self, msg):
        """`{"station": 3, "done": true, "why": "..."}` from the inspection recorder."""
        try:
            d = json.loads(msg.data)
            self._coverage[int(d["station"])] = (bool(d.get("done")), str(d.get("why", "")))
        except Exception:
            pass

    def _station_done(self, station: int):
        return self._coverage.get(int(station), (False, "the recorder has not reported this "
                                                        "station yet"))

    def _log(self, level: str, msg: str) -> None:
        getattr(self._node.get_logger(), level, self._node.get_logger().info)(msg)

    # ---------------------------------------------------------------- lifecycle
    def setup(self) -> None:
        return self._client._setup(num_iters=5)

    def initialise(self) -> None:
        latch = get_or_create(self._task_handler)
        self._runner = CloseInspectionRunner(
            goal_state=self._driver.state,
            link=self._link,
            now=lambda: self._bt.now_seconds,
            sonar_mode=lambda: self._sonar_mode,
            coverage=self._station_done,
            params=self._task_handler.get_adaptive_policy() or {},
            log=self._log)
        latch.phase = "divert"
        self._task_handler.publish_feedback_to_current_task(
            "target candidate latched — diverting; the interrupted waypoint stays queued")

    def update(self) -> Status:
        step = self._runner.tick()
        latch = get_or_create(self._task_handler)
        latch.phase = self._runner.phase
        if step.action == SEND_GOAL:
            self._driver.send(step.params)
        elif step.action == SET_SONAR_MODE:
            # A SENSOR MODE, NOT AN ACTUATOR. A plain String on the topic the sonar already
            # listens on; the runner waits for the announcement before going on.
            self._mode_pub.publish(String(data=step.mode))
        elif step.action == START_BURST:
            self._burst_pub.publish(String(data=json.dumps(
                {"station": step.station, "candidate": (latch.candidate or {}).get("id")})))
        now = self._bt.now_seconds
        if now - self._last_feedback_at > 1.0 or step.action in (SUCCESS, FAILURE):
            self._last_feedback_at = now
            self._task_handler.publish_feedback_to_current_task(step.feedback)
        self.feedback_message = step.feedback
        if step.action == SUCCESS:
            self._close(latch)
            return Status.SUCCESS
        if step.action == FAILURE:
            self._task_handler.publish_feedback_to_current_task(step.detail or step.feedback)
            self._close(latch)
            return Status.FAILURE
        return Status.RUNNING

    def _close(self, latch: DiversionLatch) -> None:
        """Record what the diversion cost and extend the mission timeout by exactly that.

        A good inspection may never be how a good mission dies (SETTLED §3p's rule), and the
        extension is what the SANCTIONED diversion actually consumed — measured, not the budget
        it was allowed.
        """
        consumed = self._runner.consumed_s()
        verdict = self._runner.verdict()
        latch.note_result(verdict, consumed)
        self._task_handler.extend_mission_timeout(consumed, reason="adaptive diversion")
        self._log("info", f"close inspection ended: {verdict} after {consumed:.0f} s; "
                          f"mission timeout extended by the same")

    def terminate(self, new_status: Status) -> None:
        """Preempted, finished or failed — get the client ready for whoever ticks next.

        NOTE WHAT IS NOT HERE. This does not cancel a goal held by another behaviour, and it is
        not what preempts the ordinary waypoint task: py_trees does that by priority. The one
        cancel below is of THIS behaviour's OWN in-flight sub-goal, when this subtree is itself
        preempted (by an abort, say) — the same thing `A_FarmInspection.terminate` does.
        """
        if new_status == Status.INVALID and self._client.state in _RUNNING_STATES:
            self._client.cancel_goal(self._client.cancel_callback)
        self._client.get_ready()


class A_ResumeAtDiversionPoint(Behaviour):
    """Fly back to P0 with the leg's heading, then let the task tree re-send the leg itself."""

    def __init__(self, client: BTActionClient, bt, task_handler: WaraPSTaskHandler,
                 node, robot_name: str):
        super().__init__("A_ResumeAtDiversionPoint")
        self._client = client
        self._bt = bt
        self._task_handler = task_handler
        self._node = node
        self._driver = _ClientDriver(client)
        self._core = None

    def setup(self) -> None:
        return self._client._setup(num_iters=5)

    def initialise(self) -> None:
        latch = get_or_create(self._task_handler)
        latch.phase = "resume"
        self._core = ResumeCore(
            goal_state=self._driver.state, latch=latch,
            now=lambda: self._bt.now_seconds,
            params=self._task_handler.get_adaptive_policy() or {},
            log=lambda lvl, m: getattr(self._node.get_logger(), lvl,
                                       self._node.get_logger().info)(m))

    def update(self) -> Status:
        step = self._core.tick()
        if step.action == SEND_GOAL:
            self._driver.send(step.params)
        self.feedback_message = step.feedback
        self._task_handler.publish_feedback_to_current_task(step.feedback)
        if step.action in (SUCCESS, FAILURE):
            # THE LATCH IS CLEARED HERE AND NOWHERE ELSE. Once it is clear,
            # `C_TargetCandidatePending` fails on the next tick, this subtree stops matching,
            # and the ordinary task tree below it re-sends the interrupted waypoint from
            # `get_current_task_params()` — the same goal, still `tasks_executing[0]`.
            get_or_create(self._task_handler).finish()
            return Status.SUCCESS if step.action == SUCCESS else Status.FAILURE
        return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        if new_status == Status.INVALID and self._client.state in _RUNNING_STATES:
            self._client.cancel_goal(self._client.cancel_callback)
        self._client.get_ready()
