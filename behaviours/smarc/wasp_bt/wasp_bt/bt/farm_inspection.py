#!/usr/bin/python3
"""Behaviour-tree shells for the algae-farm inspection: `A_FarmInspection` and
`A_SurfaceAndReport`.

EVERY DECISION IN HERE LIVES IN `farm_inspection_core.py`, which is pure python and is
tested without a graph. These classes do three things and nothing else: translate ticks into
core calls, hand the core's goal dicts to the ONE existing action client, and publish
feedback. If you find yourself adding an `if` to this file, it belongs in the core.

`A_FarmInspection` takes the SAME `BTActionClient` instance the ordinary
`auv-depth-move-to` task uses — `ros_bt` passes it the cached one — because two clients (or,
worse, two servers) on `auv_depth_move_to` is the defect that stopped every mission at
waypoint 1 for a week (SETTLED §1c).
"""
import json

from py_trees.behaviour import Behaviour
from py_trees.common import Status

from smarc_action_base.smarc_action_base import ActionClientState
from smarc_msgs.action import BaseAction
from std_msgs.msg import Bool, String
from sensor_msgs.msg import NavSatFix
from smarc_mission_msgs.msg import GotoWaypoint

from wasp_bt.bt.client import BTActionClient
from wasp_bt.bt.farm_inspection_core import (FAILURE, RUNNING, SEND_GOAL, SUCCESS,
                                             FarmInspectionRunner, SurfaceAndReportCore)
from wasp_bt.waraps.waraps_task_handler import WaraPSTaskHandler

#: The WARA-PS task name this tree provides itself. It is NOT an action server: see
#: `WaraPSTaskHandler.BT_PROVIDED_TASKS`.
FARM_INSPECTION_TASK = "auv-farm-inspection"

_RUNNING_STATES = (ActionClientState.SENT, ActionClientState.ACCEPTED,
                   ActionClientState.RUNNING, ActionClientState.CANCELLING)
_FAILED_STATES = (ActionClientState.DISCONNECTED, ActionClientState.ERROR,
                  ActionClientState.REJECTED, ActionClientState.CANCELLED)


class PlannerLink:
    """Request/answer with the farm planner node over two topics.

    Not a service, deliberately: a service call blocks a tick, and the planner may legitimately
    take a moment (it runs a fit). Requests carry a sequence number and the core discards any
    answer that does not match — the correlation is the point, not the transport.
    """

    def __init__(self, node, robot_name: str):
        ns = f"/{robot_name}"
        self._pub = node.create_publisher(String, f"{ns}/perception/farm/plan/request", 5)
        node.create_subscription(String, f"{ns}/perception/farm/plan/answer",
                                 self._answer_cb, 5)
        self._latest = None

    def _answer_cb(self, msg):
        try:
            self._latest = json.loads(msg.data)
        except Exception:
            # A malformed answer is NOT an answer, and must not overwrite a good one with
            # None: the core would then read "the planner has not replied" and eventually
            # time out with a refusal that names the planner. That is the right refusal for
            # the wrong reason, so the parse failure is kept out of the state entirely.
            self._latest = self._latest

    def request(self, payload: dict) -> None:
        self._pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def latest(self):
        return self._latest


class _ClientDriver:
    """Adapts a `BTActionClient` to the core's `send_goal` / `goal_state` pair."""

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
        if s in _RUNNING_STATES:
            return "running"
        return "running"


class A_FarmInspection(Behaviour):
    """The `auv-farm-inspection` task: stream the planner's sub-goals through one client."""

    def __init__(self, client: BTActionClient, bt, task_handler: WaraPSTaskHandler,
                 node, robot_name: str):
        super().__init__(f"A_FarmInspection({client.get_action_name()})")
        self._client = client
        self._bt = bt
        self._task_handler = task_handler
        self._node = node
        self._driver = _ClientDriver(client)
        self._link = PlannerLink(node, robot_name)
        self._runner = None
        self._last_feedback_at = 0.0

    def setup(self) -> None:
        return self._client._setup(num_iters=5)

    def initialise(self) -> None:
        params = self._task_handler.get_current_task_params() or {}
        self._runner = FarmInspectionRunner(
            goal_state=self._driver.state,
            link=self._link,
            now=lambda: self._bt.now_seconds,
            params=params,
            log=self._log)
        self._task_handler.set_current_task_status("started")
        self._task_handler.publish_feedback_to_current_task(
            "farm inspection starting — asking the planner for the first sub-goal")

    def _log(self, level: str, msg: str) -> None:
        getattr(self._node.get_logger(), level, self._node.get_logger().info)(msg)

    def update(self) -> Status:
        step = self._runner.tick()
        if step.action == SEND_GOAL:
            self._driver.send(step.params)
        now = self._bt.now_seconds
        if now - self._last_feedback_at > 1.0 or step.action in (SUCCESS, FAILURE):
            self._last_feedback_at = now
            self._task_handler.publish_feedback_to_current_task(step.feedback)
        self.feedback_message = step.feedback
        if step.action == SUCCESS:
            return Status.SUCCESS
        if step.action == FAILURE:
            # A refused inspection is not a completed survey. The detail (what the mission
            # should DO about it) goes out with the feedback so it reaches the operator's
            # record and not only the vehicle's log.
            self._task_handler.publish_feedback_to_current_task(step.detail or step.feedback)
            return Status.FAILURE
        return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        if new_status == Status.INVALID and self._client.state in _RUNNING_STATES:
            self._client.cancel_goal(self._client.cancel_callback)
        self._client.get_ready()


class A_SurfaceAndReport(Behaviour):
    """Invariant 5b's behaviour-tree half: mission complete -> surface -> report.

    Reads the vehicle's own position (for the surfacing waypoint), the protective stop (the
    gate), and the diving controller's `ctrl/neutral_handoff` line (the confirmation). It
    writes nothing except one goal through the same action client — no new actuator writer
    (invariant 12), no new command path.
    """

    def __init__(self, client: BTActionClient, bt, task_handler: WaraPSTaskHandler,
                 node, robot_name: str):
        super().__init__("A_SurfaceAndReport")
        self._client = client
        self._bt = bt
        self._task_handler = task_handler
        self._node = node
        self._driver = _ClientDriver(client)
        self._fix = None
        self._last_wp = None
        self._stop = False
        self._handoff = None

        ns = f"/{robot_name}"
        node.create_subscription(NavSatFix, f"{ns}/core/gps", self._gps_cb, 5)
        node.create_subscription(GotoWaypoint, f"{ns}/mission/last_wp", self._last_wp_cb, 5)
        node.create_subscription(Bool, f"{ns}/perception/obstacle/stop", self._stop_cb, 5)
        node.create_subscription(String, f"{ns}/ctrl/neutral_handoff", self._handoff_cb, 5)

        self._core = SurfaceAndReportCore(
            goal_state=self._driver.state,
            position=self._surface_position, handoff=lambda: self._handoff,
            stop_active=lambda: self._stop, now=lambda: self._bt.now_seconds,
            log=self._log)

    # ---------------------------------------------------------------- inputs
    def _gps_cb(self, msg):
        if msg.status.status >= 0:
            self._fix = (msg.latitude, msg.longitude)
        else:
            # NO_FIX is the normal submerged state, and a LATCHED old fix is worse than
            # none: at the end of this mission the last fix is the launch point 237 m away,
            # and surfacing "there" would be a transit home dressed as a surfacing.
            self._fix = None

    def _last_wp_cb(self, msg):
        self._last_wp = (msg.lat, msg.lon)

    def _surface_position(self):
        """Where to put the surfacing waypoint: a live fix, else the last commanded waypoint.

        There is no `dr/lat_lon` in this tree (the estimator's geographic output is not
        published on the vehicle graph today), so the honest second-best is the waypoint the
        vehicle has just been flown to — it is at most one goal tolerance away. Returning
        None when there is neither is deliberate: the core refuses rather than surfacing at
        (0, 0), which would be a transit across the sea.
        """
        return self._fix or self._last_wp

    def _stop_cb(self, msg):
        self._stop = bool(msg.data)

    def _handoff_cb(self, msg):
        # "RELEASED|confirmed|reason" / "HOLDING|...|reason" — see DivePub.
        parts = str(msg.data).split("|", 2)
        self._handoff = {
            "released": parts[0].strip().upper() == "RELEASED",
            "confirmed": len(parts) > 1 and parts[1].strip().lower() == "confirmed",
            "reason": parts[2] if len(parts) > 2 else "",
        }

    def _log(self, level: str, msg: str) -> None:
        getattr(self._node.get_logger(), level, self._node.get_logger().info)(msg)

    # ---------------------------------------------------------------- tick
    def update(self) -> Status:
        step = self._core.tick()
        if step.action == SEND_GOAL:
            self._driver.send(step.params)
        self.feedback_message = step.feedback
        self._task_handler.publish_feedback_to_current_task(step.feedback)
        if step.action == SUCCESS:
            return Status.SUCCESS
        if step.action == FAILURE:
            return Status.FAILURE
        return Status.RUNNING


class A_EndOfMissionSurface(A_SurfaceAndReport):
    """`A_SurfaceAndReport` for the ORDINARY path — invariant 5b, finally closed (2026-08-29).

    Identical behaviour to its parent; the only difference is that it tells the end-of-mission
    latch it has finished, so the tree surfaces ONCE per mission and then resolves to
    `A_Chilling` like any idle vehicle.

    It is a subclass rather than a copy on purpose: the surfacing logic, the protective-stop
    gate and the confirmed-vs-timeout distinction are the hard parts and they are already
    written and tested (SurfaceAndReportCore). A second implementation of "how to surface"
    would be a second thing to keep right.

    ON THE MISSING TASK FEEDBACK, stated rather than discovered later: the parent publishes its
    feedback to the CURRENT task, and at end of mission the queue is empty by definition, so
    that call returns None and the WARA-PS task feedback channel says nothing. The report still
    reaches an operator through the behaviour-tree tip (`feedback_message`, which MC and VC
    read from `bt_status`) and through the node log. Nothing is silently dropped, but the
    channel differs from the farm-task case and consumers should not expect a task feedback
    message for an end-of-mission surfacing -- there is no task to attach it to.
    """

    def __init__(self, client: BTActionClient, bt, task_handler: WaraPSTaskHandler,
                 node, robot_name: str):
        super().__init__(client, bt, task_handler, node, robot_name)
        self.name = "A_EndOfMissionSurface"
        self._task_handler_ref = task_handler

    def update(self) -> Status:
        status = super().update()
        if status in (Status.SUCCESS, Status.FAILURE):
            from .end_of_mission_core import get_or_create
            get_or_create(self._task_handler_ref).note_surface_finished(
                getattr(self._core, "outcome", None))
        return status
