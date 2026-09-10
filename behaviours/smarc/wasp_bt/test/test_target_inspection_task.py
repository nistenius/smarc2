"""THE ADAPTIVE CLOSE INSPECTION, DRIVEN THROUGH A REAL py_trees TREE. (letter C, 2026-09-09)

WHY THIS FILE BUILDS A REAL TREE. The one thing that had to be got right this round is the
PREEMPTION, and it is a property of py_trees' own invalidation, not of any code written here.
A test that called the behaviours directly would prove nothing about it. So this file assembles
the actual Fallback the vehicle runs — the inspection subtree ahead of the mission tree — with
the REAL `A_ActionClient` from `bt/actions.py` in the mission tree, a fake action client under
it, and a real `WaraPSTaskHandler`, and it ticks.

THE FACT BEING GUARDED (SETTLED §3ad, measured 2026-09-09). `A_ActionClient` lists
`ActionClientState.CANCELLED` among its FAILURE states, and its failure branch calls
`clear_current_task()`. So a diversion that cancelled the running goal from the side would
DELETE THE INTERRUPTED WAYPOINT from the queue and the lawnmower would silently lose a leg. The
path that works is a higher-priority sibling: py_trees terminates the running `A_ActionClient`
with `Status.INVALID`, and `bt_action_client_action.py:110-136` does the cancelling and the
`get_ready()` itself.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/behaviours/smarc/wasp_bt/test/test_target_inspection_task.py
"""
import ast
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# --------------------------------------------------------------------------- ROS stubs
# Duplicated from `test_abort_names_its_origin.py` rather than imported, deliberately and for
# the reason that file gives: two test modules sharing stubs interact through `sys.modules` and
# that has already cost this project a debugging round.
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _PermissiveModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        v = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, v)
        return v


def _permissive(name):
    sys.modules[name] = _PermissiveModule(name)
    return sys.modules[name]


class _RosNamespaceFinder:
    ROOTS = ("rclpy", "smarc_msgs", "smarc_mission_msgs", "smarc_action_base",
             "geometry_msgs", "std_msgs", "std_srvs", "nav_msgs", "sensor_msgs",
             "geographic_msgs", "builtin_interfaces", "action_msgs",
             "unique_identifier_msgs", "visualization_msgs")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root not in self.ROOTS or fullname in sys.modules:
            return None
        import importlib.machinery
        return importlib.machinery.ModuleSpec(fullname, _RosNamespaceLoader(), is_package=True)


class _RosNamespaceLoader:
    def create_module(self, spec):
        return _PermissiveModule(spec.name)

    def exec_module(self, module):
        return None


class _Msg:
    def __init__(self, *a, **kw):
        self.data = kw.get("data", a[0] if a else None)


class ActionClientState:
    """The real enum's members, as plain sentinels. The VALUES do not matter; what matters is
    which of them `A_ActionClient` files under failure — and this test asserts that CANCELLED is
    one of them, straight off the class under test."""
    READY = "READY"
    SENT = "SENT"
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    DONE = "DONE"
    ERROR = "ERROR"
    REJECTED = "REJECTED"
    DISCONNECTED = "DISCONNECTED"


class _GoalData:
    def __init__(self):
        self.data = ""


class _BaseActionGoal:
    def __init__(self):
        self.goal = _GoalData()


class BaseAction:
    Goal = _BaseActionGoal


def _install_stubs():
    sys.meta_path.insert(0, _RosNamespaceFinder())
    for _root in _RosNamespaceFinder.ROOTS:
        _permissive(_root)
    _stub("rclpy.node", Node=object)
    _stub("std_msgs.msg", String=_Msg, Int8=_Msg, Empty=_Msg, Bool=_Msg, Float32=_Msg)
    _stub("std_srvs.srv", Trigger=MagicMock())
    _stub("geographic_msgs.msg", GeoPoint=_Msg)
    _stub("sensor_msgs.msg", NavSatFix=_Msg)
    _stub("smarc_action_base.smarc_action_base",
          ActionClientState=ActionClientState, ActionType=MagicMock(), SMARCActionClient=object,
          ActionFeedback=object, ActionResult=object)
    _stub("smarc_msgs.action", BaseAction=BaseAction)

    class Topics:
        VEHICLE_HEALTH_READY = 0
        VEHICLE_HEALTH_WAITING = 1
        VEHICLE_HEALTH_ERROR = 2

    class _TopicsMeta(type):
        def __getattr__(cls, name):
            return name.lower()

    Topics = _TopicsMeta("Topics", (), dict(Topics.__dict__))
    _stub("smarc_msgs.msg", Topics=Topics)
    _stub("smarc_mission_msgs.msg", Topics=Topics, GotoWaypoint=_Msg)

    class SensorNames:
        pass
    _stub("wasp_bt.vehicles")
    _stub("wasp_bt.vehicles.sensor", Sensor=object, SensorNames=SensorNames)
    _stub("wasp_bt.vehicles.vehicle",
          IVehicleStateContainer=object, IVehicleState=object, IVehicle=object)
    return Topics


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TOPICS = _install_stubs()

py_trees = pytest.importorskip("py_trees", reason="python3 -m pip install --user py_trees")
from py_trees.common import Status                                        # noqa: E402
from py_trees.composites import Selector as Fallback                      # noqa: E402
from py_trees.composites import Sequence                                  # noqa: E402
from py_trees.behaviours import Success                                   # noqa: E402

from wasp_bt.waraps.waraps_task_handler import WaraPSTaskHandler          # noqa: E402
from wasp_bt.bt.actions import A_ActionClient                             # noqa: E402
from wasp_bt.bt import target_inspection as TI                            # noqa: E402
from wasp_bt.bt import target_inspection_core as TIC                      # noqa: E402


# --------------------------------------------------------------------------- doubles
class FakeClient:
    """The ONE action client. Records every goal, every cancel, every get_ready."""

    def __init__(self, ticks_to_finish=2):
        self.state = ActionClientState.READY
        self.sent = []
        self.cancels = 0
        self.readies = 0
        self.feedback_message = ""
        self.ticks_to_finish = ticks_to_finish
        self._n = 0
        self._node = MagicMock()
        self.cancel_callback = MagicMock()
        self.outcome = ActionClientState.DONE

    # -- the BTActionClient surface the behaviours use -----------------------------
    def get_action_name(self):
        return "auv_depth_move_to"

    def _setup(self, num_iters=1, timeout=None):
        return True

    def send_goal(self, msg):
        self.sent.append(json.loads(msg.goal.data))
        self.state = ActionClientState.RUNNING
        self._n = 0

    def cancel_goal(self, cb=None):
        self.cancels += 1
        self.state = ActionClientState.CANCELLED

    def get_ready(self):
        self.readies += 1
        self.state = ActionClientState.READY
        return True

    # -- the fake server's own progress --------------------------------------------
    def step(self):
        if self.state == ActionClientState.RUNNING:
            self._n += 1
            if self._n >= self.ticks_to_finish:
                self.state = self.outcome


class FakeLink:
    """The inspection planner, scripted."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []
        self._latest = None

    def request(self, payload):
        self.requests.append(payload)
        if not self.answers:
            return
        ans = dict(self.answers.pop(0))
        ans["seq"] = payload["seq"]
        self._latest = ans

    def latest(self):
        return self._latest


def _wp(name, lat, lon, depth=2.0, heading=None):
    wp = {"latitude": lat, "longitude": lon, "target_depth": depth, "rpm": 500.0,
          "tolerance": 2.0}
    if heading is not None:
        wp["heading"] = heading
    return {"waypoint": wp, "name": name, "timeout": 300.0}


PLAN = [
    {"kind": "goal", "phase": "approach", "params": _wp("approach_abeam", 58.8214, 17.6349),
     "reason": "approach"},
    {"kind": "sonar_mode", "phase": "close_ops", "mode": "inspection", "reason": "entry"},
    {"kind": "goal", "phase": "close_ops", "params": _wp("station_00_r0", 58.8215, 17.6348),
     "station": 0, "reason": "station 0"},
    {"kind": "burst", "phase": "close_ops", "station": 0, "min_frames": 6, "reason": "burst"},
    {"kind": "sonar_mode", "phase": "verify", "mode": "navigation", "reason": "exit"},
    {"kind": "mission_done", "phase": "resume", "reason": "inspection complete"},
]


def _leg(desc, lat, lon):
    return {"name": "auv-depth-move-to", "description": desc, "task-uuid": f"uuid-{desc}",
            "params": {"waypoint": {"latitude": lat, "longitude": lon, "target_depth": 2.0,
                                    "rpm": 500.0, "tolerance": 2.0, "timeout": 300.0}}}


LEGS = [_leg("1", 58.8215, 17.6348), _leg("2", 58.8216, 17.6335),
        _leg("3", 58.8218, 17.6321), _leg("4", 58.8215, 17.6308)]


def _start_tst(adaptive=None, timeout=1800.0):
    params = {"timeout": timeout}
    if adaptive is not None:
        params["adaptive"] = adaptive
    return {"command": "start-tst", "com-uuid": "com-1",
            "tst": {"name": "seq", "tst-uuid": "tree-1", "params": params,
                    "common-params": {}, "children": [dict(l) for l in LEGS]}}


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Rig:
    """The vehicle, as far as the tree can tell: one handler, one client, one tree."""

    def __init__(self, adaptive=None, plan=None, ticks_to_finish=4):
        self.node = MagicMock()
        self.clock = _Clock()
        self.h = WaraPSTaskHandler(self.node, {
            "name": "sam_auv_v1", "agent-uuid": "a-1",
            "levels": ["sensor", "direct_execution"], "pulse_rate": 1.0})
        self.h.current_time = self.clock
        self.h.tasks_available = [{"name": "auv-depth-move-to"}]
        self.h.health_status = TOPICS.VEHICLE_HEALTH_READY
        self.h.emergency_flag = False
        self.h._send_tst_response = MagicMock()
        self.h._handle_tst_command(_start_tst(adaptive))
        # the estimator is alive and the vehicle is on leg 2
        self.h._last_lat, self.h._last_lon = 58.82155, 17.63410
        self.h._last_depth, self.h._last_course = 2.0, 270.0
        self.h._last_pose_at = self.clock()

        self.client = FakeClient(ticks_to_finish=ticks_to_finish)
        self.bt = MagicMock()
        type(self.bt).now_seconds = property(lambda _s: self.clock())

        self.cond = TI.C_TargetCandidatePending(self.h, self.node, "sam_auv_v1")
        self.inspect = TI.A_CloseInspection(self.client, self.bt, self.h, self.node,
                                            "sam_auv_v1")
        self.inspect._link = FakeLink(plan if plan is not None else PLAN)
        self.inspect._sonar_mode = None
        self.resume = TI.A_ResumeAtDiversionPoint(self.client, self.bt, self.h, self.node,
                                                  "sam_auv_v1")
        self.leg = A_ActionClient(self.client, self.bt, self.h)

        self.root = Fallback("F_Task_Handler", memory=False, children=[
            Sequence("S_TargetInspection", memory=False, children=[
                self.cond,
                Sequence("S_TargetInspection_phases", memory=True, children=[
                    Fallback("F_InspectOrResumeAnyway", memory=False, children=[
                        self.inspect, Success(name="A_InspectionRefused_ResumeAnyway")]),
                    self.resume,
                ]),
            ]),
            Sequence("S_Mission", memory=False, children=[self.leg]),
        ])
        self.root.setup_with_descendants()

    # -- driving -------------------------------------------------------------------
    def tick(self, n=1, dt=0.5, announce_mode=True, station_done=True):
        for _ in range(n):
            self.clock.t += dt
            self.client.step()
            if announce_mode and self.inspect._runner is not None:
                want = self.inspect._runner._mode_wanted
                if want:
                    self.inspect._sonar_mode = f"{want}|4.0|20.0"
            if station_done and self.inspect._runner is not None:
                st = self.inspect._runner._burst_station
                if st is not None:
                    self.inspect._coverage[st] = (True, "6 frames accepted")
            self.root.tick_once()

    def offer_candidate(self, cid="C1"):
        self.cond._request_cb(_Msg(data=json.dumps(
            {"id": cid, "lat": 58.8215, "lon": 17.6348, "sigma_m": 1.5})))

    @property
    def latch(self):
        return TIC.get_or_create(self.h)


# --------------------------------------------------------------------------- the fact
class TestWhatASideCancelActuallyDoes(unittest.TestCase):
    """MEASURED 2026-09-09 BY DRIVING THE REAL `A_ActionClient`, and it CORRECTS SETTLED §3ad.

    §3ad says: "`A_ActionClient` lists `ActionClientState.CANCELLED` among its failure states,
    and a failure calls `clear_current_task()`" — so a goal cancelled from the side would delete
    the interrupted leg. The first half is true and is asserted below. **The second half does
    not reproduce.** `actions.py:406-412` intercepts CANCELLED at the TOP of `update()`, before
    the failure check:

        if s == ActionClientState.CANCELLED:
            self._client.get_ready()
            return Status.RUNNING

    so with CANCELLED the failure branch is never reached and the queue is untouched. Driven
    here: the task count is 4 before and 4 after. CANCELLED's membership of `_failure_states` is
    DEAD for that value.

    THIS DOES NOT CHANGE THE DESIGN, and the reason is worth writing down. Preempting by tree
    priority is still the right mechanism — it is the one `bt_action_client_action.py` documents
    and handles explicitly, it needs no reliance on a subtle early return that a future edit
    could remove, and a cancel that FAILS still lands the client in REJECTED or ERROR, which
    ARE reachable failure states and do clear the task (driven below). What changes is the
    sentence in SETTLED §3ad, and it is corrected in this round's entry rather than repeated.
    """

    def test_A_ActionClient_files_CANCELLED_under_FAILURE(self):
        rig = Rig(adaptive={"max_diversions": 1})
        self.assertIn(ActionClientState.CANCELLED, rig.leg._failure_states,
                      "if CANCELLED is no longer a failure state, the reason this feature "
                      "preempts by tree priority has changed and the design should be re-read")

    def test_but_CANCELLED_is_intercepted_before_the_failure_branch_so_the_queue_survives(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.tick(1)
        self.assertEqual(len(rig.h.tasks_executing), 4)
        rig.client.state = ActionClientState.CANCELLED
        rig.tick(1, announce_mode=False)
        self.assertEqual(len(rig.h.tasks_executing), 4,
                         "SETTLED §3ad's claim now reproduces — a side cancel DOES delete the "
                         "leg. If this fires, the early return at actions.py:406 has gone and "
                         "the correction recorded here must be un-made.")
        self.assertEqual(rig.leg.status, Status.RUNNING)

    def test_a_REACHABLE_failure_state_DOES_delete_the_interrupted_leg(self):
        """REJECTED and ERROR are the failure states the early return does not shadow — and a
        cancel that fails is exactly how a client lands in one. So the danger §3ad names is
        real; only the route to it is different."""
        rig = Rig(adaptive={"max_diversions": 1})
        rig.tick(1)
        self.assertEqual(len(rig.h.tasks_executing), 4)
        rig.client.state = ActionClientState.REJECTED
        rig.tick(1, announce_mode=False)
        self.assertEqual(len(rig.h.tasks_executing), 3,
                         "the failure branch no longer clears the current task; the whole "
                         "reason this subtree never touches the client from the side has "
                         "changed and the design should be re-read")


class TestPreemptionKeepsTheLeg(unittest.TestCase):

    def test_a_candidate_on_leg_2_preempts_by_priority_and_keeps_the_waypoint(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.tick(2)
        self.assertEqual(rig.leg.status, Status.RUNNING)
        first_goal = dict(rig.client.sent[0])
        rig.h.move_task_to_past()               # leg 1 done; the vehicle is on leg 2
        rig.tick(2)
        leg2 = dict(rig.h.get_current_task_params())
        n_queued = len(rig.h.tasks_executing)

        rig.offer_candidate()
        rig.tick(1)
        # the ordinary waypoint behaviour has been INVALIDATED by the higher-priority sibling
        self.assertEqual(rig.leg.status, Status.INVALID,
                         "the waypoint behaviour was not preempted by tree priority")
        # ... and the queue is untouched
        self.assertEqual(len(rig.h.tasks_executing), n_queued,
                         "the interrupted leg was deleted from the queue")
        self.assertEqual(rig.h.get_current_task_params(), leg2)
        self.assertIsNotNone(first_goal)

    def test_the_leg_is_re_sent_with_IDENTICAL_params_after_the_diversion(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.tick(2)
        rig.h.move_task_to_past()
        rig.tick(2)
        leg2_before = json.loads(json.dumps(rig.h.get_current_task_params()))
        rig.offer_candidate()
        rig.tick(60)
        self.assertFalse(rig.latch.active, "the diversion never finished")
        rig.tick(3)
        self.assertEqual(rig.h.get_current_task_params(), leg2_before)
        resent = [g for g in rig.client.sent
                  if g.get("waypoint", {}).get("latitude") == leg2_before["waypoint"]["latitude"]]
        self.assertTrue(resent, "the interrupted leg was never re-sent")
        self.assertEqual(resent[-1]["waypoint"], leg2_before["waypoint"],
                         "the leg came back with different parameters than it went in with")

    def test_the_inspection_never_calls_cancel_on_the_client_while_it_runs(self):
        """The one cancel that IS allowed is `terminate(INVALID)` on this subtree's own goal.
        While the diversion runs and completes normally, nothing here cancels anything."""
        rig = Rig(adaptive={"max_diversions": 1})
        rig.tick(2)
        rig.h.move_task_to_past()
        rig.tick(2)
        before = rig.client.cancels
        rig.offer_candidate()
        rig.tick(60)
        self.assertEqual(rig.client.cancels, before,
                         "the inspection cancelled a goal from the side; that is the path that "
                         "deletes the interrupted leg (SETTLED §3ad)")


class TestTheDiversionRuns(unittest.TestCase):

    def test_the_planner_answers_are_flown_in_order(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.offer_candidate()
        rig.tick(60)
        names = [g.get("name") for g in rig.client.sent]
        self.assertIn("approach_abeam", names)
        self.assertIn("station_00_r0", names)
        self.assertIn("resume_at_diversion_point", names)
        self.assertLess(names.index("approach_abeam"), names.index("station_00_r0"))
        self.assertLess(names.index("station_00_r0"),
                        names.index("resume_at_diversion_point"))

    def test_the_sonar_mode_is_published_and_WAITED_for(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.offer_candidate()
        # do NOT announce the mode: the runner must sit there
        rig.tick(12, announce_mode=False)
        published = [c.args[0].data for c in rig.inspect._mode_pub.publish.call_args_list]
        self.assertIn("inspection", published)
        self.assertIsNotNone(rig.inspect._runner._mode_wanted,
                             "the runner went on without the sonar's announcement; the "
                             "protective stop bounds itself by the ANNOUNCED range")
        names = [g.get("name") for g in rig.client.sent]
        self.assertNotIn("station_00_r0", names)

    def test_a_sonar_that_never_announces_ends_the_diversion_by_name(self):
        rig = Rig(adaptive={"max_diversions": 1, "mode_timeout_s": 3.0})
        rig.offer_candidate()
        rig.tick(30, dt=1.0, announce_mode=False)
        self.assertIsNotNone(rig.inspect._runner.refusal)
        self.assertIn("announce", rig.inspect._runner.refusal[1])
        self.assertFalse(rig.h.emergency_flag, "a refused diversion raised an emergency")

    def test_a_station_is_finished_by_the_recorders_COUNT_not_by_a_stopwatch(self):
        rig = Rig(adaptive={"max_diversions": 1, "burst_timeout_s": 1e9})
        rig.offer_candidate()
        rig.tick(20, station_done=False)
        self.assertIsNotNone(rig.inspect._runner._burst_station,
                             "the burst ended without the recorder ever accepting a frame")
        rig.tick(30)
        self.assertIsNone(rig.inspect._runner._burst_station)


class TestTheResume(unittest.TestCase):

    def test_the_resume_goal_is_at_P0_with_the_legs_heading(self):
        rig = Rig(adaptive={"max_diversions": 1})
        p0_lat, p0_lon = rig.h._last_lat, rig.h._last_lon
        rig.offer_candidate()
        rig.tick(60)
        goal = next(g for g in rig.client.sent if g["name"] == "resume_at_diversion_point")
        self.assertAlmostEqual(goal["waypoint"]["latitude"], p0_lat, places=9)
        self.assertAlmostEqual(goal["waypoint"]["longitude"], p0_lon, places=9)
        self.assertAlmostEqual(goal["waypoint"]["heading"], 270.0, places=6)

    def test_P0_is_the_position_at_the_MOMENT_of_the_diversion_not_a_later_one(self):
        """A recomputed P0 is a P0 that moved while the vehicle was away."""
        rig = Rig(adaptive={"max_diversions": 1})
        p0_lat = rig.h._last_lat
        rig.offer_candidate()
        rig.tick(4)
        rig.h._last_lat = p0_lat + 0.01          # the vehicle has moved to the ring
        rig.h._last_pose_at = rig.clock()
        rig.tick(60)
        goal = next(g for g in rig.client.sent if g["name"] == "resume_at_diversion_point")
        self.assertAlmostEqual(goal["waypoint"]["latitude"], p0_lat, places=9)

    def test_the_latch_clears_only_AFTER_the_resume(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.offer_candidate()
        for _ in range(80):
            rig.tick(1)
            if rig.inspect._runner is not None and rig.inspect._runner.finished_ok:
                break
        self.assertTrue(rig.inspect._runner.finished_ok, "the inspection never completed")
        self.assertTrue(rig.latch.active,
                        "the latch cleared at the end of the inspection, so the subtree stops "
                        "matching and the resume leg never flies")
        rig.tick(20)
        self.assertFalse(rig.latch.active)
        self.assertEqual(rig.latch.diversions_used, 1)


class TestTheRulesItMayNotBreak(unittest.TestCase):

    def test_a_candidate_with_no_policy_does_not_divert_and_says_so_once(self):
        rig = Rig(adaptive=None)
        rig.offer_candidate()
        rig.offer_candidate("C2")
        rig.tick(10)
        self.assertFalse(rig.latch.active, "a mission with no `adaptive` block diverted")
        # The vehicle keeps flying its plan, so goals ARE sent — the operator's own waypoints.
        # What must not appear is any goal the inspection planner would have produced.
        names = [g.get("name") for g in rig.client.sent]
        self.assertNotIn("approach_abeam", names,
                         "goals were sent for a diversion that was never authorised")
        self.assertNotIn("resume_at_diversion_point", names)
        self.assertEqual(rig.inspect._link.requests, [],
                         "the inspection planner was asked for a plan with no policy in force")
        logged = [str(c.args[0]) for c in rig.node.get_logger.return_value.info.call_args_list
                  if c.args and "adaptive" in str(c.args[0])]
        said = [l for l in logged if "no `adaptive` block" in l]
        self.assertEqual(len(said), 2,
                         "the absence must be said once on start-tst and once when the first "
                         "candidate arrives, and not once per candidate")

    def test_a_diversion_is_not_an_emergency(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.offer_candidate()
        rig.tick(60)
        self.assertFalse(rig.h.emergency_flag)
        self.assertIsNone(rig.h.last_abort_origin)
        self.assertIsNone(rig.h.last_abort_detail)
        self.assertFalse(rig.h.aborted_flag)

    def test_the_mission_timeout_is_extended_by_what_the_diversion_consumed(self):
        rig = Rig(adaptive={"max_diversions": 1})
        before = rig.h.mission_timeout
        rig.offer_candidate()
        rig.tick(60)
        after = rig.h.mission_timeout
        self.assertGreater(after, before, "the mission timeout was not extended")
        self.assertAlmostEqual(after - before, rig.h.mission_timeout_extension_s, places=6)
        self.assertLessEqual(rig.h.mission_timeout_extension_s, rig.clock(),
                             "the extension is larger than the whole run so far — it is not "
                             "measuring what the diversion consumed")

    def test_an_untimed_mission_is_not_made_timed_by_a_diversion(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.h.mission_timeout = None
        rig.offer_candidate()
        rig.tick(60)
        self.assertIsNone(rig.h.mission_timeout)

    def test_the_latch_lives_on_the_task_handler_and_survives_a_subtree_rebuild(self):
        """`_update_task_handler_tree` rebuilds this subtree whenever an action server's
        heartbeat changes (SETTLED §3f0p). A flag on a node is erased by that."""
        rig = Rig(adaptive={"max_diversions": 1})
        rig.offer_candidate()
        rig.tick(4)
        self.assertTrue(rig.latch.active)
        # rebuild: brand-new behaviours over the same handler
        new_cond = TI.C_TargetCandidatePending(rig.h, rig.node, "sam_auv_v1")
        self.assertEqual(new_cond.update(), Status.SUCCESS,
                         "the diversion was forgotten when the subtree was rebuilt")
        self.assertIs(TIC.get_or_create(rig.h), rig.latch)

    def test_the_phase_word_crosses_on_the_mission_timer_as_an_additive_key(self):
        rig = Rig(adaptive={"max_diversions": 1})
        self.assertEqual(rig.h.mission_timer_state()["adaptive_phase"], "scan")
        rig.offer_candidate()
        rig.tick(4)
        st = rig.h.mission_timer_state()
        self.assertIn(st["adaptive_phase"], TIC.PHASES)
        self.assertNotEqual(st["adaptive_phase"], "scan")
        self.assertEqual(st["state"], "running",
                         "`adaptive_phase` must be a FIELD; the state word may not change "
                         "(§3s5: an unknown `state` renders as nothing at all)")
        self.assertIn("extended_s", st)

    def test_the_phase_word_is_present_even_when_idle(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.h.mission_start_time = None
        st = rig.h.mission_timer_state()
        self.assertEqual(st["state"], "idle")
        self.assertEqual(st["adaptive_phase"], "scan")
        self.assertIn("extended_s", st)

    def test_the_latch_status_payload_reports_the_latchs_OWN_phase(self):
        """`as_status()` is what the bridge forwards to the station as declared status fields
        (letter E). It must report the latch's real phase, not a constant."""
        rig = Rig(adaptive={"max_diversions": 1})
        self.assertEqual(rig.latch.as_status()["adaptive_phase"], "scan")
        rig.offer_candidate()
        rig.tick(4)
        st = rig.latch.as_status()
        self.assertEqual(st["adaptive_phase"], rig.latch.phase)
        self.assertNotEqual(st["adaptive_phase"], "scan")
        self.assertEqual(set(st), {"adaptive_phase", "diversions_used",
                                   "diversion_consumed_s", "last_verdict"})

    def test_a_second_candidate_during_a_diversion_does_not_restart_it(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.offer_candidate("C1")
        rig.tick(4)
        first = rig.latch.candidate["id"]
        rig.offer_candidate("C2")
        self.assertEqual(rig.latch.candidate["id"], first)
        self.assertIn("already running", rig.latch.last_refusal)

    def test_a_STALE_position_is_no_position_at_all(self):
        """The estimator said something once, five minutes ago. Recording that as the diversion
        point would send the vehicle back to where it used to be — the same failure as latching
        an old GPS fix and calling it a surfacing position (`A_SurfaceAndReport._gps_cb`)."""
        rig = Rig(adaptive={"max_diversions": 1})
        rig.h._last_pose_at = rig.clock() - 300.0
        self.assertIsNone(rig.h.diversion_point(),
                          "a five-minute-old position was offered as the diversion point")
        rig.offer_candidate()
        rig.tick(6)
        self.assertFalse(rig.latch.active)
        self.assertIn("diversion point", rig.latch.last_refusal)

    def test_a_candidate_with_no_position_refuses_by_name_and_does_not_divert(self):
        rig = Rig(adaptive={"max_diversions": 1})
        rig.h._last_lat = rig.h._last_lon = None
        rig.offer_candidate()
        rig.tick(6)
        self.assertFalse(rig.latch.active)
        self.assertEqual(rig.latch.verdict, "not_inspected")
        self.assertIn("diversion point", rig.latch.last_refusal)
        self.assertFalse(rig.h.emergency_flag)


# --------------------------------------------------------------------------- structure
def _src(mod):
    return Path(mod.__file__).read_text()


def _called_names(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


class TestStructure(unittest.TestCase):
    """Properties of the CODE, read as syntax so comments and prose cannot satisfy them."""

    def test_the_core_never_calls_anything_that_cancels(self):
        names = _called_names(ast.parse(_src(TIC)))
        self.assertEqual(names & {"cancel_goal", "cancel", "clear_current_task"}, set(),
                         "the pure core reaches for a cancel; preemption is py_trees' job")

    def test_the_only_cancel_in_the_shell_is_inside_terminate_on_INVALID(self):
        tree = ast.parse(_src(TI))
        offenders = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            if "cancel_goal" not in _called_names(fn):
                continue
            if fn.name != "terminate":
                offenders.append(fn.name)
        self.assertEqual(offenders, [],
                         f"cancel_goal is called outside terminate(): {offenders}. That path "
                         f"puts the client into CANCELLED, which A_ActionClient treats as a "
                         f"FAILURE, and the failure branch deletes the interrupted leg.")

    def test_neither_file_touches_the_emergency_or_abort_machinery(self):
        for mod in (TI, TIC):
            src = _src(mod)
            tree = ast.parse(src)
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif isinstance(node, ast.Name):
                    names.add(node.id)
            forbidden = {"emergency_flag", "abort", "_apply_abort", "aborted_flag",
                         "A_Abort", "A_Chilling", "A_EmergencyParked",
                         "last_abort_origin", "set_mission_status"}
            self.assertEqual(names & forbidden, set(),
                             f"{mod.__name__} reaches into the emergency/abort machinery: "
                             f"{sorted(names & forbidden)}. A diversion is a mission phase.")

    def test_the_answer_kinds_match_the_planners_own_constants(self):
        """The strings are duplicated on purpose (wasp_bt must build with no perception package
        installed). Pinned here so the two cannot drift silently."""
        planner = (Path(__file__).resolve().parents[4] / "perception" / "sam" /
                   "sam_target_inspection" / "sam_target_inspection" / "inspection_planner.py")
        if not planner.exists():
            self.skipTest(f"planner not found at {planner}")
        tree = ast.parse(planner.read_text())
        theirs = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)):
                theirs[node.targets[0].id] = node.value.value
        for name in ("GOAL", "PHASE_DONE", "MISSION_DONE", "REFUSED", "WAIT", "SONAR_MODE",
                     "BURST"):
            self.assertIn(name, theirs, f"the planner no longer defines {name}")
            self.assertEqual(getattr(TIC, name), theirs[name],
                             f"the tree and the planner disagree about {name!r}")

    def test_the_subtree_is_inserted_AHEAD_of_the_mission_tree(self):
        """Order is the mechanism. Read off `ros_bt.py`'s own syntax: the append of the
        inspection subtree must come before the append of the mission tree in
        `_task_handler_tree`."""
        ros_bt = Path(__file__).resolve().parents[1] / "wasp_bt" / "bt" / "ros_bt.py"
        tree = ast.parse(ros_bt.read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_task_handler_tree")
        # BY LINE NUMBER, not by `ast.walk` order: walk is breadth-first and says nothing
        # about where in the source a call sits, which is the only thing this test is about.
        calls = [n for n in ast.walk(fn)
                 if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "append" and isinstance(n.func.value, ast.Name)
                     and n.func.value.id == "task_children")]
        order = [ast.unparse(n) for n in sorted(calls, key=lambda n: n.lineno)]
        insp = [i for i, o in enumerate(order) if "_target_inspection_tree" in o]
        miss = [i for i, o in enumerate(order) if "mission_tree" in o]
        self.assertTrue(insp, "the inspection subtree is not appended to task_children at all")
        self.assertTrue(miss, "the mission tree is no longer appended to task_children")
        self.assertLess(min(insp), min(miss),
                        "the inspection subtree is appended AFTER the mission tree, so it has "
                        "LOWER priority and can never preempt anything")
        # ... and the append must be REACHABLE. `if False:` leaves the call in the syntax tree
        # and satisfies every check above while building nothing, which is precisely the shape
        # of a feature that is present in the diff and absent from the vehicle (SETTLED §1c:
        # a structural test cannot see whether a branch runs — so make the branch itself the
        # thing that is checked).
        dead = [ast.unparse(n.test) for n in ast.walk(fn)
                if isinstance(n, ast.If) and isinstance(n.test, ast.Constant)
                and not n.test.value]
        self.assertEqual(dead, [], f"a branch in _task_handler_tree is dead-coded: {dead}")
        guard = next(n for n in ast.walk(fn)
                     if isinstance(n, ast.If)
                     and any("_target_inspection_tree" in ast.unparse(b) for b in n.body))
        self.assertIsInstance(guard.test, ast.Name,
                              "the inspection subtree is built under a constant condition")
        self.assertEqual(guard.test.id, "action_client_list",
                         "the subtree must be built exactly when there is a cached client to "
                         "stream through — one client, one server")

    def test_ros_bt_passes_the_inspection_the_cached_client_and_creates_no_new_one(self):
        ros_bt = Path(__file__).resolve().parents[1] / "wasp_bt" / "bt" / "ros_bt.py"
        tree = ast.parse(ros_bt.read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_target_inspection_tree")
        self.assertNotIn("BTActionClient", _called_names(fn),
                         "the inspection subtree builds its own action client — one client, "
                         "one server (SETTLED §1c)")


if __name__ == "__main__":
    unittest.main()
