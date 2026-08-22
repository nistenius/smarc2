"""THE `auv-farm-inspection` TASK, AND THE BT HALF OF INVARIANT 5b.  (P5, 2026-08-17)

Three things are guarded here, and each one is a defect this project has already paid for:

  1. **ONE ACTION CLIENT.** The inspection streams many goals through the *existing*
     `auv_depth_move_to` client. Two servers on that action name is what stopped every
     mission at waypoint 1 for a week (SETTLED §1c), so the task is registered as available
     only because its PROVIDER's heartbeat arrived, carrying the provider's own `ros_name`.

  2. **A REFUSED SURVEY IS NOT A COMPLETED ONE.** Every refusal — the planner going quiet,
     an answer that is not a waypoint, the planner's own named refusals — ends the task as a
     FAILURE with the reason and the response carried out to the operator. A stopwatch may
     not certify a flight (SETTLED §3e) and neither may a plan that never ran.

  3. **A HAND-OFF ON A TIMEOUT IS NOT A CONFIRMED SURFACING.** `neutral_handoff` releases the
     actuators either way and says which; `A_SurfaceAndReport` keeps the two apart all the
     way out to the mission record.

Runs with no ROS graph and no py_trees: the logic under test is in `farm_inspection_core.py`,
which imports neither. The task-handler cases stub rclpy exactly as
`test_abort_names_its_origin.py` does — the same machinery, deliberately duplicated rather
than imported, because two test modules that share stubs interact through `sys.modules` and
that has already cost this project a debugging round.

    python3 -m pytest test/test_farm_inspection_task.py
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wasp_bt.bt.farm_inspection_core import (  # noqa: E402
    FAILURE, GOAL, MISSION_DONE, PHASE_DONE, REFUSED, RUNNING, SEND_GOAL, SUCCESS, WAIT,
    FarmInspectionRunner, SurfaceAndReportCore)


# ------------------------------------------------------------------ test doubles
class FakeLink:
    """The planner, as far as the runner can tell. Scripted answers, in order."""

    def __init__(self, answers=None, echo_seq=True):
        self.answers = list(answers or [])
        self.requests = []
        self._latest = None
        self.echo_seq = echo_seq
        self.silent = False

    def request(self, payload):
        self.requests.append(payload)
        if self.silent or not self.answers:
            return
        ans = dict(self.answers.pop(0))
        if self.echo_seq:
            ans["seq"] = payload["seq"]
        self._latest = ans

    def latest(self):
        return self._latest


class FakeClient:
    """The one action client. Records what was sent and how many times."""

    def __init__(self, outcome="done", ticks_to_finish=1):
        self.sent = []
        self.outcome = outcome
        self.ticks_to_finish = ticks_to_finish
        self._n = 0
        self._busy = False

    def send(self, params):
        self.sent.append(params)
        self._busy = True
        self._n = 0

    def state(self):
        if not self._busy:
            return "idle"
        self._n += 1
        if self._n >= self.ticks_to_finish:
            self._busy = False
            return self.outcome
        return "running"


def _wp(name="T1_standoff", lat=58.25, lon=11.45, depth=0.0):
    return {"waypoint": {"latitude": lat, "longitude": lon, "target_depth": depth,
                         "rpm": 500.0, "speed": 0.7, "tolerance": 3.0},
            "name": name, "timeout": 900.0}


def _goal(name="T1_standoff", phase="T1_approach"):
    return {"kind": GOAL, "phase": phase, "reason": f"why {name}", "params": _wp(name)}


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _drive(runner, client, max_ticks=200):
    """Tick the runner as the behaviour would: it sends what the core hands back."""
    steps = []
    for _ in range(max_ticks):
        step = runner.tick()
        steps.append(step)
        if step.action == SEND_GOAL:
            client.send(step.params)
        elif step.action in (SUCCESS, FAILURE):
            return steps
    raise AssertionError("runner did not terminate")


# ============================================================== the happy sequence
class TestTheInspectionStreamsThroughOneClient(unittest.TestCase):

    def test_every_sub_goal_goes_through_the_same_client_and_none_is_skipped(self):
        link = FakeLink([_goal("g1"), _goal("g2"), {"kind": PHASE_DONE, "phase": "T1_approach",
                                                    "reason": "T1 done"},
                         _goal("g3", "T2_encircle"),
                         {"kind": MISSION_DONE, "phase": "T4_return", "reason": "done"}])
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        steps = _drive(r, client)
        self.assertEqual([p["name"] for p in client.sent], ["g1", "g2", "g3"])
        self.assertEqual(steps[-1].action, SUCCESS)
        self.assertTrue(r.finished_ok)
        self.assertEqual(r.goals_sent, 3)

    def test_the_next_goal_is_only_asked_for_after_the_previous_one_is_reached(self):
        """A planner asked for the whole plan up front cannot adapt, and a tree that ticks
        faster than the vehicle flies would send every waypoint in one second."""
        link = FakeLink([_goal("g1"), _goal("g2"),
                         {"kind": MISSION_DONE, "phase": "T4_return", "reason": "done"}])
        client = FakeClient(ticks_to_finish=5)
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        _drive(r, client)
        reached = [q for q in link.requests if q.get("event") == "reached"]
        self.assertEqual([q["name"] for q in reached], ["g1", "g2"])

    def test_a_stale_answer_is_ignored_rather_than_flown(self):
        """An answer to the PREVIOUS question is not an answer. Acting on it is how a plan
        silently skips a leg."""
        link = FakeLink([_goal("g1")], echo_seq=False)
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock(),
                                 params={"planner_timeout_s": 5.0})
        clock = _Clock()
        r.now = clock
        for _ in range(3):
            r.tick()
        self.assertEqual(client.sent, [], "a mismatched-seq answer was flown")
        clock.t = 99.0
        r.tick()
        self.assertIsNotNone(r.refusal, "a planner that only ever answers stale must refuse")


# ============================================================== refusals
class TestRefusalsAreNamedAndFatalToTheTask(unittest.TestCase):

    def test_a_silent_planner_becomes_a_named_refusal_not_a_stalled_tree(self):
        link = FakeLink([])
        link.silent = True
        clock = _Clock()
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=clock,
                                 params={"planner_timeout_s": 20.0})
        self.assertEqual(r.tick().action, RUNNING)
        clock.t = 10.0
        self.assertEqual(r.tick().action, RUNNING)
        clock.t = 30.0
        step = r.tick()
        self.assertEqual(step.action, FAILURE)
        self.assertIn("did not answer", step.feedback)
        self.assertIn("farm planner node is running", step.detail)

    def test_the_planners_own_refusal_is_carried_out_verbatim_with_its_response(self):
        link = FakeLink([{"kind": REFUSED, "phase": "T2_encircle",
                          "reason": "the encircle finished and no usable farm fix arrived",
                          "response": "hold off the farm, surface and report"}])
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        steps = _drive(r, client)
        self.assertEqual(steps[-1].action, FAILURE)
        self.assertIn("no usable farm fix", steps[-1].feedback)
        self.assertIn("surface and report", steps[-1].detail)
        self.assertEqual(client.sent, [], "nothing should have been flown after a refusal")

    def test_an_answer_that_is_not_a_waypoint_is_refused_before_it_is_sent(self):
        """`ActionServerDiveSub.goal_callback` reads latitude/longitude/rpm/target_depth/
        tolerance with no default, so a missing key raises INSIDE the goal callback and
        rclpy turns that into a rejected goal with nothing on the wire to say why. Catching
        it here means the vehicle names the planner's defect instead."""
        bad = _wp()
        del bad["waypoint"]["tolerance"]
        link = FakeLink([{"kind": GOAL, "phase": "T1_approach", "reason": "x", "params": bad}])
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        steps = _drive(r, client)
        self.assertEqual(steps[-1].action, FAILURE)
        self.assertIn("not a waypoint", steps[-1].feedback)
        self.assertEqual(client.sent, [])

    def test_a_failed_move_ends_the_inspection_rather_than_scanning_from_somewhere_else(self):
        link = FakeLink([_goal("g1"), _goal("g2")])
        client = FakeClient(outcome="failed")
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        steps = _drive(r, client)
        self.assertEqual(steps[-1].action, FAILURE)
        self.assertIn("failed at the action client", steps[-1].feedback)
        self.assertIn("do not treat the partial scan as a survey", steps[-1].detail)

    def test_a_refusal_latches_and_the_task_never_reports_success_afterwards(self):
        link = FakeLink([{"kind": REFUSED, "phase": "T3_lanes", "reason": "no line fitted",
                          "response": "fly the encircle again"}])
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        _drive(r, client)
        link.answers = [{"kind": MISSION_DONE, "phase": "T4_return", "reason": "done"}]
        for _ in range(5):
            self.assertEqual(r.tick().action, FAILURE)

    def test_an_unknown_answer_kind_is_refused_and_not_ignored(self):
        link = FakeLink([{"kind": "carry_on", "phase": "T1_approach", "reason": "?"}])
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        steps = _drive(r, client)
        self.assertEqual(steps[-1].action, FAILURE)
        self.assertIn("unknown kind", steps[-1].feedback)

    def test_a_wait_answer_keeps_asking_and_does_not_fly_anything(self):
        link = FakeLink([{"kind": WAIT, "phase": "T2_encircle", "reason": "no map yet"},
                         {"kind": WAIT, "phase": "T2_encircle", "reason": "no map yet"},
                         _goal("g1", "T2_encircle"),
                         {"kind": MISSION_DONE, "phase": "T4_return", "reason": "done"}])
        client = FakeClient()
        r = FarmInspectionRunner(goal_state=client.state, link=link, now=_Clock())
        steps = _drive(r, client)
        self.assertEqual(steps[-1].action, SUCCESS)
        self.assertEqual([p["name"] for p in client.sent], ["g1"])


# ============================================================== A_SurfaceAndReport
class _Surface:
    """Handles for one SurfaceAndReportCore under test."""

    def __init__(self, pos=(58.25, 11.45), stop=False, handoff=None, outcome="running"):
        self.pos = pos
        self.stop = stop
        self.handoff = handoff
        self.client = FakeClient(outcome=outcome, ticks_to_finish=1)
        self.clock = _Clock()
        self.core = SurfaceAndReportCore(
            goal_state=self._state, position=lambda: self.pos,
            handoff=lambda: self.handoff, stop_active=lambda: self.stop,
            now=self.clock)

    def _state(self):
        return "running" if self.client._busy else "idle"

    def tick(self):
        step = self.core.tick()
        if step.action == SEND_GOAL:
            self.client.send(step.params)
        return step


class TestSurfaceAndReport(unittest.TestCase):

    def test_it_commands_a_depth_zero_waypoint_at_the_vehicles_own_position(self):
        s = _Surface()
        step = s.tick()
        self.assertEqual(step.action, SEND_GOAL)
        wp = s.client.sent[0]["waypoint"]
        self.assertEqual(wp["target_depth"], 0.0)
        self.assertEqual((wp["latitude"], wp["longitude"]), (58.25, 11.45))
        self.assertEqual(s.client.sent[0]["name"], "mission_complete_surface")

    def test_the_protective_stop_holds_the_surfacing_and_says_so(self):
        """Invariant 5b: surfacing under an overhang damages the hull. The proxy is a
        forward-looking sonar and the reason says so — nothing here claims an overhead
        clearance measurement this vehicle cannot make."""
        s = _Surface(stop=True)
        step = s.tick()
        self.assertEqual(step.action, FAILURE)
        self.assertEqual(s.client.sent, [], "the tank was blown with the stop active")
        self.assertIn("protective stop", step.feedback)
        self.assertIn("no overhead clearance sensor", step.feedback)

    def test_no_position_means_no_surfacing_command(self):
        """A waypoint at (0, 0) is a transit across the sea, not a surfacing."""
        s = _Surface(pos=None)
        step = s.tick()
        self.assertEqual(step.action, FAILURE)
        self.assertEqual(s.client.sent, [])
        self.assertIn("no position available", step.feedback)

    def test_only_a_confirmed_handoff_reports_success(self):
        s = _Surface()
        s.tick()
        s.handoff = {"released": True, "confirmed": True,
                     "reason": "surfaced at 0.04 m, VBS 1.2 %"}
        step = s.tick()
        self.assertEqual(step.action, SUCCESS)
        self.assertEqual(s.core.outcome, "confirmed")

    def test_a_timeout_release_is_reported_as_a_release_and_never_as_a_surfacing(self):
        """`neutral_handoff` lets go on a timeout deliberately — a controller that never
        lets go cannot be taken over — but that is not evidence the vehicle surfaced."""
        s = _Surface()
        s.tick()
        s.handoff = {"released": True, "confirmed": False,
                     "reason": "TIMEOUT after 600 ticks, depth unknown"}
        step = s.tick()
        self.assertEqual(step.action, FAILURE)
        self.assertEqual(s.core.outcome, "timeout")
        self.assertIn("TIMEOUT", step.feedback)

    def test_a_controller_that_never_speaks_is_reported_not_assumed(self):
        s = _Surface()
        s.tick()
        s.clock.t = 500.0
        step = s.tick()
        self.assertEqual(step.action, FAILURE)
        self.assertIn("never reported letting go", step.feedback)
        self.assertIn("NOT a confirmed surfacing", step.feedback)

    def test_holding_is_not_read_as_released(self):
        s = _Surface()
        s.tick()
        s.handoff = {"released": False, "confirmed": False, "reason": "VBS still at 38 %"}
        self.assertEqual(s.tick().action, RUNNING)

    def test_a_rejected_surfacing_goal_is_a_failure_with_its_reason(self):
        s = _Surface()
        s.tick()
        s.client._busy = False
        s.core.goal_state = lambda: "failed"
        step = s.tick()
        self.assertEqual(step.action, FAILURE)
        self.assertIn("rejected or failed", step.feedback)


# ============================================================== task registration
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


class _RosNamespaceLoader:
    def create_module(self, spec):
        return _PermissiveModule(spec.name)

    def exec_module(self, module):
        return None


class _RosNamespaceFinder:
    ROOTS = ("rclpy", "smarc_msgs", "smarc_mission_msgs", "smarc_action_base",
             "geometry_msgs", "std_msgs", "std_srvs", "nav_msgs", "sensor_msgs",
             "builtin_interfaces", "action_msgs", "unique_identifier_msgs")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root not in self.ROOTS or fullname in sys.modules:
            return None
        import importlib.machinery
        return importlib.machinery.ModuleSpec(fullname, _RosNamespaceLoader(), is_package=True)


class _Msg:
    def __init__(self, *a, **kw):
        self.data = kw.get("data", a[0] if a else None)


def _install_stubs():
    sys.meta_path.insert(0, _RosNamespaceFinder())
    for root in _RosNamespaceFinder.ROOTS:
        sys.modules.setdefault(root, _PermissiveModule(root))
    _stub("rclpy.node", Node=object)
    _stub("std_msgs.msg", String=_Msg, Int8=_Msg, Empty=_Msg)
    _stub("std_srvs.srv", Trigger=MagicMock())

    class _TopicsMeta(type):
        def __getattr__(cls, name):
            return name.lower()

    Topics = _TopicsMeta("Topics", (), {"VEHICLE_HEALTH_READY": 0,
                                        "VEHICLE_HEALTH_WAITING": 1,
                                        "VEHICLE_HEALTH_ERROR": 2})
    _stub("smarc_msgs.msg", Topics=Topics)
    _stub("smarc_mission_msgs.msg", Topics=Topics)
    _stub("wasp_bt.vehicles")
    _stub("wasp_bt.vehicles.sensor", Sensor=object, SensorNames=type("S", (), {}))
    _stub("wasp_bt.vehicles.vehicle", IVehicleStateContainer=object, IVehicleState=object,
          IVehicle=object)
    return Topics


_install_stubs()
from wasp_bt.waraps.waraps_task_handler import (  # noqa: E402
    BT_PROVIDED_TASKS, WaraPSTaskHandler)


class TestTheTaskIsAvailableOnlyBecauseItsProviderIs(unittest.TestCase):
    """`auv-farm-inspection` has no action server. It must appear in `tasks-available`
    exactly when the server it STREAMS THROUGH is heard from, carrying that server's own
    `ros_name` — which is what makes the tree hand it the cached client instead of building
    a second one (SETTLED §1c)."""

    def setUp(self):
        self.node = MagicMock()
        self.node.get_clock.return_value.now.return_value.to_msg.return_value.sec = 100
        self.node.get_clock.return_value.now.return_value.to_msg.return_value.nanosec = 0
        self.h = WaraPSTaskHandler(self.node, {
            "name": "sam_auv_v1", "agent-uuid": "a-1",
            "levels": ["sensor", "direct_execution"], "pulse_rate": 1.0})

    def _heartbeat(self, action="/sam_auv_v1/auv_depth_move_to"):
        self.h._action_hb_callback(_Msg(data=action))

    def test_it_is_absent_until_the_depth_move_to_server_is_heard_from(self):
        self.assertNotIn("auv-farm-inspection",
                         [t["name"] for t in self.h.tasks_available])

    def test_one_heartbeat_from_the_provider_is_enough(self):
        """The FIRST heartbeat, not the second: a stack that has just come up must offer the
        task straight away, and the callback used to `return` before reaching this."""
        self._heartbeat()
        names = [t["name"] for t in self.h.tasks_available]
        self.assertIn("auv-depth-move-to", names)
        self.assertIn("auv-farm-inspection", names)

    def test_it_carries_the_providers_own_ros_name_so_the_client_is_shared(self):
        self._heartbeat()
        by = {t["name"]: t for t in self.h.tasks_available}
        self.assertEqual(by["auv-farm-inspection"]["ros_name"],
                         by["auv-depth-move-to"]["ros_name"])
        self.assertEqual(by["auv-farm-inspection"]["provided_by"], "behaviour_tree")

    def test_it_is_not_duplicated_by_repeated_heartbeats(self):
        for _ in range(5):
            self._heartbeat()
        names = [t["name"] for t in self.h.tasks_available]
        self.assertEqual(names.count("auv-farm-inspection"), 1)

    def test_an_unrelated_action_server_does_not_conjure_it(self):
        self._heartbeat("/sam_auv_v1/auv_move_to")
        self.assertNotIn("auv-farm-inspection",
                         [t["name"] for t in self.h.tasks_available])

    def test_it_ages_out_with_its_provider(self):
        """The liveliness timeout drops it like any other task: if the server it needs is
        gone, the task it provides is gone. Consumer-side, not a flag."""
        self._heartbeat()
        self.h._direct_execution_info_data = {"stamp": 0, "tasks-available": [],
                                              "tasks-executing": []}
        self.h._wara_ps_direct_execution_info_pub = MagicMock()
        self.h._wara_ps_task_list_pub = MagicMock()
        self.h.lvl_2_heartbeat(100.0 + self.h._task_liveliness_timeout + 1.0)
        self.assertNotIn("auv-farm-inspection",
                         [t["name"] for t in self.h.tasks_available])

    def test_a_start_tst_naming_it_is_accepted_like_any_other_task(self):
        self._heartbeat()
        self.h.health_status = 0
        self.h.emergency_flag = False
        self.h._send_tst_response = MagicMock()
        self.h._handle_tst_command({
            "command": "start-tst", "com-uuid": "c1",
            "tst": {"name": "seq", "tst-uuid": "t1", "params": {"timeout": 3600.0},
                    "common-params": {},
                    "children": [{"name": "auv-farm-inspection", "task-uuid": "u1",
                                  "description": "farm", "params": {"rpm": 500.0}}]}})
        self.h._send_tst_response.assert_called_with("c1", "accepted")
        self.assertEqual(len(self.h.tasks_executing), 1)
        self.assertEqual(self.h.tasks_executing[0]["task"]["name"], "auv-farm-inspection")

    def test_the_provider_map_names_a_task_that_actually_exists(self):
        self.assertEqual(BT_PROVIDED_TASKS["auv-farm-inspection"], "auv-depth-move-to")


class TestTheAnswerVocabularyMatchesThePlanners(unittest.TestCase):
    """The runner duplicates the planner's answer-kind strings rather than importing them
    (wasp_bt must build on a hull with no perception package installed). Duplication is only
    safe if something checks it, so this reads the planner's own constants when they are
    importable and skips honestly when they are not."""

    def test_the_kinds_are_the_same_strings_the_planner_emits(self):
        root = Path(__file__).resolve().parents[4]
        pkg = root / "perception" / "sam" / "sam_farm_inspection"
        if not pkg.exists():
            self.skipTest(f"sam_farm_inspection not beside wasp_bt at {pkg}")
        sys.path.insert(0, str(pkg))
        from sam_farm_inspection import farm_mission as fm
        self.assertEqual((GOAL, PHASE_DONE, MISSION_DONE, REFUSED, WAIT),
                         (fm.GOAL, fm.PHASE_DONE, fm.MISSION_DONE, fm.REFUSED, fm.WAIT))


if __name__ == "__main__":
    unittest.main()
