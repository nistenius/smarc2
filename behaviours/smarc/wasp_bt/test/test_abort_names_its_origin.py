"""EVERY ABORT NAMES WHERE IT CAME FROM, AND EMERGENCY-PARKED IS NOT IDLE.  (#29, 2026-08-15)

Two defects, one root shape: the vehicle reported a state it had not observed.

  1. ABORT ORIGIN MASQUERADE. `WaraPSTaskHandler.abort()` -- whose only in-process caller is the
     tree's own health fallback, `A_Abort` -- called `_bigredbutton_cb(String("Big Red Button
     pressed"))`. So a vehicle-internal health abort logged as "from MQTT/C2" and published a
     WARA-PS tst/response whose `response-to` was a big red button nobody had pressed. The
     operator's own record then said a HUMAN aborted a mission the VEHICLE aborted on itself,
     which points the next investigation at exactly the wrong place: someone goes looking for an
     operator action instead of a health fault. Same family as SETTLED §1's "never put a guess in
     a status string" -- except here the guess was baked into a protocol message and travelled.

  2. `A_Chilling` MEANT TWO OPPOSITE THINGS. It is instantiated as the last child of
     F_Task_Handler (healthy, resting, no mission -- what the mission gate wants to see) AND
     inside F_HandleEmergency, which is reached only when C_NoEmergencyAbortSignalDetected has
     already FAILED (the vehicle is emergency parked). Both reported the identical tip,
     `A_Chilling (Status.RUNNING)`, and identical "Just chillin'..." feedback. Mission Control and
     Vehicle Control both had to render that tip as AMBIGUOUS, because the vehicle would not say
     which of the two it was.

Runs without a ROS graph: rclpy and the message packages are stubbed, exactly as
test_start_tst_replaces_queue.py does -- the logic under test is bookkeeping and strings.

    python3 -m pytest test/test_abort_names_its_origin.py
"""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _PermissiveModule(types.ModuleType):
    """A stub module that answers any attribute with a fresh MagicMock.

    `__path__` is set so it counts as a PACKAGE: without it, `import rclpy.type_support` fails
    with "rclpy is not a package" even though rclpy itself is stubbed.
    """
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
    """Auto-stub any submodule under a ROS-build-only namespace.

    Naming every module one at a time was whack-a-mole three rounds deep (rclpy.type_support,
    smarc_mission_msgs, rclpy.action, ...) and would break again on any unrelated smarc2 import.
    A test that breaks on unrelated changes gets deleted, which costs more than it saves. The
    namespace list is explicit and short, so nothing real is ever accidentally stubbed -- in
    particular `wasp_bt` itself is NOT in it, and must not be: the code under test is the point.
    """
    ROOTS = ("rclpy", "smarc_msgs", "smarc_mission_msgs", "smarc_action_base",
             "geometry_msgs", "std_msgs", "std_srvs", "nav_msgs", "sensor_msgs",
             "builtin_interfaces", "action_msgs", "unique_identifier_msgs")

    def find_module(self, fullname, path=None):   # py2-style hook, still honoured via find_spec
        return None

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


def _install_stubs():
    sys.meta_path.insert(0, _RosNamespaceFinder())
    # Roots first, and as PACKAGES. A plain module named `rclpy` in sys.modules makes
    # `import rclpy.task` fail before the meta-path finder is ever consulted -- the machinery
    # resolves the parent, finds no __path__, and gives up. Then override the specific leaves
    # this test needs to be real (Node must be a class: things subclass it).
    for _root in _RosNamespaceFinder.ROOTS:
        _permissive(_root)
    _stub("rclpy.node", Node=object)
    _stub("std_msgs.msg", String=_Msg, Int8=_Msg, Empty=_Msg, Float32=_Msg)
    _stub("std_srvs.srv", Trigger=MagicMock())

    class Topics:
        VEHICLE_HEALTH_READY = 0
        VEHICLE_HEALTH_WAITING = 1
        VEHICLE_HEALTH_ERROR = 2

    class _TopicsMeta(type):
        def __getattr__(cls, name):
            return name.lower()

    Topics = _TopicsMeta("Topics", (), dict(Topics.__dict__))
    _stub("smarc_msgs.msg", Topics=Topics)

    class SensorNames:
        pass
    _stub("wasp_bt.vehicles")
    _stub("wasp_bt.vehicles.sensor", Sensor=object, SensorNames=SensorNames)
    # Stubbing the `wasp_bt.vehicles` PACKAGE shadows its real submodules, so anything importing
    # `..vehicles.vehicle` (bt.actions does, transitively) fails with "not a package". The task
    # handler tests never noticed because they never import the tree. Stub the leaf too.
    _stub("wasp_bt.vehicles.vehicle",
          IVehicleStateContainer=object, IVehicleState=object, IVehicle=object)

    # bt.actions reaches into packages that only exist after a colcon build (the action client,
    # generated action types). None of A_Chilling touches them, but they are imported at module
    # scope, so they must exist for the import to succeed at all.
    _stub("smarc_mission_msgs.msg", Topics=Topics)
    # Everything else under those namespaces is handled by _RosNamespaceFinder above.
    return Topics


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TOPICS = _install_stubs()
from wasp_bt.waraps.waraps_task_handler import WaraPSTaskHandler  # noqa: E402

# IMPORT WHILE OUR STUBS ARE INSTALLED, NOT LAZILY IN setUp(). pytest imports every test module
# in the directory before running anything, and test_start_tst_replaces_queue.py installs its own
# `rclpy` stub over ours at ITS import time -- a plain module, not a package. A later
# `import wasp_bt.bt.actions` inside a test then failed with "rclpy is not a package", and only
# when the two files were run together: each passed alone. Two test files that stub the same
# modules interact through sys.modules, so take what you need while your own stubs are up.
try:
    from wasp_bt.bt.actions import A_Chilling  # noqa: E402
except ImportError as _e:                      # py_trees absent -> the A_Chilling cases skip
    A_Chilling = None
    _A_CHILLING_IMPORT_ERROR = _e


def _leg(desc):
    return {"name": "auv-depth-move-to", "description": desc, "task-uuid": f"uuid-{desc}",
            "params": {"waypoint": {"latitude": 59.32, "longitude": 18.10}}}


def _start_tst(*legs):
    return {"command": "start-tst", "com-uuid": "com-1",
            "tst": {"name": "seq", "tst-uuid": "tree-1", "params": {"timeout": 300.0},
                    "common-params": {}, "children": list(legs)}}


class _HandlerCase(unittest.TestCase):
    def setUp(self):
        self.node = MagicMock()
        self.h = WaraPSTaskHandler(self.node, {
            "name": "sam_auv_v1", "agent-uuid": "a-1",
            "levels": ["sensor", "direct_execution"], "pulse_rate": 1.0,
        })
        self.h.tasks_available = [{"name": "auv-depth-move-to"}]
        self.h.health_status = TOPICS.VEHICLE_HEALTH_READY
        self.h.emergency_flag = False
        self.h._send_tst_response = MagicMock()
        self.h._wara_ps_tst_response_pub = MagicMock()

    def _published_responses(self):
        return [json.loads(c.args[0].data)
                for c in self.h._wara_ps_tst_response_pub.publish.call_args_list]

    def _logged(self):
        lg = self.node.get_logger.return_value
        return "\n".join(str(c.args[0]) for c in
                         list(lg.warn.call_args_list) + list(lg.info.call_args_list) if c.args)


class TestAbortNamesItsOrigin(_HandlerCase):

    def test_the_trees_own_health_abort_is_not_reported_as_an_operator(self):
        """THE BUG. A_Abort -> abort() used to say 'Big Red Button pressed', from MQTT/C2."""
        self.h.abort()
        self.assertEqual(self.h.last_abort_origin, WaraPSTaskHandler.ABORT_ORIGIN_BT_HEALTH)
        log = self._logged()
        self.assertNotIn("Big Red Button pressed", log,
                         "an internal health abort is still masquerading as the operator")
        self.assertNotIn("MQTT/C2", log,
                         "the vehicle's own abort is being attributed to command and control")

    def test_an_internal_abort_never_claims_to_answer_a_message(self):
        """`response-to` is a reply. There was nothing to reply to."""
        self.h.abort()
        published = self._published_responses()
        self.assertTrue(published, "the internal abort went silent -- it must still be reported")
        self.assertNotIn("response-to", published[0],
                         "claims to be replying to a WARA-PS request that was never sent")
        self.assertEqual(published[0].get("abort-origin"),
                         WaraPSTaskHandler.ABORT_ORIGIN_BT_HEALTH)

    def test_an_internal_abort_is_not_quieter_than_it_used_to_be(self):
        """Regression guard on the FIX, not the bug: removing the lie must not remove the report.

        The obvious way to fix a false `response-to` is to stop publishing. That would hide every
        health abort from the operator -- strictly worse than mislabelling one.
        """
        self.h.abort()
        self.assertEqual(len(self._published_responses()), 1)

    def test_an_operator_abort_still_reports_as_an_operator_abort(self):
        self.h._bigredbutton_cb(_Msg(data="Big Red Button pressed"))
        self.assertEqual(self.h.last_abort_origin, WaraPSTaskHandler.ABORT_ORIGIN_OPERATOR_C2)
        pub = self._published_responses()[0]
        self.assertEqual(pub["response-to"], "Big Red Button pressed",
                         "a real request must still be answered -- WARA-PS contract")
        self.assertEqual(pub["abort-origin"], WaraPSTaskHandler.ABORT_ORIGIN_OPERATOR_C2)

    def test_a_vehicle_stack_abort_reports_as_the_vehicle_stack(self):
        """smarc/abort: the obstacle detector and health checker come in here."""
        self.h._emptybigredbutton_cb(_Msg())
        self.assertEqual(self.h.last_abort_origin, WaraPSTaskHandler.ABORT_ORIGIN_VEHICLE_STACK)

    def test_the_three_origins_are_distinguishable(self):
        """The whole point: three causes, three answers, no two the same."""
        seen = []
        for fire in (lambda: self.h._bigredbutton_cb(_Msg(data="x")),
                     lambda: self.h._emptybigredbutton_cb(_Msg()),
                     lambda: self.h.abort()):
            fire()
            seen.append(self.h.last_abort_origin)
        self.assertEqual(len(set(seen)), 3, f"origins collapsed together: {seen}")

    def test_every_abort_still_actually_aborts(self):
        """Naming the origin must not have cost the aborting."""
        for fire in (lambda: self.h._bigredbutton_cb(_Msg(data="x")),
                     lambda: self.h._emptybigredbutton_cb(_Msg()),
                     lambda: self.h.abort()):
            self.setUp()
            self.h._handle_tst_command(_start_tst(_leg("1"), _leg("2")))
            self.assertEqual(len(self.h.tasks_executing), 2)
            fire()
            self.assertTrue(self.h.emergency_flag)
            self.assertEqual(self.h.tasks_executing, [])
            self.assertEqual([t["status"] for t in self.h.past_tasks], ["aborted", "aborted"])

    def test_clearing_the_emergency_names_the_cause_it_cleared(self):
        """Spec invariant 4b: one manual clear, cause named."""
        self.h.abort()
        resp = self.h._reset_emergency_cb(MagicMock(), MagicMock())
        self.assertFalse(self.h.emergency_flag)
        self.assertIn(WaraPSTaskHandler.ABORT_ORIGIN_BT_HEALTH, resp.message)

    def test_an_unrecorded_cause_is_reported_as_unrecorded_never_guessed(self):
        """Invariant 4b again: 'not recorded' is an answer; inventing one is not."""
        self.assertIsNone(self.h.last_abort_origin)
        resp = self.h._reset_emergency_cb(MagicMock(), MagicMock())
        self.assertNotIn("bt_health", resp.message)
        self.assertNotIn("operator_c2", resp.message)


class TestEmergencyParkedIsNotIdle(unittest.TestCase):
    """A_Chilling's two roles. py_trees is the only extra dependency; skip if absent."""

    def setUp(self):
        if A_Chilling is None:
            self.skipTest(f"wasp_bt.bt.actions unavailable: {_A_CHILLING_IMPORT_ERROR}")
        self.A_Chilling = A_Chilling
        self.handler = MagicMock()
        self.handler.mission_status = None
        self.handler.last_abort_origin = None
        self.handler.last_abort_detail = None
        self.bt = MagicMock()
        self.bt._task_handler = self.handler

    def _node(self, role):
        return self.A_Chilling(self.bt, role)

    def test_the_two_roles_do_not_report_the_same_name(self):
        """`tip` is built from the name, and it was identical for both -- so the tip said nothing.

        This is the assertion the GUIs needed and could not have: on a live vehicle the only
        evidence available was the tip string, and tree topology is not visible from shore.
        """
        idle = self._node(self.A_Chilling.ROLE_IDLE)
        parked = self._node(self.A_Chilling.ROLE_EMERGENCY_PARKED)
        self.assertNotEqual(idle.name, parked.name)
        self.assertEqual(idle.name, "A_Chilling",
                         "the idle name must not change -- every existing reader keys on it")

    def test_an_emergency_parked_vehicle_does_not_say_it_is_chilling(self):
        parked = self._node(self.A_Chilling.ROLE_EMERGENCY_PARKED)
        parked.update()
        self.assertNotIn("chillin", parked.feedback_message.lower())
        self.assertIn("EMERGENCY PARKED", parked.feedback_message)

    def test_an_emergency_parked_vehicle_says_why_when_it_knows(self):
        self.handler.last_abort_origin = "bt_health"
        self.handler.last_abort_detail = "health checks failed"
        parked = self._node(self.A_Chilling.ROLE_EMERGENCY_PARKED)
        parked.update()
        self.assertIn("bt_health", parked.feedback_message)
        self.assertIn("health checks failed", parked.feedback_message)

    def test_an_emergency_parked_vehicle_never_invents_a_cause(self):
        parked = self._node(self.A_Chilling.ROLE_EMERGENCY_PARKED)
        parked.update()
        self.assertIn("not recorded", parked.feedback_message)

    def test_idle_is_unchanged(self):
        """The resting state is what the mission gate looks for. Do not disturb it."""
        idle = self._node(self.A_Chilling.ROLE_IDLE)
        status = idle.update()
        self.assertIn("chillin", idle.feedback_message.lower())
        self.assertEqual(status.name, "RUNNING")

    def test_the_default_role_is_idle(self):
        """An un-migrated call site must get the harmless meaning, not the alarming one."""
        self.assertEqual(self.A_Chilling(self.bt).role, self.A_Chilling.ROLE_IDLE)


class TestTheEmergencyTreeUsesTheParkedRole(unittest.TestCase):
    """Wiring, asserted on the source of ros_bt.py.

    Deliberately labelled as a supplement, not a guard: SETTLED §1c records a structural test
    that passed while the branch it described was mutated to `if False`. Constructing the real
    tree needs a live ROS node, so what this can honestly check is that no A_Chilling is left
    in F_HandleEmergency without a role -- and it says so rather than implying more.
    """

    def test_no_emergency_chilling_is_left_unroled(self):
        src = (Path(__file__).resolve().parents[1] / "wasp_bt" / "bt" / "ros_bt.py").read_text()
        start = src.index("def _handle_emergency_tree")
        end = src.index("def _one_task_tree", start)
        block = src[start:end]
        self.assertIn("A_Chilling(self, A_Chilling.ROLE_EMERGENCY_PARKED)", block)
        self.assertNotIn("A_Chilling(self)", block,
                         "an emergency-tree A_Chilling still reports itself as ordinary idle")


if __name__ == "__main__":
    unittest.main()
