"""A start-tst REPLACES the mission queue. Regression test for the dry-dock wall strike.

2026-08-13: an earlier mission left legs 2/3/4 in `tasks_executing`. The next mission was
`extend`ed onto the end of them, so the vehicle flew the OLD waypoints first -- they
crossed land -- and drove into the dry-dock wall with the props still turning. From the
operator's seat this presented as "the dead reckoning is wrong", which is the most
expensive possible way to report a queue that was never emptied.

Runs without a ROS graph: rclpy and the message packages are stubbed, because the logic
under test is list handling and none of it needs a middleware.
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


class _Msg:
    def __init__(self, *a, **kw):
        self.data = None


def _install_stubs():
    _stub("rclpy")
    _stub("rclpy.node", Node=object)
    _stub("std_msgs")
    _stub("std_msgs.msg", String=_Msg, Int8=_Msg, Empty=_Msg)
    _stub("std_srvs")
    _stub("std_srvs.srv", Trigger=MagicMock())

    class Topics:
        VEHICLE_HEALTH_READY = 0
        VEHICLE_HEALTH_WAITING = 1
        VEHICLE_HEALTH_ERROR = 2

        # Every other attribute on the real Topics is a topic-name constant. Returning a
        # plausible string for any of them keeps this stub honest without pinning the test
        # to a list that smarc2 is free to grow.
        def __class_getitem__(cls, item):
            return str(item)

    class _TopicsMeta(type):
        def __getattr__(cls, name):
            return name.lower()

    Topics = _TopicsMeta("Topics", (), dict(Topics.__dict__))
    _stub("smarc_msgs")
    _stub("smarc_msgs.msg", Topics=Topics)

    class SensorNames:
        pass
    # Only the SUBMODULE is stubbed. Stubbing `wasp_bt` itself would shadow the real
    # package and the module under test could not be imported at all.
    _stub("wasp_bt.vehicles")
    _stub("wasp_bt.vehicles.sensor", Sensor=object, SensorNames=SensorNames)
    return Topics


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TOPICS = _install_stubs()
from wasp_bt.waraps.waraps_task_handler import WaraPSTaskHandler  # noqa: E402


def _leg(desc, lat, lon):
    return {"name": "auv-depth-move-to", "description": desc,
            "task-uuid": f"uuid-{desc}",
            "params": {"waypoint": {"latitude": lat, "longitude": lon}}}


def _start_tst(*legs):
    return {"command": "start-tst", "com-uuid": f"com-{legs[0]['description']}",
            "tst": {"name": "seq", "tst-uuid": "tree-1", "params": {"timeout": 300.0},
                    "common-params": {}, "children": list(legs)}}


class TestStartTstReplacesQueue(unittest.TestCase):
    def setUp(self):
        node = MagicMock()
        self.h = WaraPSTaskHandler(node, {
            "name": "sam_auv_v1", "agent-uuid": "a-1",
            "levels": ["sensor", "direct_execution"], "pulse_rate": 1.0,
        })
        # The handler only accepts tasks it advertises, and it must be healthy to accept
        # a mission at all -- both are separate gates, deliberately.
        self.h.tasks_available = [{"name": "auv-depth-move-to"}]
        self.h.health_status = TOPICS.VEHICLE_HEALTH_READY
        self.h.emergency_flag = False
        self.h._send_tst_response = MagicMock()

    def _descs(self):
        return [t["description"] for t in self.h.tasks_executing]

    def test_a_second_mission_replaces_the_first(self):
        self.h._handle_tst_command(_start_tst(_leg("1", 59.32, 18.10),
                                              _leg("2", 59.33, 18.11)))
        self.assertEqual(self._descs(), ["1", "2"])

        self.h._handle_tst_command(_start_tst(_leg("A", 59.30, 18.09)))
        self.assertEqual(self._descs(), ["A"],
                         "the new mission was appended behind the old one — this is the "
                         "dry-dock wall strike")

    def test_the_waypoints_flown_are_the_new_ones(self):
        """The failure that mattered: it flew the PREVIOUS mission's coordinates."""
        self.h._handle_tst_command(_start_tst(_leg("old", 59.9999, 18.9999)))
        self.h._handle_tst_command(_start_tst(_leg("new", 59.3206, 18.1013)))
        wp = self.h.tasks_executing[0]["task"]["params"]["waypoint"]
        self.assertAlmostEqual(wp["latitude"], 59.3206)
        self.assertAlmostEqual(wp["longitude"], 18.1013)

    def test_replaced_legs_are_kept_as_history_not_dropped(self):
        """A leg that was abandoned still happened. Debrief needs to see it."""
        self.h._handle_tst_command(_start_tst(_leg("1", 59.32, 18.10)))
        self.h._handle_tst_command(_start_tst(_leg("A", 59.30, 18.09)))
        self.assertEqual([t["description"] for t in self.h.past_tasks], ["1"])
        self.assertEqual(self.h.past_tasks[0]["status"], "aborted")

    def test_a_REJECTED_start_does_not_wipe_a_running_mission(self):
        """The clear sits after every validation, so a bad upload is inert.

        Wiping the queue on a malformed command would turn a typo into an abandoned
        vehicle -- strictly worse than the bug being fixed.
        """
        self.h._handle_tst_command(_start_tst(_leg("1", 59.32, 18.10)))
        self.h._handle_tst_command({"command": "start-tst", "com-uuid": "bad",
                                    "tst": {"name": "seq", "params": {},
                                            "common-params": {},
                                            "children": "not-a-list"}})
        self.assertEqual(self._descs(), ["1"])

    def test_an_unavailable_task_also_leaves_the_queue_alone(self):
        self.h._handle_tst_command(_start_tst(_leg("1", 59.32, 18.10)))
        bad = _leg("X", 59.30, 18.09)
        bad["name"] = "fly-to-the-moon"
        self.h._handle_tst_command(_start_tst(bad))
        self.assertEqual(self._descs(), ["1"])

    def test_the_emergency_flag_still_refuses_before_any_of_this(self):
        self.h._handle_tst_command(_start_tst(_leg("1", 59.32, 18.10)))
        self.h.emergency_flag = True
        self.h._handle_tst_command(_start_tst(_leg("A", 59.30, 18.09)))
        self.assertEqual(self._descs(), ["1"], "a refused start must change nothing")


if __name__ == "__main__":
    unittest.main()
