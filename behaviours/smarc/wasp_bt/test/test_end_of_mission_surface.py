"""Invariant 5b for the ORDINARY path: a plain mission must surface when it ends.

    python3 -m pytest test/test_end_of_mission_surface.py

WHY THIS EXISTS. `A_SurfaceAndReport` has surfaced the vehicle correctly since 2026-08-17 --
but only from inside the `auv-farm-inspection` subtree. Every ordinary waypoint mission ended
by falling through to `A_Chilling` with the diving controller still holding its last commanded
depth, indefinitely. SYSTEMS_SPEC invariant 5b has said this since 2026-08-14 and it stayed
open because the farm task made it *look* closed.

It is not only a recovery problem. `bridge_node`'s end-of-run rule requires *mission ended AND
SURFACED AND idle*, where "surfaced" is the diving controller's own `ctrl/neutral_handoff`
verdict, never a depth reading. A plain mission produces no verdict, so the 300 s grace never
starts, the recorder never stops, and the bag never gets a `metadata.yaml`. Ivan hit that twice
on 2026-08-29, and the hand workaround -- killing the recorder -- destroyed a flown mission,
because `bridge_node` owns that recorder and restarts it on the same path.

These tests are on the pure decision core, so they run without ROS, py_trees or a rig.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wasp_bt.bt.end_of_mission_core import EndOfMissionSurfaceCore, get_or_create


class _Handler:
    """Just enough task handler for get_or_create to hang state on."""


class TestEndOfMissionSurface(unittest.TestCase):

    # ---------------------------------------------------------------- the edge, not the level
    def test_a_vehicle_that_never_flew_does_not_surface(self):
        """THE SAFETY CASE. An empty queue is also true on power-up; blowing the tank then
        would be a surfacing nobody asked for."""
        c = EndOfMissionSurfaceCore()
        for _ in range(50):
            c.note_executing(0)
            self.assertFalse(c.should_surface())

    def test_a_mission_that_ran_and_stopped_surfaces_once(self):
        c = EndOfMissionSurfaceCore()
        for _ in range(5):
            c.note_executing(0)
        for _ in range(20):                      # mission running
            c.note_executing(3)
            self.assertFalse(c.should_surface())
        c.note_executing(0)                      # queue empties
        self.assertTrue(c.should_surface())

    def test_it_stays_pending_until_acted_on(self):
        """The tree may take many ticks to reach it; the request must not evaporate."""
        c = EndOfMissionSurfaceCore()
        c.note_executing(2)
        c.note_executing(0)
        for _ in range(100):
            c.note_executing(0)
            self.assertTrue(c.should_surface())

    def test_it_fires_once_per_mission_not_once_per_tick(self):
        c = EndOfMissionSurfaceCore()
        c.note_executing(2); c.note_executing(0)
        self.assertTrue(c.should_surface())
        c.note_surface_finished("confirmed")
        for _ in range(100):
            c.note_executing(0)
            self.assertFalse(c.should_surface(), "the tree would surface in a loop")

    def test_a_second_mission_re_arms(self):
        c = EndOfMissionSurfaceCore()
        c.note_executing(1); c.note_executing(0)
        c.note_surface_finished("confirmed")
        c.note_executing(4)                      # a new mission
        self.assertFalse(c.should_surface())
        c.note_executing(0)
        self.assertTrue(c.should_surface(), "the second mission owes a surfacing too")

    def test_a_new_mission_cancels_an_unserved_surfacing(self):
        """If the operator starts another mission before the tree got to it, the old
        recovery is moot -- the vehicle is going somewhere, not coming up."""
        c = EndOfMissionSurfaceCore()
        c.note_executing(2); c.note_executing(0)
        self.assertTrue(c.should_surface())
        c.note_executing(3)
        self.assertFalse(c.should_surface())

    # ------------------------------------------------------- every terminal outcome disarms
    def test_held_by_the_protective_stop_still_disarms(self):
        """A surfacing refused by the protective stop has been ATTEMPTED and REPORTED.
        Retrying every tick would loop the tree against a vehicle stopped on something."""
        for outcome in ("confirmed", "timeout", "held", "no_position", None):
            c = EndOfMissionSurfaceCore()
            c.note_executing(1); c.note_executing(0)
            c.note_surface_finished(outcome)
            self.assertFalse(c.should_surface(), f"{outcome} left it armed")
            self.assertEqual(c.last_outcome, outcome)

    def test_the_outcome_is_recorded_but_never_decides(self):
        """`timeout` is a release, not evidence of surfacing (SurfaceAndReportCore keeps the
        two apart). This core stores it for the report and treats all outcomes alike."""
        c = EndOfMissionSurfaceCore()
        c.note_executing(1); c.note_executing(0); c.note_surface_finished("timeout")
        self.assertEqual(c.last_outcome, "timeout")
        self.assertEqual(c.missions_surfaced, 1)

    # ------------------------------------------------------------- survives a tree rebuild
    def test_the_latch_lives_on_the_handler_so_a_rebuild_cannot_erase_it(self):
        """`ros_bt._update_task_handler_tree` rebuilds this subtree whenever an action
        server's heartbeat changes -- at arbitrary moments, including mid-mission. A flag on
        the behaviour would be lost and the vehicle would surface twice or not at all."""
        h = _Handler()
        a = get_or_create(h)
        a.note_executing(2)
        b = get_or_create(h)                     # as if the tree were rebuilt here
        self.assertIs(a, b, "a rebuild would get a fresh core and lose the mission")
        b.note_executing(0)
        self.assertTrue(get_or_create(h).should_surface())

    def test_two_vehicles_do_not_share_a_latch(self):
        h1, h2 = _Handler(), _Handler()
        get_or_create(h1).note_executing(2)
        get_or_create(h1).note_executing(0)
        self.assertTrue(get_or_create(h1).should_surface())
        self.assertFalse(get_or_create(h2).should_surface(),
                         "one vehicle's mission armed another's surfacing")

    # ------------------------------------------------------------------------ the reporting
    def test_it_says_which_state_it_is_in(self):
        c = EndOfMissionSurfaceCore()
        self.assertIn("no mission flown", c.describe())
        c.note_executing(2)
        self.assertIn("running", c.describe())
        c.note_executing(0)
        self.assertIn("owed", c.describe())
        c.note_surface_finished("confirmed")
        self.assertIn("confirmed", c.describe())


class TestWiring(unittest.TestCase):
    """Static checks on the tree wiring, since building a real tree needs ROS."""

    def _src(self, rel):
        with open(os.path.join(os.path.dirname(__file__), "..", "wasp_bt", rel)) as f:
            return f.read()

    def test_the_ordinary_path_has_a_surfacing_and_it_precedes_chilling(self):
        s = self._src("bt/ros_bt.py")
        self.assertIn("A_EndOfMissionSurface", s, "the ordinary path still never surfaces")
        i_surface = s.index("S_EndOfMissionSurface")
        i_chill = s.index("A_Chilling(self, A_Chilling.ROLE_IDLE)")
        self.assertLess(i_surface, i_chill,
                        "idle is reached before the surfacing, so it would never run")

    def test_it_is_gated_on_the_edge_condition(self):
        s = self._src("bt/ros_bt.py")
        block = s[s.index("S_EndOfMissionSurface"):s.index("A_Chilling(self, A_Chilling.ROLE_IDLE)")]
        self.assertIn("C_MissionJustEnded", block,
                      "ungated, this would surface a vehicle that never flew")

    def test_the_phases_are_a_memory_sequence(self):
        s = self._src("bt/ros_bt.py")
        block = s[s.index("S_EndOfMissionSurface"):s.index("A_Chilling(self, A_Chilling.ROLE_IDLE)")]
        self.assertIn("memory=True", block,
                      "without memory the condition is re-evaluated and the surfacing restarts")

    def test_the_surfacing_reuses_the_existing_action_client(self):
        """One client, one server, one writer -- invariant 12. A second client to the same
        action would be a second writer to the actuators."""
        s = self._src("bt/ros_bt.py")
        block = s[s.index("S_EndOfMissionSurface"):s.index("A_Chilling(self, A_Chilling.ROLE_IDLE)")]
        self.assertIn("surface_client", block)
        self.assertIn("action_client_list[0]", s)

    def test_it_subclasses_rather_than_reimplements_the_surfacing(self):
        s = self._src("bt/farm_inspection.py")
        self.assertIn("class A_EndOfMissionSurface(A_SurfaceAndReport)", s,
                      "a second implementation of 'how to surface' is a second thing to keep right")


if __name__ == "__main__":
    unittest.main()
