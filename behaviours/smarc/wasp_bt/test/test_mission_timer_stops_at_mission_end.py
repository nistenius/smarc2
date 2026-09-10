"""A mission that has ENDED stops being timed, whatever else is true.

WHAT THIS IS FOR (2026-08-19, measured on mission #37's bag -- SETTLED 3s4).

`ctrl/mission_timer` read

    elapsed_s 8001.4, limit_s 2995.0, remaining_s -5006.4, fraction 2.668, state: running

while the vehicle sat SURFACED AND IDLE, 6,856 s after the task queue emptied at +1145.3. So the
2026-08-18 fix (SETTLED 3p) replaced a wrong limit with a right one that nothing enforced and
nothing terminated -- the same shape as 3h: a number that reads like a gate and is not one.

THE MECHANISM, ENTIRELY FROM THAT ONE LINE. `state: running` with a non-None `limit_s` means
`mission_start_time` and `mission_timeout` were both set, so the first two clauses of

    if self.mission_start_time is not None and self.mission_timeout is not None \
            and self.emergency_flag is False:

held, and only the third can have been false. And it was: at +1145.1, the instant the vehicle
surfaced to end the mission, `smarc/vehicle_health` went 0 -> 2 with

    Fault detected: low altitude! Current altitude: 0.28, Min altitude: 0.5

-- a surfaced vehicle reading its own depth as bottom clearance, which is 2026-08-18's "altitude
meant two things" arriving through the health checker rather than through min_altitude. The flag
latches (only `_reset_emergency_cb` clears it), so from that moment the timer could neither fire
NOR STOP, because ENFORCEMENT AND RETIREMENT SHARED ONE GUARD.

Enforcement staying behind the emergency flag is arguable and is Ivan's call (a mission that has
already been aborted arguably needs no second abort). Retirement is not arguable: a mission that
has ended is not a mission that is running.

Run: python3 -m pytest test/test_mission_timer_stops_at_mission_end.py -q
"""
import ast
import textwrap
import types
from pathlib import Path

import pytest

SRC = (Path(__file__).resolve().parents[1] / "wasp_bt" / "waraps"
       / "waraps_task_handler.py").read_text()


def _method(name: str):
    """Bind the REAL method onto a bare class, without importing rclpy or the msg universe."""
    fn = next(n for n in ast.walk(ast.parse(SRC))
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(textwrap.dedent(ast.get_source_segment(SRC, fn)), f"<{name}>", "exec"),
         {"String": lambda: types.SimpleNamespace(data=None), "json": __import__("json")}, ns)
    return ns[name]


LVL3 = _method("lvl_3_heartbeat")


def _class_const(name: str) -> str:
    """The real value, read from the class -- never a copy typed into this file. The origin
    strings are what MC's readiness row shows an operator, and a test asserting its own copy of
    one would go green while the vehicle reported something else (SETTLED 3l's two registries)."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return node.value.value
    raise AssertionError(f"{name} is gone from waraps_task_handler.py")


ABORT_ORIGIN_MISSION_TIMEOUT = _class_const("ABORT_ORIGIN_MISSION_TIMEOUT")


class Fake:
    """Only what lvl_3_heartbeat touches. Everything it publishes is recorded, not sent."""

    def __init__(self, *, start, timeout, now, executing, emergency=False):
        self.mission_start_time = start
        self.mission_timeout = timeout
        self._now = now
        self.tasks_executing = list(executing)
        self.emergency_flag = emergency
        self.aborts = []
        self.ABORT_ORIGIN_MISSION_TIMEOUT = ABORT_ORIGIN_MISSION_TIMEOUT
        self.timer_published = 0
        # 2026-09-09, the adaptive close inspection: the level-3 heartbeat now also publishes
        # the phase word beside the timer. Recorded here, not sent, like everything else.
        self.phase_published = 0
        self._direct_execution_info_data = {}
        self._wara_ps_dict = {"agent-uuid": "u"}
        self._node = types.SimpleNamespace(
            get_logger=lambda: types.SimpleNamespace(
                info=lambda m: None, warn=lambda m: None, error=lambda m: None))
        self._wara_ps_tst_exec_info_pub = types.SimpleNamespace(publish=lambda m: None)
        self._wara_ps_tst_feedback_pub = types.SimpleNamespace(publish=lambda m: None)

    def current_time(self):
        return self._now

    def _publish_mission_timer(self):
        self.timer_published += 1

    def _publish_adaptive_phase(self):
        self.phase_published += 1

    def _apply_abort(self, origin, detail, respond=False, respond_to=None):
        self.aborts.append((origin, detail))
        self.emergency_flag = True

    lvl_3_heartbeat = LVL3


TASK = {"task-uuid": "t1", "description": "1", "status": "running"}


# ---- the state the fix must PASS on, run first (SETTLED 3s3's rule) --------------------------

def test_a_running_mission_inside_its_limit_is_left_alone():
    f = Fake(start=0.0, timeout=2995.0, now=1000.0, executing=[TASK])
    f.lvl_3_heartbeat(1000.0)
    assert f.mission_start_time == 0.0 and f.mission_timeout == 2995.0
    assert f.aborts == []


def test_a_mission_that_runs_out_of_time_is_still_aborted():
    """The 2026-08-18 behaviour, re-proven -- so a failure below means the RETIREMENT change."""
    f = Fake(start=0.0, timeout=2995.0, now=3005.0, executing=[TASK])
    f.lvl_3_heartbeat(3005.0)
    assert len(f.aborts) == 1
    assert f.aborts[0][0] == ABORT_ORIGIN_MISSION_TIMEOUT
    assert f.mission_start_time is None and f.mission_timeout is None, \
        "a timeout is an EVENT: having fired once it must not stay armed (2026-08-06)"


def test_a_completed_mission_retires_its_timers():
    f = Fake(start=0.0, timeout=2995.0, now=1145.3, executing=[])
    f.lvl_3_heartbeat(1145.3)
    assert f.mission_start_time is None and f.mission_timeout is None
    assert f.aborts == [], "an ordinary completion is not an abort"


# ---- mission #37 ----------------------------------------------------------------------------

def test_mission_37_the_measured_case():
    """Queue empty, and an emergency latched at the same instant by the surfacing health fault.
    Before today this returned with both timers still armed and the clock still counting."""
    f = Fake(start=0.0, timeout=2995.0, now=1145.3, executing=[], emergency=True)
    f.lvl_3_heartbeat(1145.3)
    assert f.mission_start_time is None, \
        "a finished mission must stop being timed even with the emergency flag up"
    assert f.mission_timeout is None


def test_the_clock_would_have_read_idle_instead_of_fraction_2_668():
    """End to end: retire the timers, then ask the clock what it says. This is the reading the
    operator actually gets, and the one #37 got wrong for nearly two hours."""
    import test_mission_timer as tmt          # the clock's own tests, for MISSION_TIMER_STATE
    f = Fake(start=0.0, timeout=2995.0, now=1145.3, executing=[], emergency=True)
    f.lvl_3_heartbeat(1145.3)
    clock = tmt.Fake(start=f.mission_start_time, timeout=f.mission_timeout, now=8001.4,
                     emergency=f.emergency_flag)
    s = clock.state()
    assert s["state"] == "idle"
    assert s["fraction"] is None


def test_an_emergency_mid_mission_still_suspends_enforcement_and_says_so():
    """The other half. With tasks still queued and the flag up, the limit is NOT enforced -- that
    is the existing design, and Ivan's to change. What must never happen again is a display
    showing a confident countdown against a limit that will not act."""
    import test_mission_timer as tmt
    f = Fake(start=0.0, timeout=2995.0, now=3005.0, executing=[TASK], emergency=True)
    f.lvl_3_heartbeat(3005.0)
    assert f.aborts == [], "enforcement is suspended under an emergency (today's design)"
    assert f.mission_start_time == 0.0, "a mission with tasks queued is not finished"
    s = tmt.Fake(start=0.0, timeout=2995.0, now=3005.0, emergency=True).state()
    assert s["emergency"] is True, \
        "a countdown that is not being enforced must say so on the wire"


def test_a_healthy_running_mission_reports_emergency_false():
    """PASS-FIRST for the new field: it must be a fact, not a constant."""
    import test_mission_timer as tmt
    assert tmt.Fake(start=0.0, timeout=2995.0, now=100.0).state()["emergency"] is False


def test_the_emergency_field_is_present_in_every_state():
    """A key that appears only sometimes forces every consumer to write `.get(...)` and guess
    what its absence means. Absent is not empty (SETTLED 3e)."""
    import test_mission_timer as tmt
    for kwargs in ({}, {"start": 0.0, "timeout": None, "now": 5.0},
                   {"start": 0.0, "timeout": 2995.0, "now": 5.0}):
        assert "emergency" in tmt.Fake(**kwargs).state(), kwargs


def test_retirement_is_not_inside_the_emergency_guard():
    """Structural, on CODE. The behavioural tests above pin what happens; this pins WHERE, so a
    later tidy-up cannot quietly move retirement back under the flag and pass everything else by
    keeping some other path that happens to clear the timers."""
    fn = next(n for n in ast.walk(ast.parse(SRC))
              if isinstance(n, ast.FunctionDef) and n.name == "lvl_3_heartbeat")
    guarded, unguarded = [], []
    for node in fn.body:
        clears = [d for d in ast.walk(node)
                  if isinstance(d, ast.Assign)
                  and any(isinstance(t, ast.Attribute) and t.attr == "mission_start_time"
                          for t in d.targets)
                  and isinstance(d.value, ast.Constant) and d.value.value is None]
        if not clears:
            continue
        mentions_emergency = any(
            isinstance(n2, ast.Attribute) and n2.attr == "emergency_flag"
            for n2 in ast.walk(node.test)) if isinstance(node, ast.If) else False
        (guarded if mentions_emergency else unguarded).append(node)
    assert unguarded, ("nothing retires the mission timers outside the emergency guard -- "
                       "mission #37's clock ran to fraction 2.668 for exactly this reason")
