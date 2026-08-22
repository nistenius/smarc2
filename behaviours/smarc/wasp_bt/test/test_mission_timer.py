"""The mission clock, as data.

WHY THIS EXISTS. `mission_start_time` and `mission_timeout` decide whether a mission lives or
dies, and until 2026-08-18 they were private attributes nothing outside could read. A 130 m
plan was therefore aborted at 301 s against a hidden 300 s limit, eight times over several
weeks, and read as a waypoint-acceptance bug, a control problem and an estimator problem in
turn. Ivan, right after the first mission that survived: "for next round of HUD it would be
useful with timer, total mission passed, countdown to end".

The tests below fix the properties a DISPLAY depends on, not the formatting:

  * three distinct states, because "no mission" and "a mission with no limit" are different
    facts and a countdown is wrong for both;
  * a negative remaining is reported, not clamped -- an overrun that has not yet been acted
    on is precisely the thing worth seeing;
  * a fraction, because seconds alone are not the reading that catches a doomed plan.

Run: python3 -m pytest test/test_mission_timer.py -q
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The handler drags in rclpy and the WARA-PS message universe at import time; none of it is
# touched by the clock. Stub only what import needs, then bind the real method onto a fake.
for name, attrs in (
    ("rclpy", {}), ("rclpy.node", {"Node": object}),
    ("std_msgs", {}), ("std_msgs.msg", {"String": object, "Empty": object, "Bool": object,
                                        "Int8": object, "Float32": object}),
    ("smarc_mission_msgs", {}), ("smarc_mission_msgs.msg", {"Topics": object,
                                                            "GotoWaypoint": object}),
):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)

try:
    from wasp_bt.waraps.waraps_task_handler import WaraPSTaskHandler as _H
    MISSION_TIMER_STATE = _H.mission_timer_state
    MISSION_PROGRESS = _H._mission_progress
except Exception:                                   # pragma: no cover
    # The class name or its imports moved. Fail loudly rather than silently skipping: a guard
    # that quietly stops running is the shape this whole register is made of.
    import re
    src = (Path(__file__).resolve().parents[1] / "wasp_bt" / "waraps"
           / "waraps_task_handler.py").read_text()
    assert "def mission_timer_state" in src, "mission_timer_state has gone missing entirely"
    ns = {}
    body = re.search(r"(    def _mission_progress.*?)\n    def _publish_mission_timer", src, re.S)
    exec("class _Shim:\n" + body.group(1), ns)
    MISSION_TIMER_STATE = ns["_Shim"].mission_timer_state
    MISSION_PROGRESS = ns["_Shim"]._mission_progress


class Fake:
    """Just the attributes the clock reads."""
    def __init__(self, start=None, timeout=None, now=0.0, wp_total=0, past=0, base=0,
                 emergency=False):
        self.mission_start_time = start
        self.mission_timeout = timeout
        # 2026-08-19: the clock now reports whether the limit it is showing is actually being
        # enforced. See test_mission_timer_stops_at_mission_end.py for what that cost.
        self.emergency_flag = emergency
        self._now = now
        self._mission_wp_total = wp_total
        self.past_tasks = [None] * past
        self._mission_past_base = base
        self._mission_wp_done_seen = 0
        self._mission_last_wp_at = start
        self._last_mission_summary = {"last_elapsed_s": None, "last_limit_s": None,
                                      "last_wp_done": None, "last_wp_total": None}

    def current_time(self):
        return self._now

    state = MISSION_TIMER_STATE
    _mission_progress = MISSION_PROGRESS


def test_no_mission_is_idle_not_a_zero_countdown():
    s = Fake().state()
    assert s["state"] == "idle"
    assert s["remaining_s"] is None and s["limit_s"] is None
    # A display that read 0 here would show a mission about to be aborted. It must not.
    assert s["elapsed_s"] is None


def test_a_mission_with_no_timeout_says_untimed_rather_than_faking_a_limit():
    s = Fake(start=0.0, timeout=None, now=42.0).state()
    assert s["state"] == "untimed"
    assert s["elapsed_s"] == 42.0
    assert s["remaining_s"] is None and s["fraction"] is None


def test_zero_timeout_is_also_untimed():
    """0 is the project's 'no limit' convention for per-waypoint timeouts; treat it the same."""
    assert Fake(start=0.0, timeout=0.0, now=5.0).state()["state"] == "untimed"


def test_a_running_mission_reports_all_four_numbers():
    s = Fake(start=100.0, timeout=1188.0, now=400.0).state()
    assert s["state"] == "running"
    assert s["elapsed_s"] == 300.0
    assert s["limit_s"] == 1188.0
    assert s["remaining_s"] == 888.0
    assert abs(s["fraction"] - 300.0 / 1188.0) < 1e-3


def test_the_run_that_kept_dying_would_have_been_visible():
    """The measured case: 301 s elapsed against the old 300 s limit."""
    s = Fake(start=0.0, timeout=300.0, now=301.0).state()
    assert s["remaining_s"] < 0
    assert s["fraction"] > 1.0


def test_an_overrun_is_reported_negative_not_clamped():
    s = Fake(start=0.0, timeout=100.0, now=140.0).state()
    assert s["remaining_s"] == -40.0, "clamping at zero hides an overrun nobody has acted on"


def test_elapsed_never_goes_negative_if_the_clock_steps_backwards():
    """Sim time can jump. A negative elapsed would render as a countdown running upwards."""
    s = Fake(start=500.0, timeout=100.0, now=490.0).state()
    assert s["elapsed_s"] == 0.0


# ------------------------------------------------------- waypoint progress and pace estimates

def test_progress_counts_only_THIS_mission_not_the_history():
    """`past_tasks` accumulates across runs. Measuring against its LENGTH would report the
    previous mission's waypoints as already flown; the baseline is where it stood at accept."""
    n = Fake(start=0.0, timeout=1000.0, now=100.0, wp_total=3, past=7, base=5)
    s = n.state()
    assert s["wp_total"] == 3 and s["wp_done"] == 2 and s["wp_current"] == 3


def test_no_estimates_before_the_first_waypoint_completes():
    """Zero completed legs is no evidence of pace. Estimating from it would print a number
    with nothing behind it -- the failure this whole day was made of."""
    s = Fake(start=0.0, timeout=1000.0, now=90.0, wp_total=4, past=0, base=0).state()
    assert s["wp_done"] == 0
    assert s["eta_finish_s"] is None and s["eta_next_s"] is None


def test_pace_estimate_extrapolates_from_completed_legs():
    # 2 of 4 legs in 200 s -> 100 s/leg -> ~200 s left
    s = Fake(start=0.0, timeout=1000.0, now=200.0, wp_total=4, past=2, base=0).state()
    assert s["eta_finish_s"] == 200.0


def test_the_next_waypoint_estimate_never_goes_negative():
    """A leg already running longer than the average is 'due', not 'overdue by a guess'. Only
    the TIMEOUT is allowed to report an overrun, because only it is a hard limit."""
    n = Fake(start=0.0, timeout=1000.0, now=200.0, wp_total=4, past=2, base=0)
    n.state()                       # marks the completion time at now=200
    n._now = 5000.0                 # a very long current leg
    assert n.state()["eta_next_s"] == 0.0


def test_a_mission_with_no_waypoint_count_still_reports_the_clock():
    """Progress is a bonus; the clock is the point. An unknown wp_total must not blank it."""
    s = Fake(start=0.0, timeout=600.0, now=60.0, wp_total=0).state()
    assert s["state"] == "running" and s["elapsed_s"] == 60.0
    assert s["wp_total"] is None and s["eta_finish_s"] is None


# ---------------------------------------------------------------- the idle row keeps its story

def test_idle_reports_the_previous_mission_rather_than_nothing():
    """Ivan, 2026-08-18: keep the row when idling, with the previous mission's data. A row that
    disappears between missions is indistinguishable from a feature that was never built."""
    n = Fake(start=0.0, timeout=1000.0, now=300.0, wp_total=3, past=3, base=0)
    n.state()                                   # fly it
    n.mission_start_time = None                 # ...and it ends
    s = n.state()
    assert s["state"] == "idle"
    assert s["last_elapsed_s"] == 300.0 and s["last_wp_done"] == 3
    assert s["last_wp_total"] == 3 and s["last_limit_s"] == 1000.0
    # but the LIVE fields stay empty: a finished mission is not a running one
    assert s["elapsed_s"] is None and s["remaining_s"] is None


def test_a_vehicle_that_has_never_flown_says_so_rather_than_inventing_a_last_run():
    s = Fake().state()
    assert s["state"] == "idle"
    assert s["last_elapsed_s"] is None and s["last_wp_total"] is None


def test_the_summary_is_flat_so_the_hud_parser_needs_no_nesting():
    """The Unity dashboard scrapes this JSON with a minimal parser on purpose (JsonUtility
    yields 0 for missing fields, which is how 'no data' becomes 'zero' on a display). A nested
    object would force nesting into that parser for one field."""
    s = Fake(start=0.0, timeout=100.0, now=10.0, wp_total=1, past=1, base=0).state()
    assert all(not isinstance(v, dict) for v in s.values()), "the payload must stay flat"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
