"""P8.2 — the mode switch is a PHASE TRANSITION with a named gate.

Ivan, 2026-08-19: once the farm has been encircled and the buoys, ropes and anchors identified
with their placement, the vehicle goes over to farm-relative navigation — following ropes and
planning paths from where the ropes actually are — and stops relying on DR-based pre-assigned
waypoints.

THE ONE PROPERTY EVERYTHING HERE PROTECTS: **a gate that cannot be satisfied falls back to
SURFACE-AND-REPORT, never silently to DR lanes.** That is not tidiness. DR lanes through a farm
whose position was never established are flown blind past ropes on a 2 m standoff budget, with
the report saying the survey happened — which is strictly more dangerous than not flying, and it
is the exact behaviour Ivan's requirement removes. SETTLED §3d's rule, at the phase level: an
unsatisfied mode is refused BY NAME, never downgraded to the nearest thing that still runs.

Run: PYTHONPYCACHEPREFIX=/tmp/pyc python3 -m pytest -p no:cacheprovider \
         test/test_farm_relative_gate.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam_farm_inspection.farm_mission import (            # noqa: E402
    ARMING_VERDICTS, DR_WAYPOINTS, FARM_RELATIVE, NAV_MODES, REFUSED, T3_LANES,
    FarmMap, FarmMissionPlanner, MissionConfig,
)
from sam_farm_inspection.farm_prior import load_farm_prior  # noqa: E402


@pytest.fixture(scope="module")
def prior():
    return load_farm_prior()


def _map(*, lines=2, seen=5, rms=0.3, ok=True):
    """A verified map with the properties the gate reads, and nothing else."""
    names = ["line_A", "line_B", "line_C"][:lines]
    buoy_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    verdicts = {n: ("confirmed" if i < seen else "missing")
                for i, n in enumerate(buoy_names)}
    return FarmMap(
        ok=ok, reason="fitted" if ok else "no fit",
        buoys_xz={n: (float(i), 0.0) for i, n in enumerate(buoy_names)},
        verdicts=verdicts,
        lines_xz={n: ((0.0, float(i)), (10.0, float(i))) for i, n in enumerate(names)},
        line_refusals={} if lines >= 2 else {"line_B": "too few detections"},
        rms_m=rms, source="test")


def _planner(prior, m):
    p = FarmMissionPlanner(prior)
    p.map = m
    return p


# ---- the state the gate must PASS on, run first (SETTLED §3s3's rule) -------------------------

def test_a_good_encircle_arms_farm_relative(prior):
    p = _planner(prior, _map())
    ok, why = p.farm_relative_gate()
    assert ok, why
    ans = p.arm_farm_relative()
    assert ans.kind != REFUSED
    assert p.nav_mode == FARM_RELATIVE


def test_the_gate_says_WHY_even_when_it_passes(prior):
    """A gate that only speaks when it refuses leaves the operator unable to tell *armed* from
    *nobody asked*. This sentence is what MC's T3 row shows and what a debrief reads back."""
    ok, why = _planner(prior, _map(lines=2, seen=5, rms=0.31)).farm_relative_gate()
    assert ok
    assert "2 line(s) fitted" in why and "5 buoy(s) seen" in why and "0.31" in why


def test_arming_is_idempotent(prior):
    p = _planner(prior, _map())
    p.arm_farm_relative()
    p.arm_farm_relative()
    assert p.nav_mode == FARM_RELATIVE


# ---- every way it must refuse, and each one names itself ---------------------------------------

def test_no_verified_map_at_all_cannot_arm(prior):
    p = FarmMissionPlanner(prior)                 # map is None: the encircle never reported
    ok, why = p.farm_relative_gate()
    assert not ok and "no verified map" in why


def test_a_map_that_did_not_fit_cannot_arm(prior):
    ok, why = _planner(prior, _map(ok=False)).farm_relative_gate()
    assert not ok and "no verified map" in why


def test_one_fitted_line_is_not_enough_and_the_refusal_names_the_other(prior):
    """A corridor lane derived from one measured line and one PRIOR line is half a measurement,
    and half a measurement presented as farm-relative is worse than an honest DR lane."""
    ok, why = _planner(prior, _map(lines=1)).farm_relative_gate()
    assert not ok
    assert "only 1 culture line(s) fitted" in why
    assert "line_A" in why, "name what DID fit"
    assert "line_B" in why, "name what did not — that is where the vehicle must look again"


def test_too_few_buoys_seen_cannot_arm_and_the_verdicts_are_listed(prior):
    ok, why = _planner(prior, _map(seen=2)).farm_relative_gate()
    assert not ok
    assert "only 2 buoy(s) came back seen" in why
    assert "C3=missing" in why, "the operator acts on WHICH buoy, not on a count"


def test_a_moved_buoy_still_counts_as_seen(prior):
    """Four verdicts, not three (SETTLED §3k). `moved` means the encircle SAW it and measured
    where it now is — a measurement, not a doubt. Treating it as `missing` would ground a
    farm-relative survey over a farm that had merely been re-anchored."""
    m = _map(seen=0)
    m.verdicts = {"C1": "moved", "C2": "moved", "C3": "moved",
                  "C4": "missing", "C5": "not_surveyed"}
    ok, why = _planner(prior, m).farm_relative_gate()
    assert ok, why


def test_missing_and_not_surveyed_are_both_excluded(prior):
    """"We looked and it is gone" and "we never went there" are different facts, and NEITHER is
    evidence about where the buoy is now."""
    m = _map(seen=0)
    m.verdicts = {"C1": "confirmed", "C2": "missing", "C3": "not_surveyed"}
    ok, why = _planner(prior, m).farm_relative_gate()
    assert not ok and "only 1 buoy(s)" in why


def test_a_loose_fit_cannot_arm_and_the_refusal_says_what_it_would_cost(prior):
    ok, why = _planner(prior, _map(rms=3.0)).farm_relative_gate()
    assert not ok
    assert "RMS 3.00 m" in why
    assert "push a lane through a rope" in why, \
        "a threshold with no consequence attached becomes a tuning knob"


# ---- THE property: the fallback is surface-and-report, never DR lanes --------------------------

def test_a_failed_gate_refuses_and_does_not_quietly_fly_dr_lanes(prior):
    p = _planner(prior, _map(lines=1))
    ans = p.arm_farm_relative()
    assert ans.kind == REFUSED
    assert ans.phase == T3_LANES
    assert p.nav_mode == DR_WAYPOINTS, "the mode must not have moved"
    assert "do NOT fly the lanes on dead reckoning" in ans.response


def test_the_refusal_names_surface_and_report_as_the_action(prior):
    """A refusal an operator cannot act on is a log line (the planner's own `_refuse` rule)."""
    ans = _planner(prior, _map(seen=0)).arm_farm_relative()
    assert "surface and report" in ans.response


def test_a_refused_arming_leaves_the_vehicle_a_way_home(prior):
    """`_refuse` replaces the remaining plan with surface-here-then-home. Arming must go through
    it rather than around it, or a refused farm-relative switch strands the vehicle at the farm."""
    p = _planner(prior, _map(lines=0))
    p.arm_farm_relative()
    rec = p.recovery_goals()
    assert len(rec) == 2
    assert all(g.depth_m == 0.0 for g in rec), "the way home is on the surface"
    assert all(g.nav_mode == DR_WAYPOINTS for g in rec), \
        "recovery is never farm-relative: it exists precisely because the farm is not trusted"


def test_a_refusal_latches_so_a_later_map_cannot_turn_it_into_carry_on(prior):
    p = _planner(prior, _map(lines=1))
    p.arm_farm_relative()
    p.map = _map()                       # a better map arrives afterwards
    assert p.next().kind == REFUSED


# ---- the mode travels on the GOAL, not on a flag beside it ------------------------------------

def test_t3_goals_carry_farm_relative_once_armed(prior):
    p = FarmMissionPlanner(prior)
    p.map = _map()
    p.arm_farm_relative()
    goals, refusal = p._plan_lanes()
    assert refusal is None, refusal
    assert goals, "the fixture must produce lanes"
    assert all(g.nav_mode == FARM_RELATIVE for g in goals)


def test_t3_goals_stay_dr_when_nothing_armed_them(prior):
    """PASS-FIRST for the default: unarmed is the state everything before P8 was in, and it must
    still work exactly as it did."""
    p = FarmMissionPlanner(prior)
    p.map = _map()
    goals, refusal = p._plan_lanes()
    assert refusal is None
    assert all(g.nav_mode == DR_WAYPOINTS for g in goals)


def test_the_approach_and_the_encircle_are_never_farm_relative(prior):
    """An approach across open water has no farm to be relative to, and the encircle is what
    MEASURES the farm. Arming before them must not change either."""
    p = FarmMissionPlanner(prior)
    p.map = _map()
    p.arm_farm_relative()
    assert all(g.nav_mode == DR_WAYPOINTS for g in p._plan_approach())
    assert all(g.nav_mode == DR_WAYPOINTS for g in p._plan_encircle())
    assert all(g.nav_mode == DR_WAYPOINTS for g in p._plan_return())


def test_only_two_nav_modes_exist(prior):
    assert set(NAV_MODES) == {DR_WAYPOINTS, FARM_RELATIVE}


def test_the_report_says_which_mode_and_why(prior):
    """P8.6's ground and MC's T3 row: the copy MC holds says NOMINAL (SETTLED §3l), so the
    vehicle's own report is the only place the flown mode is stated at all."""
    p = _planner(prior, _map())
    r = p.report()
    assert r["nav_mode"] == DR_WAYPOINTS
    p.arm_farm_relative()
    r = p.report()
    assert r["nav_mode"] == FARM_RELATIVE
    assert "line(s) fitted" in r["farm_relative_gate"]


def test_the_arming_verdicts_are_ones_the_localizer_actually_produces(prior):
    from sam_farm_inspection import localizer
    src = open(localizer.__file__).read()
    for v in ARMING_VERDICTS:
        assert f'"{v}"' in src or f"'{v}'" in src, f"{v!r} is not a verdict that exists"


def test_the_gate_thresholds_live_in_MissionConfig_not_in_the_method(prior):
    """A threshold buried in a method is a threshold nobody can change for a different farm —
    and a farm with three culture lines is a real possibility."""
    p = FarmMissionPlanner(prior, MissionConfig(min_fitted_lines_for_farm_relative=3))
    p.map = _map(lines=2)
    ok, why = p.farm_relative_gate()
    assert not ok and "needs 3" in why
