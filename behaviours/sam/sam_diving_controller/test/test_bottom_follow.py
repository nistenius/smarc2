"""BOTTOM FOLLOWING ANTICIPATES A RISE AND REFUSES TO BE PULLED INTO A HOLE.  (2026-08-15)

Ivan asked for bottom tracking that looks ahead with the 3D sonar "to plan better the path to
follow to track a varying bottom contour". The policy that makes that safe rather than merely
clever is one line: over the lookahead window take the SHALLOWEST seabed, never the nearest and
never the mean.

That asymmetry is the whole argument. Rising terrain has a deadline — at 1 m/s over a 30° slope
an altimeter-only loop is permanently behind, because the altimeter reads what is already
underneath. Descending terrain has none: nothing bad happens if the vehicle is temporarily higher
above a trench than asked. Averaging would trade the first for the second, and a "nearest return"
rule would drive the vehicle down into every hole it flew over.

The module is a pure function on purpose — it decides where the vehicle goes, so it must be
testable without a ROS graph, a sonar, or water.

    python3 -m pytest test/test_bottom_follow.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_diving_controller.bottom_follow import (  # noqa: E402
    BottomFollowConfig, BottomSample, lookahead_distance, plan_depth)


CFG = BottomFollowConfig()


def flat(depth, n=10, spacing=3.0, age=0.0):
    return [BottomSample(range_ahead_m=i * spacing, seabed_depth_m=depth, age_s=age)
            for i in range(n)]


def plan(samples, **kw):
    kw.setdefault("target_altitude_m", 3.0)
    kw.setdefault("speed_mps", 1.0)
    # Deep by default. `fallback_depth_m` is the leg's `target_depth`, and it doubles as the
    # CEILING in bottom-follow mode — an early version of this file defaulted it to 20 m and then
    # asserted the vehicle settled at 25 m over a 30 m seabed, which the ceiling was correctly
    # preventing. The fixture was wrong, not the clamp; a shallow default here silently tests the
    # clamp instead of the policy.
    kw.setdefault("current_depth_m", 10.0)
    kw.setdefault("fallback_depth_m", 100.0)
    kw.setdefault("dt_s", 1.0)
    return plan_depth(samples, **kw)


# A window long enough to contain the fixtures below. The DEFAULT window is 8 s — at 1 m/s that
# is 8 m, and a ridge 20 m ahead is genuinely outside it. That is a real tuning question, not a
# fixture detail, so it gets its own test (see "the window and the climb rate must agree") rather
# than being hidden by quietly widening the default.
WIDE = BottomFollowConfig(lookahead_s=30.0)


# ------------------------------------------------------------------ the lookahead window

def test_the_window_scales_with_speed():
    slow = lookahead_distance(0.5, CFG)
    fast = lookahead_distance(2.0, CFG)
    assert fast > slow, "the useful horizon is how long there is to react, so it scales with speed"


def test_the_window_never_collapses_when_crawling():
    """At 0.1 m/s an 8 s window is 0.8 m — no lookahead at all, and it would degrade to
    altimeter-only behaviour without saying so."""
    assert lookahead_distance(0.05, CFG) >= CFG.min_lookahead_m


def test_the_window_never_exceeds_what_the_sonar_can_resolve():
    assert lookahead_distance(50.0, CFG) <= CFG.max_lookahead_m


# ------------------------------------------------------------------ the core asymmetry

def test_a_rise_ahead_is_anticipated_before_it_is_underneath():
    """THE FEATURE. The seabed is 40 m down here and 12 m down 20 m ahead."""
    samples = [BottomSample(0.0, 40.0), BottomSample(5.0, 40.0), BottomSample(10.0, 30.0),
               BottomSample(15.0, 20.0), BottomSample(20.0, 12.0)]
    r = plan(samples, cfg=WIDE, current_depth_m=37.0, previous_setpoint_m=37.0, dt_s=100.0)
    assert r.seabed_ahead_m == 12.0
    assert r.depth_setpoint_m == pytest.approx(9.0), \
        "commanded depth must clear the SHALLOWEST seabed in the window by target_altitude"


def test_a_hole_ahead_does_not_pull_the_vehicle_down():
    """The mirror image, and the reason this is `min` rather than `mean` or `nearest`."""
    samples = [BottomSample(0.0, 20.0), BottomSample(5.0, 20.0), BottomSample(10.0, 60.0),
               BottomSample(15.0, 60.0), BottomSample(20.0, 20.0)]
    r = plan(samples, cfg=WIDE, current_depth_m=17.0, previous_setpoint_m=17.0, dt_s=100.0)
    assert r.depth_setpoint_m == pytest.approx(17.0), \
        "a trench in the window must not become a dive command"


def test_the_mean_would_have_been_wrong_and_this_is_not_the_mean():
    """Pins the choice, not just today's number: a mean over this window is 30 m of seabed and
    would command 27 m — flying the vehicle straight into the 12 m ridge."""
    samples = [BottomSample(0.0, 40.0), BottomSample(10.0, 40.0), BottomSample(20.0, 12.0)]
    r = plan(samples, cfg=WIDE, current_depth_m=9.0, previous_setpoint_m=9.0, dt_s=100.0)
    assert r.depth_setpoint_m < 27.0


# ------------------------------------------------------------------ no bottom is not zero bottom

def test_no_returns_holds_the_legs_own_target_depth():
    r = plan([], fallback_depth_m=15.0)
    assert r.depth_setpoint_m == 15.0, \
        "returning None would leave the vehicle with no z command at all"


def test_too_few_returns_is_treated_as_no_bottom():
    r = plan(flat(30.0, n=CFG.min_samples - 1))
    assert r.depth_setpoint_m == 100.0
    assert "need" in r.reason


def test_stale_returns_are_not_evidence_about_what_is_ahead_now():
    r = plan(flat(30.0, n=10, age=CFG.max_sample_age_s + 1))
    assert r.seabed_ahead_m is None
    assert "no usable bottom" in r.reason


def test_returns_beyond_the_window_are_ignored():
    """A ridge 200 m away must not make the vehicle climb now."""
    far = [BottomSample(range_ahead_m=200.0 + i, seabed_depth_m=5.0) for i in range(5)]
    r = plan(flat(40.0, n=5, spacing=2.0) + far, cfg=WIDE, current_depth_m=37.0,
             previous_setpoint_m=37.0, dt_s=100.0)
    assert r.seabed_ahead_m == 40.0


def test_every_outcome_names_its_reason():
    """A 'no bottom' with no reason is the output this component exists to prevent: out of range,
    silted water, a blocked beam and a genuinely flat seabed read identically otherwise."""
    for samples in ([], flat(30.0), flat(30.0, age=99.0)):
        assert plan(samples).reason.strip(), "an empty reason is never acceptable"


# ------------------------------------------------------------------ clamps

def test_bottom_following_never_surfaces_the_vehicle_mid_mission():
    """Seabed 1 m down, asked to hold 3 m above it — the arithmetic wants -2 m."""
    r = plan(flat(1.0), target_altitude_m=3.0, current_depth_m=2.0,
             previous_setpoint_m=2.0, dt_s=100.0)
    assert r.depth_setpoint_m >= CFG.min_depth_m and r.clamped


def test_the_legs_target_depth_is_a_ceiling_not_a_suggestion():
    """A seabed reading that would drive the vehicle below what the leg asked for is more likely
    a bad return than an instruction."""
    r = plan(flat(100.0), target_altitude_m=3.0, fallback_depth_m=12.0,
             current_depth_m=12.0, previous_setpoint_m=12.0, dt_s=100.0)
    assert r.depth_setpoint_m == 12.0 and r.clamped


def test_the_setpoint_is_rate_limited():
    """A step change in the setpoint is how the 2026-08-09 dive limit cycle was provoked."""
    r = plan(flat(10.0), target_altitude_m=3.0, current_depth_m=30.0,
             previous_setpoint_m=30.0, dt_s=1.0)
    assert r.depth_setpoint_m == pytest.approx(30.0 - CFG.max_climb_rate_mps)
    assert r.clamped


def test_the_rate_limit_follows_the_vehicle_when_there_is_no_previous_setpoint():
    """Limiting from a stale setpoint the vehicle never reached lets it run away and hands the
    PID a growing error."""
    r = plan(flat(10.0), current_depth_m=30.0, previous_setpoint_m=None, dt_s=1.0)
    assert r.depth_setpoint_m == pytest.approx(30.0 - CFG.max_climb_rate_mps)


def test_climbing_is_allowed_to_be_faster_than_descending():
    assert CFG.max_climb_rate_mps > CFG.max_descend_rate_mps, \
        "failing to climb has a deadline; failing to descend does not"


# ------------------------------------------------------------------ steady state

def test_over_flat_ground_it_settles_at_exactly_the_asked_altitude():
    depth = 20.0
    for _ in range(200):
        r = plan(flat(30.0), target_altitude_m=5.0, current_depth_m=depth,
                 previous_setpoint_m=depth, dt_s=0.5)
        depth = r.depth_setpoint_m
    assert depth == pytest.approx(25.0, abs=0.01), \
        f"settled at {depth} m over a 30 m seabed asked to hold 5 m above it"


def test_it_does_not_oscillate_once_settled():
    seq = []
    depth = 25.0
    for _ in range(20):
        depth = plan(flat(30.0), target_altitude_m=5.0, current_depth_m=depth,
                     previous_setpoint_m=depth, dt_s=0.5).depth_setpoint_m
        seq.append(depth)
    assert max(seq) - min(seq) < 1e-6


# ------------------------------------------------- the window and the climb rate must agree

def test_a_rise_the_vehicle_cannot_climb_in_time_is_CLAMPED_AND_SAYS_SO():
    """The tuning constraint, made visible rather than assumed away.

    The default window is 8 s. At 1 m/s and a 0.3 m/s climb limit, that buys 2.4 m of climb — so
    any ridge rising faster than ~17 degrees ahead of the vehicle cannot be cleared by looking
    8 s ahead. The right response is not to silently command the impossible depth: it is to
    command what the vehicle can actually reach and REPORT that it was limited, so a rig session
    can lengthen `lookahead_s` (or the operator can slow the leg down) on evidence.

    This test exists because the first version of this file quietly used a 30 s window in its
    fixtures and would have shipped an 8 s default nobody had questioned.
    """
    steep = [BottomSample(0.0, 40.0), BottomSample(3.0, 30.0), BottomSample(6.0, 12.0)]
    r = plan(steep, target_altitude_m=3.0, current_depth_m=37.0,
             previous_setpoint_m=37.0, dt_s=1.0)
    assert r.seabed_ahead_m == 12.0, "the ridge IS seen"
    assert r.depth_setpoint_m == pytest.approx(37.0 - CFG.max_climb_rate_mps), \
        "and the command is what the vehicle can reach this tick, not the unreachable ideal"
    assert r.clamped, "a limited command that does not report being limited is a silent failure"


def test_the_default_window_is_recorded_as_needing_a_rig_number():
    """Not a behaviour test — a standing note in executable form. lookahead_s and the climb rates
    are the three numbers a rig session must set, and nothing has flown yet."""
    assert CFG.lookahead_s == 8.0 and CFG.max_climb_rate_mps == 0.3, (
        "these defaults are provisional and unflown; if a rig session changed them, update "
        "ADR-004 and SETTLED in the same session rather than only here")
