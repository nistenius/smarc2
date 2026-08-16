"""Side-scan geometry and the protective-stop envelope.

2026-08-16, algae-farm inspection mission. Two properties dominate this file:

  * a side scan cannot see anything at or above its own depth, which is what makes the
    naive reading of mission requirement 7 ("fly at the rope depth") unflyable; and
  * the envelope numbers this package plans against must be the SAME numbers the
    obstacle detector triggers on. That is checked by parsing the detector's own
    `declare_parameter` calls, not by trusting a comment — copying a safety constant into
    a second file is how `STOP_M` inherited a suppression value and the protective stop
    became a report of the collision (spec invariant 8).

Run: python3 -m pytest smarc2/perception/sam/sam_farm_inspection/test/test_sss_geometry.py
"""
import ast
import math
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from sam_farm_inspection import envelope as ENV            # noqa: E402
from sam_farm_inspection.sss_geometry import (             # noqa: E402
    REGRESSED_BEAM_2026_08_16, SIM_BEAM_AS_SHIPPED, BeamGeometry,
    choose_lane_geometry, first_bottom_return_slant_m, range_per_bin,
    required_off_nadir_deg, slant_to_ground_range)

DETECTOR = PKG.parent / "sam_perception" / "sam_perception" / "obstacle_detector.py"


# --------------------------------------------------------------- the beam
def test_the_beam_window_matches_the_simulators_own_ray_construction():
    """`Sonar.cs` casts SSS rays at `rayAngle + side*(90 - tilt - breadth/2)` about the
    forward axis, starting from nadir, with rayAngle in [-B/2, +B/2]. So the angle from
    nadir spans [90 - tilt - breadth, 90 - tilt]. If this ever stops matching, every
    visibility answer in the mission is wrong in a way nothing else would notice."""
    b = BeamGeometry(tilt_deg=30.0, breadth_deg=40.0, max_range_m=100.0, num_buckets=2000)
    assert b.theta_min_deg == pytest.approx(20.0)
    assert b.theta_max_deg == pytest.approx(60.0)
    assert b.nadir_gap_deg == pytest.approx(40.0)


def test_nothing_at_or_above_the_sonars_own_depth_is_visible():
    """THE fact that changed the mission profile. A target level with the sonar is at 90
    degrees off nadir, and no side scan reaches 90 degrees. The answer must be None —
    'no geometry at all' — and not an empty or zero-width window, because a caller that
    reads a number will plan a lane that can never see anything."""
    for beam in (SIM_BEAM_AS_SHIPPED, REGRESSED_BEAM_2026_08_16):
        assert beam.lateral_window_m(0.0) is None
        assert beam.lateral_window_m(-1.0) is None
        assert beam.sees(0.0, 2.0) is False


def test_a_beam_that_cannot_reach_far_enough_off_nadir_is_refused_by_name():
    """The refusal path, exercised with a beam that genuinely cannot reach — the
    pre-7226e43 mount, tilt 45 / breadth 45, i.e. 0..45 degrees off nadir, so the visible
    lateral band never exceeds the vertical separation and the IROS survey's 2 m standoff
    does not fit inside it at any depth.

    Deliberately NOT written against whatever SIM_BEAM_AS_SHIPPED happens to be: on
    2026-08-16 this test asserted that the SHIPPED beam was unflyable, which quietly
    turned a guard into a rubber stamp for a regression — the shipped prefab had been
    reverted by Unity and the test agreed with it. A refusal test needs a beam chosen to
    fail, not the current one.

    The refusal must NAME the limit, because 'the lane could not be planned' is not
    actionable and 'the beam stops at 45 degrees off nadir' is."""
    g = choose_lane_geometry(2.0, REGRESSED_BEAM_2026_08_16, preferred_standoff_m=2.0,
                             min_standoff_m=1.9)
    assert g.ok is False
    assert "off-nadir 0..45 deg" in g.reason
    assert g.standoff_m == 0.0 and g.scan_depth_m == 0.0, \
        "a refusal must not carry numbers that look usable"


def test_the_shipped_simulator_beam_flies_the_two_metre_lane():
    """The committed SAMSensorsV2 mount (tilt 0 / breadth 60, commit 7226e43) reaches
    30..90 degrees off nadir, so the IROS survey's 2 m standoff fits with margin. This is
    the assertion that would have caught the 2026-08-16 prefab regression on the day it
    happened: it fails the moment the shipped beam stops being able to do the mission's
    own headline job."""
    g = choose_lane_geometry(2.0, SIM_BEAM_AS_SHIPPED, preferred_standoff_m=2.0,
                             min_standoff_m=1.9)
    assert g.ok is True
    assert g.standoff_m == pytest.approx(2.0)
    assert g.scan_depth_m < 2.0, "the vehicle must fly ABOVE the ropes to see them"
    assert g.beam_margin_m > 0.0, "a lane on the beam edge is not a lane"


def test_the_chooser_never_returns_the_beam_edge():
    """A lane flown on the last ray of the beam sees the ropes on a good day and nothing
    on a normal one, so a requested margin must be HONOURED — by moving the lane if that
    is possible, and by refusing if it is not.

    The margins are chosen against the beam's binding limit, which is not the same limit
    for every beam. The committed mount reaches 90 deg off nadir, so its outer edge is
    MaxRange, not an angle: squeezing the margin there does not refuse, it walks the
    standoff outwards, which is correct and was invisible while this test ran against a
    beam that ran out of angle first."""
    moved = choose_lane_geometry(2.0, SIM_BEAM_AS_SHIPPED, preferred_standoff_m=2.0,
                                 min_beam_margin_m=10.0)
    assert moved.ok is True
    assert moved.beam_margin_m >= 10.0, "the requested margin was not honoured"
    assert moved.standoff_m > 2.0, "honouring the margin must move the lane, not shrink it"

    impossible = choose_lane_geometry(
        2.0, SIM_BEAM_AS_SHIPPED, preferred_standoff_m=2.0,
        min_beam_margin_m=SIM_BEAM_AS_SHIPPED.max_range_m * 2.0)
    assert impossible.ok is False, "a margin wider than the beam must refuse, not graze"
    assert impossible.standoff_m == 0.0 and impossible.scan_depth_m == 0.0


def test_the_lane_depth_is_derived_from_the_rope_depth():
    """Mission requirement 7. Move the ropes and the lane must move; a hardcoded scan
    depth would pass every other test here."""
    shallow = choose_lane_geometry(2.0, SIM_BEAM_AS_SHIPPED)
    deep = choose_lane_geometry(6.0, SIM_BEAM_AS_SHIPPED)
    assert shallow.ok and deep.ok
    assert deep.depth_below_sonar_m > shallow.depth_below_sonar_m


def test_the_required_angles_are_a_property_not_a_constant():
    """The two acceptance angles handed to the Unity verifier must follow the rope depth
    and the standoff, so the in-editor check keeps meaning 'can this sonar see a rope
    from a flyable lane' whatever beam a future session picks."""
    a_min, a_max = required_off_nadir_deg(2.0, standoff_m=2.0)
    b_min, b_max = required_off_nadir_deg(2.0, standoff_m=4.0)
    assert b_max > a_max, "a wider standoff must need a beam that reaches further out"
    assert SIM_BEAM_AS_SHIPPED.theta_max_deg >= a_max
    assert REGRESSED_BEAM_2026_08_16.theta_max_deg < a_max


# --------------------------------------------------------------- slant range
def test_the_slant_correction_is_the_iros_equation():
    """IROS 2025 Eq. 1: d_2D = sqrt(d_3D^2 - dz^2)."""
    assert slant_to_ground_range(5.0, 3.0) == pytest.approx(4.0)
    assert slant_to_ground_range(2.0, 0.0) == pytest.approx(2.0)


def test_an_impossible_slant_range_is_refused_not_rounded_to_zero():
    """A slant range shorter than the depth difference is geometrically impossible.
    Clamping it to 0 puts a phantom detection directly under the vehicle, which then
    enters the map as a real object."""
    assert slant_to_ground_range(1.0, 3.0) is None


def test_the_first_bottom_return_accounts_for_the_nadir_gap():
    """With a nadir gap the first bottom return is not at the vertical distance but at
    that distance over cos(theta_min). Ignoring it puts the water column's end too close
    and clips the outer part of it."""
    flat = BeamGeometry(45.0, 45.0, 100.0, 2000)          # no gap
    gapped = BeamGeometry(20.0, 60.0, 100.0, 2000)        # 20 deg gap
    assert first_bottom_return_slant_m(9.0, 0.5, flat) == pytest.approx(8.5)
    assert first_bottom_return_slant_m(9.0, 0.5, gapped) > 8.5
    assert first_bottom_return_slant_m(1.0, 5.0, gapped) is None   # seabed above the sonar


# --------------------------------------------------------------- range scale
def test_the_range_scale_prefers_the_ping_and_says_so():
    res, src, complaint = range_per_bin(2 * 100.0 / 1500.0, 2000, 100.0)
    assert src == "message" and complaint is None
    assert res == pytest.approx(0.05)


def test_a_disagreeing_parameter_is_a_complaint_not_a_silent_reconciliation():
    res, src, complaint = range_per_bin(2 * 100.0 / 1500.0, 2000, 30.0)
    assert src == "message", "the publisher's own statement wins"
    assert res == pytest.approx(0.05)
    assert complaint and "30.0 m" in complaint and "100.0 m" in complaint


def test_a_ping_with_no_duration_falls_back_and_names_the_fallback():
    res, src, _ = range_per_bin(0.0, 2000, 100.0)
    assert src == "parameter" and res == pytest.approx(0.05)


def test_no_duration_and_no_parameter_is_a_refusal():
    """Rather than a guessed scale. Every detection range multiplies by this number."""
    res, src, _ = range_per_bin(0.0, 2000, 0.0)
    assert res is None and "no range scale" in src


def test_the_round_trip_through_travel_time_is_exact():
    """SSS_Pub writes 2*range/c; this reads c*duration/2. The two constants must be the
    same or every range is off by their ratio — which looks like a farm of the wrong
    size, i.e. exactly the finding this mission is designed to report."""
    for rng in (10.0, 100.0, 200.0):
        res, _, _ = range_per_bin(2 * rng / 1500.0, 1000)
        assert res * 1000 == pytest.approx(rng, rel=1e-9)


# --------------------------------------------------------------- the envelope
def test_the_envelope_constants_are_the_detectors_own():
    """Parsed out of obstacle_detector.py's `declare_parameter` calls. If someone retunes
    the envelope there, this fails here rather than the planner quietly planning lanes
    against a stop distance that no longer exists."""
    if not DETECTOR.exists():
        pytest.skip(f"obstacle_detector.py not found at {DETECTOR}")
    tree = ast.parse(DETECTOR.read_text())
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "declare_parameter"):
            continue
        if len(node.args) != 2 or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if name in ("stop_margin", "stop_t_react", "stop_a_stop"):
            found[name] = ast.literal_eval(node.args[1])
    assert set(found) == {"stop_margin", "stop_t_react", "stop_a_stop"}, \
        f"could not read all three envelope parameters from {DETECTOR}: got {found}"
    assert found["stop_margin"] == ENV.STOP_MARGIN_M
    assert found["stop_t_react"] == ENV.STOP_T_REACT_S
    assert found["stop_a_stop"] == ENV.STOP_A_STOP_MS2


def test_r_stop_is_quadratic_in_speed():
    """It is not a margin plus a bit. Doubling the speed more than doubles the distance,
    which is why the scan speed is a safety choice and not a schedule choice."""
    assert ENV.r_stop_m(0.0) == pytest.approx(0.5)
    assert ENV.r_stop_m(0.5) == pytest.approx(1.357, abs=1e-3)
    assert ENV.r_stop_m(0.7) == pytest.approx(1.9, abs=1e-3)
    assert ENV.r_stop_m(1.0) == pytest.approx(2.929, abs=1e-3)
    assert ENV.r_stop_m(1.0) > 2 * ENV.r_stop_m(0.5)


def test_the_instructions_figure_omitted_the_reaction_term():
    """The mission instructions quote 'R_stop ~ 0.9 m at 0.5 m/s'. Measured against the
    shipped parameters that is the value at 0.3 m/s; at 0.5 m/s the real trigger is
    1.36 m and at the 0.7 m/s scan speed it is 1.90 m. Recorded as a test so the smaller
    number cannot come back as a planning assumption."""
    assert ENV.r_stop_m(0.3) == pytest.approx(0.93, abs=0.01)
    assert ENV.r_stop_m(0.5) > 1.3


def test_only_the_forward_cone_triggers_the_stop():
    """Which is what makes a 2 m lane standoff flyable at all: a rope line parallel to
    the lane sits abeam. It comes back inside the cone at every lane end."""
    assert ENV.stop_cone_applies(0.0) is True
    assert ENV.stop_cone_applies(30.0) is True
    assert ENV.stop_cone_applies(90.0) is False
    assert ENV.stop_cone_applies(-20.0) is True
    assert ENV.stop_cone_applies(350.0) is True
