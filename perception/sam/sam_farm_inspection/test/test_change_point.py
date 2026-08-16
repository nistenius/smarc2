"""The change-point detector, driven on synthetic pings with known answers.

2026-08-16, algae-farm inspection mission. These tests build a ping, hand it to the real
`detect_ping`, and assert what came out. Nothing here inspects the source text: a
structural check cannot see whether a branch runs (SETTLED §1c), and every property
below is one a plausible-looking refactor could break silently.

THE PLANTED POSITIONS IN THESE FIXTURES ARE GROUND TRUTH AND ARE NAMED AS SUCH. They
exist only in the test. The detector's input is a vector of intensities and it has no
access to them — which is the same rule that keeps it off the simulator's material
labels (mission design decision D4, spec invariant 11's family).

Mutation-tested 2026-08-16, seven mutations. Six are caught:
  * dropping the NEAR sustain window in find_nadir  -> the rope and buoy detection tests
  * dropping the FAR sustain window                 -> test_class_follows_EXTENT_not_amplitude
  * taking the global max instead of the first qualifying rise
                                                    -> test_the_nadir_is_the_FIRST_bottom_return
  * baselining objects over the whole ping instead of the searched region
                                                    -> test_nothing_is_reported_in_an_empty_water_column
  * returning [] instead of a refusal when there is no bottom
                                                    -> test_a_ping_with_no_bottom_return_refuses_by_name
  * reporting the change point as the object's range
                                                    -> test_the_reported_range_is_the_object_centre...
  * classifying by which window scored higher instead of by extent
                                                    -> test_class_survives_a_single_target_class

The seventh, removing the `mean_delta > 0` pre-filter from `find_nadir`, is NOT caught
and is recorded here rather than papered over: it is subsumed by the sustain test, since
a falling edge cannot produce a positive median step either. It is a pre-filter, not a
guard, and the module says so. Two of the comments in `change_point.py` were rewritten
after this exercise because the reasoning in them was wrong — the surviving mutations
were how that was discovered.

Run: python3 -m pytest smarc2/perception/sam/sam_farm_inspection/test/test_change_point.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam_farm_inspection.change_point import (  # noqa: E402
    TargetClass, detect_ping, discrepancy_profile, find_nadir, targets_from_metres)

RES = 0.05          # HF680 range bin, metres
N_BINS = 2000       # HF680 buckets per beam


def make_ping(nadir_bin=170, objects=(), water=6.0, seabed=90.0, noise=1.0, seed=0,
              n=N_BINS):
    """A side-scan ping with a quiet water column, planted objects, and a seabed step.

    GROUND TRUTH: the caller knows `nadir_bin` and every entry of `objects`. The
    detector does not, and must not.

    `objects` is [(bin, amplitude, half_width_bins)]. Shapes are rectangular on purpose —
    a Gaussian bump would let the detector's window size be tuned to the fixture rather
    than to the physics.
    """
    rng = np.random.default_rng(seed)
    y = np.full(n, water, dtype=np.float64)
    y[nadir_bin:] = seabed
    # Seabed returns fall off with range; without this the "seabed" is a flat plateau
    # and the trailing-edge question the rise test exists for never arises.
    tail = np.arange(n - nadir_bin, dtype=np.float64)
    y[nadir_bin:] = seabed * np.exp(-tail / 600.0) + water
    for b, amp, hw in objects:
        y[max(0, b - hw): b + hw + 1] += amp
    y += rng.normal(0.0, noise, n)
    return np.clip(y, 0, 255)


def default_targets():
    # Rope ~10 cm of slant extent, buoy ~40 cm. Both are physical sizes; the helper
    # turns them into bins with the ping's own resolution.
    return targets_from_metres(0.10, 0.40, RES)


# ------------------------------------------------------------------ the arithmetic
def test_the_discrepancy_matches_the_papers_formula_computed_directly():
    """The prefix-sum implementation must equal the O(T*t) definition it replaced.

    This is the one place a naive translation is worth pinning: the fast form is
    unreadable next to the paper and a sign or an off-by-one in it would look like a
    detector that is merely 'a bit noisy'.
    """
    rng = np.random.default_rng(7)
    y = rng.normal(50, 10, 60)
    t = 5

    def c(a, b):
        w = y[a:b]
        return float(np.sum((w - w.mean()) ** 2))

    d, delta = discrepancy_profile(y, t)
    assert d.size == len(y) - 2 * t + 1
    for i in range(d.size):
        want = c(i, i + 2 * t) - c(i, i + t) - c(i + t, i + 2 * t)
        assert d[i] == pytest.approx(want, rel=1e-9, abs=1e-6)
        assert delta[i] == pytest.approx(y[i + t:i + 2 * t].mean() - y[i:i + t].mean(),
                                         rel=1e-9, abs=1e-9)


def test_a_flat_signal_has_no_change_points():
    d, _ = discrepancy_profile(np.full(100, 42.0), 8)
    assert float(np.max(d)) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ the nadir
def test_the_nadir_lands_on_the_planted_seabed_step():
    truth_nadir = 170                                   # GROUND TRUTH
    y = make_ping(nadir_bin=truth_nadir)
    got, reason, _ = find_nadir(y, 20, 8.0, blank_bins=10)
    assert reason == "ok"
    assert abs(got - truth_nadir) <= 20, f"nadir {got} vs planted {truth_nadir}"


def test_the_nadir_is_a_rise_not_a_fall():
    """The discrepancy is symmetric: it fires equally on a step DOWN. Only the sign of
    the mean difference separates the seabed's start from anything that gets quieter.

    A ping that gets quieter and never louder has no bottom return, and the detector
    must say so rather than name the drop as the seabed.
    """
    y = np.full(600, 90.0)
    y[300:] = 6.0                                       # a pure fall, nothing else
    got, reason, _ = find_nadir(y, 20, 8.0, blank_bins=10)
    assert got is None, f"a falling edge was named as the bottom return at bin {got}"
    assert "rising" in reason


def test_the_nadir_is_the_FIRST_bottom_return():
    """Two rises: the seabed, then a stronger specular return further out. The paper's
    nadir is the FIRST bottom-hitting return; taking the global maximum would put the
    water-column window past the seabed and feed every seabed return to the rope
    detector."""
    y = make_ping(nadir_bin=200, seabed=40.0)
    y[600:640] += 200.0                                 # a much louder later return
    got, reason, _ = find_nadir(y, 20, 8.0, blank_bins=10)
    assert reason == "ok"
    assert got < 400, f"picked the later, louder return at bin {got}"


def test_the_near_field_blanking_is_honoured():
    """Transmit ring-down at bin 0 is a colossal change point and is not the seabed."""
    y = make_ping(nadir_bin=300)
    y[:15] = 255.0
    got, _, _ = find_nadir(y, 20, 8.0, blank_bins=40)
    assert got >= 40


# ------------------------------------------------------------------ objects
def test_a_rope_in_the_water_column_is_found_at_its_planted_range():
    truth_rope_bin = 90                                 # GROUND TRUTH
    y = make_ping(nadir_bin=170, objects=[(truth_rope_bin, 45.0, 1)])
    res = detect_ping(y, RES, default_targets(), blank_bins=10)
    assert res.ok, res.reason
    hits = [d for d in res.detections if abs(d.bin_index - truth_rope_bin) <= 6]
    assert hits, f"nothing within 6 bins of the planted rope; got {res.detections}"
    assert hits[0].slant_range_m == pytest.approx(truth_rope_bin * RES, abs=0.4)
    assert hits[0].confidence > 0.0


def test_nothing_is_reported_in_an_empty_water_column():
    """The expensive failure mode is the opposite one — an inspection that reports a
    farm where there is none. A quiet ping must come back ok-and-empty, not ok-and-noisy."""
    y = make_ping(nadir_bin=170, objects=[])
    res = detect_ping(y, RES, default_targets(), blank_bins=10)
    assert res.ok, res.reason
    assert res.detections == [], f"invented {len(res.detections)} detections in open water"


def test_water_column_objects_survive_a_loud_seabed():
    """The seabed return is orders of magnitude louder than a rope. If the object
    baseline were taken over the whole ping it would be set by the seabed and the water
    column would read as silent — the detector would work perfectly in a tank and find
    nothing at sea."""
    truth_rope_bin = 100                                # GROUND TRUTH
    y = make_ping(nadir_bin=180, objects=[(truth_rope_bin, 40.0, 1)], seabed=250.0)
    res = detect_ping(y, RES, default_targets(), blank_bins=10)
    assert res.ok, res.reason
    assert any(abs(d.bin_index - truth_rope_bin) <= 6 for d in res.detections), \
        f"the seabed drowned the rope; got {res.detections}"


def test_detections_never_come_from_beyond_the_nadir():
    """Everything past the first bottom return is seabed. A 'rope' there is a seabed
    feature wearing a rope's label, and it would be mapped as farm structure."""
    y = make_ping(nadir_bin=150, objects=[(80, 45.0, 1), (400, 120.0, 6)])
    res = detect_ping(y, RES, default_targets(), blank_bins=10)
    assert res.ok, res.reason
    assert all(d.bin_index < res.nadir_bin for d in res.detections), \
        f"a detection at or past the nadir {res.nadir_bin}: {res.detections}"


def test_a_bigger_brighter_object_reads_as_a_buoy_and_the_tie_is_declared():
    """Rope and buoy are the same algorithm at two scales, so a real object trips both.
    The stronger class wins and `ambiguous` records that the other fired — a buoy
    silently filed as a rope is a mapping error, and hiding the tie is how it happens."""
    truth_buoy_bin = 110                                # GROUND TRUTH
    y = make_ping(nadir_bin=200, objects=[(truth_buoy_bin, 120.0, 8)])
    res = detect_ping(y, RES, default_targets(), blank_bins=10)
    assert res.ok, res.reason
    near = [d for d in res.detections if abs(d.bin_index - truth_buoy_bin) <= 12]
    assert near, f"the buoy was not detected at all; got {res.detections}"
    assert any(d.ambiguous for d in near), \
        "both classes fired on one object and the detector did not say so"


def test_class_survives_a_single_target_class():
    """Classification must not depend on two windows racing each other.

    Run the detector with ONLY the rope window and put a 1 m object in front of it. A
    window comparison has no vote here — there is nothing to compare — so a detector that
    classifies by "which window scored higher" reports a metre-wide buoy as a rope, and
    the map gains a rope where a mooring buoy is. The extent rule has an answer either
    way. (Found by mutation-testing: with both windows present the two rules agree, so
    this single-class case is the only place the difference is visible.)
    """
    y = make_ping(nadir_bin=260, objects=[(120, 40.0, 10)])   # GROUND TRUTH buoy, ~1 m
    rope_only = [targets_from_metres(0.10, 0.40, RES)[0]]
    got = [d for d in detect_ping(y, RES, rope_only, blank_bins=10).detections
           if abs(d.bin_index - 120) <= 15]
    assert got, "the object was not detected at all"
    assert got[0].target == "buoy", (
        f"a {got[0].extent_m:.2f} m object found by the rope window was reported as "
        f"{got[0].target}")


def test_class_follows_EXTENT_not_amplitude():
    """A bright narrow object is a rope; a dim wide one is a buoy — and the amplitudes
    are the wrong way round on purpose, so a rule that read brightness would fail."""
    bright_narrow = make_ping(nadir_bin=220, objects=[(100, 200.0, 1)])   # GROUND TRUTH rope
    dim_wide = make_ping(nadir_bin=220, objects=[(100, 30.0, 10)])        # GROUND TRUTH buoy

    r = [d for d in detect_ping(bright_narrow, RES, default_targets(), blank_bins=10).detections
         if abs(d.bin_index - 100) <= 12]
    b = [d for d in detect_ping(dim_wide, RES, default_targets(), blank_bins=10).detections
         if abs(d.bin_index - 100) <= 15]
    assert r and b, f"one of the two objects was missed: rope={r} buoy={b}"
    assert r[0].target == "rope", f"a 15 cm object at amplitude 200 was called {r[0].target}"
    assert b[0].target == "buoy", f"a 1 m object at amplitude 30 was called {b[0].target}"
    assert b[0].extent_bins > r[0].extent_bins


def test_the_reported_range_is_the_object_centre_not_the_change_point():
    """A change point marks an EDGE. Reporting it as the object's range biases every
    detection by half the object's width plus the window — systematically, in one
    direction, which is exactly the kind of error a map fit will absorb as a shifted
    farm rather than reject as noise."""
    truth_bin = 120                                     # GROUND TRUTH
    y = make_ping(nadir_bin=240, objects=[(truth_bin, 60.0, 6)])   # 13 bins = 0.65 m wide
    res = detect_ping(y, RES, default_targets(), blank_bins=10)
    near = [d for d in res.detections if abs(d.bin_index - truth_bin) <= 15]
    assert near, res.detections
    det = near[0]
    # No escape clause: the change point for a 13-bin object is at least 4 bins off its
    # centre, so "centre == change point" is a real failure and not a coincidence to
    # forgive. An earlier version of this test allowed that case and let the mutation
    # through.
    assert abs(det.bin_index - truth_bin) <= 2, \
        f"reported centre {det.bin_index} is {abs(det.bin_index - truth_bin)} bins off truth {truth_bin}"
    assert abs(det.change_bin - truth_bin) >= 4, \
        (f"the fixture is not discriminating: the change point {det.change_bin} is already "
         f"on the object's centre {truth_bin}, so this test cannot see the difference")


# ------------------------------------------------------------------ refusals
def test_a_ping_with_no_bottom_return_refuses_by_name():
    """`ok=False` with a reason, never an empty list. 'No ropes here' and 'I could not
    find where the water column ends' are different answers and only one is information.
    This is the ADR-004 refused-never-downgraded rule applied to perception."""
    y = np.full(N_BINS, 6.0) + np.random.default_rng(1).normal(0, 1, N_BINS)
    res = detect_ping(y, RES, default_targets(), blank_bins=10)
    assert res.ok is False
    assert res.detections == []
    assert "bottom return" in res.reason
    assert "baseline" in res.reason or "baseline discrepancy" in res.reason


def test_an_empty_ping_refuses_rather_than_dividing_by_zero():
    res = detect_ping(np.empty(0), RES, default_targets())
    assert res.ok is False and "empty" in res.reason


def test_a_zero_range_resolution_refuses():
    """Range resolution reaches this function from a message field. Zero would put every
    detection at 0 m, i.e. directly under the vehicle, which is a plausible-looking
    catastrophe."""
    res = detect_ping(make_ping(), 0.0, default_targets())
    assert res.ok is False and "resolution" in res.reason


# ------------------------------------------------------------------ configuration
def test_window_sizes_are_configured_in_metres_not_bins():
    """The two side-scan modes differ 4x in bin size (HF680 5 cm, LF340 20 cm). A window
    expressed in bins is right for one mode and meaningless for the other."""
    hf = targets_from_metres(0.10, 0.40, 0.05)
    lf = targets_from_metres(0.10, 0.40, 0.20)
    assert hf[0].window_bins == 2 and hf[1].window_bins == 8
    assert lf[0].window_bins == 2 and lf[1].window_bins == 2
    # ... and the physical extent each represents is the same to within one bin.
    assert abs(hf[1].window_bins * 0.05 - 0.40) <= 0.05


def test_the_change_ratio_actually_gates():
    """A threshold that does not threshold is the shape of half the bugs in SETTLED."""
    y = make_ping(nadir_bin=170, objects=[(90, 25.0, 1)])
    loose = TargetClass("rope", 2, 2.0)
    tight = TargetClass("rope", 2, 1e6)
    assert detect_ping(y, RES, [loose], blank_bins=10).detections != []
    assert detect_ping(y, RES, [tight], blank_bins=10).detections == []
