"""Farm localization, driven on synthetic surveys of the REAL Kristineberg prior.

2026-08-16. Every fixture below is built by taking the seven surveyed buoy positions,
applying a known rigid transform, and sampling noisy detections from the result. The
transform is GROUND TRUTH and exists only in the test; `localize_farm` sees detections.

The properties worth guarding are the ones where a wrong answer looks like a right one:
a transform fitted from collinear points, a farm matched to the wrong farm, a buoy the
vehicle never visited reported as missing, and a line offset fitted from one side.

Mutation-tested 2026-08-16 — see the individual docstrings; every mutation listed there
was applied and reproduced the failure.

Run: python3 -m pytest smarc2/perception/sam/sam_farm_inspection/test/test_localizer.py
"""
import math
import pathlib
import sys

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from sam_farm_inspection.localizer import (  # noqa: E402
    WorldDetection, cluster_buoys, coverage_bearings_ok, fit_culture_lines,
    fit_rigid_2d, localize_farm, match_prior_to_clusters)

#: The seven surveyed buoys, Unity metres — the same numbers `site_frame.py` holds and
#: `farm_prior.yaml` is generated from. Copied here as a TEST FIXTURE, which is the one
#: place a copy is allowed: a test that regenerated the prior would be testing the
#: generator, and `test_farm_prior_is_generated.py` already does that.
PRIOR = {
    "C2_corner_SW": (238.17, 108.30),
    "C0_south_mid": (248.17, 110.30),
    "C1_corner_SE": (264.17, 105.30),
    "M2_east_mid":  (263.17, 121.30),
    "M1_west_mid":  (242.67, 123.30),
    "C4_corner_NW": (247.17, 138.30),
    "C3_corner_NE": (262.17, 137.30),
}
LINES = [
    ("west", PRIOR["C2_corner_SW"], PRIOR["C4_corner_NW"]),
    ("east", PRIOR["C1_corner_SE"], PRIOR["C3_corner_NE"]),
]


def rot(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def make_survey(rotate_deg=0.0, shift=(0.0, 0.0), noise=0.6, per_buoy=14,
                rope_per_line=60, seed=1, drop=(), bearings=None):
    """A synthetic encircle: buoy and rope detections from a transformed farm.

    GROUND TRUTH: `rotate_deg` and `shift`. The detector never sees them.
    """
    rng = np.random.default_rng(seed)
    R = rot(rotate_deg)
    T = np.array(shift, dtype=np.float64)
    ctr = np.mean(np.array(list(PRIOR.values())), axis=0)

    def move(p):
        return R @ (np.array(p) - ctr) + ctr + T

    bearings = bearings if bearings is not None else list(range(0, 360, 30))
    dets = []
    for name, xy in PRIOR.items():
        if name in drop:
            continue
        p = move(xy)
        for i in range(per_buoy):
            dets.append(WorldDetection(
                float(p[0] + rng.normal(0, noise)), float(p[1] + rng.normal(0, noise)),
                "buoy", 1.0, float(bearings[i % len(bearings)])))
    for _, a, b in LINES:
        pa, pb = move(a), move(b)
        for i in range(rope_per_line):
            f = rng.random()
            p = pa + f * (pb - pa)
            dets.append(WorldDetection(
                float(p[0] + rng.normal(0, noise)), float(p[1] + rng.normal(0, noise)),
                "rope", 1.0, float(bearings[i % len(bearings)])))
    return dets, R, T, ctr


def track_all():
    """A track that visited every buoy, so `missing` means missing."""
    return [tuple(v) for v in PRIOR.values()]


# ------------------------------------------------------------------ Kabsch
def test_the_rigid_fit_recovers_a_known_transform():
    src = np.array(list(PRIOR.values()))
    R = rot(23.0)
    dst = src @ R.T + np.array([4.0, -7.0])
    Rf, tf, rms = fit_rigid_2d(src, dst)
    assert rms < 1e-9
    assert math.degrees(math.atan2(Rf[1, 0], Rf[0, 0])) == pytest.approx(23.0, abs=1e-6)


def test_the_fit_refuses_to_absorb_a_scale_change():
    """Allowing scale would let a bad correspondence set resize the farm until its own
    residual looked small — a wrong map with a convincing number attached. A farm that
    appears 20 % larger must show up as residual, not be fitted away.

    Mutation: adding a scale term to `fit_rigid_2d` drops the residual to ~0 here.
    """
    src = np.array(list(PRIOR.values()))
    dst = (src - src.mean(0)) * 1.2 + src.mean(0)
    _, _, rms = fit_rigid_2d(src, dst)
    assert rms > 1.0, "a 20 % size change was absorbed instead of reported"


# ------------------------------------------------------------------ clustering
def test_seven_buoys_come_back_as_seven_clusters():
    dets, _, _, _ = make_survey()
    pts = [(d.x, d.y) for d in dets if d.target == "buoy"]
    clusters, why = cluster_buoys(pts, max_classes=7, obs_sigma_m=0.8)
    assert why == "ok"
    assert len(clusters) == 7, f"got {len(clusters)} clusters: {clusters}"


def test_the_maximum_class_count_is_a_ceiling_not_a_target():
    """A farm that has lost a buoy must come back with SIX clusters, not seven with one
    invented in open water. The Dirichlet prior is what decays an unclaimed component;
    mutation: raising alpha0 to 5.0 produces a spurious seventh."""
    dets, _, _, _ = make_survey(drop=("M1_west_mid",))
    pts = [(d.x, d.y) for d in dets if d.target == "buoy"]
    clusters, why = cluster_buoys(pts, max_classes=7, obs_sigma_m=0.8)
    assert why == "ok"
    assert len(clusters) == 6, f"invented a cluster: {len(clusters)}"


def test_outliers_go_to_the_background_and_do_not_drag_a_cluster():
    """Robustness to outliers is the stated reason the 2022 paper uses a VGMM at all.

    Mutation: removing the uniform background component pulls the nearest cluster
    several metres toward the strays and this fails.
    """
    dets, _, _, _ = make_survey(noise=0.5)
    pts = [(d.x, d.y) for d in dets if d.target == "buoy"]
    clean, _ = cluster_buoys(pts, max_classes=7, obs_sigma_m=0.8)
    rng = np.random.default_rng(4)
    strays = [(float(x), float(y)) for x, y in
              rng.uniform([230, 100], [275, 145], size=(25, 2))]
    dirty, _ = cluster_buoys(pts + strays, max_classes=7, obs_sigma_m=0.8)

    def nearest(cs, p):
        return min(math.hypot(c.x - p[0], c.y - p[1]) for c in cs)

    for name, xy in PRIOR.items():
        assert nearest(dirty, xy) < 2.0, \
            f"{name}: {nearest(dirty, xy):.2f} m from any cluster once outliers were added"
    assert len(clean) == 7


def test_too_few_detections_refuse_rather_than_cluster_noise():
    clusters, why = cluster_buoys([(1.0, 1.0)], max_classes=7)
    assert clusters == [] and "minimum" in why
    clusters, why = cluster_buoys([], max_classes=7)
    assert clusters == [] and "no buoy detections" in why


# ------------------------------------------------------------------ matching
def test_a_displaced_farm_is_localized_and_the_displacement_is_reported():
    """Design decision D5: displacement is a FINDING, not a failure. The mission carries
    on against the observed map."""
    truth_rot, truth_shift = 12.0, (6.0, -4.0)          # GROUND TRUTH
    dets, R_true, T_true, ctr = make_survey(rotate_deg=truth_rot, shift=truth_shift)
    fix = localize_farm(dets, PRIOR, LINES, covered_points=track_all(), obs_sigma_m=0.8)
    assert fix.ok, fix.reason
    assert fix.rotation_deg == pytest.approx(truth_rot, abs=2.0)
    # The fixture rotates ABOUT THE FARM CENTRE and then shifts, so the ground-truth
    # translation of the equivalent `R p + t` form is `ctr - R ctr + T`, not `T`. An
    # earlier version of this test compared against `T` and read a perfectly correct fit
    # as 30 m out — the algebra was the test's, not the localizer's.
    t_true = ctr - R_true @ ctr + T_true
    assert fix.translation_m[0] == pytest.approx(float(t_true[0]), abs=2.0)
    assert fix.translation_m[1] == pytest.approx(float(t_true[1]), abs=2.0)
    # And the check that does not depend on any convention at all: where does the
    # transform put a buoy, against where the fixture put it?
    for name, xy in PRIOR.items():
        want = R_true @ (np.array(xy) - ctr) + ctr + T_true
        got = np.array([v.observed_xy for v in fix.verdicts if v.name == name][0])
        assert np.linalg.norm(got - want) < 2.0, f"{name}: {np.linalg.norm(got - want):.2f} m"
    assert fix.n_matched == 7
    assert fix.rms_m < 1.5
    assert all(v.status == "confirmed" for v in fix.verdicts), \
        [(v.name, v.status, v.residual_m) for v in fix.verdicts]


def test_a_buoy_that_moved_is_reported_as_moved_and_the_rest_stay_confirmed():
    """One storm-dragged buoy must not be absorbed into the transform, and must not
    invalidate the other six."""
    dets, R, T, ctr = make_survey()
    moved_true = np.array(PRIOR["C0_south_mid"]) + np.array([7.0, 5.0])   # GROUND TRUTH
    dets = [d for d in dets
            if math.hypot(d.x - PRIOR["C0_south_mid"][0],
                          d.y - PRIOR["C0_south_mid"][1]) > 3.0 or d.target != "buoy"]
    rng = np.random.default_rng(9)
    for i in range(14):
        dets.append(WorldDetection(float(moved_true[0] + rng.normal(0, 0.6)),
                                   float(moved_true[1] + rng.normal(0, 0.6)),
                                   "buoy", 1.0, float((i * 30) % 360)))
    fix = localize_farm(dets, PRIOR, LINES, covered_points=track_all(), obs_sigma_m=0.8)
    assert fix.ok, fix.reason
    by = {v.name: v for v in fix.verdicts}
    assert by["C0_south_mid"].status == "moved", by["C0_south_mid"]
    assert by["C0_south_mid"].residual_m > 5.0
    others = [v.status for n, v in by.items() if n != "C0_south_mid"]
    assert all(s == "confirmed" for s in others), others


def test_a_collinear_set_is_refused_because_its_rotation_is_a_guess():
    """The most dangerous answer shape there is: a small residual and an arbitrary
    heading. Mutation: dropping the collinearity check returns ok=True here with a
    rotation that changes with the seed."""
    from sam_farm_inspection.localizer import Cluster
    line = {f"b{i}": (240.0 + 5.0 * i, 110.0) for i in range(5)}
    clusters = [Cluster(240.0 + 5.0 * i, 110.0, 10.0, 0.4) for i in range(5)]
    R, t, pairs, why = match_prior_to_clusters(line, clusters)
    assert R is None
    assert "collinear" in why and "unconstrained" in why


def test_two_clusters_are_not_enough_to_check_a_transform():
    """A rigid transform from two points is exactly determined, so its residual is zero
    by construction and proves nothing."""
    from sam_farm_inspection.localizer import Cluster
    R, t, pairs, why = match_prior_to_clusters(
        PRIOR, [Cluster(238.0, 108.0, 10, 0.3), Cluster(264.0, 105.0, 10, 0.3)])
    assert R is None and "at least 3" in why


def test_a_different_farm_is_refused_rather_than_fitted():
    """The refusal a mission needs in order to hold, surface and report instead of
    scanning something else."""
    from sam_farm_inspection.localizer import Cluster
    square = [Cluster(0.0, 0.0, 10, 0.3), Cluster(60.0, 0.0, 10, 0.3),
              Cluster(60.0, 60.0, 10, 0.3), Cluster(0.0, 60.0, 10, 0.3)]
    R, t, pairs, why = match_prior_to_clusters(PRIOR, square, inlier_m=3.0)
    assert R is None
    assert "does not look like the farm in the prior" in why


# ------------------------------------------------------------------ verdicts
def test_a_buoy_nobody_looked_for_is_not_surveyed_not_missing():
    """"We looked and it is gone" and "we never went there" are different facts, and
    reporting the second as the first sends someone to recover a buoy that is fine.

    Mutation: collapsing `not_surveyed` into `missing` fails here.
    """
    dets, _, _, _ = make_survey(drop=("C3_corner_NE",))
    partial_track = [xy for n, xy in PRIOR.items() if n != "C3_corner_NE"]
    partial_track = [(x - 25.0, y - 25.0) for x, y in partial_track]   # never went near
    fix = localize_farm(dets, PRIOR, LINES, covered_points=partial_track, obs_sigma_m=0.8)
    assert fix.ok, fix.reason
    by = {v.name: v for v in fix.verdicts}
    assert by["C3_corner_NE"].status == "not_surveyed", by["C3_corner_NE"]


def test_a_buoy_that_was_looked_for_and_not_found_is_missing():
    dets, _, _, _ = make_survey(drop=("C3_corner_NE",))
    fix = localize_farm(dets, PRIOR, LINES, covered_points=track_all(), obs_sigma_m=0.8)
    assert fix.ok, fix.reason
    by = {v.name: v for v in fix.verdicts}
    assert by["C3_corner_NE"].status == "missing", by["C3_corner_NE"]


def test_with_no_track_at_all_nothing_is_called_not_surveyed():
    """Absent coverage information must not become an excuse. With no track the honest
    reading is that we cannot claim the vehicle failed to look, so the verdict falls back
    to `missing` and the operator sees it."""
    dets, _, _, _ = make_survey(drop=("C3_corner_NE",))
    fix = localize_farm(dets, PRIOR, LINES, covered_points=None, obs_sigma_m=0.8)
    assert {v.name: v.status for v in fix.verdicts}["C3_corner_NE"] == "missing"


def test_one_stray_cluster_cannot_rescue_two_missing_buoys():
    """A displaced buoy is matched to a leftover cluster only on a MUTUAL nearest basis.

    Without mutuality, the same lone cluster is handed to every unmatched buoy that
    happens to be within the search radius, and two genuinely missing buoys both come
    back "moved" — the mission would report a farm that is merely rearranged when half of
    it is gone. Mutation: replacing the mutual check with `if True` reports both as moved.
    """
    from sam_farm_inspection.localizer import Cluster, verify_against_prior
    prior = {"near": (0.0, 0.0), "far": (10.0, 0.0)}
    clusters = [Cluster(2.0, 0.0, 12.0, 0.4)]
    verdicts = {v.name: v for v in verify_against_prior(
        prior, clusters, np.eye(2), np.zeros(2), pairs={}, covered_points=[(0, 0), (10, 0)])}
    assert verdicts["near"].status == "moved", verdicts["near"]
    assert verdicts["far"].status == "missing", verdicts["far"]


# ------------------------------------------------------------------ coverage
def test_a_farm_seen_from_one_side_only_is_refused():
    """The 2022 initialization is a circumnavigation. Detections spanning 60 deg of look
    bearing are one pass, and the far edge of the farm is then pure prior wearing a
    measurement's label."""
    dets, _, _, _ = make_survey(bearings=[0, 15, 30, 45, 60])
    fix = localize_farm(dets, PRIOR, LINES, covered_points=track_all(), obs_sigma_m=0.8)
    assert fix.ok is False
    assert "coverage refusal" in fix.reason and "one side" in fix.reason


def test_coverage_is_measured_as_the_largest_GAP_not_the_count():
    """Two hundred detections from one heading are still one heading. The measure has to
    be angular, and the largest gap is the thing that says whether the circle closed."""
    ok, why = coverage_bearings_ok(
        [WorldDetection(0, 0, "buoy", 1.0, b) for b in [0, 1, 2, 3, 4] * 40])
    assert ok is False and "largest gap" in why
    ok, why = coverage_bearings_ok(
        [WorldDetection(0, 0, "buoy", 1.0, b) for b in range(0, 360, 30)])
    assert ok is True


# ------------------------------------------------------------------ line fits
def test_the_line_offset_is_measured_with_the_orientation_held_to_the_prior():
    """A pass down ONE side of a rope gives detections with no leverage on direction. The
    constrained fit still recovers the perpendicular offset; an unconstrained one would
    take its direction from noise."""
    dets, R, T, ctr = make_survey(rotate_deg=0.0, shift=(0.0, 0.0), rope_per_line=80)
    fits = fit_culture_lines([d for d in dets if d.target == "rope"], LINES,
                             np.eye(2), np.zeros(2))
    assert all(f.ok for f in fits), [(f.name, f.reason) for f in fits]
    for f in fits:
        assert abs(f.offset_m) < 0.5, f"{f.name} offset {f.offset_m:.2f} m on a centred fixture"
        assert f.rms_m < 1.5
        assert f.n_points >= 8


def test_a_line_whose_detections_run_the_wrong_way_is_refused_not_offset():
    """If the ropes are not where the prior says they run, the orientation constraint has
    stopped describing this farm. Reporting an offset anyway is a confidently wrong
    number, which is worse than none.

    Mutation: deleting the free-direction cross-check makes this return ok with an
    offset near zero.
    """
    rng = np.random.default_rng(2)
    a, b = PRIOR["C2_corner_SW"], PRIOR["C4_corner_NW"]
    mid = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
    cross = [WorldDetection(float(mid[0] + s), float(mid[1] + rng.normal(0, 0.3)), "rope")
             for s in np.linspace(-14, 14, 60)]
    fits = fit_culture_lines(cross, [("west", a, b)], np.eye(2), np.zeros(2))
    assert fits[0].ok is False
    assert "orientation constraint" in fits[0].reason


def test_each_line_refuses_on_its_own():
    """Losing the east line is not losing the farm, and a mission that can still scan one
    corridor should be told which one."""
    dets, _, _, _ = make_survey()
    west_only = [d for d in dets
                 if d.target == "rope" and d.x < PRIOR["M1_west_mid"][0] + 4.0]
    fits = {f.name: f for f in fit_culture_lines(west_only, LINES, np.eye(2), np.zeros(2))}
    assert fits["west"].ok is True
    assert fits["east"].ok is False and "rope detection" in fits["east"].reason


# ------------------------------------------------------------------ determinism
def test_the_same_survey_gives_the_same_answer():
    """A localizer whose answer moves between runs on one bag makes every disagreement
    unarguable. The seeding is deterministic on purpose."""
    dets, _, _, _ = make_survey(rotate_deg=8.0, shift=(3.0, 2.0))
    a = localize_farm(dets, PRIOR, LINES, covered_points=track_all(), obs_sigma_m=0.8)
    b = localize_farm(dets, PRIOR, LINES, covered_points=track_all(), obs_sigma_m=0.8)
    assert a.rotation_deg == pytest.approx(b.rotation_deg, abs=1e-9)
    assert a.translation_m == pytest.approx(b.translation_m, abs=1e-9)
