"""The forward-sonar proud-object detector, driven — letter B, strategy §4b.

The synthetic clouds are ray-cast through the Sonar3D15 prefab's OWN fan (150 × 17 rays over
90° × 40°, tilt −20°), so what is being exercised is the geometry the simulator will actually
produce, not a convenient point set.

The properties:

  * IT FIRES ON THE BOX AND ON NOTHING ELSE — a ridge, a tall buoy and a rope are each refused
    by a NAMED gate;
  * THE PLANE FIT IS PRIOR-CONSTRAINED. Without the altitude and tilt priors an unconstrained
    RANSAC fits a car's ROOF and then reports a bare seabed;
  * EVERY THRESHOLD IS DERIVED from the sensor's beam geometry, and the derivations are checked
    against the strategy's own quoted numbers;
  * THE HEIGHT IS MEASURED PERPENDICULAR TO THE FITTED PLANE, so a sloping seabed does not turn
    every flat patch into a proud object of exactly the slope's height;
  * PERSISTENCE IS TESTED IN THE WORLD FRAME — a real object sits still while the vehicle moves.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_fls_target_core.py
"""
import math

import numpy as np
import pytest

from sam_target_inspection import fls_target_core as F
from synthetic import MINI_H, MINI_L, MINI_W, cloud

CAR = (7.0, 0.0, MINI_L, MINI_W, MINI_H)
CFG = F.FlsConfig(altitude_m=4.0)


# ------------------------------------------------------------------ fires on the box only
def test_it_finds_a_car_sized_box_on_the_seabed():
    r = F.detect_cloud(cloud((CAR,)), CFG)
    assert r.ok and len(r.boxes) == 1, r.reason
    b = r.boxes[0]
    assert F.FOOTPRINT_MIN_M <= b.footprint_m <= F.FOOTPRINT_MAX_M
    assert b.height_m == pytest.approx(MINI_H, abs=0.25)
    assert math.hypot(b.centroid[0] - 7.0, b.centroid[1]) < 2.0


def test_a_bare_seabed_produces_nothing_and_says_it_is_bare():
    r = F.detect_cloud(cloud(()), CFG)
    assert r.ok and r.boxes == ()
    assert "bare" in r.reason
    assert r.plane is not None and r.plane.n_inliers > 1000


def test_a_ridge_fails_the_footprint_bound_by_name():
    r = F.detect_cloud(cloud(((8.0, 0.0, 14.0, 2.0, 1.0),)), CFG)
    assert r.boxes == ()
    assert r.n_killed_footprint >= 1
    assert "footprint" in r.reason


def test_a_tall_buoy_fails_a_size_bound_by_name():
    r = F.detect_cloud(cloud(((7.0, 0.0, 0.6, 0.6, 3.0),)), CFG)
    assert r.boxes == ()
    assert r.n_killed_footprint + r.n_killed_height >= 1


def test_an_object_too_TALL_is_refused_by_the_height_band_alone():
    """A 2 m x 2 m object standing 3 m proud: its footprint is comfortably inside the band, so
    the only gate that can refuse it is the height one, and the report must say so."""
    r = F.detect_cloud(cloud(((7.0, 0.0, 2.0, 2.0, 3.0),)), CFG)
    assert r.boxes == (), r.reason
    assert r.n_killed_height >= 1 and r.n_killed_footprint == 0
    assert "height band" in r.reason


def test_an_object_too_SHORT_is_refused_by_the_height_band():
    r = F.detect_cloud(cloud(((7.0, 0.0, 3.0, 1.5, 0.25),)), CFG)
    assert r.boxes == (), r.reason


def test_a_rope_produces_nothing():
    r = F.detect_cloud(cloud(((7.0, 0.0, 10.0, 0.06, 0.06),)), CFG)
    assert r.boxes == ()


def test_a_car_beside_a_ridge_yields_the_car_only():
    r = F.detect_cloud(cloud(((7.0, -6.0, MINI_L, MINI_W, MINI_H),
                              (9.0, 6.0, 14.0, 2.0, 1.0))), CFG)
    assert len(r.boxes) == 1, r.reason
    assert r.boxes[0].centroid[1] < 0, "the surviving box must be the CAR, not the ridge"
    assert r.n_killed_footprint >= 1


def test_the_car_is_one_cluster_and_not_several():
    """THE MEASUREMENT THAT CHANGED THE CLUSTERING. With the link distance taken from the
    datasheet beam width alone, the sim's 2.5° elevation ray spacing broke this exact car into
    three clusters of 96, 20 and 18 points, and a 14 m ridge into seven — so the footprint band,
    whose only job is to tell a car from a ridge, was being applied to fragments of both."""
    r = F.detect_cloud(cloud((CAR,)), CFG)
    assert r.n_clusters == 1, f"the car came back as {r.n_clusters} clusters"


# ------------------------------------------------------------------ derived thresholds
def test_the_beam_footprint_is_two_r_tan_half_beam():
    assert F.beam_footprint_m(10.0, 0.85) == pytest.approx(2 * 10.0 * math.tan(math.radians(0.85) / 2))
    assert F.beam_footprint_m(20.0, 0.85) == pytest.approx(2 * F.beam_footprint_m(10.0, 0.85))


def test_the_height_gate_reproduces_the_strategys_thirty_centimetres_at_seven_metres():
    """Strategy §4b quotes 0.3 m. Its own justification ("~5 × the 1.6° beam's 0.2 m footprint")
    does not give 0.3 — 7 m × tan 1.60° is 0.196 m, so 0.3 m is 1.53 ×, not 5 ×. The number
    survives by a different route: beam footprint plus three sigma of a good plane fit. This
    test pins the ROUTE, so the gate scales with range instead of being right at exactly 7 m."""
    got = F.min_height_above_plane_m(7.0, plane_rms_m=0.03)
    assert got == pytest.approx(0.29, abs=0.02)
    assert F.min_height_above_plane_m(14.0, 0.03) > 2 * F.beam_footprint_m(7.0, F.NAV_BEAM_EL_DEG)
    # a worse plane fit demands more height, which is the whole reason the residual is a term
    assert F.min_height_above_plane_m(7.0, 0.20) > F.min_height_above_plane_m(7.0, 0.03)


def test_the_footprint_floor_reproduces_the_strategys_six_beams_at_ten_metres():
    assert F.min_footprint_m(10.0) == pytest.approx(0.89, abs=0.02)
    assert F.min_footprint_m(10.0) == pytest.approx(6 * F.beam_footprint_m(10.0, F.NAV_BEAM_AZ_DEG))


def test_the_cluster_link_is_two_sample_steps_and_the_step_is_the_coarser_of_two():
    assert F.sample_step_deg(90.0, 150, 0.85) == pytest.approx(0.85), \
        "150 beams over 90 deg is 0.60 deg apart, finer than the 0.85 deg beam — the BEAM wins"
    assert F.sample_step_deg(40.0, 17, 1.60) == pytest.approx(2.5), \
        "17 rays over 40 deg is 2.5 deg apart, coarser than the 1.6 deg beam — SAMPLING wins"
    assert F.cluster_link_m(10.0, 0.85) == pytest.approx(0.30, abs=0.01), \
        "strategy §4b's 0.3 m, derived"


def test_the_grazing_stretch_is_one_at_nadir_and_grows_with_range():
    assert F.grazing_stretch(4.0, 4.0) == pytest.approx(1.0)
    assert F.grazing_stretch(12.0, 4.0) == pytest.approx(3.0)
    assert F.grazing_stretch(100.0, 4.0) == pytest.approx(6.0), "clamped; a smear is not a surface"
    assert F.grazing_stretch(7.0, 0.0) == pytest.approx(1.0), "no altitude, no stretch claim"


# ------------------------------------------------------------------ the plane
def test_the_plane_fit_refuses_a_cloud_with_too_few_returns():
    p, why = F.fit_seabed_plane(np.zeros((5, 3)), altitude_m=4.0)
    assert p is None and "too few" in why


def test_the_plane_fit_refuses_a_non_ne3_array_by_name():
    p, why = F.fit_seabed_plane(np.zeros((10, 2)), altitude_m=4.0)
    assert p is None and "(N,3)" in why


def test_the_altitude_prior_stops_a_car_roof_being_fitted_as_the_seabed():
    """Points from a car roof ONLY, 1.28 m above a seabed the altimeter puts at 4 m. Without
    the prior an unconstrained fit accepts the roof, and everything standing on the real seabed
    then reads as BELOW it — i.e. a detector that reports nothing while looking straight at a
    car."""
    rng = np.random.default_rng(0)
    roof = np.stack([6.0 + 3.0 * rng.random(400), -0.7 + 1.4 * rng.random(400),
                     np.full(400, -2.72) + 0.01 * rng.standard_normal(400)], axis=1)
    p, why = F.fit_seabed_plane(roof, altitude_m=4.0, altitude_tol_m=0.5)
    assert p is None, "a plane 1.28 m off the altimeter's answer was accepted as the seabed"
    assert "altimeter" in why
    # MEASURED while writing this test, and it is worth stating: on a roof-only cloud the
    # RANSAC LOOP does find a plane the offset prior admits — a TILTED one whose intercept
    # happens to land near 4 m — and it is the post-refit re-check that refuses. So this
    # refusal is the refit's, and the loop's own offset check is isolated in the test below.
    assert "the refined plane sits at" in why, why
    # ... and with a tolerance loose enough to admit it, it IS admitted — so the test above is
    # measuring the prior and not some other refusal.
    p2, _ = F.fit_seabed_plane(roof, altitude_m=4.0, altitude_tol_m=2.0)
    assert p2 is not None


def test_the_ransac_loops_own_offset_prior_refuses_when_no_tilted_dodge_exists():
    """The loop's offset check, isolated. With the tilt prior tightened to 3° the "a steeply
    tilted plane through the roof happens to intercept near the altimeter's depth" escape is
    unavailable, so it is the LOOP that must refuse — and its sentence names how many draws it
    rejected for standing off the altimeter."""
    rng = np.random.default_rng(0)
    roof = np.stack([6.0 + 3.0 * rng.random(400), -0.7 + 1.4 * rng.random(400),
                     np.full(400, -2.72) + 0.01 * rng.standard_normal(400)], axis=1)
    p, why = F.fit_seabed_plane(roof, altitude_m=4.0, altitude_tol_m=0.5, max_tilt_deg=3.0)
    assert p is None
    assert "RANSAC draws satisfied the priors" in why, why
    assert "standing off the altimeter" in why


def test_the_tilt_prior_refuses_a_wall():
    """A vertical plane is not a seabed. Points on a wall at x = 8 m."""
    rng = np.random.default_rng(1)
    wall = np.stack([np.full(300, 8.0) + 0.01 * rng.standard_normal(300),
                     -3.0 + 6.0 * rng.random(300),
                     -4.0 + 4.0 * rng.random(300)], axis=1)
    p, why = F.fit_seabed_plane(wall, altitude_m=4.0)
    assert p is None
    assert "RANSAC draws satisfied the priors" in why, why
    assert "rejected for tilt" in why, why


def test_the_fitted_plane_is_refit_by_least_squares_so_its_residual_is_real():
    """RANSAC picks WHICH points, not the best plane through them: a plane through three points
    has an rms of exactly zero, which would collapse the height gate to the beam term alone."""
    p, why = F.fit_seabed_plane(cloud(()), altitude_m=4.0)
    assert p is not None, why
    assert p.rms_m > 0.0
    assert p.rms_m < 0.10
    assert p.tilt_deg < 5.0


def test_the_plane_normal_points_up():
    p, _ = F.fit_seabed_plane(cloud(()), altitude_m=4.0)
    assert p.normal[2] > 0.9


def test_a_sloping_seabed_does_not_become_a_field_of_proud_objects():
    """Height is measured PERPENDICULAR TO THE FITTED PLANE. Measured along body-z, a 10° slope
    over a 10 m strip would present 1.7 m of "height" and every patch of it would be a car."""
    pts = cloud(())
    slope = math.radians(10.0)
    tilted = pts.copy()
    tilted[:, 2] = pts[:, 2] + pts[:, 0] * math.tan(slope)
    r = F.detect_cloud(tilted, F.FlsConfig(altitude_m=4.0, pitch_rad=0.0))
    assert r.boxes == (), r.reason
    # The sharp assertion: on a bare slope essentially NOTHING may stand above the fitted
    # plane. Measuring height along body-z instead would put the far end of a 10 deg slope
    # 1.7 m "above" the seabed and every patch of it would be a candidate.
    assert r.n_above < 0.02 * r.n_points, f"{r.n_above} of {r.n_points} returns above a slope"
    assert "bare" in r.reason


# ------------------------------------------------------------------ clustering
def test_the_link_cluster_accepts_a_per_point_link_distance():
    pts = np.array([[0., 0., 0.], [0.5, 0., 0.], [3.0, 0., 0.], [3.4, 0., 0.]])
    tight = F.link_cluster(pts, 0.6, min_points=2)
    assert len(tight) == 2
    loose = F.link_cluster(pts, np.array([0.6, 0.6, 3.0, 3.0]), min_points=2)
    assert len(loose) == 1, "the larger of the two link distances must decide an edge"


def test_an_empty_point_set_clusters_to_nothing_without_raising():
    assert F.link_cluster(np.zeros((0, 3)), 0.3) == []


# ------------------------------------------------------------------ box fitting
def test_the_box_fit_recovers_a_known_rectangle():
    rng = np.random.default_rng(2)
    plane = F.Plane((0.0, 0.0, 1.0), 4.0, 0.02, 100, 100)
    x = 6.0 + 3.0 * rng.random(500)
    y = -0.7 + 1.4 * rng.random(500)
    z = np.full(500, -2.8)
    bf = F.box_fit(np.stack([x, y, z], axis=1), plane)
    assert bf.length_m == pytest.approx(3.0, abs=0.3)
    assert bf.width_m == pytest.approx(1.4, abs=0.3)
    assert bf.height_m == pytest.approx(1.2, abs=0.1)
    assert bf.length_m >= bf.width_m


# ------------------------------------------------------------------ persistence
def test_three_consecutive_clouds_make_one_candidate():
    tr = F.FlsTargetTracker(persistence_pings=3)
    ripe = []
    for i in range(4):
        # the vehicle closes on the target at 0.5 m per ping; the world position is unchanged
        r = F.detect_cloud(cloud(((7.0 - 0.5 * i, 0.0, MINI_L, MINI_W, MINI_H),)), CFG)
        ripe += tr.update(i, r, (0.5 * i, 0.0), 0.0)
    assert len(ripe) == 1
    assert ripe[0].n_pings >= 3


def test_the_persistence_gate_is_derived_from_the_sonars_own_rate():
    assert F.PERSISTENCE_PINGS == F.persistence_pings_for()
    assert F.persistence_pings_for(ping_hz=20.0) > F.persistence_pings_for(ping_hz=5.0)
    assert F.PERSISTENCE_PINGS == 3


def test_the_default_tracker_uses_the_derived_persistence():
    tr = F.FlsTargetTracker()
    assert tr.persistence_pings == F.PERSISTENCE_PINGS
    ripe = []
    for i in range(F.PERSISTENCE_PINGS - 1):
        r = F.detect_cloud(cloud(((7.0 - 0.5 * i, 0.0, MINI_L, MINI_W, MINI_H),)), CFG)
        ripe += tr.update(i, r, (0.5 * i, 0.0), 0.0)
    assert ripe == [], "the default tracker fired before its own derived persistence"


def test_a_single_cloud_cannot_make_a_candidate():
    tr = F.FlsTargetTracker(persistence_pings=3)
    r = F.detect_cloud(cloud((CAR,)), CFG)
    assert tr.update(0, r, (0.0, 0.0), 0.0) == []


def test_a_detection_that_does_not_hold_still_in_the_world_is_not_a_track():
    """A real object sits still while the vehicle moves. A body-frame gate cannot tell that from
    speckle; a world-frame gate can, and it has no free parameter."""
    tr = F.FlsTargetTracker(persistence_pings=3, gate_m=1.0)
    ripe = []
    for i in range(5):
        # the same BODY position every ping while the vehicle travels: in the world this thing
        # is running away from us at the vehicle's own speed, so it is not an object.
        r = F.detect_cloud(cloud((CAR,)), CFG)
        ripe += tr.update(i, r, (3.0 * i, 0.0), 0.0)
    assert ripe == []


def test_the_world_transform_rotates_with_the_course():
    assert F.FlsTargetTracker.to_world((10.0, 0.0), (0.0, 0.0), 0.0) == pytest.approx((10.0, 0.0))
    assert F.FlsTargetTracker.to_world((10.0, 0.0), (0.0, 0.0), 90.0) == pytest.approx((0.0, 10.0), abs=1e-9)


def test_the_candidate_record_scores_against_the_band_not_the_exact_dimensions():
    """A forward sonar on one pass sees the near face and part of the roof: at 7 m and the sim's
    2.5° elevation sampling the along-track extent recovered from a 3.08 m car is 1–2 m. Scoring
    that against 3.08 would rank a correct single-aspect detection as a bad one."""
    tr = F.FlsTargetTracker(persistence_pings=3)
    ripe = []
    for i in range(4):
        r = F.detect_cloud(cloud(((7.0 - 0.5 * i, 0.0, MINI_L, MINI_W, MINI_H),)), CFG)
        ripe += tr.update(i, r, (0.5 * i, 0.0), 0.0)
    rec = F.candidate_from_track(ripe[0], cid="F1", t=10.0, sigma_m=0.4)
    assert rec["sensor"] == "fls"
    assert rec["score"] == pytest.approx(0.0, abs=0.01), rec
    assert "confidence" not in rec
    assert rec["n_pings"] >= 3
    assert rec["bbox_m"]["width"] > 0


def test_the_band_distance_is_zero_inside_and_grows_outside():
    assert F._outside_band(2.0, 0.8, 8.0) == 0.0
    assert F._outside_band(0.4, 0.8, 8.0) > 0.0
    assert F._outside_band(20.0, 0.8, 8.0) > F._outside_band(9.0, 0.8, 8.0)


# ------------------------------------------------------------------ refusals
def test_a_malformed_array_refuses_by_name():
    r = F.detect_cloud(np.zeros((10, 2)), CFG)
    assert not r.ok and "(N,3)" in r.reason


def test_returns_inside_the_near_field_are_dropped_at_the_detectors_own_range_min():
    """0.8 m, the same number and the same reason as `obstacle_detector`'s `range_min`:
    a surfaced hull sees itself (SETTLED §3u)."""
    pts = np.concatenate([cloud(()), np.zeros((50, 3)) + 0.1])
    r = F.detect_cloud(pts, CFG)
    assert r.ok, r.reason
    assert r.n_points <= cloud(()).shape[0] + 1
