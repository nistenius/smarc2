"""The capture recorder's decisions — letter B, strategy §6.1, ADR-010.

The properties:

  * IT IS IDLE BETWEEN BURSTS. ADR-010: a payload node's cost is bounded and measured, and a
    recorder that quietly kept gating frames between stations would be spending the budget on
    nothing. The refusal is in the core, not in the shell's good intentions.
  * A STATION IS DONE ON A COUNT OF ACCEPTED EVIDENCE, never on a stopwatch (SETTLED §3e).
  * EITHER FRAMES OR SCANS SUFFICE. At the site's own measured clarity the R0 model puts 3.6 %
    of the red channel at the ring radius; requiring camera frames would make the mission fail
    for a reason the water decided.
  * A SCAN WITHOUT A FRESH POSE IS NOT A SCAN — it is worse than a missing one, because it
    will be trusted.
  * THINNING IS DETERMINISTIC AND BOUNDED.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_inspection_recorder_core.py
"""
import numpy as np
import pytest

from sam_target_inspection import inspection_recorder_core as R

POSE = {"x": 1.0, "y": 2.0, "z": -3.0, "yaw": 0.5}


def sharp(seed=0, shape=(64, 64)):
    return np.random.default_rng(seed).integers(0, 255, shape).astype(float)


def flat(value=128.0, shape=(64, 64)):
    return np.full(shape, float(value))


# ------------------------------------------------------------------ idleness
def test_it_refuses_frames_while_no_burst_is_open():
    r = R.InspectionRecorderCore("C1")
    assert r.busy is False
    ok, why = r.offer_frame(sharp(), t=0.0, pose=POSE)
    assert ok is False and "idle" in why and "ADR-010" in why
    assert r.frames == [], "a frame refused while idle must not enter the manifest either"


def test_it_refuses_scans_while_no_burst_is_open():
    r = R.InspectionRecorderCore("C1")
    ok, why, pts = r.offer_scan(np.zeros((500, 3)), t=0.0, pose=POSE, pose_age_s=0.1)
    assert ok is False and pts is None and "idle" in why


def test_it_goes_idle_again_after_the_burst_ends():
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    assert r.busy
    r.end_burst()
    assert r.busy is False
    assert r.offer_frame(sharp(), t=1.0, pose=POSE)[0] is False


def test_two_open_bursts_are_refused_by_name():
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    with pytest.raises(R.RecorderRefusal):
        r.start_burst(1)


def test_ending_a_burst_that_was_never_started_is_refused():
    with pytest.raises(R.RecorderRefusal):
        R.InspectionRecorderCore("C1").end_burst()


# ------------------------------------------------------------------ the frame gate
def test_a_blurred_frame_is_rejected_relative_to_the_bursts_own_median():
    """Relative, not absolute: absolute sharpness depends on the scene and on the water, and a
    fixed number would accept everything in clear water and nothing in Baltic autumn."""
    r = R.InspectionRecorderCore("C1", min_frames=3)
    r.start_burst(0)
    for i in range(4):
        assert r.offer_frame(sharp(i), t=float(i), pose=POSE)[0] is True
    ok, why = r.offer_frame(flat(), t=9.0, pose=POSE)
    assert ok is False and "blurred" in why and "median" in why


def test_a_softer_frame_in_a_LOW_CONTRAST_burst_is_still_rejected():
    """The floor must be RELATIVE. In turbid water a whole burst has low Laplacian variance;
    an absolute floor would then accept every frame in it, including the one that is ten times
    softer than its neighbours. This burst's frames all sit far above any plausible absolute
    floor, and the soft one must still be refused."""
    rng = np.random.default_rng(7)
    soft_burst = [128.0 + 8.0 * rng.standard_normal((64, 64)) for _ in range(4)]
    softest = 128.0 + 0.8 * rng.standard_normal((64, 64))
    assert R.laplacian_variance(softest) > 10.0, "the fixture must clear any absolute floor"
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    for i, f in enumerate(soft_burst):
        assert r.offer_frame(f, t=float(i), pose=POSE)[0] is True
    ok, why = r.offer_frame(softest, t=9.0, pose=POSE)
    assert ok is False and "blurred" in why


def test_an_underexposed_and_an_overexposed_frame_are_each_rejected_by_name():
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    r.offer_frame(sharp(), t=0.0, pose=POSE)
    ok, why = r.offer_frame(sharp(1) * 0.0 + 2.0, t=1.0, pose=POSE)
    assert ok is False and "underexposed" in why
    ok, why = r.offer_frame(sharp(2) * 0.0 + 254.0, t=2.0, pose=POSE)
    assert ok is False and "overexposed" in why


def test_a_frame_with_no_pose_is_refused():
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    ok, why = r.offer_frame(sharp(), t=0.0, pose=None)
    assert ok is False and "pose" in why


def test_the_laplacian_variance_falls_with_blur_and_is_zero_on_a_flat_frame():
    assert R.laplacian_variance(flat()) == pytest.approx(0.0)
    a = sharp(3)
    blurred = 0.25 * (a + np.roll(a, 1, 0) + np.roll(a, 1, 1) + np.roll(a, (1, 1), (0, 1)))
    assert R.laplacian_variance(blurred) < R.laplacian_variance(a)


def test_the_gate_uses_luma_not_a_channel_mean_on_colour_frames():
    """A mean weights blue as heavily as green, and in water the blue channel carries the least
    information (SETTLED §3f0u's transmittance)."""
    img = np.zeros((32, 32, 3))
    img[..., 2] = 200.0            # blue only
    assert R.mean_intensity(img) == pytest.approx(0.114 * 200.0)


def test_a_too_small_array_is_refused_rather_than_scored():
    with pytest.raises(R.RecorderRefusal):
        R.laplacian_variance(np.zeros((2, 2)))


# ------------------------------------------------------------------ the scan gate
def test_a_scan_with_a_stale_pose_is_refused_by_name():
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    pts = np.random.default_rng(0).normal(size=(3000, 3)) * 2.0
    ok, why, out = r.offer_scan(pts, t=0.0, pose=POSE, pose_age_s=5.0)
    assert ok is False and "old" in why and out is None


def test_a_scan_with_no_pose_is_refused_and_says_why_a_guess_would_be_worse():
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    ok, why, _ = r.offer_scan(np.zeros((3000, 3)), t=0.0, pose=None, pose_age_s=None)
    assert ok is False and "guessed pose is worse" in why


def test_a_thin_scan_is_refused_with_the_count():
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    ok, why, _ = r.offer_scan(np.random.default_rng(1).normal(size=(20, 3)) * 3,
                              t=0.0, pose=POSE, pose_age_s=0.1)
    assert ok is False and "register" in why


# ------------------------------------------------------------------ thinning
def test_voxel_thinning_keeps_one_point_per_voxel_and_is_deterministic():
    pts = np.repeat(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), 100, axis=0)
    a, note = R.voxel_thin(pts, voxel_m=0.05)
    b, _ = R.voxel_thin(pts, voxel_m=0.05)
    assert a.shape[0] == 2
    assert np.array_equal(a, b), "a scan written twice must be the same scan"
    assert "->" in note


def test_voxel_thinning_keeps_the_first_point_not_a_centroid():
    """A centroid of two returns from two different surfaces is a point on neither, and the
    station's registration would be fitting invented geometry."""
    pts = np.array([[0.01, 0.0, 0.0], [0.04, 0.0, 0.0]])
    kept, _ = R.voxel_thin(pts, voxel_m=0.05)
    assert kept.shape[0] == 1
    assert kept[0][0] == pytest.approx(0.01)


def test_the_cap_bounds_a_scan_and_says_so():
    pts = np.random.default_rng(2).normal(size=(50000, 3)) * 20.0
    kept, note = R.voxel_thin(pts, voxel_m=0.05, max_points=1000)
    assert kept.shape[0] == 1000 and "capped" in note


def test_an_empty_scan_thins_to_nothing_without_raising():
    kept, note = R.voxel_thin(np.zeros((0, 3)))
    assert kept.shape == (0, 3) and note == "empty"


def test_a_malformed_point_array_is_refused_by_name():
    with pytest.raises(R.RecorderRefusal):
        R.voxel_thin(np.zeros((10, 2)))


# ------------------------------------------------------------------ coverage
def test_a_station_is_done_on_accepted_frames_not_on_elapsed_time():
    r = R.InspectionRecorderCore("C1", min_frames=3, min_scans=99)
    r.start_burst(0)
    for i in range(2):
        r.offer_frame(sharp(i), t=float(i), pose=POSE)
    done, why = r.station_done(0)
    assert done is False and "2/3 frames" in why
    r.offer_frame(sharp(9), t=9.0, pose=POSE)
    done, why = r.station_done(0)
    assert done is True and "3 frames accepted" in why


def test_a_station_is_done_on_sonar_scans_when_the_camera_produced_nothing():
    """At Baltic visibility the camera may produce nothing usable. Requiring both would make
    the mission fail for a reason the water decided."""
    r = R.InspectionRecorderCore("C1", min_frames=6, min_scans=2)
    r.start_burst(0)
    pts = np.random.default_rng(3).normal(size=(4000, 3)) * 3.0
    for i in range(2):
        assert r.offer_scan(pts, t=float(i), pose=POSE, pose_age_s=0.1)[0] is True
    done, why = r.station_done(0)
    assert done is True and "sonar scans accepted" in why


def test_a_station_nobody_visited_is_not_done_and_says_so():
    r = R.InspectionRecorderCore("C1")
    done, why = r.station_done(7)
    assert done is False and "nothing at all" in why


def test_the_rejection_reasons_are_counted_and_reported():
    r = R.InspectionRecorderCore("C1", min_frames=99)
    r.start_burst(0)
    r.offer_frame(sharp(), t=0.0, pose=POSE)
    for i in range(3):
        r.offer_frame(flat(), t=float(i), pose=POSE)
    _, why = r.station_done(0)
    assert "blurred x3" in why


def test_the_coverage_line_counts_only_accepted_frames():
    r = R.InspectionRecorderCore("C1", min_frames=2, min_scans=99)
    r.start_burst(0)
    for i in range(3):
        r.offer_frame(sharp(i), t=float(i), pose=POSE)
    r.offer_frame(flat(), t=9.0, pose=POSE)
    line = r.coverage_line(12)
    assert "stations 1/12" in line
    assert "frames accepted 3" in line and "rejected 1" in line


def test_stations_accepted_is_what_the_ledgers_rung_two_reads():
    r = R.InspectionRecorderCore("C1", min_frames=1, min_scans=99)
    for s in range(3):
        r.start_burst(s)
        r.offer_frame(sharp(s), t=float(s), pose=POSE)
        r.end_burst()
    assert r.stations_accepted() == 3


# ------------------------------------------------------------------ manifests
def test_the_manifests_are_one_json_object_per_line_and_carry_the_verdict_and_the_pose():
    import json
    r = R.InspectionRecorderCore("C1", min_frames=1)
    r.start_burst(4)
    r.offer_frame(sharp(), t=1.5, pose=POSE, altitude_m=2.0)
    r.offer_frame(flat(), t=2.5, pose=POSE)
    r.offer_scan(np.random.default_rng(4).normal(size=(4000, 3)) * 3,
                 t=3.5, pose=POSE, pose_age_s=0.05)
    rows = [json.loads(l) for l in r.frames_jsonl().splitlines()]
    assert len(rows) == 2
    assert rows[0]["accepted"] is True and rows[1]["accepted"] is False
    assert rows[0]["station"] == 4 and rows[0]["pose"] == POSE
    assert rows[0]["candidate"] == "C1" and rows[0]["altitude_m"] == 2.0
    assert "laplacian_var" in rows[0] and "mean_intensity" in rows[0]
    srows = [json.loads(l) for l in r.scans_jsonl().splitlines()]
    assert len(srows) == 1 and srows[0]["accepted"] is True and srows[0]["n_points"] > 0


def test_rejected_frames_are_recorded_with_their_reason_not_dropped():
    """A rejected frame is a measurement of the water, and dropping it silently is how a
    coverage figure becomes a claim about the camera rather than about the site."""
    r = R.InspectionRecorderCore("C1")
    r.start_burst(0)
    r.offer_frame(sharp(), t=0.0, pose=POSE)
    r.offer_frame(flat(), t=1.0, pose=POSE)
    assert any(not f["accepted"] and f["reason"] for f in r.frames)
