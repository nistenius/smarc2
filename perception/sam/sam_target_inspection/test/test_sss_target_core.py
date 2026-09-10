"""The side-scan car detector, driven — letter B of the 2026-09-09 work order.

The properties, and why each is a guard rather than a decoration:

  * HIGHLIGHT AND SHADOW ARE REQUIRED TOGETHER. That is the whole discriminating power of the
    detector: bedrock reads DARKER than sand in the real record (SETTLED §3f0h), so brightness
    alone discriminates nothing. Two tests remove one half each and require silence.
  * THE THRESHOLD IS DERIVED FROM A FALSE-ALARM BUDGET, and the budget is checked by counting
    false alarms on noise-only pings. A hand-picked threshold is what gave the first farm
    detector hundreds of detections per noise ping (SETTLED §3k).
  * THE SHADOW LENGTH IS GEOMETRY. `H·r/h` — so a target twice as far casts a shadow twice as
    long, and a fixed length in metres would be right at exactly one range.
  * LEVELS ARE MEASURED AGAINST A LOCAL BACKGROUND, not against the smoothed envelope. The
    object is inside its own smoothing window; measured, that costs 2.4 dB of a 9.2 dB
    highlight and two thirds of the shadow.
  * THE CENTRE IS REPORTED, NOT THE EDGE (SETTLED §3k).
  * SIGMA IS DECOMPOSED. A sigma that cannot be taken apart cannot be checked (§3s7).

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_sss_target_core.py
"""
import math

import numpy as np
import pytest

from sam_target_inspection import sss_target_core as S
from synthetic import (ACROSS_M, HIGHLIGHT_DB, MINI_H, RANGE_RES_M, SHADOW_DB, SPECKLE_CV,
                       gain_curve, ping)

CFG = S.SssConfig(range_res_m=RANGE_RES_M, altitude_m=6.0)


# ------------------------------------------------------------------ the signature
def test_it_fires_on_the_measured_ideal_fan_signature():
    r = S.detect_ping(ping(seed=0), CFG)
    assert r.ok and len(r.hits) == 1, r.reason
    h = r.hits[0]
    assert h.highlight_db == pytest.approx(HIGHLIGHT_DB, abs=1.5)
    assert h.shadow_db == pytest.approx(SHADOW_DB, abs=1.5)
    assert h.across_m == pytest.approx(ACROSS_M, abs=0.2)


def test_one_object_produces_exactly_one_hit():
    """A real target raises the discrepancy at every window position that straddles either of
    its edges — 52 of them at 4 cm bins, measured. Reporting each would let a single ping count
    as fifty and make the persistence gate meaningless."""
    for seed in range(6):
        r = S.detect_ping(ping(seed=seed), CFG)
        assert len(r.hits) == 1, f"seed {seed}: {len(r.hits)} hits"


def test_a_highlight_with_no_shadow_is_bottom_texture_and_is_refused():
    y = ping(seed=1, shadow_len_scale=0.0)
    r = S.detect_ping(y, CFG)
    assert r.hits == ()
    assert "shadow" in r.reason


def test_a_shadow_with_no_highlight_is_not_a_target():
    """A dark run on its own is a hole or the display's own nadir gap, not a proud object."""
    y = ping(seed=2, target_ground_m=None)
    env = gain_curve(y.size, CFG.altitude_m)
    b0 = int(math.hypot(25.0, 6.0) / RANGE_RES_M)
    n_sh = int((MINI_H * 25.0 / 6.0) / RANGE_RES_M)
    y[b0:b0 + n_sh] = env[b0:b0 + n_sh] * 10 ** (SHADOW_DB / 20.0)
    assert S.detect_ping(y, CFG).hits == ()


def test_a_dim_highlight_with_a_shadow_is_refused_by_the_contrast_gate():
    """A highlight bright enough to raise a change point but below the +6 dB gate, WITH a
    correctly-sized shadow — so the only thing that can refuse it is the contrast gate, and
    the report must say that is what refused it."""
    r = S.detect_ping(ping(seed=3, highlight_db=4.0), CFG)
    assert r.hits == ()
    assert r.n_rises > 0, "the fixture must reach the change-point stage or it tests nothing"
    assert "dimmer than 6 dB" in r.reason


def test_an_object_too_wide_across_track_is_a_ridge_not_a_car():
    r = S.detect_ping(ping(seed=4, across_m=6.0), CFG)
    assert r.hits == ()
    assert "across-track" in r.reason or "shadow" in r.reason


def test_an_object_too_narrow_across_track_is_refused():
    r = S.detect_ping(ping(seed=5, across_m=0.2), CFG)
    assert r.hits == ()


# ------------------------------------------------------------------ the budget
def test_the_false_alarm_count_on_noise_matches_the_declared_budget():
    """The threshold is DERIVED from `false_alarms_per_ping`; this measures the result.

    Two-sided: a detector that fires on nothing is as suspect as one that fires on everything,
    so the SAME fixture is checked to fire on the car above. Over 300 noise pings at a budget of
    0.05/ping the change-point stage alone would admit ~15 rises; the signature gates take that
    to zero, which is the number recorded here."""
    n_hits = 0
    n_rises = 0
    for s in range(300):
        r = S.detect_ping(ping(seed=2000 + s, target_ground_m=None), CFG)
        n_hits += len(r.hits)
        n_rises += r.n_rises
    assert n_hits == 0, f"{n_hits} false alarms over 300 noise pings"
    assert n_rises > 0, ("not one rise passed the change-point threshold over 300 noise pings — "
                         "the threshold is not being derived from the budget at all, it is just "
                         "very high, and the detector would be silent on a real target too")


def test_raising_the_budget_admits_more_rises():
    """The budget is a knob with a measurable effect, which is what makes it a budget."""
    tight = sum(S.detect_ping(ping(seed=3000 + s, target_ground_m=None),
                              S.SssConfig(range_res_m=RANGE_RES_M, altitude_m=6.0,
                                          false_alarms_per_ping=0.001)).n_rises
                for s in range(40))
    loose = sum(S.detect_ping(ping(seed=3000 + s, target_ground_m=None),
                              S.SssConfig(range_res_m=RANGE_RES_M, altitude_m=6.0,
                                          false_alarms_per_ping=5.0)).n_rises
                for s in range(40))
    assert loose > tight


# ------------------------------------------------------------------ geometry
def test_the_shadow_length_is_geometry_and_scales_with_range():
    assert S.expected_shadow_len_m(25.0, 6.0, 1.278) == pytest.approx(1.278 * 25.0 / 6.0)
    assert S.expected_shadow_len_m(50.0, 6.0, 1.278) == pytest.approx(
        2.0 * S.expected_shadow_len_m(25.0, 6.0, 1.278))
    assert S.expected_shadow_len_m(25.0, 3.0, 1.278) > S.expected_shadow_len_m(25.0, 6.0, 1.278)


def test_a_zero_altitude_refuses_rather_than_dividing_by_zero():
    with pytest.raises(S.SssRefusal):
        S.expected_shadow_len_m(25.0, 0.0, 1.278)


def test_it_fires_at_two_different_ranges_with_the_right_shadow_each_time():
    """The same object at 15 m and at 35 m casts shadows of 3.2 m and 7.5 m. A detector with a
    fixed shadow length would find one of them."""
    for gr in (15.0, 30.0):
        r = S.detect_ping(ping(seed=7, target_ground_m=gr), CFG)
        assert len(r.hits) == 1, f"{gr} m: {r.reason}"
        h = r.hits[0]
        assert h.ground_range_m == pytest.approx(gr, abs=1.5)
        assert h.shadow_len_expected_m == pytest.approx(MINI_H * gr / 6.0, rel=0.1)


def test_the_reported_bin_is_the_CENTRE_of_the_highlight_and_not_its_leading_edge():
    """A change point marks an EDGE, so reporting it as the object's range is biased by half
    the object's width plus the window (SETTLED §3k). The hit carries its own measured extent,
    so the property is checkable without a fixture constant."""
    r = S.detect_ping(ping(seed=8, across_m=1.9), CFG)
    assert len(r.hits) == 1, r.reason
    h = r.hits[0]
    assert h.centre_bin == (h.start_bin + h.end_bin - 1) // 2
    assert h.start_bin < h.centre_bin < h.end_bin
    # ... and the centre is at least a third of the object's width past its leading edge
    assert (h.centre_bin - h.start_bin) * RANGE_RES_M > 0.3 * h.across_m


def test_the_background_guard_puts_the_window_ENTIRELY_off_the_object():
    """With a short window the guard's SIGN is load-bearing: a window that reaches past the
    object's leading edge is measuring the object."""
    y = np.concatenate([np.full(200, 10.0), np.full(200, 100.0)])
    bg, src = S.local_background(y, 200, 400, guard_bins=8, window_bins=16)
    assert src == "before"
    assert bg == pytest.approx(10.0),         "the background window overlapped the object it is the background for"


def test_the_persistence_gate_is_derived_from_the_objects_own_along_track_extent():
    """Not hand-picked. The Mini is 3.078 m; at 18.55 Hz and 0.7 m/s it occupies 82 pings, so
    half a pass is 41 and the shipped gate is that number."""
    assert S.PERSISTENCE_PINGS == S.persistence_pings_for()
    assert S.persistence_pings_for(speed_ms=0.5) > S.persistence_pings_for(speed_ms=0.7)
    assert S.persistence_pings_for(length_m=6.0) == 2 * S.persistence_pings_for(length_m=3.0)
    assert 35 <= S.PERSISTENCE_PINGS <= 55, S.PERSISTENCE_PINGS


def test_the_default_tracker_uses_the_derived_persistence():
    tr = S.SssTargetTracker()
    assert tr.persistence_pings == S.PERSISTENCE_PINGS
    ripe = []
    for i in range(S.PERSISTENCE_PINGS - 1):
        ripe += tr.update(i, "starboard", S.detect_ping(ping(seed=300 + i), CFG))
    assert ripe == [], "the default tracker fired before its own derived persistence"
    ripe += tr.update(S.PERSISTENCE_PINGS - 1, "starboard",
                      S.detect_ping(ping(seed=400), CFG))
    assert len(ripe) == 1


def test_a_shadow_that_runs_off_the_end_of_the_record_is_counted_separately():
    """MEASURED while writing this suite. At the record's own geometry (40 m range, 4 cm bins)
    a 1.28 m target at 35 m ground range under 6 m altitude casts a 7.5 m shadow that ends 113
    bins past the last bin. "The recorder's range was too short" is not "there was no shadow",
    and blaming the target for the range setting is how a detector gets tuned in the wrong
    direction."""
    r = S.detect_ping(ping(seed=7, target_ground_m=35.0), CFG)
    assert r.hits == ()
    assert r.n_killed_truncated >= 1 and r.n_killed_no_shadow == 0
    assert "RECORD ENDED" in r.reason
    # ... and with a longer record the same target is found.
    long_cfg = S.SssConfig(range_res_m=RANGE_RES_M, altitude_m=6.0)
    r2 = S.detect_ping(ping(seed=7, target_ground_m=35.0, n_bins=1400), long_cfg)
    assert len(r2.hits) == 1, r2.reason


def test_nothing_inside_the_nadir_gap_gets_a_ground_range():
    assert S.slant_to_ground_m(3.0, 6.0) is None
    assert S.slant_to_ground_m(10.0, 6.0) == pytest.approx(8.0)
    assert S.nadir_gap_m(6.0) == pytest.approx(6.9)


# ------------------------------------------------------------------ the background
def test_the_local_background_is_taken_beside_the_object_not_through_it():
    """THE MEASUREMENT THAT CHANGED THE DESIGN. A ±64-bin envelope is dragged up by a 39-bin
    highlight and down inside a 133-bin shadow, so the measured +9.2 dB reads +6.8 and the
    shadow run terminates at 52 of 133 bins. The background must come from beside."""
    y = ping(seed=11)
    env = S.gain_envelope(y, S.ENVELOPE_HALF_BINS)
    b0 = int(math.hypot(25.0, 6.0) / RANGE_RES_M)
    peak = b0 + 10
    through = S.db_over(float(y[peak]), float(env[peak]))
    bg, src = S.local_background(y, b0, b0 + 200, CFG.background_guard_bins,
                                 2 * CFG.envelope_half_bins)
    beside = S.db_over(float(y[peak]), bg)
    assert src == "before"
    assert beside > through + 1.0
    assert beside == pytest.approx(HIGHLIGHT_DB, abs=1.5)


def test_the_background_falls_back_to_the_far_side_and_says_so():
    y = ping(seed=12)
    _, src = S.local_background(y, 5, 400, 8, 128)
    assert "nadir" in src, "a fallback background must announce that it is one"


# ------------------------------------------------------------------ refusals
def test_an_empty_channel_refuses_by_name():
    r = S.detect_ping(np.zeros(0), CFG)
    assert not r.ok and "empty" in r.reason


def test_a_non_positive_range_resolution_refuses_by_name():
    r = S.detect_ping(ping(), S.SssConfig(range_res_m=0.0, altitude_m=6.0))
    assert not r.ok and "resolution" in r.reason


def test_an_altitude_past_the_end_of_the_ping_refuses_rather_than_reporting_a_bare_seabed():
    """"I could not tell where the seabed starts" and "there is nothing on this seabed" are
    different answers and only one of them is information (SETTLED §3e)."""
    r = S.detect_ping(ping(n_bins=200), S.SssConfig(range_res_m=RANGE_RES_M, altitude_m=50.0))
    assert not r.ok and "seabed" in r.reason


def test_a_quiet_ping_is_ok_with_a_reason_not_a_failure():
    r = S.detect_ping(ping(seed=13, target_ground_m=None), CFG)
    assert r.ok is True, "a featureless seabed is an ANSWER, not a failure"
    assert r.hits == () and r.reason


# ------------------------------------------------------------------ the score
def test_the_score_is_zero_at_the_measured_signature_and_grows_away_from_it():
    assert S.signature_score(HIGHLIGHT_DB, SHADOW_DB, ACROSS_M) == pytest.approx(0.0)
    assert S.signature_score(3.0, -2.0, 3.0) > 1.0


def test_the_score_is_one_sided_on_contrast():
    """A brighter highlight and a darker shadow are MORE car-like, not less. Penalising them
    would rank the strongest evidence worst."""
    assert S.signature_score(HIGHLIGHT_DB + 6, SHADOW_DB, ACROSS_M) == pytest.approx(0.0)
    assert S.signature_score(HIGHLIGHT_DB, SHADOW_DB - 6, ACROSS_M) == pytest.approx(0.0)
    assert S.signature_score(HIGHLIGHT_DB - 6, SHADOW_DB, ACROSS_M) > 0.0


def test_nothing_maps_the_score_into_zero_to_one():
    """A number in [0,1] invites a consumer to multiply it by something. This one is a distance
    and may exceed 1."""
    assert S.signature_score(0.0, 0.0, 8.0) > 1.0


# ------------------------------------------------------------------ persistence
def _feed(tracker, n, side="starboard", make=lambda i: ping(seed=100 + i)):
    ripe = []
    for i in range(n):
        ripe += tracker.update(i, side, S.detect_ping(make(i), CFG))
    return ripe


def test_a_single_bright_ping_cannot_make_a_candidate():
    tr = S.SssTargetTracker(persistence_pings=40)
    ripe = tr.update(0, "starboard", S.detect_ping(ping(seed=0), CFG))
    assert ripe == []


def test_forty_consecutive_pings_make_one_candidate_and_not_two():
    tr = S.SssTargetTracker(persistence_pings=40)
    ripe = _feed(tr, 45)
    assert len(ripe) == 1
    assert ripe[0].n_pings if hasattr(ripe[0], "n_pings") else len(ripe[0].hits) >= 40


def test_a_track_that_stops_being_seen_is_closed_and_does_not_resurrect():
    tr = S.SssTargetTracker(persistence_pings=40, max_gap_pings=3)
    _feed(tr, 20)
    for i in range(20, 30):
        tr.update(i, "starboard", S.detect_ping(ping(seed=1, target_ground_m=None), CFG))
    ripe = []
    for i in range(30, 60):
        ripe += tr.update(i, "starboard", S.detect_ping(ping(seed=100 + i), CFG))
    assert ripe == [], "a gap of ten pings must not be bridged into one 40-ping track"


def test_port_and_starboard_tracks_do_not_merge():
    tr = S.SssTargetTracker(persistence_pings=5)
    for i in range(4):
        tr.update(i, "port", S.detect_ping(ping(seed=200 + i), CFG))
    ripe = tr.update(4, "starboard", S.detect_ping(ping(seed=204), CFG))
    assert ripe == []


# ------------------------------------------------------------------ sigma and georeference
def test_the_sigma_is_decomposed_and_the_parts_sum_in_quadrature():
    sig = S.sigma_for(1.5, RANGE_RES_M, 0.4, 25.0, 6.0)
    d = sig.as_dict()
    assert set(d) == {"dr_m", "bin_quantisation_m", "along_track_m", "slant_geometry_m", "total_m"}
    assert d["total_m"] == pytest.approx(
        math.sqrt(sum(d[k] ** 2 for k in
                      ("dr_m", "bin_quantisation_m", "along_track_m", "slant_geometry_m"))))
    assert d["total_m"] >= d["dr_m"]


def test_the_bin_quantisation_grows_at_grazing_incidence():
    """At grazing incidence a slant bin covers far more ground than at nadir. Quoting the slant
    bin would understate the error exactly where the detector works."""
    near = S.sigma_for(0.0, RANGE_RES_M, 0.0, 8.0, 6.0).bin_quantisation_m
    far = S.sigma_for(0.0, RANGE_RES_M, 0.0, 40.0, 6.0).bin_quantisation_m
    assert near > far


def test_a_candidate_in_the_nadir_gap_is_refused_rather_than_placed_at_the_vehicle():
    with pytest.raises(S.SssRefusal):
        S.sigma_for(1.0, RANGE_RES_M, 0.4, 0.0, 6.0)


def test_the_georeference_uses_course_and_refuses_without_it():
    b = int(math.hypot(25.0, 6.0) / RANGE_RES_M)
    la, lo, why = S.georeference(b, "starboard", CFG, lat=58.82, lon=17.63, course_deg=None)
    assert (la, lo) == (None, None) and "course" in why
    la, lo, why = S.georeference(b, "starboard", CFG, lat=None, lon=None, course_deg=0.0)
    assert (la, lo) == (None, None) and "position" in why


def test_starboard_and_port_land_on_opposite_sides_of_the_track():
    b = int(math.hypot(25.0, 6.0) / RANGE_RES_M)
    s_lat, s_lon, _ = S.georeference(b, "starboard", CFG, lat=58.82, lon=17.63, course_deg=0.0)
    p_lat, p_lon, _ = S.georeference(b, "port", CFG, lat=58.82, lon=17.63, course_deg=0.0)
    # course 0 = north, so starboard is EAST and port is WEST, and both at the same latitude.
    assert s_lon > 17.63 > p_lon
    assert s_lat == pytest.approx(p_lat, abs=1e-9)


def test_the_candidate_record_carries_the_centre_and_the_decomposed_sigma():
    tr = S.SssTargetTracker(persistence_pings=40)
    ripe = _feed(tr, 45)
    c = S.candidate_from_track(ripe[0], CFG, cid="C1", t=123.0, lat=58.82, lon=17.63,
                               course_deg=0.0, dr_since_fix_m=1.5, along_track_m=0.5)
    d = c.as_dict()
    assert d["sensor"] == "sss" and d["id"] == "C1"
    assert d["sigma_parts"]["total_m"] == pytest.approx(c.sigma_m)
    assert d["n_pings"] >= 40
    assert 0.0 <= d["score"] < 2.0
    # the reported bin is the CENTRE of the highlight: its ground range must be the target's
    assert S.slant_to_ground_m((d["bin"] + 0.5) * RANGE_RES_M, 6.0) == pytest.approx(25.0, abs=1.5)


def test_the_candidate_record_never_carries_a_confidence_key():
    """`score` is a normalised distance. A key called `confidence` invites a consumer to treat
    it as a probability, which is the rule §3s7 exists to enforce."""
    tr = S.SssTargetTracker(persistence_pings=40)
    ripe = _feed(tr, 45)
    d = S.candidate_from_track(ripe[0], CFG, cid="C1", t=1.0, lat=58.82, lon=17.63,
                               course_deg=0.0, dr_since_fix_m=1.0, along_track_m=0.5).as_dict()
    assert "confidence" not in d and "probability" not in d


# ---------------------------------------------------------- round 2: the gap-tolerant extent
#: MEASURED 2026-09-10 on the Ideal-fan car bag, starboard distinct pings 2142/2143 against a
#: local background of 65: the car's near facet cluster ends at bin 378 and its far cluster
#: begins at bin 404, so the largest gap INSIDE the object is 404 - 378 = 26 bins. The record's
#: own geometry: 4 cm bins, 5.87 m altitude, 14.6 m ground range.
CAR_GAP_BINS = 26
CAR_RES_M, CAR_ALT_M, CAR_GROUND_M = 0.04, 5.87, 14.6


def test_the_gap_tolerance_covers_the_gap_actually_measured_inside_the_car():
    """The derivation must contain the measurement it was derived for.

    This is the check that separates the two candidate bounds. `across_min_m` (1.00 m = 25 bins
    at this record's resolution) is ONE BIN SHORT of the car's own 26-bin self-shadow, which is
    why it is the wrong bound: the smallest object this detector accepts is not a limit on how
    big a hole one object may contain. `across_max_m` (2.00 m) is, because the gap sits inside
    a span the extent gate already caps at that.
    """
    cfg = S.SssConfig(range_res_m=CAR_RES_M, altitude_m=CAR_ALT_M)
    g = S.gap_tolerance_bins(cfg, CAR_GROUND_M)
    assert g >= CAR_GAP_BINS, (
        f"the gap tolerance is {g} bins at the car's own measured geometry, and the gap "
        f"measured INSIDE the car on that record is {CAR_GAP_BINS} bins — the walk cannot "
        f"cross its own target")
    tight = int(round(cfg.across_min_m / CAR_RES_M))
    assert tight < CAR_GAP_BINS, (
        "across_min_m is no longer below the measured gap, so the comment above and the "
        "derivation in gap_tolerance_bins both need re-measuring against this record")


def test_the_gap_tolerance_is_still_bounded_by_the_shadow_at_short_range():
    """The shadow bound must actually bite somewhere, or `min(...)` is decoration. At short
    ground range the predicted shadow is short and takes over from the size bound."""
    cfg = S.SssConfig(range_res_m=CAR_RES_M, altitude_m=6.0)
    near = S.gap_tolerance_bins(cfg, 7.0)
    far = S.gap_tolerance_bins(cfg, CAR_GROUND_M)
    assert near < far, "the tolerance does not tighten as the predicted shadow shortens"
    assert near < int(round(cfg.across_max_m / CAR_RES_M)), \
        "at 7 m ground range the size bound is still winning, so the shadow bound is inert"


def test_the_extent_walk_grows_nearest_first_and_does_not_spend_its_span_on_speckle():
    """Left-to-right growth lets one speckle spike on the near side spend the whole span budget
    before the object's own far facets are ever offered.

    The fixture is arithmetic, not a picture: a solid object at bins 50-58, one isolated spike
    at bin 44, a gap tolerance that can reach either, and a span cap of 10 bins that cannot
    hold both. THE PEAK IS AT THE OBJECT'S NEAR EDGE (bin 50), which is where the two policies
    differ and where a real faceted target routinely puts it — with the peak in the middle both
    policies land on the object and the fixture proves nothing.

    Nearest-first takes bin 51 (one away) before bin 44 (six away) and returns 50-58, the
    object. Walking the left side to exhaustion takes the spike first, spends 7 of its 10 bins
    getting there, and returns 44-53: speckle included and the object cut in half.
    """
    y = np.full(80, 1.0)
    y[50:59] = 10.0          # the object
    y[44] = 10.0             # an isolated spike, 6 bins away
    a, b = S._measure_extent(y, 1.0, 50, gap_bins=6, max_span_bins=10)
    assert (a, b) == (50, 59), (a, b)
    assert 44 not in range(a, b), "the walk spent its span budget on the speckle spike"


def test_gap_zero_and_no_cap_reproduce_the_old_contiguous_walk_exactly():
    """The round-1 behaviour must still be reachable, because every claim about what the gap
    tolerance changed is a comparison against it."""
    y = np.full(80, 1.0)
    y[50:59] = 10.0
    y[44] = 10.0
    assert S._measure_extent(y, 1.0, 54, gap_bins=0, max_span_bins=0) == (50, 59)
    assert S._measure_extent(y, 1.0, 44, gap_bins=0, max_span_bins=0) == (44, 45)


def test_a_short_bright_interruption_does_not_end_a_shadow_but_an_object_sized_one_does():
    """`_shadow_after`'s counterpart tolerance, and the bound that separates the two cases.

    MEASURED on port ping 5160 of the car bag: six bins of 61-77 sitting inside 95 bins of
    exact zero. Six bins is 0.24 m; `across_min_m` is 1.00 m, so it is too small to be an
    object and does not end the shadow. A bright run as long as `across_min_m` does.
    """
    cfg = S.SssConfig(range_res_m=CAR_RES_M, altitude_m=CAR_ALT_M)
    slack = int(round(cfg.across_min_m / CAR_RES_M))
    base = 100.0
    y = np.full(400, 1.0)
    y[:60] = base
    short = y.copy()
    short[120:126] = base            # 6 bins, well under `slack`
    n_short, _db = S._shadow_after(short, base, 60, cfg, 200)
    assert n_short > 126 - 60, ("a 6-bin interruption ended the shadow; that is the defect that "
                                "killed the car on every port ping")
    longer = y.copy()
    longer[120:120 + slack] = base    # exactly object-sized
    n_long, _db2 = S._shadow_after(longer, base, 60, cfg, 200)
    assert n_long == 120 - 60, ("an object-sized bright run did not end the shadow, so the "
                                "shadow can now swallow whatever lies beyond the object")
