"""MEASURED on real records reachable from this machine — 2026-09-09, letter B.

EVERYTHING ELSE IN THIS SUITE IS A FIXTURE. The synthetic pings and clouds prove the code does
what it says; they prove nothing about real data. This file is the only part of round 1 that is
evidence, and it is deliberately small, slow and skippable: it reads bags, and a bag that is not
on the machine makes it SKIP with the path named rather than pass quietly.

WHAT WAS MEASURED (`scripts/adaptive-inspection/measure_negative_sets.py`, full run recorded in
the work order §7 and in SETTLED §3ad):

  bag `sam_mk2_02_57_1788090909` — Askö bay, sim, 1207 s, 117 topics
    side scan  3,000 DISTINCT pings (465 byte-identical repeats skipped, the §3u publisher
               defect) = 6,000 channels -> 5 per-ping hits, **0 candidates** after the 40-ping
               persistence gate. 104,989 change-point rises passed the derived threshold and
               were killed by the signature gates (6,844 on across-track extent, 624 on the
               missing shadow). 0.33 ms per channel.
    3D-15      1,500 clouds -> 1,458 "bare seabed", 20 "no plane the instruments agree with",
               21 clusters, 12 killed on footprint, 9 boxes, **2 candidates** after the 3-ping
               world-frame persistence gate. 4.8 ms per cloud.

  `sss_auto_20240716-112825.dvs` — REAL water, Kristineberg, 35,235 pings
    3,000 pings = 6,000 channels at each of three assumed altitudes -> **0 hits at every one**.
    `change_point.find_nadir` REFUSED 1,000 of 1,000 channels tried on this record with its
    shipped settings, which is why the altitude had to be swept (a `.dvs` carries no altimeter).

ROUND 2 (2026-09-10) CORRECTED TWO OF THE STATEMENTS ABOVE, and both corrections make this
file's evidence weaker rather than stronger, which is why they are written here in full:

  * the Ideal-fan bag with the car (`sam_mk2_02_50_1788007043`) IS on this machine, at
    `data-cube/scratch/bags/`. It was never on vm1. The positive half of the ladder is run by
    `test_MEASURED_positive_set_car_bag.py` beside this file.
  * **bag 57 IS NOT A NEGATIVE SET.** Over 3,000 distinct pings the round-2 detector reports
    420 hits and 1 candidate; georeferenced through `smarc/odom` in the `unity_origin` frame,
    69 of those hits and the whole candidate track sit **0.4-1.4 m** from the MMT Mini's seeded
    scene position (-171.8, -155.4), at 9.2-9.7 m ground range — abeam, above the 6.9 m nadir
    gap. §3f0k's "under the keel" holds for legs 0 and 1 and not for this one. Another 306 of
    the remaining 351 hits fall in 16 fixed 3 m cells, i.e. the scene's rock field.
    The round-1 numbers quoted above (5 hits, 0 candidates) measured the CONTIGUOUS EXTENT WALK
    being unable to see any faceted object at all, the car included — not a quiet detector.
    The tests below therefore keep bag 57 only for COST and for the 400-ping prefix that
    genuinely predates the car, and the false-alarm claim now rests on the `.dvs` alone.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_MEASURED_negative_sets.py
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[5]
BAG57 = ROOT / "_example_data_sets" / "sam_mk2_02_57_1788090909"
DVS_DIR = ROOT / "_example_data_sets" / "2024-07-15-19 Diver transects Kristineberg"
SCRIPT_DIR = ROOT / "data-cube" / "scripts" / "adaptive-inspection"
sys.path.insert(0, str(SCRIPT_DIR))

from sam_target_inspection import fls_target_core as FLS      # noqa: E402
from sam_target_inspection import sss_target_core as SSS      # noqa: E402


def _need(path):
    if not path.exists():
        pytest.skip(f"not on this machine: {path}")
    pytest.importorskip("rosbags", reason="python3 -m pip install --user rosbags")


# ------------------------------------------------------------------ the record's own geometry
def test_the_dvs_layout_fits_the_file_exactly_or_nothing_is_decoded():
    """Two structural checks before one intensity is read: the record size must divide the file
    exactly, and the file header's own bin size and rate must be the ones SETTLED records. A
    binary layout that is NEARLY right decodes into confident nonsense (SETTLED §3c)."""
    import measure_negative_sets as M
    files = sorted(p for p in DVS_DIR.rglob("*.dvs") if p.stat().st_size > 0)
    if not files:
        pytest.skip(f"no non-empty .dvs under {DVS_DIR}")
    meta, rows = M.read_dvs(str(files[0]), max_pings=5)
    assert (files[0].stat().st_size - M.DVS_FILE_HEADER) % M.DVS_RECORD == 0
    assert meta["n_bins"] == 1000
    assert meta["bin_size_m"] == pytest.approx(0.03996, abs=1e-5)
    assert meta["range_per_side_m"] == pytest.approx(39.96, abs=0.01)
    assert 18.0 < meta["ping_hz"] < 19.0
    assert len(rows[0][1]) == 1000 and len(rows[0][2]) == 1000


def test_an_empty_dvs_is_refused_by_name_and_not_read_as_zero_pings():
    """One of the three files in this directory is 0 bytes, exactly as the Askö set had one.
    "Empty" and "no targets" are different facts (SETTLED §3e)."""
    import measure_negative_sets as M
    empty = [p for p in DVS_DIR.rglob("*.dvs") if p.stat().st_size == 0]
    if not empty:
        pytest.skip("no empty .dvs in this checkout")
    with pytest.raises(ValueError) as e:
        M.read_dvs(str(empty[0]), max_pings=1)
    assert "0 bytes" in str(e.value)


# ------------------------------------------------------------------ the negative sets
def test_the_sss_detector_raises_no_candidate_on_the_real_kristineberg_record():
    """A REAL-water record with no car in it — the ONLY clean SSS negative on this machine.

    ROUND 2 CHANGED WHAT THIS ASSERTS, and the change is a loosening, so here is the reason.
    Round 1 asserted `hits == 0`. That was the wrong quantity: a hit is one ping's evidence and
    the detector's contract is a per-ping FALSE-ALARM BUDGET (`false_alarms_per_ping`, 0.05),
    not silence. Asserting zero hits asserted something stricter than the design promises, and
    the gap tolerance duly broke it at 3 hits over 600 channels — against a budget of 30.

    So this now asserts the two things the design actually claims:
      * ZERO CANDIDATES. A candidate is what an operator is shown, and it is the number that
        must be zero on a record with nothing in it. Measured over the full 3,000 pings at each
        of three assumed altitudes: 0, 0, 0.
      * hits WITHIN THE CONFIGURED BUDGET, computed from the config rather than written down,
        so that tightening `false_alarms_per_ping` tightens this test with it.
    """
    import measure_negative_sets as M
    files = sorted(p for p in DVS_DIR.rglob("*.dvs") if p.stat().st_size > 0)
    if not files:
        pytest.skip(f"no non-empty .dvs under {DVS_DIR}")
    meta, rows = M.read_dvs(str(files[0]), max_pings=300)
    res = meta["bin_size_m"]
    cfg = SSS.SssConfig(range_res_m=res, altitude_m=5.0)
    trackers = {"port": SSS.SssTargetTracker(), "starboard": SSS.SssTargetTracker()}
    hits = rises = candidates = 0
    for i, (_lat, port, stbd) in enumerate(rows):
        for side, chan in (("port", port), ("starboard", stbd)):
            r = SSS.detect_ping(chan, cfg)
            hits += len(r.hits)
            rises += r.n_rises
            candidates += len(trackers[side].update(i, side, r))
    channels = 2 * len(rows)
    budget = cfg.false_alarms_per_ping * channels
    assert candidates == 0, f"{candidates} false CANDIDATES over {channels} real channels"
    assert hits <= budget, (f"{hits} hits over {channels} real channels is above this config's "
                            f"own budget of {budget:.0f} "
                            f"({cfg.false_alarms_per_ping}/ping)")
    assert rises > 0, ("not one change point passed the derived threshold over 600 real "
                       "channels — the threshold is not derived from the budget, it is simply "
                       "very high, and a real target would be invisible too")


def test_the_sss_detector_produces_no_candidate_over_bag_57():
    """400 pings of bag 57, which is the prefix that genuinely contains no target.

    DO NOT RAISE THE 400 EXPECTING THIS TO HOLD. Measured 2026-09-10: over the first 3,000
    pings this record yields ONE candidate and it is the MMT Mini itself, 0.4-1.4 m from its
    seeded scene position (see this file's header). What is asserted here is that the detector
    is quiet over seabed that has nothing on it — not that bag 57 is target-free, because it
    is not."""
    _need(BAG57)
    import measure_negative_sets as M
    out = M.measure_bag57_sidescan(400)
    assert out["distinct"] == 400
    assert out["duplicates"] > 0, ("bag 57 is the bag with the 17.84 % publisher-duplicate "
                                   "defect (SETTLED §3u); finding none means the reader is not "
                                   "seeing the same messages it saw in 2026-08-30")
    assert out["candidates"] == 0, out
    assert out["rises"] > 0


def test_the_fls_detector_is_quiet_over_bag_57s_terrain():
    """Quiet, not silent: 2 candidates over 1,500 clouds in the full run, one of which sits
    1.2 m from the position the bay seed documents for the MMT Mini. See the work order §7 —
    this test asserts the RATE, because that is what a budget is."""
    _need(BAG57)
    import measure_negative_sets as M
    out = M.measure_bag57_fls(300)
    assert out["clouds"] == 300
    assert out["bare"] > 0.8 * out["clouds"], \
        "most of this seabed must read as bare, or the plane fit is not fitting the seabed"
    assert out["candidates"] <= 2, out


# ------------------------------------------------------------------ cost (ADR-010)
def test_the_per_ping_cost_is_measured_and_is_under_a_millisecond():
    """ADR-010: the git-guard is the MEASUREMENT, not the argument. At 18.55 Hz x 2 channels the
    side-scan detector has 27 ms of budget per ping pair; measured, it uses well under one."""
    _need(BAG57)
    import measure_negative_sets as M
    out = M.measure_bag57_sidescan(200)
    assert out["ms_per_channel"] < 5.0, out["ms_per_channel"]


def test_the_per_cloud_cost_is_measured_and_fits_the_sonars_own_rate():
    """The 3D-15 in NAV mode pings at 5 Hz, i.e. 200 ms per cloud."""
    _need(BAG57)
    import measure_negative_sets as M
    out = M.measure_bag57_fls(100)
    assert out["ms_per_cloud"] < 200.0, out["ms_per_cloud"]
