"""MEASURED on the ONE record that contains the car — 2026-09-10, round 2.

THE POSITIVE HALF OF STRATEGY §4's VALIDATION LADDER. Round 1 did not run it because the
opening session recorded that this bag lived on vm1; it does not, it is at
`data-cube/scratch/bags/sam_mk2_02_50_1788007043`, and everything below is a first measurement
rather than a re-measurement.

WHAT THE ROUND-1 DETECTOR DID ON IT: 8,386 distinct pings, 16,772 channels, **24 hits, ZERO
candidates**, none in either of SETTLED §3f0s's windows. The cause was measured, not guessed:
in the Ideal fan the car is a COMB OF FACETS separated by its own self-shadow, and
`_measure_extent` walked a CONTIGUOUS half-maximum run from the peak, so it measured one facet
(bins 411-415 = 0.16 m on starboard ping 2170) and `across_min_m` killed the car on every ping.

WHAT THIS FILE ASSERTS, and the honest shape of the result:

  * STARBOARD (§3f0s pings 2144-2217): the detector now produces a candidate, measured at
    distinct ping 2171, from a track of 78 hits spanning pings 2119-2211 at 13.9 m ground range
    with a 1.5-1.7 m across-track extent. ASSERTED.
  * OUTSIDE BOTH WINDOWS: zero candidates over all 8,386 pings. ASSERTED — this is the
    false-alarm half and it is the reason the run is not truncated to the windows.
  * PORT (§3f0s pings 5151-5214): the detector now SEES the car — 28 hits over pings 5150-5179
    at 33.7 m ground range, 1.0-1.4 m across, +15 to +19 dB — but does NOT reach the 40-ping
    persistence gate, so there is NO PORT CANDIDATE. That is asserted as what it is. The two
    measured reasons, neither of which is fixable by a number:
      - on pings 5119-5143 the car's shadow measures 80-90 bins where half the flat-seabed
        prediction `H*r/h` demands 94-96. The shadow is cut short by a bright seabed step about
        4 m beyond the car (the return jumps from 24 to 103 at a FIXED slant bin ~967 while the
        car walks from bin 888 to 838), so the shadow the detector can see is 3.2-3.8 m against
        a 7.5 m prediction. `shadow_fraction_min` is 0.5 and this record sits at 0.44-0.51 of
        prediction, i.e. exactly ON the gate.
      - on pings 5180-5211 the across-track extent falls through `across_min_m` (1.00, 0.96,
        0.88, 0.80, 0.60, 0.48 m) as the aspect closes.
    `shadow_fraction_min = 0.5` is a CHOSEN number, not a derived one — its comment says so —
    and moving it to 0.45 would produce a port candidate. That is precisely the hand-tuning the
    method forbids, so it was not done and this file records the shortfall instead.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider \
        smarc2/perception/sam/sam_target_inspection/test/test_MEASURED_positive_set_car_bag.py
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[5]
CAR_BAG = ROOT / "data-cube" / "scratch" / "bags" / "sam_mk2_02_50_1788007043"

from sam_target_inspection import sss_target_core as SSS      # noqa: E402

#: SETTLED §3f0s, by DISTINCT-ping index. The tolerance is §3f0s's own ±40.
EXPECTED = {"starboard": (2144, 2217), "port": (5151, 5214)}
WINDOW_SLACK = 40
SOUND_SPEED_MS = 1500.0


def _in_window(side: str, ping: int) -> bool:
    lo, hi = EXPECTED[side]
    return (lo - WINDOW_SLACK) <= ping <= (hi + WINDOW_SLACK)


def _run_the_bag():
    """Every distinct ping of the car bag through `detect_ping` and a per-side tracker.

    Duplicates are dropped on (port bytes, starboard bytes, stamp) — the §3u publisher defect
    republishes a ping under its own stamp, and counting one look twice would let the
    persistence gate be reached by a recorder bug rather than by the vehicle moving.
    """
    if not CAR_BAG.exists():
        pytest.skip(f"the Ideal-fan car bag is not on this machine: {CAR_BAG}")
    pytest.importorskip("rosbags", reason="python3 -m pip install --user rosbags")
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    ts = get_typestore(Stores.ROS2_HUMBLE)
    out = {"distinct": 0, "duplicates": 0, "channels": 0, "hits": [], "candidates": []}
    with Reader(str(CAR_BAG)) as r:
        ss = [c for c in r.connections if c.topic.endswith("payload/sidescan")]
        al = [c for c in r.connections if c.topic.endswith("smarc/altitude")]
        if not ss:
            pytest.skip(f"{CAR_BAG} carries no payload/sidescan topic")
        d = ss[0].msgdef
        text = d.data if hasattr(d, "data") else d
        ts.register(get_types_from_msg(text.split("=" * 80)[0], ss[0].msgtype))
        alts = [(t, float(ts.deserialize_cdr(raw, c.msgtype).data))
                for c, t, raw in r.messages(connections=al)]
        a_t = np.array([x[0] for x in alts], dtype=float)
        a_v = np.array([x[1] for x in alts], dtype=float)
        trackers = {"port": SSS.SssTargetTracker(), "starboard": SSS.SssTargetTracker()}
        last = None
        ping_i = 0
        for conn, tns, raw in r.messages(connections=ss):
            m = ts.deserialize_cdr(raw, conn.msgtype)
            key = (bytes(m.port_channel), bytes(m.starboard_channel),
                   m.header.stamp.sec, m.header.stamp.nanosec)
            if key == last:
                out["duplicates"] += 1
                continue
            last = key
            out["distinct"] += 1
            res = (m.max_duration * SOUND_SPEED_MS / 2.0) / max(len(m.port_channel), 1)
            if res <= 0:
                continue
            alt = float(np.interp(float(tns), a_t, a_v)) if a_t.size else 6.0
            cfg = SSS.SssConfig(range_res_m=res, altitude_m=alt)
            ping_i += 1
            for side, chan in (("port", m.port_channel), ("starboard", m.starboard_channel)):
                y = np.frombuffer(bytes(chan), dtype=np.uint8).astype(np.float64)
                if y.size == 0:
                    continue
                out["channels"] += 1
                rep = SSS.detect_ping(y, cfg)
                if rep.ok:
                    for h in rep.hits:
                        out["hits"].append((ping_i, side, h))
                for _track in trackers[side].update(ping_i, side, rep):
                    out["candidates"].append((ping_i, side))
    return out


@pytest.fixture(scope="module")
def measured():
    return _run_the_bag()


def test_the_bag_is_the_one_settled_measured_and_the_duplicates_are_still_there(measured):
    """Identity before evidence. If this is a different record, or the reader has stopped
    seeing the §3u duplicate republishes, then every window index below means nothing."""
    assert measured["distinct"] > 8000, measured["distinct"]
    assert measured["duplicates"] > 0, ("no duplicate republishes found — this reader is not "
                                        "seeing what SETTLED §3u measured, so the distinct-ping "
                                        "indices the windows are quoted in do not line up")
    assert measured["channels"] == 2 * measured["distinct"]


def test_the_detector_produces_a_candidate_inside_the_starboard_window(measured):
    """THE REGRESSION THIS FILE EXISTS FOR. Round 1 produced zero candidates anywhere in this
    bag; restoring the contiguous extent walk puts it back to zero and fails here."""
    inside = [p for p, side in measured["candidates"]
              if side == "starboard" and _in_window("starboard", p)]
    assert inside, (
        f"no candidate in the starboard window {EXPECTED['starboard']} ±{WINDOW_SLACK}; "
        f"candidates found: {measured['candidates']}")


def test_no_candidate_appears_outside_either_window(measured):
    """The false-alarm half, over the whole 8,386-ping record and not just the windows."""
    outside = [(p, side) for p, side in measured["candidates"] if not _in_window(side, p)]
    assert outside == [], f"{len(outside)} candidate(s) outside both windows: {outside}"


def test_the_port_car_is_seen_even_though_it_does_not_reach_persistence(measured):
    """PORT IS A SHORTFALL, NOT A MISS, and this asserts the difference.

    The floor is `PERSISTENCE_PINGS // 2` — half the gate the port pass fails — and it is
    written in terms of that gate rather than in terms of the 28 that was measured, so it
    cannot silently become a record of this bag's numbers. What it forbids is the car going
    back to being INVISIBLE on port, which is what round 1 had.
    """
    port_hits = [p for p, side, _h in measured["hits"]
                 if side == "port" and _in_window("port", p)]
    floor = SSS.PERSISTENCE_PINGS // 2
    assert len(port_hits) >= floor, (
        f"only {len(port_hits)} port hits inside {EXPECTED['port']} ±{WINDOW_SLACK}; the car "
        f"is effectively invisible on the port pass again")
    assert not [p for p, side in measured["candidates"] if side == "port"], (
        "a port candidate appeared. That is an IMPROVEMENT, not a failure — but it means the "
        "shadow-gate shortfall this file documents has been changed, so update the docstring, "
        "SETTLED §3ad and the work order §7 with the new measurement before deleting this line")


def test_the_port_hits_are_at_the_cars_own_range_and_not_scattered(measured):
    """A shortfall is only informative if the hits are the car. Measured: 33.7 m ground range,
    which is where §3f0s puts the port pass; scattered hits at random ranges would mean the
    port count above is speckle that happens to land in the window."""
    ranges = [h.ground_range_m for p, side, h in measured["hits"]
              if side == "port" and _in_window("port", p)]
    assert ranges
    spread = max(ranges) - min(ranges)
    assert spread < 2.0, (f"port hits inside the window span {spread:.1f} m of ground range "
                          f"({min(ranges):.1f}-{max(ranges):.1f}) — that is not one object")
