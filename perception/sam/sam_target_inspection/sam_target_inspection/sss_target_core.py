#!/usr/bin/env python3
"""The side-scan car detector — highlight THEN shadow, on intensities only.

Strategy §4. Pure arithmetic over one intensity vector plus a small tracker across pings: no
ROS, no I/O, no simulator. `sss_target_detector.py` is the shell.

WHAT THE SIGNATURE IS, AND WHERE EACH NUMBER CAME FROM. Measured on the Ideal fan with the
MMT Mini at 15–35 m abeam (SETTLED §3f0s):

  * a HIGHLIGHT of about +9.2 dB over the local gain-envelope baseline, 1.56 m across track
    (the car is 1.42 m wide, 3.08 m long);
  * IMMEDIATELY FOLLOWED IN RANGE by an acoustic SHADOW at −11.3 dB with 72 % of its bins
    exactly zero in the Ideal fan; in the real 2024-12-10 record a shadow is
    "darker-than-floor" rather than zero (§3f0e), so the test here is a ratio against the
    envelope, never an absolute count of zeros;
  * the shadow's LENGTH is geometry, not a parameter: a target of height H at ground range r
    under altitude h casts `L_shadow ≈ H·r/h`.

THE TWO ARE REQUIRED TOGETHER, and that is the whole discriminating power of the detector. A
bright patch without a shadow is bottom texture — in the real record bedrock reads 0.2–0.9 dB
DARKER than sand (§3f0h), so brightness alone discriminates nothing. A dark run without a
preceding highlight is a hole, or the display's own nadir gap.

THRESHOLDS ARE DERIVED FROM A FALSE-ALARM BUDGET, not chosen. The change-point statistic and
its budget mapping are IMPORTED from `sam_farm_inspection.change_point` — not re-implemented.
That import is deliberate and is the §3d lesson: a second copy of the same maths is a second
thing to keep right, and the first farm detector's hand-picked ratio of 4 produced hundreds of
detections per noise ping.

WHAT THIS MODULE NEVER DOES
  * it never reads a simulator material label (SETTLED §3k). Its input is a vector of numbers;
  * it never mints a confidence. `score` is a NORMALISED DISTANCE from the expected signature,
    with every sigma in the denominator named in `SIGNATURE_SIGMAS` (§3s7);
  * it never reports the leading edge of the highlight as the object's range. It reports the
    CENTRE, because a change point marks an EDGE and reporting it is biased by half the
    object's width plus the window (§3k).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam_farm_inspection.change_point import (change_ratio_for_false_alarm,
                                              discrepancy_profile)

# --------------------------------------------------------------------------------------
# The measured signature, and the spread of each part. Both halves are needed: a distance
# without a sigma is not a distance, it is a number with units nobody agreed on.
#
# EVERY VALUE HERE IS THE 2026-08-29 IDEAL-FAN MEASUREMENT (SETTLED §3f0s) EXCEPT THE SIGMAS,
# which are the honest spread of ONE measurement of ONE car and are marked PROVISIONAL. They
# widen the score's denominator, so getting them wrong makes the detector less decisive, never
# more — which is the safe direction for a number nobody has measured twice.
# --------------------------------------------------------------------------------------
SIGNATURE = {
    "highlight_db": 9.2,        # measured, Ideal fan
    "shadow_db": -11.3,         # measured, Ideal fan
    "across_m": 1.56,           # measured (car is 1.416 m wide, SETTLED §3f0d)
}
SIGNATURE_SIGMAS = {
    "highlight_db": 3.0,        # PROVISIONAL: one car, one fidelity, one aspect
    "shadow_db": 4.0,           # PROVISIONAL: a real shadow is darker-than-floor, not zero
    "across_m": 0.5,            # PROVISIONAL: bin quantisation + aspect
}

#: The gain-envelope half-window, in BINS. §3f0e measured the record's own gain envelope over
#: ±64 columns and that is the window this detector removes before looking for anything. A
#: shorter window follows the object and erases it; a longer one leaves the TVG ramp in.
ENVELOPE_HALF_BINS = 64

def persistence_pings_for(length_m: float = 3.078, speed_ms: float = 0.7,
                          ping_hz: float = 18.55, fraction: float = 0.5) -> int:
    """How many consecutive pings a target must persist for — DERIVED, not chosen.

    The object's own along-track extent is `length/speed * ping_hz` pings; at the measured
    record geometry (18.55 Hz, SETTLED §3f0e) and the Mini's 3.078 m (SETTLED §3f0d) that is
    82 pings at 0.7 m/s and 114 at 0.5 m/s — strategy §4's "80-110". The gate is `fraction` of
    the FASTER case, so half a pass at the fastest scan speed is still a detection while a
    single bright ping cannot be one. Faster, not slower: a threshold set from the slow case
    would need more pings than the fast case ever produces.
    """
    return max(2, int(round(fraction * length_m / max(speed_ms, 1e-6) * ping_hz)))


#: 40 at the shipped numbers. See the derivation above; do not hand-edit this.
PERSISTENCE_PINGS = persistence_pings_for()

#: Across-track extent band, metres. The car is 1.42 m wide and 3.08 m long, so its across-track
#: signature is 1–2 m at any aspect. A rock ridge is wider, a rope is narrower.
ACROSS_MIN_M, ACROSS_MAX_M = 1.0, 2.0

#: A bin is "in shadow" when it falls to this fraction of the local gain envelope. Derived, not
#: chosen: the measured shadow is −11.3 dB, i.e. a ratio of 10**(−11.3/20) = 0.272. Half way (in
#: dB) between the envelope and the measured shadow is −5.65 dB = 0.52, which is the loosest
#: test that still cannot be satisfied by ordinary speckle at the measured CV of 0.28 (a 0.52
#: ratio is 1.7 sigma below the mean for CV 0.28, and the run-length requirement below turns
#: that into a vanishing probability).
SHADOW_RATIO = 10.0 ** (-5.65 / 20.0)

#: Measured speckle coefficient of variation on the real 2024-12-10 record (SETTLED §3f0h,
#: `docs/asko/sss_real_targets_2024-12-10.json`). Used only to state, in the refusal, how many
#: sigma a shadow run is — never to synthesise anything.
REAL_RECORD_SPECKLE_CV = 0.28


class SssRefusal(RuntimeError):
    """Raised with an operator-readable reason. Never caught and defaulted."""


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PingHit:
    """One ping's evidence for one object. NOT a candidate — a candidate needs persistence."""

    #: CENTRE of the highlight, in bins. Not its leading edge (SETTLED §3k).
    centre_bin: int
    start_bin: int
    end_bin: int
    highlight_db: float
    shadow_db: float
    shadow_bins: int
    across_m: float
    shadow_len_m: float
    #: Shadow length the geometry predicts for a target of the configured height at this range.
    shadow_len_expected_m: float
    slant_range_m: float
    ground_range_m: float
    score: float


@dataclass(frozen=True)
class PingReport:
    """What one channel of one ping produced. `ok=False` ALWAYS carries a reason."""

    ok: bool
    reason: str
    hits: Tuple[PingHit, ...] = ()
    nadir_bin: Optional[int] = None
    #: How many rises passed the derived change-point threshold before the signature gates.
    #: This is the number that says whether a quiet result means "nothing there" or "the
    #: threshold is wrong", and it is on the health line for exactly that reason.
    n_rises: int = 0
    n_killed_no_shadow: int = 0
    n_killed_extent: int = 0
    #: Objects whose predicted shadow ran past the end of the record. Counted separately
    #: because "the recorder's range was too short" is not "there was no shadow".
    n_killed_truncated: int = 0


@dataclass(frozen=True)
class SigmaBreakdown:
    """Where a candidate's positional uncertainty comes from. Never a single opaque number.

    The rule (§3s7, and the station's "MEASURED ±0.014 m" that was a constant): a sigma that
    does not say what it is made of cannot be checked, and one that cannot be checked will be
    trusted. `total_m` is the quadrature sum and nothing else.
    """

    dr_m: float                 # dead-reckoned drift since the last fix
    bin_quantisation_m: float   # half a range bin, projected to ground range
    along_track_m: float        # centring error along the swath
    slant_geometry_m: float     # error from the altimeter's own uncertainty

    @property
    def total_m(self) -> float:
        return math.sqrt(self.dr_m ** 2 + self.bin_quantisation_m ** 2 +
                         self.along_track_m ** 2 + self.slant_geometry_m ** 2)

    def as_dict(self) -> Dict[str, float]:
        return {"dr_m": self.dr_m, "bin_quantisation_m": self.bin_quantisation_m,
                "along_track_m": self.along_track_m,
                "slant_geometry_m": self.slant_geometry_m, "total_m": self.total_m}


@dataclass(frozen=True)
class SssCandidate:
    """The record strategy §4 specifies. `score` is a distance, never a confidence."""

    id: str
    t: float
    side: str                       # port | starboard
    ping: int
    bin: int
    lat: Optional[float]
    lon: Optional[float]
    sigma_m: float
    sigma_parts: Dict[str, float]
    highlight_db: float
    shadow_db: float
    extent_m: Dict[str, float]      # across, along, shadow_len
    score: float
    n_pings: int

    def as_dict(self) -> Dict[str, object]:
        return {"id": self.id, "t": self.t, "side": self.side, "ping": self.ping,
                "bin": self.bin, "lat": self.lat, "lon": self.lon,
                "sigma_m": round(self.sigma_m, 3), "sigma_parts": self.sigma_parts,
                "highlight_db": round(self.highlight_db, 2),
                "shadow_db": round(self.shadow_db, 2),
                "extent_m": {k: round(v, 3) for k, v in self.extent_m.items()},
                "score": round(self.score, 3), "n_pings": self.n_pings,
                "sensor": "sss"}


# --------------------------------------------------------------------------------------
# the gain envelope
# --------------------------------------------------------------------------------------
def gain_envelope(signal: Sequence[float], half_bins: int = ENVELOPE_HALF_BINS) -> np.ndarray:
    """The record's own smoothed gain envelope: a centred moving mean over ±`half_bins`.

    §3f0e measured the envelope over ±64 columns and every level in this file is expressed
    against it rather than against an absolute intensity. That is not a stylistic choice: the
    simulator's reflectivities are documented wild guesses (KRISTINEBERG_SITE.md §5h) and the
    real board applies its own TVG, so any rule with an absolute number in it is measuring the
    guess. Ratios against a locally-estimated envelope survive both.

    Edges are handled by shrinking the window rather than by padding: padding with zeros
    invents a dark band at both ends of every ping, which is exactly the thing a shadow test
    would then fire on.

    WHAT IT IS FOR, AND WHAT IT IS NOT FOR — measured while writing this file, and it changed
    the design. A ±64-bin MEAN is a fine detrender: subtract it and the change-point statistic
    sees a flat residual with the TVG ramp gone. It is NOT a usable reference level for
    measuring a target's contrast, because the target is inside its own window. With the
    measured signature (a 1.56 m highlight and a 5.3 m shadow at 4 cm bins, i.e. 39 and 133
    bins against a 129-bin window) the envelope is dragged UP by the highlight and DOWN inside
    the shadow: the +9.2 dB highlight measured against it reads +6.8 dB, and the shadow test
    against it terminates after 52 of 133 bins because the envelope has fallen to meet it.
    Every LEVEL in this file is therefore measured against `local_background`, which is the
    reverb floor taken from beside the object with a guard gap — the CFAR discipline, and the
    only one that survives an object big enough to make its own background.
    """
    y = np.asarray(signal, dtype=np.float64)
    n = y.size
    if n == 0:
        return y
    w = max(1, int(half_bins))
    c = np.concatenate(([0.0], np.cumsum(y)))
    lo = np.maximum(np.arange(n) - w, 0)
    hi = np.minimum(np.arange(n) + w + 1, n)
    out = (c[hi] - c[lo]) / (hi - lo)
    # A floor so a channel of exact zeros cannot divide by zero and read as infinite contrast.
    return np.maximum(out, 1e-9)


def local_background(signal: Sequence[float], start_bin: int, end_bin: int,
                     guard_bins: int, window_bins: int) -> Tuple[float, str]:
    """The reverb floor BESIDE an object, with a guard gap. Returns (level, where it came from).

    Constant-false-alarm-rate discipline, and here it is not a refinement but a requirement:
    the objects this detector looks for are large compared with any smoothing window, so a
    background estimated through the object is an estimate of the object (see `gain_envelope`).

    The window is taken BEFORE the highlight — nearer the vehicle — because that is the side
    that is guaranteed not to be the object's own shadow. If there is not enough ping before
    it (a target close to the nadir), the window is taken AFTER the predicted shadow instead
    and the returned reason says so, because a background measured on the far side of a
    5 m shadow at a different TVG point is a weaker measurement and the consumer should know.

    MEDIAN, not mean: one bright multipath return inside the background window would raise a
    mean and quietly suppress the very contrast being measured.
    """
    y = np.asarray(signal, dtype=np.float64)
    hi = int(start_bin) - int(guard_bins)
    lo = hi - int(window_bins)
    if lo >= 0 and hi - lo >= 8:
        return max(float(np.median(y[lo:hi])), 1e-9), "before"
    lo2 = int(end_bin) + int(guard_bins)
    hi2 = min(y.size, lo2 + int(window_bins))
    if hi2 - lo2 >= 8:
        return max(float(np.median(y[lo2:hi2])), 1e-9), "after (too close to the nadir for a near-side window)"
    return max(float(np.median(y)), 1e-9), "whole ping (no room for a local window)"


def db_over(value: float, reference: float) -> float:
    """20·log10(value/reference) — the amplitude convention every SSS tool in this repo uses
    (`scripts/sss-tuning/layer_ab.py`, `diff_target.py`, `model_ideal_fan.py`)."""
    return 20.0 * math.log10(max(float(value), 1e-9) / max(float(reference), 1e-9))


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def slant_to_ground_m(slant_m: float, altitude_m: float) -> Optional[float]:
    """sqrt(slant² − h²), or None inside the nadir gap where there is no ground range at all."""
    if slant_m <= altitude_m:
        return None
    return math.sqrt(slant_m * slant_m - altitude_m * altitude_m)


def expected_shadow_len_m(ground_range_m: float, altitude_m: float,
                          target_height_m: float) -> float:
    """L_shadow ≈ H·r/h — the shadow a target of height H casts at ground range r under h.

    Straight similar triangles: the grazing ray that just clears the target's top continues to
    the seabed at r + H·r/h. It is the ONE length in this detector that is not a parameter, and
    the reason a shadow test can be scale-free: a target twice as far casts a shadow twice as
    long, so a fixed shadow length in metres would be right at exactly one range.
    """
    if altitude_m <= 0:
        raise SssRefusal("altitude must be positive to predict a shadow length; a shadow at "
                         "zero altitude is not geometry, it is a division by zero")
    return target_height_m * ground_range_m / altitude_m


def nadir_gap_m(altitude_m: float) -> float:
    """1.15·h — the across-track strip the side scan cannot see (SETTLED §3f0e)."""
    return 1.15 * altitude_m


# --------------------------------------------------------------------------------------
# the per-ping detector
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SssConfig:
    """Everything the per-ping detector needs, with the derivation of each default in place."""

    range_res_m: float
    altitude_m: float
    target_height_m: float = 1.278            # the Mini's height, SETTLED §3f0d
    across_min_m: float = ACROSS_MIN_M
    across_max_m: float = ACROSS_MAX_M
    #: Minimum highlight over the envelope. The measurement is +9.2 dB; the gate is +6, one
    #: measured sigma below it, so a weaker aspect of the same object is not thrown away.
    highlight_db_min: float = 6.0
    envelope_half_bins: int = ENVELOPE_HALF_BINS
    #: The threshold is DERIVED from this budget through change_point's chi-square tail. It is
    #: the same 0.05/ping the farm detector ships with, and it is the number the 2024-12-10
    #: negative record is used to check.
    false_alarms_per_ping: float = 0.05
    #: Fraction of the predicted shadow length that must actually be dark. Half: a real shadow
    #: is broken up by the far edge of the target and by multipath, and requiring the whole
    #: predicted length would make the detector fail on exactly the geometry it predicts.
    shadow_fraction_min: float = 0.5
    shadow_ratio: float = SHADOW_RATIO
    #: Search starts this many bins beyond the detected seabed return. The bottom return has a
    #: leading edge of a few bins and the rise detector will happily fire on it.
    nadir_guard_bins: int = 4
    #: Altimeter uncertainty, metres. Used only in the sigma decomposition.
    altitude_sigma_m: float = 0.2
    #: Bins between an object's measured edge and its background window. Sized so a highlight
    #: that spills one across-track window past its half-maximum edge still cannot land inside
    #: its own background estimate.
    background_guard_bins: int = 8
    #: Hard cap on window positions evaluated per ping. Cost, and ADR-010: this node's own
    #: per-ping cost is on its health line, and an unbounded inner loop over a noisy ping is
    #: how a perception node becomes the reason a flight computer misses its control rate.
    max_evaluations_per_ping: int = 16

    def across_window_bins(self) -> int:
        """The change-point half-window, in bins, from the object's PHYSICAL across-track size.

        Configured in metres and converted here for the same reason `targets_from_metres` does
        it in the farm detector: the two side-scan modes differ 4× in bin size and a bin count
        that is right for one is meaningless for the other.
        """
        m = 0.5 * (self.across_min_m + self.across_max_m)
        return max(2, int(round(m / max(self.range_res_m, 1e-9))))


def _find_seabed_bin(y: np.ndarray, env: np.ndarray, cfg: SssConfig) -> Optional[int]:
    """First bin whose slant range exceeds the altitude — the geometric start of the seabed.

    DELIBERATELY GEOMETRIC, not a change-point search. The farm detector hunts the nadir because
    its targets are in the WATER COLUMN and the seabed must be excluded. Here the target is ON
    the seabed, so the water column is what must be excluded, and the vehicle's own altimeter
    already says where it ends: the first return from the bottom is at slant range h. Using the
    altimeter is the same rule §3f0h set for georeferencing — the vehicle's own measurement of
    its own geometry, not a feature hunted in the data.
    """
    if cfg.altitude_m <= 0:
        return None
    b = int(math.ceil(cfg.altitude_m / max(cfg.range_res_m, 1e-9))) + cfg.nadir_guard_bins
    return b if b < y.size - 2 else None


def gap_tolerance_bins(cfg: "SssConfig", ground_range_m: float) -> int:
    """How long a gap inside an object may be before it is the END of the object.

    THE MEASUREMENT THAT FORCED THIS (2026-09-10, the Ideal-fan car bag
    `sam_mk2_02_50_1788007043`). The car is not one bright block. It is a COMB of bright facets
    — roof line, windscreen, wheel arches, exactly the structure SETTLED §3f0s reports — with
    SELF-SHADOW between them. Starboard distinct ping 2170, bins 370-420, against a local
    background of 62:

        377 378  380 381 382 383            403   405 406      411 412 413 414
        151 156  177 175 142 132            132   153 153      199 199 198 198

    A CONTIGUOUS half-maximum walk from the peak at bin 411 measures **bins 411-414 = 0.16 m**
    — one facet — and `across_min_m` (1.0 m) then kills the car on every single ping. That is
    what it did: 8,386 distinct pings, 16,772 channels, 24 channel-hits, **zero candidates**,
    none of them in the car's own windows. §3f0s's 1.56 m is the span of the WHOLE COMB
    (377-414 = 38 bins = 1.52 m), and only a gap-tolerant walk recovers it.

    **THE GAP IS 20 BINS, NOT 7.** The longest EXACT-ZERO run inside the object is 7 bins
    (394-400), but bins 384-393 are dim-and-nonzero (128 down to 39) and still below half
    maximum, so the largest gap between ABOVE-HALF bins is 383 -> 403 = **20 bins = 0.80 m**. A
    tolerance derived from the zero runs alone would not have fixed this, and the walk is over
    the half-maximum criterion, not over zeros.

    **THE GAP IS AS LONG AS THE OBJECT'S OWN SELF-SHADOW, WHICH IS 26 BINS = 1.04 m.** The first
    derivation of this function bounded the tolerance by `across_min_m` (1.0 m = 25 bins) and was
    WRONG BY ONE BIN, which is measurable rather than arguable: it raised the channel-hit count
    from 24 to 86 and put 47 hits inside the starboard window, but the run broke apart every few
    pings. Read off starboard distinct pings 2142 and 2143 against a background of 65, the car
    resolves into TWO facet clusters — a near one ending at bin 378 and a far one starting at bin
    404 — with its own hull self-shadow between them:

        ping 2142  368..378: 148 148 115 162 175 165 134 125 136 142 123
                   379..403: 106 83 .. 92, then EXACT ZERO from 394
                   404..407: 158 163 158 124        408+: exact zero (the true shadow)
        ping 2143  368..378: 116 152 143 108 171 170 138 136 127 123 118
                   404..407: 181 181 162 143        408+: exact zero

    The gap between the two clusters is 404 - 378 = **26 bins = 1.04 m**, and a 25-bin tolerance
    bridges it on ping 2142 (peak 163, half 114, so bin 378 at 123 is still above half and sits
    exactly 26 back from 404) and FAILS on ping 2143 (peak 181, half 123, so the rightmost
    above-half bin is 376, 28 back). That knife-edge is the whole dropout pattern.

    THE DERIVATION, and it introduces NO NEW CONSTANT. A gap may be internal only if it is too
    short to be either of the two things that would end the object:

      1. **the object would be too big.** The gap lies INSIDE the span, and `detect_ping` only
         accepts a span of at most `across_max_m`. So a tolerance beyond `across_max_m` cannot
         admit anything the extent gate would pass, and it is the correct upper bound, not
         `across_min_m`: `across_min_m` bounded the wrong quantity — it is the smallest object,
         not the largest hole one can have, and a car seen broadside is exactly an object whose
         self-shadow is longer than the smallest object this detector accepts.

         This bound is NOT free, and saying so here was wrong until seed 311 of
         `test_sss_target_core` was measured: a gap tolerance with no cap on the resulting span
         walked off a real target into a speckle spike 45 bins away and reported 3.36 m, so the
         extent gate then discarded a TRUE detection. A gap tolerance costs false negatives, not
         only false alarms. `_measure_extent` therefore carries `max_span_bins` as well, and the
         two are different statements: this one is how far the walk may LOOK across a dark run,
         that one is how wide the object it builds is allowed to END UP.
      2. **this object's own shadow.** `_shadow_after` calls a dark run a shadow once it reaches
         `shadow_fraction_min` of the predicted `H*r/h`. A gap that long IS the shadow by the
         detector's own rule, so it must terminate the object — otherwise the walk could step
         over an object's shadow and swallow whatever lies beyond it.

    The tolerance is the smaller of the two, so the two gates can never disagree about the same
    dark run, and bound 2 is what actually does the work at every geometry this detector sees:
    it TIGHTENS on its own at short range (at r = 7 m under h = 6 m the shadow is 1.49 m and the
    tolerance falls to 0.75 m — below the measured 1.04 m, i.e. a car at 7 m would have to be
    found by its two clusters separately, which is the honest answer at that geometry). At the
    car's own geometry — r = 14.6 m, h = 5.87 m, H = `target_height_m` — the predicted shadow is
    3.18 m, half of it is 1.59 m = 39 bins, and that is under `across_max_m` (50 bins), so the
    tolerance is 39 bins. It clears the measured 26 and was not chosen to: no term in it was
    fitted to this record, and both terms already existed as gates.
    """
    shadow_m = expected_shadow_len_m(ground_range_m, cfg.altitude_m, cfg.target_height_m)
    g_m = min(cfg.across_max_m, cfg.shadow_fraction_min * shadow_m)
    return max(1, int(round(g_m / max(cfg.range_res_m, 1e-9))))


def _measure_extent(y: np.ndarray, level: float, peak: int, gap_bins: int = 0,
                    max_span_bins: int = 0) -> Tuple[int, int]:
    """Run around `peak` that stays above half its height over `level`, TOLERATING GAPS.

    The half-maximum rule, scale-free and needing no calibrated amplitude — the same rule
    `change_point.measure_extent` uses and for the same reason. `level` is the local background,
    never the smoothed envelope: see `gain_envelope`.

    `gap_bins` is what makes it usable on a real target: the walk continues past a gap when
    another above-half bin lies within `gap_bins` of the current edge. A gap INSIDE the span is
    self-shadow between facets; the object's own shadow is the dark run AFTER the last facet,
    and `_shadow_after` starts where this returns. See `gap_tolerance_bins` for the measurement
    that made this necessary and for where the number comes from.

    `max_span_bins` is the OTHER half of the tolerance and it is not optional. MEASURED
    2026-09-10 on synthetic seed 311 of `test_sss_target_core`: with a gap tolerance and no span
    cap, the walk left a real 1.56 m target at bin 642 and bridged 45 bins of seabed to a single
    bright speckle bin at 597, reporting 3.36 m — and `across_max_m` then threw the TRUE target
    away. A gap tolerance therefore does not only risk false alarms, it manufactures false
    NEGATIVES, and an earlier version of this docstring claimed it "loosens no gate", which that
    measurement disproves. The cap is `across_max_m` in bins, i.e. the walk may never build a
    span the extent gate downstream would reject anyway: the walk now respects the gate it
    feeds instead of handing it a span it built by wandering.

    The two edges therefore grow NEAREST-FIRST rather than left-to-right: at each step the walk
    takes whichever candidate — left or right — is closer to the span it already has, and stops
    when neither fits under the cap. Nearest-first is not a tie-break convenience; walking one
    side to exhaustion first would let speckle on that side spend the whole span budget before
    the object's own facets on the other side were ever offered.

    `gap_bins = 0` with `max_span_bins = 0` is the old contiguous, uncapped behaviour exactly,
    and is kept so the two can be compared — that comparison is the mutation this function's
    test is built around.
    """
    half = level + 0.5 * (y[peak] - level)
    g = max(0, int(gap_bins))
    cap = int(max_span_bins) if max_span_bins and max_span_bins > 0 else 0
    a = b = int(peak)

    def _next_left(edge: int) -> Optional[int]:
        stop = max(0, edge - 1 - g)
        for j in range(edge - 1, stop - 1, -1):
            if y[j] > half:
                return j
        return None

    def _next_right(edge: int) -> Optional[int]:
        stop = min(y.size - 1, edge + 1 + g)
        for j in range(edge + 1, stop + 1):
            if y[j] > half:
                return j
        return None

    while True:
        l = _next_left(a)
        r = _next_right(b)
        if cap:
            if l is not None and (b - l + 1) > cap:
                l = None
            if r is not None and (r - a + 1) > cap:
                r = None
        if l is None and r is None:
            break
        if r is None or (l is not None and (a - l) <= (r - b)):
            a = l
        else:
            b = r
    return a, b + 1


def _shadow_after(y: np.ndarray, level: float, start: int, cfg: SssConfig,
                  expect_bins: int) -> Tuple[int, float]:
    """(length in bins, mean dB) of the dark run that begins at `start`.

    "Begins at" is still literal in the sense that matters: the shadow is the geometric
    continuation of the target and this never searches down range for the next dark patch it
    likes. But two things are allowed, and they are the SAME allowance made twice, bounded by
    the same number — `across_min_m`, the smallest across-track extent this detector will call
    an object at all:

      * **the dark may begin up to `across_min_m` past `start`.** `start` is one bin past the
        last ABOVE-HALF-MAXIMUM bin, which is not the target's trailing edge: the dim skirt of
        the object continues past half maximum. That skirt cannot be longer than the smallest
        object, or it would be one.
      * **a bright interruption shorter than `across_min_m` does not end the run.** MEASURED
        2026-09-10 on the Ideal-fan car bag, port distinct ping 5160: the object ends at bin
        871, the return is exact zero from 872 to 951, there are SIX bins of 61-77 at 952-957,
        and then exact zero again to 966 before the seabed resumes at 103. The old rule stopped
        at 951 and reported 80 bins where the geometry demands 93, so the car was killed as
        "carried no shadow of the predicted length" on every port ping. Six bins is 0.24 m —
        a scatterer lying in the shadow, not the end of it, and by this detector's own extent
        gate it is too small to be an object.

    This is the exact counterpart of the gap tolerance in `_measure_extent`, and the two use
    DIFFERENT bounds on purpose: an object's internal gap is bounded by how big the object may
    be (`across_max_m`, see `gap_tolerance_bins`), while a shadow's internal interruption is
    bounded by how small a separate object may be (`across_min_m`). Confusing the two is what
    the first version of the gap tolerance did.

    `length` is the SPAN from the first dark bin to the last, interruptions included, because
    that is the quantity the geometric prediction `H*r/h` is a prediction of — and it is the
    same convention `_measure_extent` uses for `across_m`. The reported dB is the mean over that
    same span, so a tolerated bright interruption makes the shadow read LESS dark, never more:
    the score is penalised for the interruption rather than being flattered by ignoring it.
    """
    n = y.size
    if start >= n:
        return 0, 0.0
    thresh = cfg.shadow_ratio * level
    slack = max(1, int(round(cfg.across_min_m / max(cfg.range_res_m, 1e-9))))
    i = start
    while i < min(n, start + slack) and y[i] > thresh:
        i += 1
    if i >= n or y[i] > thresh:
        return 0, 0.0
    begin = i
    limit = min(n, begin + max(4, 3 * expect_bins))
    last_dark = begin
    while i < limit:
        if y[i] <= thresh:
            last_dark = i
            i += 1
            continue
        j = i
        while j < limit and y[j] > thresh:
            j += 1
        if j >= limit or j - i >= slack:
            break
        i = j
    length = last_dark + 1 - begin
    if length <= 0:
        return 0, 0.0
    return length, db_over(float(np.mean(y[begin:last_dark + 1])), level)


def signature_score(highlight_db: float, shadow_db: float, across_m: float) -> float:
    """Normalised distance from the measured signature. NOT a confidence (§3s7).

    Root-mean-square of three z-scores, each against a named sigma in `SIGNATURE_SIGMAS`. Zero
    is "exactly the Ideal-fan car"; one is "one sigma out on average". Nothing maps it to 0..1
    and nothing calls it a probability, because a consumer handed a 0.87 will multiply it by
    something.

    ONE-SIDED ON PURPOSE for the two contrast terms: a highlight BRIGHTER than the measured car
    and a shadow DARKER than the measured car are more car-like, not less, and penalising them
    would rank the strongest evidence worst.
    """
    zs = [
        max(0.0, SIGNATURE["highlight_db"] - highlight_db) / SIGNATURE_SIGMAS["highlight_db"],
        max(0.0, shadow_db - SIGNATURE["shadow_db"]) / SIGNATURE_SIGMAS["shadow_db"],
        abs(across_m - SIGNATURE["across_m"]) / SIGNATURE_SIGMAS["across_m"],
    ]
    return math.sqrt(sum(z * z for z in zs) / len(zs))


def detect_ping(channel: Sequence[float], cfg: SssConfig) -> PingReport:
    """One channel of one ping: gain envelope, then a rise, then the shadow it must carry.

    Returns a report whose `ok=False` always carries a reason. "I could not tell where the
    seabed starts" and "there is nothing on this seabed" are different answers and only one of
    them is information (SETTLED §3e).
    """
    y = np.asarray(channel, dtype=np.float64)
    if y.size == 0:
        return PingReport(False, "empty channel")
    if cfg.range_res_m <= 0:
        return PingReport(False, f"range resolution {cfg.range_res_m} m is not positive")
    env = gain_envelope(y, cfg.envelope_half_bins)

    seabed = _find_seabed_bin(y, env, cfg)
    if seabed is None:
        return PingReport(False,
                          f"the altimeter reports {cfg.altitude_m:.2f} m, which puts the first "
                          f"seabed return at or past the end of this {y.size}-bin ping — there "
                          f"is no seabed in it to look at")

    t = cfg.across_window_bins()
    seg = y[seabed:] - env[seabed:]     # envelope-removed: the rise test is on the residual
    d, mean_delta = discrepancy_profile(seg, t)
    if d.size == 0:
        return PingReport(False,
                          f"only {y.size - seabed} bins of seabed after the nadir, too few for "
                          f"a {t}-bin across-track window", nadir_bin=seabed)
    base = max(float(np.median(d)), 1e-9)
    # DERIVED, never chosen: the chi-square tail that admits `false_alarms_per_ping` noise peaks
    # over the number of window positions ACTUALLY tested in this seabed region.
    ratio = change_ratio_for_false_alarm(int(d.size), cfg.false_alarms_per_ping)
    thresh = base * ratio

    rises = np.nonzero((d >= thresh) & (mean_delta > 0.0))[0]
    n_rises = int(rises.size)
    hits: List[PingHit] = []
    killed_shadow = killed_extent = killed_dim = killed_truncated = 0
    taken: List[int] = []
    evaluated: List[int] = []
    for i in sorted(rises, key=lambda j: -d[j]):
        change = int(i) + t + seabed
        # ONE EVALUATION PER OBJECT. A real target raises the discrepancy at every window
        # position that straddles either of its edges — 52 of them for the measured signature
        # at 4 cm bins — and evaluating each would multiply the work by 50 and report the same
        # object dozens of times. Separation is enforced on EVALUATIONS, not only on accepted
        # hits, so a killed candidate also suppresses its own neighbours.
        if any(abs(change - o) < t for o in evaluated):
            continue
        evaluated.append(change)
        if len(evaluated) > cfg.max_evaluations_per_ping:
            break
        # The peak near the change point, then the object's own half-maximum extent around it.
        lo = max(seabed, change - 2 * t)
        hi = min(y.size, change + 2 * t)
        if hi - lo < 2:
            continue
        peak = int(np.argmax(y[lo:hi] - env[lo:hi])) + lo
        # The gap tolerance needs the predicted shadow, which needs a ground range, which needs
        # the extent — so it is derived at the PEAK's own range rather than at the centre's.
        # Over an object of the admissible size the two differ by under a metre of ground range
        # and the tolerance by under a bin; the CENTRE's range is used for the shadow test
        # itself, below, where the difference would matter.
        peak_ground = slant_to_ground_m((peak + 0.5) * cfg.range_res_m, cfg.altitude_m)
        if peak_ground is None:
            continue
        gap_bins = gap_tolerance_bins(cfg, peak_ground)
        # Provisional extent against the smoothed envelope, only to find where the object ends
        # so the background window can be placed off THE WHOLE SPAN; every LEVEL below is then
        # measured against that background.
        span_cap = max(1, int(round(cfg.across_max_m / max(cfg.range_res_m, 1e-9))))
        a0, b0 = _measure_extent(y, float(env[peak]), peak, gap_bins, span_cap)
        bg, bg_src = local_background(y, a0, b0 + 4 * t, cfg.background_guard_bins,
                                      2 * cfg.envelope_half_bins)
        a, b = _measure_extent(y, bg, peak, gap_bins, span_cap)
        across_m = (b - a) * cfg.range_res_m
        if not (cfg.across_min_m <= across_m <= cfg.across_max_m):
            killed_extent += 1
            continue
        centre = (a + b - 1) // 2
        slant = (centre + 0.5) * cfg.range_res_m
        ground = slant_to_ground_m(slant, cfg.altitude_m)
        if ground is None:
            continue
        expect_m = expected_shadow_len_m(ground, cfg.altitude_m, cfg.target_height_m)
        expect_bins = max(2, int(round(expect_m / cfg.range_res_m)))
        # THE RECORD CAN END BEFORE THE SHADOW DOES, and that is a different fact from "there
        # is no shadow". At the measured record geometry (40 m range, 4 cm bins) a 1.28 m
        # target at 35 m ground range under 6 m of altitude casts a 7.5 m shadow that runs 113
        # bins past the end of the ping. Reporting that as "no shadow" would blame the target
        # for the recorder's range setting.
        if b + int(cfg.shadow_fraction_min * expect_bins) >= y.size:
            killed_truncated += 1
            continue
        n_shadow, shadow_db = _shadow_after(y, bg, b, cfg, expect_bins)
        if n_shadow < cfg.shadow_fraction_min * expect_bins:
            killed_shadow += 1
            continue
        highlight_db = db_over(float(y[peak]), bg)
        if highlight_db < cfg.highlight_db_min:
            killed_dim += 1
            continue
        # Two window positions on opposite edges of one object can both survive the separation
        # check on CHANGE POINTS and then resolve to the same peak. Dedupe on the reported
        # CENTRE as well: one object, one hit, or the tracker counts a single ping twice and
        # persistence stops meaning what it says.
        if any(abs(centre - o) < t for o in taken):
            continue
        taken.append(centre)
        hits.append(PingHit(
            centre_bin=centre, start_bin=a, end_bin=b,
            highlight_db=highlight_db, shadow_db=shadow_db, shadow_bins=n_shadow,
            across_m=across_m,
            shadow_len_m=n_shadow * cfg.range_res_m,
            shadow_len_expected_m=expect_m,
            slant_range_m=slant, ground_range_m=ground,
            score=signature_score(highlight_db, shadow_db, across_m)))

    hits.sort(key=lambda h: h.centre_bin)
    if not hits:
        why = (f"{n_rises} rise(s) passed the derived {ratio:.1f}x threshold, "
               f"{len(evaluated)} evaluated; {killed_shadow} carried no shadow of the "
               f"predicted length, {killed_extent} were the wrong across-track size and "
               f"{killed_dim} were dimmer than {cfg.highlight_db_min:.0f} dB over the local "
               f"background, and for {killed_truncated} the RECORD ENDED before their "
               f"predicted shadow did (the range setting, not the target). Nothing on this "
               f"seabed has the highlight-and-shadow signature") if n_rises else \
              (f"no rise reached the derived {ratio:.1f}x threshold over the seabed region "
               f"({d.size} window positions, budget "
               f"{cfg.false_alarms_per_ping}/ping) — this seabed is featureless")
        return PingReport(True, why, nadir_bin=seabed, n_rises=n_rises,
                          n_killed_no_shadow=killed_shadow, n_killed_extent=killed_extent,
                          n_killed_truncated=killed_truncated)
    return PingReport(True, "ok", tuple(hits), nadir_bin=seabed, n_rises=n_rises,
                      n_killed_no_shadow=killed_shadow, n_killed_extent=killed_extent,
                      n_killed_truncated=killed_truncated)


# --------------------------------------------------------------------------------------
# persistence across pings
# --------------------------------------------------------------------------------------
@dataclass
class _Track:
    side: str
    first_ping: int
    last_ping: int
    bins: List[int] = field(default_factory=list)
    hits: List[PingHit] = field(default_factory=list)
    emitted: bool = False


class SssTargetTracker:
    """Turns per-ping hits into candidates by requiring PERSISTENCE.

    "Sustained, not a spike" is the same rule the nadir search uses and the FLS detector uses,
    and it is the second gate the strategy names: a single bright ping cannot make a candidate.
    `PERSISTENCE_PINGS` is half the car's own along-track extent at scan speed.

    Association across pings is by RANGE BIN within one across-track window: the vehicle moves
    along the swath, not across it, so a real object's slant range changes slowly while a
    speckle spike lands anywhere. A track that misses more than `max_gap_pings` consecutive
    pings is closed, because a shadow that reappears 200 pings later is a different object.
    """

    def __init__(self, *, persistence_pings: int = PERSISTENCE_PINGS,
                 gate_bins: int = 24, max_gap_pings: int = 6):
        self.persistence_pings = int(persistence_pings)
        self.gate_bins = int(gate_bins)
        self.max_gap_pings = int(max_gap_pings)
        self._tracks: List[_Track] = []
        self.closed: List[_Track] = []

    def update(self, ping_index: int, side: str, report: PingReport) -> List[_Track]:
        """Feed one ping. Returns the tracks that JUST reached persistence this ping.

        THE GAP RULE IS CHECKED TWICE, deliberately and redundantly: once when deciding whether
        a hit may join an existing track, and once when deciding whether a track is still open.
        Mutation-tested 2026-09-09: removing EITHER check alone leaves the behaviour unchanged,
        because the other still ends the track. Both are kept because they answer different
        questions — "is this the same object?" and "is this track still alive?" — and a later
        change to one is not a licence to drop the other. Recorded here so neither reads as
        dead code.
        """
        for h in report.hits:
            match = None
            for tr in self._tracks:
                if tr.side != side:
                    continue
                if ping_index - tr.last_ping > self.max_gap_pings:
                    continue
                if abs(tr.bins[-1] - h.centre_bin) <= self.gate_bins:
                    match = tr
                    break
            if match is None:
                match = _Track(side=side, first_ping=ping_index, last_ping=ping_index)
                self._tracks.append(match)
            match.last_ping = ping_index
            match.bins.append(h.centre_bin)
            match.hits.append(h)

        ripe: List[_Track] = []
        keep: List[_Track] = []
        for tr in self._tracks:
            if ping_index - tr.last_ping > self.max_gap_pings:
                self.closed.append(tr)
                continue
            keep.append(tr)
            if not tr.emitted and len(tr.hits) >= self.persistence_pings:
                tr.emitted = True
                ripe.append(tr)
        self._tracks = keep
        return ripe

    @property
    def open_tracks(self) -> List[_Track]:
        return list(self._tracks)


# --------------------------------------------------------------------------------------
# georeferencing and the candidate record
# --------------------------------------------------------------------------------------
def sigma_for(dr_since_fix_m: float, range_res_m: float, along_track_m: float,
              ground_range_m: float, altitude_m: float,
              altitude_sigma_m: float = 0.2) -> SigmaBreakdown:
    """The decomposed positional sigma of a side-scan candidate.

    Four parts, each of which is a different thing that can go wrong:

      * `dr_m` — how far the dead reckoning may have drifted since the last fix. Handed in,
        because this module does not own the estimator and must not invent its error.
      * `bin_quantisation_m` — half a range bin, projected into GROUND range. The projection
        matters: at grazing incidence a slant bin covers far more ground than it does at
        nadir, and quoting the slant bin here would understate the error exactly where the
        detector works.
      * `along_track_m` — the centring error along the swath, i.e. how well the track's
        along-track midpoint locates the object.
      * `slant_geometry_m` — what the altimeter's own uncertainty does to the ground range,
        d(ground)/dh · sigma_h = (h/ground)·sigma_h.

    Quadrature, and the parts travel with the total (§3s7): a sigma that cannot be taken apart
    cannot be checked.
    """
    if ground_range_m <= 0:
        raise SssRefusal("a candidate inside the nadir gap has no ground range and therefore "
                         "no position; it must be refused, not placed at the vehicle")
    slant = math.sqrt(ground_range_m ** 2 + altitude_m ** 2)
    dground_dslant = slant / ground_range_m
    return SigmaBreakdown(
        dr_m=float(dr_since_fix_m),
        bin_quantisation_m=0.5 * range_res_m * dground_dslant,
        along_track_m=float(along_track_m),
        slant_geometry_m=abs(altitude_m / ground_range_m) * float(altitude_sigma_m),
    )


def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> Tuple[float, float]:
    """Local flat-earth offset. Explicit rather than borrowed so the constants are visible.

    111320·cos(lat) m per degree of longitude and 110540 m per degree of latitude — the same
    pair Mission Control uses (SETTLED §3u measured MC's metric math as ~0.8 % off a geodesic,
    which over the 15–40 m offsets this function is used for is under 0.3 m, i.e. well inside
    the sigma above; recorded here so nobody has to re-derive whether it matters).
    """
    dlat = north_m / 110540.0
    dlon = east_m / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lat + dlat, lon + dlon


def georeference(track_bin: int, side: str, cfg: SssConfig, *,
                 lat: Optional[float], lon: Optional[float],
                 course_deg: Optional[float]) -> Tuple[Optional[float], Optional[float], str]:
    """Place a candidate abeam of the vehicle, using COURSE OVER GROUND — never raw yaw.

    §3f0h's method note, and it is not pedantry: the yaw published on this vehicle comes from a
    frame whose sign convention has already been wrong twice (SETTLED §3t on `smarc/depth`,
    §3s on the ocean transform), and near a steel car the compass is unreliable by construction
    (memory `heading_observability_strategy`). Course over ground is derived from successive DR
    POSITIONS, which is a different and more robust measurement.

    Returns (lat, lon, reason). A missing course or fix yields (None, None, reason) — a
    candidate with no position is still a candidate, and placing it at the vehicle would be a
    fabrication.
    """
    if lat is None or lon is None:
        return None, None, "no vehicle position for this ping; the candidate has no georeference"
    if course_deg is None:
        return None, None, ("no course over ground for this ping; the swath cannot be placed "
                            "and raw yaw is not a substitute (SETTLED §3f0h)")
    slant = (track_bin + 0.5) * cfg.range_res_m
    ground = slant_to_ground_m(slant, cfg.altitude_m)
    if ground is None:
        return None, None, (f"slant range {slant:.1f} m is inside the {cfg.altitude_m:.1f} m "
                            f"nadir gap; there is no ground range to place")
    # starboard is +90 deg from course, port is -90.
    sign = 1.0 if side == "starboard" else -1.0
    bearing = math.radians(course_deg + sign * 90.0)
    north = ground * math.cos(bearing)
    east = ground * math.sin(bearing)
    la, lo = offset_latlon(lat, lon, north, east)
    return la, lo, "ok"


def candidate_from_track(track: _Track, cfg: SssConfig, *, cid: str, t: float,
                         lat: Optional[float], lon: Optional[float],
                         course_deg: Optional[float], dr_since_fix_m: float,
                         along_track_m: float) -> SssCandidate:
    """Build the strategy §4 record from a persisted track. Reports the CENTRE, everywhere."""
    bins = sorted(track.bins)
    centre_bin = bins[len(bins) // 2]                  # median bin: robust to a stray ping
    hits = track.hits
    la, lo, _why = georeference(centre_bin, track.side, cfg, lat=lat, lon=lon,
                                course_deg=course_deg)
    slant = (centre_bin + 0.5) * cfg.range_res_m
    ground = slant_to_ground_m(slant, cfg.altitude_m) or 0.0
    sig = sigma_for(dr_since_fix_m, cfg.range_res_m, along_track_m, max(ground, 1e-6),
                    cfg.altitude_m, cfg.altitude_sigma_m)
    med = lambda xs: float(np.median(np.asarray(xs, dtype=np.float64)))   # noqa: E731
    return SssCandidate(
        id=cid, t=t, side=track.side, ping=track.last_ping, bin=centre_bin,
        lat=la, lon=lo, sigma_m=sig.total_m, sigma_parts=sig.as_dict(),
        highlight_db=med([h.highlight_db for h in hits]),
        shadow_db=med([h.shadow_db for h in hits]),
        extent_m={"across": med([h.across_m for h in hits]),
                  "along": along_track_m * 2.0,
                  "shadow_len": med([h.shadow_len_m for h in hits])},
        score=med([h.score for h in hits]),
        n_pings=len(hits))
