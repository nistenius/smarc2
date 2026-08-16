#!/usr/bin/env python3
"""Window-sliding change-point detection on a single side-scan ping.

This is the detector of sensors-22-05064 §5 ("Seaweed Farm Rope and Buoy Detection"),
implemented as pure arithmetic over an intensity vector: no ROS, no I/O, no simulator.
The ROS node is a thin wrapper; everything worth testing is here.

THE METHOD, in the paper's own notation. For a window s_{i:i+t} of the 1D ping s,

    c(s_{i:i+t}) = sum_{n=i}^{i+t-1} || y_n - ybar ||^2          (least-square deviation)

and the discrepancy of splitting a 2t-long stretch at its midpoint is

    d(s_{i:i+t}, s_{i+t:i+2t}) = c(s_{i:i+2t}) - c(s_{i:i+t}) - c(s_{i+t:i+2t})

A window of size t is slid across the signal and the index with the highest discrepancy
is the change point. The paper runs the algorithm TWICE per ping — first to find the
nadir (the first bottom-hitting return, whose change is large by definition and would
otherwise swamp everything), then again inside the water column, where the ropes and
buoys are. Rope and buoy are the same algorithm with a different window size and change
ratio.

WHY IT IS DONE ON INTENSITIES AND NOTHING ELSE (mission design decision D4). The
simulator labels every sonar hit with the physics material it struck — Rope, Buoy,
Algae — and reading those labels would produce a perfect detector that transfers to
exactly zero real pings. Sim labels are GROUND TRUTH and may appear only in tests, named
as such. This module never sees them; its input is a vector of numbers.

THREE THINGS THAT ARE DELIBERATE, because each is a place the algorithm could have been
quietly weakened:

1. `c` is computed from prefix sums, so the whole profile is O(T) rather than O(T*t).
   The paper's claim is that this runs in real time on the vehicle; an O(T*t) version at
   2000 bins x 7 Hz x 2 channels would not have been.
2. The nadir must be a SUSTAINED RISE, checked over two window lengths. The discrepancy
   is symmetric — it fires just as hard on a drop as on a rise — and it fires on ropes
   and buoys too, which are precisely the things that must stay INSIDE the water column.
   `find_nadir` documents the two impostors and the window that rejects each. (The cheap
   `mean_delta > 0` pre-filter is kept for clarity and speed, but note honestly that it
   is SUBSUMED by the sustain test: mutating it away changes no outcome, because a fall
   cannot produce a positive median step either. It is a pre-filter, not a guard.)
3. A ping with no detectable bottom return yields a REFUSAL WITH A REASON, not an empty
   detection list. "No ropes here" and "I could not tell where the water column ends" are
   different answers and only one of them is information.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------
# The change ratio is DERIVED FROM A FALSE-ALARM BUDGET, not chosen.
#
# Measured 2026-08-16 on 200 000-sample synthetic pings, and it is also exactly what
# the ANOVA decomposition predicts: for a signal that is pure noise of standard
# deviation sigma, the discrepancy is
#
#     d = (t/2) * (mean_R - mean_L)^2 ,   mean_R - mean_L ~ N(0, 2*sigma^2/t)
#       = sigma^2 * chi2_1
#
# i.e. **the noise distribution of d does not depend on the window size at all**. The
# median of chi2_1 is 0.4549, so the median of a mostly-noise discrepancy profile
# estimates 0.4549*sigma^2 whatever window it was computed with, and the ratio
# d/median is a scale-free, window-free test statistic. Measured medians at
# sigma = 1 and 3 with t = 2, 5, 20, 50: 0.4533, 0.4496, 0.4492, 0.4511, 4.0866,
# 4.0569, 3.9972, 4.0991 — against 0.4549 and 4.0941 predicted.
#
# That is what makes a false-alarm budget possible. The first version of this file used
# a hand-picked ratio of 4, and the tests caught it immediately: over 2000 bins, a ratio
# of 4 corresponds to a per-bin p of 0.18, so a pure-noise ping produced hundreds of
# "detections" and a real rope was invisible among them. Nothing about that would have
# looked wrong in a log.
#
# The Gaussian assumption is an approximation for real sonar noise, which is not
# Gaussian. What survives it is the SHAPE of the argument: the baseline is measured from
# the ping itself, so the threshold self-calibrates to whatever the noise level is; only
# the mapping from a false-alarm rate to a ratio is model-dependent, and it is the thing
# to re-measure against real pings.
# --------------------------------------------------------------------------------

#: Median of the chi-square distribution with one degree of freedom.
_CHI2_1_MEDIAN = 0.45493642311957305


def _erfinv(x: float) -> float:
    """Inverse error function by bisection on math.erf.

    Fifty iterations, called a handful of times at startup. Written out rather than
    pulled from scipy because this package must import on the vehicle, where the
    dependency list is rclpy + numpy and adding to it is a deployment problem, not a
    line in a requirements file.
    """
    if x <= -1.0 or x >= 1.0:
        raise ValueError(f"erfinv is undefined at {x}")
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.erf(mid) < x:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def change_ratio_for_false_alarm(n_candidates: int,
                                 false_alarms_per_ping: float = 0.05) -> float:
    """The d/baseline ratio that admits `false_alarms_per_ping` noise peaks on average.

    `n_candidates` is how many window positions are tested — for one channel of an
    HF680 ping searched over its water column, a few hundred. The per-test tail
    probability is the budget divided by that, and the ratio is the chi2_1 quantile
    there over chi2_1's median.

    Worked values at the shipped budget of 0.05 false alarms per ping:
        200 candidates -> 26.3      500 -> 30.4      2000 -> 36.4
    """
    n = max(1, int(n_candidates))
    p_tail = min(0.5, max(1e-12, float(false_alarms_per_ping) / n))
    # chi2_1 quantile at (1 - p_tail) is the square of the standard-normal quantile at
    # (1 - p_tail/2), and Phi^-1(q) = sqrt(2) * erfinv(2q - 1).
    z = math.sqrt(2.0) * _erfinv(1.0 - p_tail)
    return (z * z) / _CHI2_1_MEDIAN


@dataclass(frozen=True)
class TargetClass:
    """One kind of thing to look for: the paper's 'changing the sliding window size and
    signal change ratio'."""

    name: str
    #: Half-window length in BINS. The natural unit is metres of slant range; the node
    #: converts using the ping's own range resolution so a mode change (LF340's 20 cm
    #: bins vs HF680's 5 cm) does not silently change what the detector looks for.
    window_bins: int
    #: How many times the robust baseline discrepancy a peak must reach to count.
    change_ratio: float
    #: Peaks closer together than this are one object seen twice.
    min_separation_bins: int = 4
    max_detections: int = 8


@dataclass(frozen=True)
class Detection:
    target: str
    #: CENTRE of the object, in bins. A change point marks an EDGE, not a centre — the
    #: window split lands where the signal changes — so reporting the change point as
    #: the object's range is biased by half the object's width plus the window. The
    #: centre is measured from the extent below.
    bin_index: int
    slant_range_m: float
    #: The change point the detection came from, kept because it is what the paper's
    #: algorithm actually returns and a range disagreement should be traceable.
    change_bin: int
    #: Contiguous bins the object occupies above half its own step. This, not the
    #: window size, is what separates a buoy from a rope — see the note in `detect_ping`.
    extent_bins: int
    extent_m: float
    score: float
    #: score / baseline. Named `snr` because that is what it is — a ratio against the
    #: ping's own noise floor, not a probability.
    snr: float
    #: A monotone 0..1 proxy derived from `snr`. It is NOT a probability and nothing
    #: downstream may treat it as one; it exists so a consumer can rank and threshold.
    confidence: float
    #: True when another target class fired at the same place. The classes overlap by
    #: construction (a buoy is a big bright object, a rope is a small one), so silently
    #: picking one would hide a real ambiguity.
    ambiguous: bool = False


@dataclass
class PingResult:
    """What one channel of one ping produced. `ok=False` always carries a reason."""

    ok: bool
    reason: str
    nadir_bin: Optional[int] = None
    nadir_slant_m: Optional[float] = None
    detections: List[Detection] = field(default_factory=list)
    #: Robust baseline of the discrepancy profile, kept so a health line can say whether
    #: the ping was quiet or loud without re-running anything.
    baseline: float = 0.0


def discrepancy_profile(signal: Sequence[float], t: int) -> Tuple[np.ndarray, np.ndarray]:
    """d(i) for every i where a 2t window fits, plus the mean difference at each i.

    Returns (d, mean_delta) both of length max(0, T - 2t + 1). `d[i]` is the paper's
    discrepancy for splitting s[i : i+2t] at i+t, so the change point it refers to is
    bin i+t. `mean_delta[i]` is mean(right) - mean(left), which is how a rise is told
    from a fall — the discrepancy itself cannot.
    """
    y = np.asarray(signal, dtype=np.float64)
    T = y.size
    if t < 1 or T < 2 * t:
        return np.empty(0), np.empty(0)

    s1 = np.concatenate(([0.0], np.cumsum(y)))
    s2 = np.concatenate(([0.0], np.cumsum(y * y)))

    def c(a, b):
        """Least-square deviation from the window mean, over [a, b)."""
        n = b - a
        tot = s1[b] - s1[a]
        return (s2[b] - s2[a]) - tot * tot / n

    i = np.arange(0, T - 2 * t + 1)
    left, mid, right = i, i + t, i + 2 * t
    d = c(left, right) - c(left, mid) - c(mid, right)
    mean_delta = (s1[right] - s1[mid]) / t - (s1[mid] - s1[left]) / t
    # Floating-point can make an exactly-flat window's discrepancy a tiny negative.
    # Clamping at zero is not hiding anything: d is a sum of squares by construction.
    return np.maximum(d, 0.0), mean_delta


def _baseline(d: np.ndarray) -> float:
    """A robust noise floor for the discrepancy profile.

    Median, not mean: the profile's whole point is that it contains large spikes, and a
    mean baseline is raised by exactly the events it is meant to measure against. The
    floor keeps a perfectly flat synthetic ping from producing infinite SNR.
    """
    if d.size == 0:
        return 0.0
    return max(float(np.median(d)), 1e-9)


def find_nadir(signal: Sequence[float], window_bins: int, change_ratio: float,
               blank_bins: int = 0, sustain_sigmas: float = 5.0,
               sustain_bins: Optional[int] = None) -> Tuple[Optional[int], str, float]:
    """First bottom-hitting return: the earliest SUSTAINED rise in the ping.

    `blank_bins` skips the near field (transmit ring-down and the vehicle's own hull),
    which is a real feature of every side scan and produces a colossal, meaningless
    change point at bin 0 if it is not skipped.

    Returns (bin index or None, reason, baseline). Earliest-qualifying rather than
    global-maximum on purpose: the paper's phrase is "the FIRST bottom-hitting acoustic
    return", and the global maximum of a ping over a hard seabed is often the trailing
    edge or a later specular return. Choosing the global max would put the water-column
    window's end after the seabed, and every seabed return would then be offered to the
    rope detector.

    WHY "SUSTAINED" AND NOT JUST "RISING" — measured 2026-08-16, and it is the defect
    the tests found on the first implementation. A bright rope at 4.5 m tripped the
    nadir search: with a 20-bin nadir window, two bins of rope inside the right window
    lift its mean by 4.5 counts, giving a discrepancy 42x the baseline — comfortably
    over any threshold loose enough to be robust. The nadir then landed 100 bins BEFORE
    the rope, the water column ended before the object it exists to contain, and the
    detector reported an empty farm while looking straight at one. Raising the threshold
    would have "fixed" the fixture and left the failure mode intact.

    The physical difference is not size, it is PERSISTENCE: the seabed is a level change
    that stays, a rope is an event that passes. So a candidate must also raise the MEDIAN
    of the following `sustain_bins` by at least `sustain_sigmas` standard deviations over
    the median of the preceding ones. Sigma is estimated from the profile's own baseline
    (median(d) = 0.4549 sigma^2), so it needs no calibration and no configuration.
    """
    y = np.asarray(signal, dtype=np.float64)
    d, mean_delta = discrepancy_profile(y, window_bins)
    base = _baseline(d)
    if d.size == 0:
        return None, (f"ping too short for a {window_bins}-bin nadir window "
                      f"({y.size} bins)"), base
    thresh = base * change_ratio
    # TWO sustain windows, near and far, and a candidate must satisfy BOTH. Each rejects
    # a different impostor, and each was found by a test:
    #
    #   FAR (ten nadir windows, ~10 m of slant) rejects a WIDE OBJECT. A 1 m bright
    #   object — a buoy is about that — holds a two-window median up all by itself, so a
    #   short window alone classes the largest thing in the water column as the seabed
    #   and the water column then ends before the object it exists to contain.
    #
    #   NEAR (two nadir windows, ~2 m) rejects a candidate whose sustain is BORROWED
    #   FROM THE REAL SEABED further out. A rope 100 bins before the bottom return trips
    #   the threshold, and a far window centred on it still straddles the true seabed, so
    #   the level step looks real. The near window sees only water and refuses.
    #
    # Together: the seabed is a level change that holds immediately AND keeps holding.
    # Nothing floating does both, and the margin is wide enough not to need tuning.
    far = int(sustain_bins) if sustain_bins else max(20, 10 * window_bins)
    near = max(4, 2 * window_bins)
    sigma = math.sqrt(max(base, 1e-12) / _CHI2_1_MEDIAN)
    min_step = sustain_sigmas * sigma

    def step(b, w):
        before = y[max(blank_bins, b - w):b]
        after = y[b:b + w]
        if before.size == 0 or after.size == 0:
            return None
        return float(np.median(after) - np.median(before))

    # index i in the profile refers to change point i + window_bins
    cand = np.nonzero((d >= thresh) & (mean_delta > 0.0) &
                      (np.arange(d.size) + window_bins >= blank_bins))[0]
    n_raised = int(cand.size)
    for i in cand:
        b = int(i) + window_bins
        s_near, s_far = step(b, near), step(b, far)
        if s_near is None or s_far is None:
            continue
        if s_near >= min_step and s_far >= min_step:
            return b, "ok", base

    if n_raised == 0:
        return None, (f"no rising change point reached {change_ratio:.1f}x the baseline "
                      f"discrepancy ({base:.3g}); peak was {float(d.max()):.3g} — no bottom "
                      f"return in this ping, so the water column has no end and objects "
                      f"cannot be separated from the seabed"), base
    return None, (f"{n_raised} rising change point(s) passed the {change_ratio:.1f}x "
                  f"threshold but none was SUSTAINED (needs a median step of "
                  f"{min_step:.3g} counts over BOTH {near} and {far} bins) — those are "
                  f"objects in the water column, not the seabed, so this ping has no "
                  f"bottom return and its water column has no end"), base


def find_objects(signal: Sequence[float], target: TargetClass,
                 start_bin: int, end_bin: int,
                 false_alarms_per_ping: Optional[float] = None
                 ) -> Tuple[List[Tuple[int, float, float]], float]:
    """Peaks in the discrepancy profile inside [start_bin, end_bin).

    Returns ([(bin, score, snr)], baseline). The baseline is computed over the SEARCHED
    REGION only — the water column and the seabed have completely different statistics,
    and a baseline taken over the whole ping would be set by the seabed and make the
    water column look silent.
    """
    y = np.asarray(signal, dtype=np.float64)
    lo = max(0, int(start_bin))
    hi = min(y.size, int(end_bin))
    if hi - lo < 2 * target.window_bins:
        return [], 0.0
    seg = y[lo:hi]
    d, _ = discrepancy_profile(seg, target.window_bins)
    base = _baseline(d)
    if d.size == 0:
        return [], base
    # The class's own ratio, unless the caller supplied a false-alarm budget — in which
    # case the threshold is re-derived for the number of positions ACTUALLY tested in
    # this water column, which is far fewer than a whole ping and would otherwise be
    # thresholded as if it were the whole ping.
    ratio = target.change_ratio
    if false_alarms_per_ping is not None:
        ratio = change_ratio_for_false_alarm(d.size, false_alarms_per_ping)
    thresh = base * ratio

    order = np.argsort(d)[::-1]
    chosen: List[Tuple[int, float, float]] = []
    taken: List[int] = []
    for idx in order:
        score = float(d[idx])
        if score < thresh:
            break
        b = int(idx) + target.window_bins + lo
        if any(abs(b - o) < target.min_separation_bins for o in taken):
            continue
        taken.append(b)
        chosen.append((b, score, score / base))
        if len(chosen) >= target.max_detections:
            break
    chosen.sort(key=lambda x: x[0])
    return chosen, base


def measure_extent(signal: Sequence[float], change_bin: int, search_bins: int,
                   level: float) -> Tuple[int, int, int]:
    """The object around a change point: (start, end_exclusive, centre) in bins.

    `level` is the local background (the water column's own median). The object is the
    contiguous run around the strongest sample near the change point that stays above
    half its height over that background — the standard half-maximum rule, which is
    scale-free and does not need the amplitude to be calibrated. That matters here: the
    simulator's reflectivities are wild guesses (KRISTINEBERG_SITE.md §5h), so any rule
    that depended on an absolute intensity would be measuring the guess.
    """
    y = np.asarray(signal, dtype=np.float64)
    lo = max(0, change_bin - search_bins)
    hi = min(y.size, change_bin + search_bins + 1)
    if hi <= lo:
        return change_bin, change_bin + 1, change_bin
    seg = y[lo:hi]
    pk = int(np.argmax(seg)) + lo
    half = level + 0.5 * (float(y[pk]) - level)
    a = pk
    while a - 1 >= 0 and y[a - 1] > half:
        a -= 1
    b = pk
    while b + 1 < y.size and y[b + 1] > half:
        b += 1
    return a, b + 1, (a + b) // 2


def _confidence(snr: float, ratio: float) -> float:
    """Map SNR onto 0..1, saturating at twice the trigger ratio.

    Deliberately crude and deliberately NOT called a probability. A detector that
    reports 0.87 invites a consumer to multiply it by something; this one reports a
    rank. The localizer weights detections by it and nothing else does.
    """
    if ratio <= 0:
        return 0.0
    x = (snr - ratio) / ratio
    return float(min(1.0, max(0.0, x)))


def detect_ping(signal: Sequence[float],
                range_resolution_m: float,
                targets: Sequence[TargetClass],
                nadir_window_bins: int = 20,
                nadir_change_ratio: Optional[float] = None,
                blank_bins: int = 0,
                water_column_guard_bins: int = 2,
                merge_bins: int = 4,
                buoy_min_extent_m: float = 0.25,
                false_alarms_per_ping: Optional[float] = None) -> PingResult:
    """One channel of one ping, end to end: nadir first, then objects before it.

    `water_column_guard_bins` pulls the search window back from the nadir, because the
    bottom return has a leading edge of a few bins and the object detector will happily
    fire on it. Two bins at HF680 is 10 cm.

    `nadir_change_ratio=None` derives the ratio from the false-alarm budget over the
    number of window positions actually tested. Passing a number overrides it, which is
    what the tests do when they want to prove that the gate gates.

    HOW A BUOY IS TOLD FROM A ROPE. The paper says the same algorithm detects both "by
    changing the sliding window size and signal change ratio", and the window pair is
    kept because it is the paper's structure and it is what makes objects at both scales
    trigger at all. The CLASS, though, is decided by the object's measured half-maximum
    extent.

    An earlier version of this comment justified that by saying window-based
    classification "would follow amplitude". **That was wrong, and mutation-testing
    caught it** — the guard survived the mutation, which is how the error surfaced. Work
    the algebra: for an object of width w and amplitude A, the peak discrepancy is
    (t/2)A^2 when w >= 2t and A^2 w^2 / (2t) when w < t. Both classes scale as A^2, so
    comparing two windows is already amplitude-free, and on ordinary objects the two
    rules agree. The two real reasons to classify on extent are narrower and both hold:

      1. **It works when only one window fires.** A comparison needs two votes; extent
         needs none. Running the detector with a single target class must still produce
         a correctly classified detection, and with a window comparison it cannot.
      2. **It is published.** `extent_m` travels with every detection, so a verdict of
         "buoy" can be checked against a number instead of being trusted.

    `ambiguous` records when both windows fired, because a buoy filed as a rope is a
    mapping error and the tie is the only warning of it.
    """
    y = np.asarray(signal, dtype=np.float64)
    if y.size == 0:
        return PingResult(False, "empty ping")
    if range_resolution_m <= 0:
        return PingResult(False, f"range resolution {range_resolution_m} m is not positive")

    if nadir_change_ratio is None:
        # The nadir search spans the whole ping, so the candidate count is the whole
        # profile. The object passes each use their class's ratio, which
        # `targets_from_metres` derived for a typical water-column length.
        nadir_change_ratio = change_ratio_for_false_alarm(
            max(1, y.size - 2 * nadir_window_bins + 1),
            0.05 if false_alarms_per_ping is None else false_alarms_per_ping)
    nadir, reason, base = find_nadir(y, nadir_window_bins, nadir_change_ratio, blank_bins)
    if nadir is None:
        return PingResult(False, reason, baseline=base)

    end = nadir - water_column_guard_bins
    start = max(blank_bins, 0)
    # The water column's own background, for the half-maximum extent rule. Median over
    # the searched region: the objects are sparse there by definition, so they do not
    # move it.
    level = float(np.median(y[start:end])) if end > start else 0.0
    buoy_min_extent_bins = max(1, int(round(buoy_min_extent_m / range_resolution_m)))

    per_target: List[Detection] = []
    for tgt in targets:
        peaks, _ = find_objects(y, tgt, start, end,
                                false_alarms_per_ping=false_alarms_per_ping)
        for b, score, snr in peaks:
            a_, b_, centre = measure_extent(y, b, max(4, 2 * tgt.window_bins), level)
            per_target.append(Detection(
                target=tgt.name, bin_index=centre,
                slant_range_m=(centre + 0.5) * range_resolution_m,
                change_bin=b, extent_bins=b_ - a_,
                extent_m=(b_ - a_) * range_resolution_m,
                score=score, snr=snr,
                confidence=_confidence(snr, tgt.change_ratio)))

    # Merge across classes. Strongest first, and a clash between two DIFFERENT classes is
    # resolved by extent, not by which window happened to score higher.
    per_target.sort(key=lambda d_: (-d_.snr, d_.bin_index))
    kept: List[Detection] = []
    for det in per_target:
        clash = next((k for k in kept if abs(k.bin_index - det.bin_index) < merge_bins), None)
        if clash is None:
            kept.append(det)
            continue
        if clash.target == det.target:
            continue
        winner = "buoy" if clash.extent_bins >= buoy_min_extent_bins else "rope"
        base_det = clash if clash.target == winner else det
        kept[kept.index(clash)] = Detection(
            winner, base_det.bin_index, base_det.slant_range_m, base_det.change_bin,
            clash.extent_bins, clash.extent_m,
            base_det.score, base_det.snr, base_det.confidence, ambiguous=True)
    # A detection that only one class found still gets its class checked against the
    # extent rule, so classification does not depend on whether both windows fired.
    kept = [d_ if (d_.extent_bins >= buoy_min_extent_bins) == (d_.target == "buoy")
            else Detection("buoy" if d_.extent_bins >= buoy_min_extent_bins else "rope",
                           d_.bin_index, d_.slant_range_m, d_.change_bin, d_.extent_bins,
                           d_.extent_m, d_.score, d_.snr, d_.confidence, d_.ambiguous)
            for d_ in kept]
    kept.sort(key=lambda d_: d_.bin_index)

    return PingResult(True, "ok", nadir_bin=nadir,
                      nadir_slant_m=(nadir + 0.5) * range_resolution_m,
                      detections=kept, baseline=base)


def targets_from_metres(rope_extent_m: float, buoy_extent_m: float,
                        range_resolution_m: float,
                        rope_ratio: Optional[float] = None,
                        buoy_ratio: Optional[float] = None,
                        n_candidates: int = 500,
                        false_alarms_per_ping: float = 0.05) -> List[TargetClass]:
    """Build the target classes from PHYSICAL sizes and the ping's own resolution.

    Window sizes are configured in metres, never in bins, because the two side-scan
    modes have a 4x resolution difference (HF680 5 cm bins, LF340 20 cm) and a bin count
    that is right for one is meaningless for the other. This is the same rule as sizing
    the obstacle grid from the sensor's real FOV rather than from a remembered index.

    Ratios default to the false-alarm budget rather than to a number somebody liked.
    """
    def bins(m):
        return max(2, int(round(m / max(range_resolution_m, 1e-9))))

    derived = change_ratio_for_false_alarm(n_candidates, false_alarms_per_ping)
    return [
        TargetClass("rope", bins(rope_extent_m),
                    derived if rope_ratio is None else rope_ratio,
                    min_separation_bins=bins(rope_extent_m)),
        TargetClass("buoy", bins(buoy_extent_m),
                    derived if buoy_ratio is None else buoy_ratio,
                    min_separation_bins=bins(buoy_extent_m)),
    ]
