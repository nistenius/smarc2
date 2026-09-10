#!/usr/bin/env python3
"""What the inspection recorder decides — strategy §6.1. Pure python + numpy, no ROS, no I/O.

ADR-010 GOVERNS THIS FILE. The recorder is a SENSOR in that document's sense: it runs only
while a station burst is commanded, it measures its own cost, and it does no reconstruction,
no feature matching and no learned detector on the flight computer. The 30 W cap
(`SAM_VEHICLE_SPECS.md` §7, written after camera-induced reboots) is the reason, and §3s7
already set the rule: measure the residual before buying the model.

WHAT IT DECIDES, AND WHY EACH DECISION IS A NUMBER RATHER THAN A FEELING

  * a FRAME is accepted on two cheap numbers — Laplacian variance (blur) and mean intensity
    (exposure). Two numbers, no model. The thresholds are RELATIVE to the burst's own frames,
    not absolute, because absolute sharpness depends on the scene and on the water, and a
    fixed number would accept everything in clear water and nothing in Baltic autumn;
  * a SCAN is accepted on point count and POSE FRESHNESS. A pose-less scan is not a scan: the
    station registration at the base station is seeded with these poses (§6.2), and a scan
    tagged with a stale pose is worse than a missing one because it will be trusted;
  * a STATION is done when `min_frames` have been ACCEPTED — a count of evidence, never a
    stopwatch (SETTLED §3e). That number is what the behaviour tree reads to move on.

AND WHAT IT REFUSES TO DO: it never reports a coverage figure that includes rejected frames,
and `station_done` is False with a REASON, never a bare False.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Voxel edge for thinning a sonar scan before it is written, metres. 5 cm: the 3D-15's
#: INSPECTION-mode range resolution is 1.5 mm and its HF beams are 0.45° × 0.85° (≈ 2.4 cm
#: lateral sampling at 3 m, strategy §4b), so 5 cm keeps roughly one point per two beam
#: footprints at the ring radius — dense enough that the box fit is unchanged, coarse enough
#: that a station's scan is bounded.
VOXEL_M = 0.05

#: Hard cap on points kept per scan after thinning. Bounded cost is the whole of ADR-010's
#: argument: an unbounded write is how a payload node becomes the reason a bag fills a disk.
MAX_POINTS_PER_SCAN = 20000

#: A pose older than this is not this frame's pose. 0.5 s at 0.3 m/s is 15 cm — under the
#: 5 cm voxel by three, which is the scale at which the registration would notice.
POSE_MAX_AGE_S = 0.5


class RecorderRefusal(RuntimeError):
    """Raised with an operator-readable reason. Never caught and defaulted."""


# --------------------------------------------------------------------------------------
# the two cheap frame numbers
# --------------------------------------------------------------------------------------
def laplacian_variance(image: Sequence[Sequence[float]]) -> float:
    """Variance of the 4-neighbour Laplacian — the standard cheap blur measure.

    Written out rather than pulled from OpenCV: this package must import on the vehicle, where
    the dependency list is rclpy + numpy and adding to it is a deployment problem rather than a
    line in a requirements file (the same reasoning `change_point._erfinv` is written out for).

    On a 848 × 480 frame this is four array shifts and a variance — a few milliseconds, and the
    node prints its own per-frame cost so the claim is measured rather than asserted.
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        # Luma, not a mean of channels: a mean weights blue as heavily as green, and in water
        # the blue channel is the one carrying the least information (§3f0u's transmittance).
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        raise RecorderRefusal(f"a {a.shape} array is not an image this gate can measure")
    lap = (-4.0 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] +
           a[1:-1, :-2] + a[1:-1, 2:])
    return float(np.var(lap))


def mean_intensity(image: Sequence[Sequence[float]]) -> float:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return float(np.mean(a))


@dataclass(frozen=True)
class FrameGate:
    """The acceptance rule for one frame, with both thresholds RELATIVE to the burst.

    `blur_floor_fraction` of the burst's own MEDIAN Laplacian variance, so a burst taken in
    turbid water is judged against itself. The exposure band is absolute because a frame that
    is clipped black or clipped white carries no information at any clarity — 8-bit, so 8 and
    247 are one step inside each rail.
    """

    blur_floor_fraction: float = 0.5
    exposure_min: float = 8.0
    exposure_max: float = 247.0

    def verdict(self, lap_var: float, mean_i: float, burst_median_lap: float) -> Tuple[bool, str]:
        if mean_i < self.exposure_min:
            return False, f"underexposed (mean {mean_i:.1f} < {self.exposure_min:.0f})"
        if mean_i > self.exposure_max:
            return False, f"overexposed (mean {mean_i:.1f} > {self.exposure_max:.0f})"
        floor = self.blur_floor_fraction * max(burst_median_lap, 1e-9)
        if lap_var < floor:
            return False, (f"blurred (Laplacian variance {lap_var:.1f} < {floor:.1f}, "
                           f"{self.blur_floor_fraction:.0%} of this burst's median)")
        return True, "ok"


# --------------------------------------------------------------------------------------
# scans
# --------------------------------------------------------------------------------------
def voxel_thin(points: Sequence[Sequence[float]], voxel_m: float = VOXEL_M,
               max_points: int = MAX_POINTS_PER_SCAN) -> Tuple[np.ndarray, str]:
    """One point per occupied voxel, then a cap. Returns (points, what happened).

    The FIRST point in each voxel, not the centroid: a centroid of two returns from two
    different surfaces is a point on neither, and the registration at the station would be
    fitting invented geometry. Deterministic for the same input order, which matters because a
    scan written twice must be the same scan.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3), "empty"
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RecorderRefusal(f"expected an (N,3) array of points, got {pts.shape}")
    keys = np.floor(pts / max(voxel_m, 1e-6)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    kept = pts[idx]
    note = f"{pts.shape[0]} -> {kept.shape[0]} at {voxel_m * 100:.0f} cm"
    if kept.shape[0] > max_points:
        kept = kept[:max_points]
        note += f", capped at {max_points}"
    return kept, note


@dataclass(frozen=True)
class ScanGate:
    min_points: int = 200
    pose_max_age_s: float = POSE_MAX_AGE_S

    def verdict(self, n_points: int, pose_age_s: Optional[float]) -> Tuple[bool, str]:
        if pose_age_s is None:
            return False, ("no pose for this scan; a scan the station cannot place is not a "
                           "scan, and one placed with a guessed pose is worse")
        if pose_age_s > self.pose_max_age_s:
            return False, (f"pose is {pose_age_s:.2f} s old (limit {self.pose_max_age_s:.2f} s)")
        if n_points < self.min_points:
            return False, f"{n_points} points is below the {self.min_points} needed to register"
        return True, "ok"


# --------------------------------------------------------------------------------------
# the burst
# --------------------------------------------------------------------------------------
@dataclass
class StationCoverage:
    station: int
    frames_accepted: int = 0
    frames_rejected: int = 0
    scans_accepted: int = 0
    scans_rejected: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)

    def note(self, ok: bool, why: str, is_scan: bool) -> None:
        if ok:
            if is_scan:
                self.scans_accepted += 1
            else:
                self.frames_accepted += 1
            return
        if is_scan:
            self.scans_rejected += 1
        else:
            self.frames_rejected += 1
        key = why.split("(")[0].strip()
        self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1


class InspectionRecorderCore:
    """One candidate's capture, station by station. Decides; the shell writes.

    IDLE BETWEEN BURSTS IS ENFORCED HERE, not by the shell's good intentions: `offer_frame`
    and `offer_scan` REFUSE while no burst is open, and `busy` is False, which is what the
    node's health line reports and what `test_inspection_recorder_is_idle_between_bursts.py`
    drives. ADR-010's rule is that a payload node's cost is bounded and measured; a recorder
    that quietly kept gating frames between stations would be spending the budget on nothing.
    """

    def __init__(self, candidate_id: str, *, min_frames: int = 6, min_scans: int = 3,
                 frame_gate: Optional[FrameGate] = None,
                 scan_gate: Optional[ScanGate] = None):
        self.candidate_id = candidate_id
        self.min_frames = int(min_frames)
        self.min_scans = int(min_scans)
        self.frame_gate = frame_gate or FrameGate()
        self.scan_gate = scan_gate or ScanGate()
        self.coverage: Dict[int, StationCoverage] = {}
        self.frames: List[Dict[str, Any]] = []
        self.scans: List[Dict[str, Any]] = []
        self._station: Optional[int] = None
        self._burst_laps: List[float] = []

    # ---------------------------------------------------------------- burst control
    @property
    def busy(self) -> bool:
        return self._station is not None

    def start_burst(self, station: int) -> None:
        if self.busy:
            raise RecorderRefusal(
                f"a burst is already open at station {self._station}; opening a second one "
                f"would mix two stations' frames into one coverage count")
        self._station = int(station)
        self._burst_laps = []
        self.coverage.setdefault(int(station), StationCoverage(int(station)))

    def end_burst(self) -> StationCoverage:
        if not self.busy:
            raise RecorderRefusal("no burst is open")
        cov = self.coverage[self._station]
        self._station = None
        self._burst_laps = []
        return cov

    # ---------------------------------------------------------------- offers
    def offer_frame(self, image, *, t: float, pose: Optional[Dict[str, float]],
                    altitude_m: Optional[float] = None) -> Tuple[bool, str]:
        """Gate one camera frame. Returns (accepted, reason). Refuses while idle."""
        if not self.busy:
            return False, ("no capture burst is open; the recorder is idle between stations "
                           "(ADR-010) and does not gate frames nobody asked for")
        if pose is None:
            return False, "no pose for this frame; a frame the reconstruction cannot seed is noise"
        lap = laplacian_variance(image)
        mean_i = mean_intensity(image)
        self._burst_laps.append(lap)
        median_lap = float(np.median(self._burst_laps))
        ok, why = self.frame_gate.verdict(lap, mean_i, median_lap)
        cov = self.coverage[self._station]
        cov.note(ok, why, is_scan=False)
        self.frames.append({
            "candidate": self.candidate_id, "station": self._station, "t": t,
            "accepted": bool(ok), "reason": why,
            "laplacian_var": round(lap, 3), "mean_intensity": round(mean_i, 2),
            "burst_median_laplacian_var": round(median_lap, 3),
            "pose": pose, "altitude_m": altitude_m,
        })
        return ok, why

    def offer_scan(self, points, *, t: float, pose: Optional[Dict[str, float]],
                   pose_age_s: Optional[float]) -> Tuple[bool, str, Optional[np.ndarray]]:
        """Gate and thin one sonar scan. Returns (accepted, reason, thinned points or None)."""
        if not self.busy:
            return False, ("no capture burst is open; the recorder is idle between stations "
                           "(ADR-010)"), None
        thinned, note = voxel_thin(points)
        ok, why = self.scan_gate.verdict(int(thinned.shape[0]), pose_age_s)
        cov = self.coverage[self._station]
        cov.note(ok, why, is_scan=True)
        self.scans.append({
            "candidate": self.candidate_id, "station": self._station, "t": t,
            "accepted": bool(ok), "reason": why, "thinning": note,
            "n_points": int(thinned.shape[0]), "pose": pose, "pose_age_s": pose_age_s,
        })
        return ok, why, (thinned if ok else None)

    # ---------------------------------------------------------------- verdicts
    def station_done(self, station: int) -> Tuple[bool, str]:
        """Is this station finished? A COUNT of accepted evidence, never a stopwatch.

        Either enough frames OR enough scans: at Baltic visibility the camera may produce
        nothing usable at all (the R0 model puts 3.6 % of the red channel at the ring radius),
        and the 3D-15 is what carries the verdict there (strategy §4b). Requiring both would
        make the mission fail for a reason the water decided.
        """
        cov = self.coverage.get(int(station))
        if cov is None:
            return False, f"station {station} has produced nothing at all"
        if cov.frames_accepted >= self.min_frames:
            return True, (f"{cov.frames_accepted} frames accepted "
                          f"(needed {self.min_frames}), {cov.frames_rejected} rejected")
        if cov.scans_accepted >= self.min_scans:
            return True, (f"{cov.scans_accepted} sonar scans accepted "
                          f"(needed {self.min_scans}); the camera produced "
                          f"{cov.frames_accepted} usable frames")
        return False, (f"station {station}: {cov.frames_accepted}/{self.min_frames} frames and "
                       f"{cov.scans_accepted}/{self.min_scans} scans accepted; rejected "
                       f"{cov.frames_rejected} frame(s) and {cov.scans_rejected} scan(s) "
                       f"[{', '.join(f'{k} x{v}' for k, v in sorted(cov.reject_reasons.items()))}]")

    def coverage_line(self, stations_planned: int) -> str:
        """The line the behaviour tree reads and the operator sees (§6.1's own example)."""
        done = sum(1 for s in self.coverage if self.station_done(s)[0])
        fa = sum(c.frames_accepted for c in self.coverage.values())
        fr = sum(c.frames_rejected for c in self.coverage.values())
        sa = sum(c.scans_accepted for c in self.coverage.values())
        return (f"stations {done}/{stations_planned} · frames accepted {fa} · rejected {fr} "
                f"· sonar scans {sa}")

    def stations_accepted(self) -> int:
        return sum(1 for s in self.coverage if self.station_done(s)[0])

    # ---------------------------------------------------------------- manifests
    def frames_jsonl(self) -> str:
        return "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in self.frames)

    def scans_jsonl(self) -> str:
        return "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in self.scans)
