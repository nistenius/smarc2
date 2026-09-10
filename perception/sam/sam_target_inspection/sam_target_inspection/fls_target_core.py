#!/usr/bin/env python3
"""Proud-object detection on the Water Linked 3D-15's point cloud — strategy §4b.

WHY THIS NODE EXISTS AND THE OBSTACLE DETECTOR IS NOT TOUCHED. `sam_perception/
obstacle_detector.py` is a SAFETY layer (spec invariants 3 and 8) and stays exactly as flown.
It deliberately throws away the returns near the seabed — its z-band keeps the walls a hull can
hit. This module keeps precisely what that one discards: the things STANDING ON the floor. Two
different questions, two different nodes, one shared cloud.

THE SENSOR AND WHY IT PAYS FOR ITSELF. The side scan is blind inside its nadir gap, 1.15·h wide
(SETTLED §3f0e) — which is exactly where the bay seed put the car (§3f0k). With the sim's −20°
tilt and a 40° elevation fan the 3D-15 in NAV mode images a strip of seabed AHEAD of the hull,
nadir included, five times a second, before the hull passes over it. Two sensors, two exclusion
zones that do not overlap.

THE PIPELINE, per cloud:
  1. fit the local seabed plane from the fan's lower returns (RANSAC, with the vehicle's own
     altitude and attitude as the prior — the same rule as §3f0h's georeferencing: the
     vehicle's own measurement of its own geometry, never a feature hunted in the data);
  2. keep points above that plane by more than the beam's own vertical footprint plus the
     plane fit's residual — the two things that could put a seabed point above the seabed;
  3. link-cluster them at a distance derived from the beam spacing;
  4. box-fit each cluster and keep those whose footprint and height are car-like;
  5. require persistence over consecutive pings with a centroid that moves consistently with
     the vehicle's own motion — "sustained, not a spike", the same rule as the SSS detector.

EVERY THRESHOLD IS DERIVED FROM THE SENSOR'S OWN BEAM GEOMETRY, and the derivation is the
function that computes it, not a comment beside a constant.

ONE MEASURED CORRECTION TO THE STRATEGY, recorded rather than quietly fixed: strategy §4b says
"0.3 m above the plane is ~5 × the NAV beam's vertical footprint at 7 m (1.6° ≈ 0.2 m)". The
arithmetic does not give 5 — 7 m × tan(1.60°) = 0.196 m, so 0.3 m is 1.53 × that footprint, not
5 ×. The 0.3 m figure itself survives by a different and better route: beam footprint (0.196 m)
plus three sigma of a good plane fit (~0.03 m each) is 0.29 m. `min_height_above_plane` computes
that, so the gate now scales with range instead of being a constant that is right at 7 m.

BODY FRAME: x forward, y left, z UP — the convention `obstacle_detector.process` uses (it keeps
|z| ≤ z_band for walls and discards the seabed at depth). The seabed is therefore at negative z.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# The sensor, from the datasheet read 2026-09-09 (SETTLED §3ad) and the sim prefab.
# Beam widths are the DATASHEET's; the sim does not model them (it casts 150 x 17 rays over the
# fan), so every threshold derived from them is a statement about the REAL unit and is
# conservative in the sim. Said out loud because a number derived from a spec the simulator does
# not implement is exactly the kind of thing that later reads as measured.
# --------------------------------------------------------------------------------------
NAV_BEAM_AZ_DEG = 0.85       # LF 1.2 MHz, datasheet
NAV_BEAM_EL_DEG = 1.60       # LF 1.2 MHz, datasheet
INS_BEAM_AZ_DEG = 0.45       # HF 2.4 MHz, datasheet
INS_BEAM_EL_DEG = 0.85       # HF 2.4 MHz, datasheet
NAV_RANGE_M = 15.0
INS_RANGE_M = 4.0

#: The MMT Mini (SETTLED §3f0d) — the object these bounds are shaped around.
MINI_L_M, MINI_W_M, MINI_H_M = 3.078, 1.416, 1.278

#: Footprint band, metres. A car is 3.1 x 1.4 in plan. A rock ridge fails the upper bound, a
#: buoy the height bound below, a rope both.
FOOTPRINT_MIN_M, FOOTPRINT_MAX_M = 0.8, 8.0
HEIGHT_MIN_M, HEIGHT_MAX_M = 0.4, 2.5

def persistence_pings_for(sustain_s: float = 0.6, ping_hz: float = 5.0) -> int:
    """Consecutive clouds a cluster must appear in — DERIVED from the sonar's own rate.

    `sustain_s` is the "sustained, not a spike" window the SSS nadir search uses and §4b names:
    0.6 s. At the prefab's NAV rate of 5 Hz that is 3 clouds. Written as a function so a change
    to `NavPingHz` moves the gate instead of leaving a constant that was right at 5 Hz.
    """
    return max(2, int(round(sustain_s * ping_hz)))


#: 3 at the prefab's NAV rate. See the derivation above; do not hand-edit this.
PERSISTENCE_PINGS = persistence_pings_for()


class FlsRefusal(RuntimeError):
    """Raised with an operator-readable reason. Never caught and defaulted."""


# --------------------------------------------------------------------------------------
# COPIED FROM sam_perception/sam_perception/obstacle_detector.py.
#
# The originals are METHODS on `ObstacleDetector(Node)`, and that module imports rclpy at
# module scope, so they cannot be imported on a machine without ROS — which is every machine
# these cores are tested on. They are therefore copied, and
# `test/test_fls_copy_matches_the_obstacle_detector.py` compares the copies against the
# ORIGINALS' own parsed syntax trees (comments stripped by `ast.unparse`), so a change to the
# cloud layout upstream fails here instead of silently producing a different point set from the
# same bytes. The copy is of the ARITHMETIC only; the originals' logger calls and health
# counters belong to that node.
# --------------------------------------------------------------------------------------
def quat_to_rot(x, y, z, w):
    """Quaternion -> 3x3 rotation matrix (numpy, no external deps)."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def parse_cloud(msg, intensity_min: int = 1):
    """Unity's SonarPointCloud_Pub: x,y,z float32 + intensity uint8, 13-byte step.

    A copy of `ObstacleDetector.parse_cloud`'s parse. Raises instead of logging-and-returning
    None, because this is a library function and swallowing the exception here would hand the
    caller an empty cloud that reads exactly like a silent seabed.
    """
    try:
        if msg.point_step == 13:
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "u1")])
            arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
            inten = arr["i"].astype(np.int32)
        else:
            from sensor_msgs_py import point_cloud2
            arr = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
            xyz = arr.astype(np.float64)
            inten = np.full(len(xyz), 255, dtype=np.int32)
    except Exception as e:
        raise FlsRefusal(f"cloud parse failed: {e}")
    keep = inten >= intensity_min
    return xyz[keep]


def apply_transform(xyz, translation, rotation_xyzw):
    """The world->body arithmetic of `ObstacleDetector.to_body`, without the TF lookup.

    THE SIM SUBTLETY, documented in the original and repeated here because it is the thing that
    makes this code look wrong: in simulation the cloud is published in a WORLD frame and is
    ground truth, so this transform is what reconstructs the sensor-relative ranges the real
    driver outputs natively. On the real unit the cloud arrives body-relative and the transform
    is the identity.
    """
    R = quat_to_rot(*rotation_xyzw)
    p = np.asarray(translation, dtype=np.float64)
    return np.asarray(xyz, dtype=np.float64) @ R.T + p


# --------------------------------------------------------------------------------------
# derived thresholds
# --------------------------------------------------------------------------------------
def beam_footprint_m(range_m: float, beam_deg: float) -> float:
    """How wide one beam is at a range: 2·r·tan(beam/2) ≈ r·tan(beam) for these angles."""
    return 2.0 * range_m * math.tan(math.radians(beam_deg) / 2.0)


def min_height_above_plane_m(range_m: float, plane_rms_m: float,
                             beam_el_deg: float = NAV_BEAM_EL_DEG,
                             n_sigma: float = 3.0) -> float:
    """How far above the fitted seabed a point must be before it is standing ON it.

    Two terms, and both are things that genuinely put a seabed return above the seabed:

      * the beam's own VERTICAL FOOTPRINT at this range — a beam that straddles the plane
        reports the whole patch at one range, so a flat seabed produces returns spread over
        `2·r·tan(el/2)` in height by construction;
      * `n_sigma` times the plane fit's own residual, because a plane fitted to a sloping or
        rippled bottom is not the bottom.

    At the strategy's worked point (r = 7 m, a good fit at 3 cm rms) this is
    0.196 + 0.09 = 0.29 m — which is where §4b's 0.3 m comes from, by a route that scales with
    range instead of being right at exactly one.
    """
    return beam_footprint_m(range_m, beam_el_deg) + n_sigma * abs(plane_rms_m)


def sample_step_deg(fan_deg: float, n_samples: int, beam_deg: float) -> float:
    """The angular spacing between adjacent RETURNS: the coarser of the sampling and the beam.

    MEASURED WHILE WRITING THIS FILE, and it changed the clustering. The datasheet's beam
    widths (0.85° azimuth, 1.60° elevation) describe the REAL unit's resolution. The SIMULATOR
    casts 150 × 17 rays over a 90° × 40° fan, i.e. 0.60° azimuth and **2.50° elevation**
    spacing — four times coarser in elevation than the datasheet beam. A link distance derived
    from the beam width alone therefore SPLITS a car in the sim into one cluster per elevation
    ray: measured, six fragments of a 3.08 × 1.42 × 1.28 m box at 7 m, none of which is
    car-sized. Taking the coarser of the two is right for both: on the real unit the beam
    dominates, in the sim the ray spacing does.
    """
    if n_samples > 1:
        return max(float(fan_deg) / (n_samples - 1), float(beam_deg))
    return float(beam_deg)


def grazing_stretch(range_m: float, altitude_m: float) -> float:
    """How much one angular step is stretched on a surface seen at grazing incidence: 1/sin(θ).

    THE SECOND THING MEASURED WHILE WRITING THIS FILE, and it also changed the clustering. A
    forward-looking fan does not see a car's roof face-on; it sees it at a grazing angle
    θ = asin(h/r). One elevation step of 2.5° at 7 m subtends 0.31 m ACROSS the beam but lands
    0.31/sin θ ≈ 0.5–1.0 m apart ALONG a near-horizontal surface. Measured: with the link
    distance taken as two un-stretched steps, a 3.08 × 1.42 × 1.28 m car at 7 m under 4 m of
    altitude broke into three clusters of 96, 20 and 18 points, and a 14 m ridge into seven —
    so the footprint band, whose whole job is to tell a car from a ridge, was being applied to
    fragments of both.

    Clamped at 1.0 (a surface cannot be sampled more finely than the beam) and at 6.0, beyond
    which the geometry is so grazing that a "surface" is a smear and no clustering scale is
    honest.
    """
    if range_m <= 0 or altitude_m <= 0:
        return 1.0
    return float(min(max(range_m / altitude_m, 1.0), 6.0))


def cluster_link_m(range_m: float, step_deg: float, stretch: float = 1.0) -> float:
    """Link distance for the clustering: two sample steps at this range, grazing-stretched.

    ONE step would split a smooth surface into one cluster per sample the moment the sampling
    is slightly uneven; three would bridge a car to the rock beside it. Two is the smallest
    that cannot split a continuously-sampled surface.

    WHY A GENEROUS LINK IS SAFE HERE, stated because it looks careless otherwise: this
    clustering runs ONLY on the points that already stand above the fitted seabed, which on a
    bare bottom is the empty set. Its job is not to separate an object from the terrain — the
    plane already did that — it is to avoid splitting ONE object. Two proud objects 1.5 m apart
    being merged is the failure mode this trades for, and the merged box then fails the
    footprint band and is refused BY NAME, which is a visible outcome rather than a silent one.

    At 10 m with the datasheet's 0.85° azimuth beam and no stretch this is 0.30 m — strategy
    §4b's number, derived.
    """
    return 2.0 * beam_footprint_m(range_m, step_deg) * max(float(stretch), 1.0)


def min_footprint_m(range_m: float, beam_az_deg: float = NAV_BEAM_AZ_DEG,
                    n_beams: int = 6) -> float:
    """Smallest footprint that is an OBJECT rather than a few adjacent returns.

    Six beams: fewer than that and a cluster is inside the sampling noise of one surface patch;
    at 10 m with the 0.85° beam this is 0.89 m, which is where §4b's 0.8 m floor comes from.
    """
    return n_beams * beam_footprint_m(range_m, beam_az_deg)


# --------------------------------------------------------------------------------------
# the plane
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Plane:
    """A fitted seabed plane in body coordinates: n·p + d = 0, |n| = 1, n pointing UP."""

    normal: Tuple[float, float, float]
    d: float
    rms_m: float
    n_inliers: int
    n_points: int

    def height_above(self, pts: np.ndarray) -> np.ndarray:
        n = np.asarray(self.normal, dtype=np.float64)
        return np.asarray(pts, dtype=np.float64) @ n + self.d

    @property
    def tilt_deg(self) -> float:
        """Angle between the fitted normal and straight up in the body frame."""
        return math.degrees(math.acos(max(-1.0, min(1.0, self.normal[2]))))


def fit_seabed_plane(points: Sequence[Sequence[float]], *, altitude_m: float,
                     pitch_rad: float = 0.0, roll_rad: float = 0.0,
                     inlier_tol_m: float = 0.15, max_tilt_deg: float = 25.0,
                     altitude_tol_m: float = 1.5, iterations: int = 120,
                     seed: int = 0) -> Tuple[Optional[Plane], str]:
    """RANSAC plane fit with the vehicle's OWN altitude and attitude as the prior.

    Returns (plane or None, reason). The prior is not decoration: a fan looking at a car sees a
    large flat-ish patch of car roof, and an unconstrained RANSAC will happily fit the roof and
    then report that there is nothing above the seabed. The two prior gates are

      * TILT: the fitted normal must be within `max_tilt_deg` of the attitude-corrected "up",
        because a seabed is not a wall and a car roof at a steep aspect is;
      * OFFSET: the plane must pass within `altitude_tol_m` of the altimeter's own answer, so
        the roof of a 1.3 m car cannot be mistaken for a floor 1.3 m too shallow.

    A fit that fails either is REFUSED BY NAME. It is not replaced by a horizontal plane at the
    altimeter's depth, tempting as that is: a refusal that quietly substitutes a guess is how a
    detector reports objects that are terrain.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None, f"expected an (N,3) array of body-frame points, got {pts.shape}"
    if pts.shape[0] < 12:
        return None, (f"only {pts.shape[0]} returns in this cloud — too few to fit a seabed "
                      f"plane, so nothing can be said to stand on it")
    # The prior normal: body "up" tilted by the vehicle's own pitch and roll.
    up = np.array([math.sin(pitch_rad), -math.sin(roll_rad), 1.0])
    up /= np.linalg.norm(up)

    rng = np.random.default_rng(seed)
    best: Optional[Plane] = None
    n_tilt_rejected = 0
    n_offset_rejected = 0
    for _ in range(int(iterations)):
        idx = rng.choice(pts.shape[0], size=3, replace=False)
        a, b, c = pts[idx]
        nv = np.cross(b - a, c - a)
        norm = np.linalg.norm(nv)
        if norm < 1e-9:
            continue
        nv = nv / norm
        if nv @ up < 0:
            nv = -nv                    # normals point UP, always
        if math.degrees(math.acos(max(-1.0, min(1.0, float(nv @ up))))) > max_tilt_deg:
            n_tilt_rejected += 1
            continue
        d = -float(nv @ a)
        # The plane's signed distance from the sensor (the origin) is `d`; the altimeter says
        # the seabed is `altitude_m` below, i.e. the plane should sit at +altitude_m.
        if abs(d - altitude_m) > altitude_tol_m:
            n_offset_rejected += 1
            continue
        dist = pts @ nv + d
        inl = np.abs(dist) <= inlier_tol_m
        n_in = int(inl.sum())
        if best is not None and n_in <= best.n_inliers:
            continue
        rms = float(np.sqrt(np.mean(dist[inl] ** 2))) if n_in else float("inf")
        best = Plane(tuple(float(v) for v in nv), d, rms, n_in, int(pts.shape[0]))
    if best is None:
        return None, (f"no plane in {iterations} RANSAC draws satisfied the priors "
                      f"({n_tilt_rejected} rejected for tilt > {max_tilt_deg:.0f}°, "
                      f"{n_offset_rejected} for standing off the altimeter's "
                      f"{altitude_m:.2f} m by more than {altitude_tol_m:.1f} m). This cloud "
                      f"does not contain a seabed the vehicle's own instruments agree with")
    if best.n_inliers < 0.2 * pts.shape[0]:
        return None, (f"the best plane explains only {best.n_inliers} of {pts.shape[0]} "
                      f"returns; that is not a seabed, and calling it one would put every "
                      f"other return above it")
    # Least-squares refit on the inliers: RANSAC picks WHICH points, not the best plane
    # through them, and a plane fitted to three points has an rms of exactly zero, which would
    # make the height gate above collapse to the beam term alone.
    nv = np.asarray(best.normal)
    inl = np.abs(pts @ nv + best.d) <= inlier_tol_m
    P = pts[inl]
    centroid = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - centroid, full_matrices=False)
    nv2 = vh[-1]
    if nv2 @ up < 0:
        nv2 = -nv2
    d2 = -float(nv2 @ centroid)
    resid = P @ nv2 + d2
    refined = Plane(tuple(float(v) for v in nv2), d2,
                    float(np.sqrt(np.mean(resid ** 2))), int(inl.sum()), int(pts.shape[0]))
    # THE PRIORS ARE RE-CHECKED AFTER THE REFIT, and this is not belt-and-braces — it is a
    # defect the tests found. The refit is unconstrained by construction, so on a cloud that is
    # ALL car roof, a tilted RANSAC plane that happened to satisfy the altitude prior selects a
    # band of roof points as its inliers and the least-squares refit through that band snaps
    # back to the roof: measured, a plane at d = 2.735 m returned for an altimeter reading of
    # 4.0 m with a 0.5 m tolerance. A refined fit that no longer agrees with the vehicle's own
    # instruments is REFUSED, not returned.
    if abs(refined.d - altitude_m) > altitude_tol_m:
        return None, (f"the refined plane sits at {refined.d:.2f} m where the altimeter reports "
                      f"{altitude_m:.2f} m (tolerance {altitude_tol_m:.1f} m). That is a flat "
                      f"surface the vehicle's own instruments say is not the seabed — most "
                      f"likely the top of the thing we are looking for")
    if refined.tilt_deg > max_tilt_deg:
        return None, (f"the refined plane is tilted {refined.tilt_deg:.0f}° from the "
                      f"attitude-corrected vertical (limit {max_tilt_deg:.0f}°); a seabed is "
                      f"not a wall")
    return refined, "ok"


# --------------------------------------------------------------------------------------
# clustering and box fitting
# --------------------------------------------------------------------------------------
def link_cluster(points: np.ndarray, link_m, min_points: int = 4) -> List[np.ndarray]:
    """Single-link clustering. Returns index arrays, largest first.

    `link_m` is a scalar OR a per-point array. Per-point is the useful case and is why this is
    not `sklearn`: the sampling scale of a fan grows with range and with grazing, so a single
    link distance is either too small at the far edge (which fragments a ridge into car-sized
    pieces — measured, five of them) or too large at the near edge. An edge between two points
    is taken when their separation is within the LARGER of the two link distances, which is the
    only choice that is symmetric.

    A grid-bucketed flood fill, so the cost is O(N) in the number of points rather than O(N²).
    That matters: a 3D-15 cloud is 150 × 17 = 2550 rays and this runs at 5 Hz on a flight
    computer under a 30 W cap (ADR-010).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return []
    link = np.full(n, float(link_m)) if np.isscalar(link_m) else np.asarray(link_m, dtype=np.float64)
    cell = max(float(link.max()), 1e-6)
    keys = np.floor(pts / cell).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)

    seen = np.zeros(n, dtype=bool)
    out: List[np.ndarray] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            kx, ky, kz = keys[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in buckets.get((kx + dx, ky + dy, kz + dz), ()):
                            if seen[j]:
                                continue
                            lim = max(link[i], link[j])
                            if float(np.sum((pts[i] - pts[j]) ** 2)) <= lim * lim:
                                seen[j] = True
                                stack.append(j)
        if len(comp) >= min_points:
            out.append(np.asarray(comp, dtype=int))
    out.sort(key=len, reverse=True)
    return out


@dataclass(frozen=True)
class BoxFit:
    """A cluster's oriented footprint and its height above the seabed plane."""

    centroid: Tuple[float, float, float]
    length_m: float          # the longer horizontal side
    width_m: float           # the shorter horizontal side
    height_m: float          # highest point above the plane
    yaw_deg: float           # bearing of the long side in the body frame
    n_points: int

    @property
    def footprint_m(self) -> float:
        return self.length_m


def box_fit(points: np.ndarray, plane: Plane) -> BoxFit:
    """Principal-axis box in the plane, height measured PERPENDICULAR TO THE PLANE.

    Perpendicular to the plane and not along body-z, because a sloping seabed would otherwise
    make every flat patch look like a proud object of exactly the slope's height — which is the
    single most likely false positive on a Baltic bottom.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = np.asarray(plane.normal, dtype=np.float64)
    h = pts @ n + plane.d
    centroid = pts.mean(axis=0)
    # Two orthonormal axes IN the plane.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uv = np.stack([(pts - centroid) @ e1, (pts - centroid) @ e2], axis=1)
    if uv.shape[0] >= 2:
        _, _, vh = np.linalg.svd(uv - uv.mean(axis=0), full_matrices=False)
        R = vh                                    # rows are the principal directions
        rot = (uv - uv.mean(axis=0)) @ R.T
        ext = rot.max(axis=0) - rot.min(axis=0)
        long_dir = R[0]
    else:
        ext = np.array([0.0, 0.0])
        long_dir = np.array([1.0, 0.0])
    length, width = float(max(ext)), float(min(ext))
    axis3 = long_dir[0] * e1 + long_dir[1] * e2
    yaw = math.degrees(math.atan2(float(axis3[1]), float(axis3[0])))
    return BoxFit(tuple(float(v) for v in centroid), length, width,
                  float(h.max()), yaw, int(pts.shape[0]))


# --------------------------------------------------------------------------------------
# the per-cloud detector
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FlsConfig:
    altitude_m: float
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    footprint_min_m: float = FOOTPRINT_MIN_M
    footprint_max_m: float = FOOTPRINT_MAX_M
    height_min_m: float = HEIGHT_MIN_M
    height_max_m: float = HEIGHT_MAX_M
    beam_az_deg: float = NAV_BEAM_AZ_DEG
    beam_el_deg: float = NAV_BEAM_EL_DEG
    #: The fan and its SAMPLING, from Sonar3D15.prefab: 90° x 40°, 150 beams x 17 rays. The
    #: clustering scale is the coarser of the beam and the sample step — see `sample_step_deg`.
    fan_az_deg: float = 90.0
    fan_el_deg: float = 40.0
    n_beams_az: int = 150
    n_rays_el: int = 17
    range_max_m: float = NAV_RANGE_M
    #: Points closer than this are the vehicle's own near field. Same value and same reason as
    #: `obstacle_detector`'s `range_min` (0.8 m) — a surfaced hull sees itself (SETTLED §3u).
    range_min_m: float = 0.8
    min_cluster_points: int = 8


@dataclass(frozen=True)
class FlsCloudReport:
    ok: bool
    reason: str
    boxes: Tuple[BoxFit, ...] = ()
    plane: Optional[Plane] = None
    n_points: int = 0
    n_above: int = 0
    n_clusters: int = 0
    n_killed_footprint: int = 0
    n_killed_height: int = 0


def detect_cloud(points: Sequence[Sequence[float]], cfg: FlsConfig) -> FlsCloudReport:
    """One cloud: plane, points above it, clusters, boxes, size gates.

    `ok=False` always carries a reason, and the reason distinguishes "I could not find the
    seabed" from "the seabed is bare" — different facts, and only one of them is information.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return FlsCloudReport(False, f"expected an (N,3) array, got {pts.shape}")
    rng = np.linalg.norm(pts, axis=1) if pts.shape[0] else np.zeros(0)
    keep = (rng >= cfg.range_min_m) & (rng <= cfg.range_max_m)
    pts = pts[keep]
    if pts.shape[0] < 12:
        return FlsCloudReport(False,
                              f"{int(keep.size)} returns, {int(pts.shape[0])} inside "
                              f"{cfg.range_min_m:.1f}–{cfg.range_max_m:.0f} m — too few to fit "
                              f"a seabed")
    plane, why = fit_seabed_plane(pts, altitude_m=cfg.altitude_m, pitch_rad=cfg.pitch_rad,
                                  roll_rad=cfg.roll_rad)
    if plane is None:
        return FlsCloudReport(False, why, n_points=int(pts.shape[0]))

    h = plane.height_above(pts)
    r = np.linalg.norm(pts, axis=1)
    gate = np.array([min_height_above_plane_m(float(ri), plane.rms_m, cfg.beam_el_deg)
                     for ri in r])
    above = h > gate
    n_above = int(above.sum())
    if n_above < cfg.min_cluster_points:
        return FlsCloudReport(True,
                              f"the seabed plane fits {plane.n_inliers}/{plane.n_points} "
                              f"returns at {plane.rms_m * 100:.0f} cm rms and only {n_above} "
                              f"return(s) stand above it — this patch of seabed is bare",
                              plane=plane, n_points=int(pts.shape[0]), n_above=n_above)

    P = pts[above]
    step = max(sample_step_deg(cfg.fan_az_deg, cfg.n_beams_az, cfg.beam_az_deg),
               sample_step_deg(cfg.fan_el_deg, cfg.n_rays_el, cfg.beam_el_deg))
    r_above = r[above]
    link = np.array([cluster_link_m(float(ri), step, grazing_stretch(float(ri), cfg.altitude_m))
                     for ri in r_above])
    comps = link_cluster(P, link, cfg.min_cluster_points)
    boxes: List[BoxFit] = []
    killed_fp = killed_h = 0
    for c in comps:
        bf = box_fit(P[c], plane)
        rc = float(np.linalg.norm(bf.centroid))
        fp_floor = max(cfg.footprint_min_m, min_footprint_m(rc, cfg.beam_az_deg))
        # `min_footprint_m` uses the BEAM, not the sample step: the floor asks "is this more
        # than a few adjacent looks of one surface patch", which is a resolution question.
        if not (fp_floor <= bf.footprint_m <= cfg.footprint_max_m):
            killed_fp += 1
            continue
        if not (cfg.height_min_m <= bf.height_m <= cfg.height_max_m):
            killed_h += 1
            continue
        boxes.append(bf)
    boxes.sort(key=lambda b: -b.n_points)
    if not boxes:
        return FlsCloudReport(True,
                              f"{len(comps)} cluster(s) stood above the seabed; {killed_fp} "
                              f"failed the footprint band and {killed_h} the height band "
                              f"({cfg.height_min_m:.1f}–{cfg.height_max_m:.1f} m). Nothing "
                              f"car-sized is standing on this seabed",
                              plane=plane, n_points=int(pts.shape[0]), n_above=n_above,
                              n_clusters=len(comps), n_killed_footprint=killed_fp,
                              n_killed_height=killed_h)
    return FlsCloudReport(True, "ok", tuple(boxes), plane, int(pts.shape[0]), n_above,
                          len(comps), killed_fp, killed_h)


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------
@dataclass
class FlsTrack:
    boxes: List[BoxFit] = field(default_factory=list)
    world_xy: List[Tuple[float, float]] = field(default_factory=list)
    first_ping: int = 0
    last_ping: int = 0
    emitted: bool = False

    @property
    def n_pings(self) -> int:
        return len(self.boxes)


class FlsTargetTracker:
    """Persistence with a MOTION-CONSISTENT centroid.

    The gate is applied in the WORLD frame, not the body frame. That is the whole point: a real
    object sits still while the vehicle moves, so its body-frame centroid marches towards the
    hull at the vehicle's own speed, and a body-frame gate either has to be as wide as the
    vehicle's travel per ping (which lets speckle in) or rejects the real thing. Converting each
    box's centroid to the world with the vehicle's own pose turns "moves consistently with our
    motion" into "does not move", which is a test with no free parameter.
    """

    def __init__(self, *, persistence_pings: int = PERSISTENCE_PINGS,
                 gate_m: float = 1.0, max_gap_pings: int = 2):
        self.persistence_pings = int(persistence_pings)
        self.gate_m = float(gate_m)
        self.max_gap_pings = int(max_gap_pings)
        self._tracks: List[FlsTrack] = []

    @staticmethod
    def to_world(centroid: Sequence[float], vehicle_xy: Tuple[float, float],
                 course_deg: float) -> Tuple[float, float]:
        """Body (x forward, y left) -> world (north, east) about the vehicle's own position."""
        c = math.radians(course_deg)
        north = vehicle_xy[0] + centroid[0] * math.cos(c) - centroid[1] * math.sin(c)
        east = vehicle_xy[1] + centroid[0] * math.sin(c) + centroid[1] * math.cos(c)
        return north, east

    def update(self, ping_index: int, report: FlsCloudReport,
               vehicle_xy: Tuple[float, float], course_deg: float) -> List[FlsTrack]:
        """Feed one cloud. Returns the tracks that JUST reached persistence."""
        for bf in report.boxes:
            w = self.to_world(bf.centroid, vehicle_xy, course_deg)
            match = None
            for tr in self._tracks:
                if ping_index - tr.last_ping > self.max_gap_pings:
                    continue
                dx = tr.world_xy[-1][0] - w[0]
                dy = tr.world_xy[-1][1] - w[1]
                if math.hypot(dx, dy) <= self.gate_m:
                    match = tr
                    break
            if match is None:
                match = FlsTrack(first_ping=ping_index, last_ping=ping_index)
                self._tracks.append(match)
            match.last_ping = ping_index
            match.boxes.append(bf)
            match.world_xy.append(w)

        ripe, keep = [], []
        for tr in self._tracks:
            if ping_index - tr.last_ping > self.max_gap_pings:
                continue
            keep.append(tr)
            if not tr.emitted and tr.n_pings >= self.persistence_pings:
                tr.emitted = True
                ripe.append(tr)
        self._tracks = keep
        return ripe


def _outside_band(value: float, lo: float, hi: float) -> float:
    """How far outside [lo, hi] a value is, in units of the band's own half-width. 0 inside."""
    half = 0.5 * (hi - lo)
    if half <= 0:
        return 0.0
    if value < lo:
        return (lo - value) / half
    if value > hi:
        return (value - hi) / half
    return 0.0


def candidate_from_track(track: FlsTrack, *, cid: str, t: float,
                         sigma_m: float) -> Dict[str, object]:
    """The strategy §4b record. `score` is a normalised distance, never a confidence.

    THE DISTANCE IS AGAINST THE CAR-SIZED BAND, NOT AGAINST THE MINI'S EXACT DIMENSIONS, and
    that is a correction to the obvious design rather than a shortcut. A forward-looking sonar
    on one pass sees the NEAR FACE and part of the roof: at 7 m and the sim's 2.5° elevation
    sampling the along-track extent it recovers from a 3.08 m car is 1–2 m, not 3.08. Scoring
    that against 3.08 would rank a correct single-aspect detection as a bad one, and the
    inspection orbit — several aspects — is precisely what earns the exact-dimension comparison
    later (rung 2 of the verdict ladder, strategy §7.1).

    So: zero while the box is inside the car-sized band, growing as it leaves it, in units of
    the band's own half-width. Nothing is minted; the numbers travel with the record so the
    ledger can compare them with the SSS extent itself.
    """
    L = np.array([b.length_m for b in track.boxes])
    W = np.array([b.width_m for b in track.boxes])
    H = np.array([b.height_m for b in track.boxes])
    xs = np.array([p[0] for p in track.world_xy])
    ys = np.array([p[1] for p in track.world_xy])
    z = [_outside_band(float(np.median(L)), FOOTPRINT_MIN_M, FOOTPRINT_MAX_M),
         _outside_band(float(np.median(W)), 0.0, FOOTPRINT_MAX_M),
         _outside_band(float(np.median(H)), HEIGHT_MIN_M, HEIGHT_MAX_M)]
    return {
        "id": cid, "t": t, "sensor": "fls",
        "centroid_body": list(track.boxes[-1].centroid),
        "north": float(np.median(xs)), "east": float(np.median(ys)),
        "bbox_m": {"length": float(np.median(L)), "width": float(np.median(W))},
        "height_m": float(np.median(H)),
        "n_points": int(np.median([b.n_points for b in track.boxes])),
        "n_pings": track.n_pings,
        "sigma_m": float(sigma_m),
        "score": float(math.sqrt(sum(v * v for v in z) / len(z))),
    }
