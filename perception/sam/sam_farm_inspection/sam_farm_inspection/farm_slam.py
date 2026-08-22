#!/usr/bin/env python3
"""Farm-relative navigation, the graph half (P8.1).

Pure python — no ROS, no I/O — the same shape as `farm_mission.py`, and for the same reason:
everything that decides where the vehicle thinks it is must be testable on a laptop.

WHAT THIS IS, AND WHY IT IS SHAPED THIS WAY
===========================================

Ivan, 2026-08-19: once the farm has been encircled and the buoys, ropes and anchors identified,
the vehicle goes over to **farm-relative navigation** — following ropes and planning paths from
where the ropes actually are as seen by the 3D sonar and side scan — and stops relying on
DR-based pre-assigned waypoints.

The method is the one in *Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring*
(Valdez, Torroba, Folkesson, Stenius; https://github.com/julRusVal/sss_farm_slam):

  * every SSS rope detection becomes **its own landmark**, with **its own instance of the prior
    belonging to the rope line it came from** — mean at the midpoint between that line's two
    mooring buoys, covariance LONG along the line and NARROW across it;
  * buoys are ordinary point landmarks;
  * dead reckoning supplies odometry factors between consecutive poses;
  * iSAM2 optimises incrementally.

The reason a per-detection landmark works at all is the covariance shape. A rope detection tells
you a great deal about your ACROSS-line position and almost nothing about your ALONG-line
position, and that is exactly what a long-thin prior encodes. Rope detections are ubiquitous
along a lane, so the online estimate is laterally constrained BY THE STRUCTURE — which is the
literal content of "farm-relative".

WHY THIS DOES NOT TOUCH THE GROSS PRIOR->OBSERVED TRANSFORM (SETTLED §3l)
========================================================================

`farm_localizer` fits the prior onto detections placed with `dr/odom`, whose origin is wherever
the vehicle started — measured at **126.92 m** at this rig. That offset is inside the fitted
translation and cannot be separated from a real rigid movement of the farm; the information is
not in the fit, and no amount of cleverness puts it there. So it must never be navigated on.

This module never sees that transform. It works entirely in the DR frame: the priors it plants
are placed by transforming the map's line endpoints into the DR frame ONCE, at arming, using the
vehicle's own DR pose at a moment when both frames are known — and after that the graph is
self-consistent. What leaves this module is `correction()`, an SE(2) to be applied **to the
GOALS** (P8.3), never a second odometry source for the controllers (invariant 12).

THE SOLVER IS NAMED, NEVER GUESSED
==================================

`hydrobatic_localization` builds GTSAM from source with **`-DGTSAM_BUILD_PYTHON=OFF`**
(`smarc2/scripts/install_gtsam_for_hydrobatic.sh` line 49), so **python-gtsam is NOT available on
the VM by that route** — checked in the repo, 2026-08-19, rather than assumed. Two honest ways
forward, and it is Ivan's call which (see the session report):

  * `pip install gtsam` — the PyPI wheel is 4.3a1, which satisfies the same `>= 4.3a0` the C++
    side already requires; separate process, separate linkage, no ABI contact with the node;
  * flip that flag to ON and rebuild (~40 min, needs pybind11 + pyparsing).

Until one of those is done on the vehicle, `backend="isam2"` REFUSES BY NAME. It does not fall
back. The `"batch"` backend is a Gauss-Newton solve of the SAME factor graph, written out in
`_solve_batch`, and it is **not iSAM2 and does not claim to be**: it re-linearises everything
every call, so it is for off-rig verification and for checking the incremental answer against a
batch one — never for a vehicle in the water. Asking for a backend that is not there is an
`UnavailableBackend`, which is §3d's "an unknown mode is refused by name, never downgraded".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

try:                                    # pragma: no cover - presence is the thing under test
    import gtsam as _gtsam
    GTSAM_AVAILABLE = True
except ImportError:                     # pragma: no cover
    _gtsam = None
    GTSAM_AVAILABLE = False

ROPE = "rope"
BUOY = "buoy"
DETECTION_KINDS = (ROPE, BUOY)

SSS = "sss"
SONAR3D = "sonar3d"
DETECTION_SOURCES = (SSS, SONAR3D)

ISAM2 = "isam2"
BATCH = "batch"
BACKENDS = (ISAM2, BATCH)

PORT = -1
STARBOARD = +1

#: WHICH WAY IS PORT, WRITTEN DOWN ONCE (2026-08-19).
#:
#: Bearing is measured in the vehicle's own frame, CCW positive, with `theta` CCW from +x — the
#: convention `Pose2` and `gtsam.Pose2::bearing` already use. So the LEFT side (port) is +pi/2
#: and the RIGHT side (starboard) is -pi/2.
#:
#: This is written as a table rather than as `side * pi/2` because that expression was the first
#: version, and it had the sign backwards: with a vehicle heading north it put every starboard
#: return to the WEST. Nothing caught it — the corridor survey is symmetric, so the pose came out
#: right in x and every existing check passed. It surfaced only as a cost breakdown showing one
#: landmark 16 m from the prior it had been given. A sign convention with no name is the same
#: family as the rebuilt-VM yaw sign inversion: invisible until something is mirrored.
SIDE_BEARING_RAD = {PORT: math.pi / 2.0, STARBOARD: -math.pi / 2.0}


class SlamRefusal(RuntimeError):
    """Raised with an operator-readable reason. Never caught and defaulted."""


class UnavailableBackend(SlamRefusal):
    """The named solver is not installed. A different one is not a substitute."""


# --------------------------------------------------------------------- the map it stands on

@dataclass(frozen=True)
class RopeLine:
    """One culture line, in the frame the graph works in.

    `a` and `b` are the line's two MOORING BUOYS, not arbitrary endpoints — the paper's rope
    prior is the midpoint between them, and calling them endpoints invites somebody to pass in a
    clipped segment, which would move every rope prior along the line.
    """
    line_id: str
    a: Tuple[float, float]
    b: Tuple[float, float]
    depth_m: float

    @property
    def midpoint(self) -> Tuple[float, float]:
        return (0.5 * (self.a[0] + self.b[0]), 0.5 * (self.a[1] + self.b[1]))

    @property
    def heading_rad(self) -> float:
        return math.atan2(self.b[1] - self.a[1], self.b[0] - self.a[0])

    @property
    def length_m(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


@dataclass(frozen=True)
class Buoy:
    buoy_id: str
    xy: Tuple[float, float]
    #: How well T2 measured it. NOT a config constant: a buoy the encircle only saw once is
    #: worth less than one it saw from three sides, and a graph told otherwise will trust the
    #: wrong landmark. `verified_map_to_slam_map` reads it off the verdicts.
    sigma_m: float
    depth_m: float = 0.0


@dataclass(frozen=True)
class FarmSlamMap:
    """What P8 is allowed to plant priors from.

    **This comes from the T2-VERIFIED map, not from the config prior.** T2's verdicts are the
    gate that arms farm-relative navigation at all (P8.2): a line that did not fit is ABSENT
    here, never straightened onto its prior, because a lane flown down a line nobody measured is
    a guess wearing a measurement's clothes (`FarmMap.lines_xz` keeps the same rule).
    """
    lines: Tuple[RopeLine, ...]
    buoys: Tuple[Buoy, ...]
    source: str = ""

    def line(self, line_id: str) -> RopeLine:
        for ln in self.lines:
            if ln.line_id == line_id:
                return ln
        raise SlamRefusal(
            f"no culture line called {line_id!r} in the verified map "
            f"(it has {[l.line_id for l in self.lines]}). A detection cannot be associated with "
            f"a line T2 did not confirm — that is how a guess becomes a landmark.")

    def buoy(self, buoy_id: str) -> Buoy:
        for b in self.buoys:
            if b.buoy_id == buoy_id:
                return b
        raise SlamRefusal(
            f"no buoy called {buoy_id!r} in the verified map "
            f"(it has {[b.buoy_id for b in self.buoys]}).")


# --------------------------------------------------------------------- what comes off the wire

@dataclass(frozen=True)
class Odometry:
    """A dead-reckoned BODY-FRAME delta between two poses, with its own sigmas.

    Body frame, not world: `dr/odom` gives world poses and the difference between two of them is
    a world delta, which is NOT what a `BetweenFactor(Pose2)` wants. `from_poses` does the
    conversion once so no caller has to remember.
    """
    dx: float
    dy: float
    dtheta: float
    sigma_xy_m: float
    sigma_theta_rad: float

    @staticmethod
    def from_poses(p0: Tuple[float, float, float], p1: Tuple[float, float, float],
                   sigma_xy_m: float, sigma_theta_rad: float) -> "Odometry":
        c, s = math.cos(p0[2]), math.sin(p0[2])
        wx, wy = p1[0] - p0[0], p1[1] - p0[1]
        return Odometry(dx=c * wx + s * wy, dy=-s * wx + c * wy,
                        dtheta=_wrap(p1[2] - p0[2]),
                        sigma_xy_m=sigma_xy_m, sigma_theta_rad=sigma_theta_rad)


@dataclass(frozen=True)
class Detection:
    """One thing a sonar saw, already associated with something in the map.

    ASSOCIATION IS NOT THIS MODULE'S JOB and is deliberately an input. `sss_farm_detector`
    produces change points on INTENSITIES (invariant 11: never material labels — those are
    ground truth and test-only) and the association to a line or buoy is the localizer's; doing
    it here would put the same decision in two places.

    `slant_range_m` is the raw slant range. The flat-bottom conversion to a ground range is
    Eq. 1 of the paper and lives in `ground_range()`, which REFUSES an impossible geometry rather
    than clamping it — a clamped slant range is a landmark planted at the vehicle's own feet.

    `bearing_rad` is left None for a side scan, which has no along-track resolution to speak of:
    the bearing is then taken as exactly abeam (+/- pi/2 by `side`). The forward 3D sonar DOES
    measure a bearing and supplies one; that is P8.5's whole contribution and the only way a
    near-vertical mooring line is ever seen at all.
    """
    t: float
    kind: str
    assoc_id: str
    slant_range_m: float
    side: int
    vehicle_depth_m: float
    sigma_range_m: float
    source: str = SSS
    bearing_rad: Optional[float] = None
    sigma_bearing_rad: float = 0.15


@dataclass(frozen=True)
class SlamConfig:
    """How much the graph believes each thing. No farm geometry here — that is the map."""

    #: The rope prior's along-line sigma, as a FRACTION of the line's own length. The paper's
    #: point is that a rope detection says nothing about where along the rope you are; making
    #: this proportional rather than absolute means a 40 m line and a 200 m line both get a prior
    #: that is honestly uninformative along its own axis.
    rope_sigma_along_frac: float = 0.5
    #: Across-line sigma. This IS the informative direction and it is what pulls the pose
    #: sideways, so it is the one number in this file that decides how hard the farm corrects the
    #: vehicle. 1.0 m is the paper's own reported rope RMSE (1.00 m) — i.e. we claim exactly the
    #: accuracy the method has been measured to have, and no more.
    rope_sigma_across_m: float = 1.0
    #: A rope prior is never tighter than this across-line, whatever the arithmetic says.
    rope_sigma_floor_m: float = 0.05
    #: Where the vehicle starts, in the graph's own frame. Loose but not free: the first pose has
    #: to be anchored or the whole graph slides along the ropes' null direction.
    origin_sigma_xy_m: float = 1.0
    origin_sigma_theta_rad: float = 0.2
    #: Gauss-Newton, for the batch backend.
    max_iterations: int = 30
    convergence_delta: float = 1e-6


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def ground_range(slant_range_m: float, vehicle_depth_m: float, target_depth_m: float) -> float:
    """Eq. 1 of the paper: flat-bottom slant range -> horizontal range.

    REFUSES rather than clamps. `sqrt(r^2 - dz^2)` with `r < |dz|` is not a small error, it is a
    statement that the target is closer than its own vertical separation — geometrically
    impossible, and every way of "handling" it (max(0, ...), abs, a small epsilon) plants a
    landmark somewhere the sonar cannot have seen. It means the association is wrong, or the
    depth is, and both are worth stopping for. Same rule as `choose_lane_geometry` refusing when
    no depth pair fits instead of returning the beam edge (SETTLED §3k).
    """
    dz = float(vehicle_depth_m) - float(target_depth_m)
    r = float(slant_range_m)
    if r < 0.0:
        raise SlamRefusal(f"a slant range cannot be negative ({r:.3f} m)")
    if r < abs(dz):
        raise SlamRefusal(
            f"impossible geometry: slant range {r:.3f} m is shorter than the vertical "
            f"separation {abs(dz):.3f} m (vehicle {vehicle_depth_m:.2f} m, target "
            f"{target_depth_m:.2f} m). The association or the depth is wrong; a ground range "
            f"cannot be recovered from this and will not be invented.")
    return math.sqrt(r * r - dz * dz)


# --------------------------------------------------------------------- the graph, as data

@dataclass
class PriorFactor:
    key: str
    mean: Tuple[float, ...]
    #: Full covariance, row-major. Not sigmas: the rope prior's whole point is that it is
    #: CORRELATED — long along the line and narrow across it — and a diagonal cannot say that
    #: unless the line happens to run along an axis.
    cov: Tuple[Tuple[float, ...], ...]
    what: str = ""


@dataclass
class BetweenFactor:
    key_from: str
    key_to: str
    delta: Tuple[float, float, float]
    sigmas: Tuple[float, float, float]


@dataclass
class BearingRangeFactor:
    pose_key: str
    landmark_key: str
    bearing_rad: float
    range_m: float
    sigma_bearing_rad: float
    sigma_range_m: float
    #: Kept for the report and for P8.4's rope estimate. The graph does not read it.
    assoc_id: str = ""
    kind: str = ROPE


@dataclass
class FarmGraph:
    """The factor graph as plain data, before any solver sees it.

    Deliberately a separate object. It makes the paper's formulation ASSERTABLE — "one landmark
    per detection, its prior taken from the originating line" is a property of this structure,
    not of a solver's output — and it means the same graph can be handed to iSAM2 on the vehicle
    and to the batch solver on a laptop and the two answers compared. A test that can only see
    the optimised numbers cannot tell a right answer from a right answer for the wrong reason.
    """
    pose_priors: List[PriorFactor] = field(default_factory=list)
    landmark_priors: List[PriorFactor] = field(default_factory=list)
    odometry: List[BetweenFactor] = field(default_factory=list)
    measurements: List[BearingRangeFactor] = field(default_factory=list)
    #: landmark key -> the map id it was instantiated from, in insertion order.
    landmark_origin: Dict[str, str] = field(default_factory=dict)
    landmark_kind: Dict[str, str] = field(default_factory=dict)

    @property
    def n_poses(self) -> int:
        return 1 + len(self.odometry)

    @property
    def n_landmarks(self) -> int:
        return len(self.landmark_priors)


def rope_prior_covariance(line: RopeLine, cfg: SlamConfig) -> Tuple[Tuple[float, float],
                                                                    Tuple[float, float]]:
    """`R diag(sigma_along^2, sigma_across^2) R^T` for the line's own heading.

    The rotation is the entire content of the paper's rope prior. Written out here rather than
    passed as three sigmas because the moment somebody hands this a diagonal, every rope prior
    silently becomes axis-aligned and the constraint stops being about the rope.
    """
    along = max(cfg.rope_sigma_floor_m, cfg.rope_sigma_along_frac * line.length_m)
    across = max(cfg.rope_sigma_floor_m, cfg.rope_sigma_across_m)
    c, s = math.cos(line.heading_rad), math.sin(line.heading_rad)
    a2, r2 = along * along, across * across
    return ((c * c * a2 + s * s * r2, c * s * (a2 - r2)),
            (c * s * (a2 - r2), s * s * a2 + c * c * r2))


class FarmSlam:
    """Build the graph, solve it, and answer where the vehicle is relative to the farm.

    NOT a ROS node, holds no publisher, writes no actuator and has no action client — the same
    shape as ADR-004's setpoint source. What it produces is a CORRECTION, consumed by whoever
    already owns the goals (P8.3, invariant 12).
    """

    def __init__(self, farm_map: FarmSlamMap, config: Optional[SlamConfig] = None,
                 backend: str = ISAM2,
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if backend not in BACKENDS:
            raise SlamRefusal(
                f"unknown SLAM backend {backend!r}; this module knows {list(BACKENDS)}. "
                f"An unrecognised mode is refused, never downgraded to the nearest thing.")
        if backend == ISAM2 and not GTSAM_AVAILABLE:
            raise UnavailableBackend(
                "backend 'isam2' asked for and python-gtsam is not importable here. "
                "install_gtsam_for_hydrobatic.sh builds GTSAM with -DGTSAM_BUILD_PYTHON=OFF, so "
                "the C++ node having it proves nothing about python. Either `pip install gtsam` "
                "(the PyPI wheel is 4.3a1, satisfying the same >= 4.3a0 the C++ side needs) or "
                "rebuild with -DGTSAM_BUILD_PYTHON=ON. Refusing rather than silently solving "
                "this batch: a batch solve is not iSAM2 and must not be handed to a vehicle "
                "under the name of one.")
        if not farm_map.lines and not farm_map.buoys:
            raise SlamRefusal(
                "the verified map has neither a culture line nor a buoy in it. Farm-relative "
                "navigation has nothing to be relative TO; this is the state that must fall back "
                "to surface-and-report, and it must do so by name (P8.2).")
        self.map = farm_map
        self.cfg = config or SlamConfig()
        self.backend = backend
        self.graph = FarmGraph()
        self._pose_keys: List[str] = ["x0"]
        self._landmark_seq = 0
        self._origin = tuple(float(v) for v in origin)
        self.graph.pose_priors.append(PriorFactor(
            key="x0", mean=self._origin,
            cov=((self.cfg.origin_sigma_xy_m ** 2, 0.0, 0.0),
                 (0.0, self.cfg.origin_sigma_xy_m ** 2, 0.0),
                 (0.0, 0.0, self.cfg.origin_sigma_theta_rad ** 2)),
            what="the pose the vehicle armed farm-relative navigation at"))
        self._values: Dict[str, Tuple[float, ...]] = {"x0": self._origin}
        self._solved = False

    # ---- building ---------------------------------------------------------------------

    @property
    def current_pose_key(self) -> str:
        return self._pose_keys[-1]

    def add_odometry(self, odo: Odometry) -> str:
        """One DR delta -> one new pose. Returns the new pose key."""
        prev = self.current_pose_key
        key = f"x{len(self._pose_keys)}"
        self._pose_keys.append(key)
        self.graph.odometry.append(BetweenFactor(
            key_from=prev, key_to=key,
            delta=(odo.dx, odo.dy, odo.dtheta),
            sigmas=(odo.sigma_xy_m, odo.sigma_xy_m, odo.sigma_theta_rad)))
        px, py, pth = self._values[prev]
        c, s = math.cos(pth), math.sin(pth)
        self._values[key] = (px + c * odo.dx - s * odo.dy,
                             py + s * odo.dx + c * odo.dy,
                             _wrap(pth + odo.dtheta))
        self._solved = False
        return key

    def add_detection(self, det: Detection) -> str:
        """One detection -> ONE NEW LANDMARK, with its own copy of the originating prior.

        This is the paper's central move and it looks wasteful until you see why. Associating
        every detection of a rope with ONE landmark would demand a correct data association
        along the rope's own length — exactly the direction in which a side scan carries no
        information — and one bad association then drags the whole line. A landmark per
        detection, each anchored by a long-thin prior, makes the along-line direction cost
        nothing and the across-line direction cost everything, which is the only part the
        measurement actually knows.
        """
        if det.kind not in DETECTION_KINDS:
            raise SlamRefusal(
                f"unknown detection kind {det.kind!r}; this module knows {list(DETECTION_KINDS)}. "
                f"Refused by name rather than treated as the nearest thing.")
        if det.source not in DETECTION_SOURCES:
            raise SlamRefusal(
                f"unknown detection source {det.source!r}; known: {list(DETECTION_SOURCES)}.")
        if det.side not in (PORT, STARBOARD):
            raise SlamRefusal(
                f"side must be PORT ({PORT}) or STARBOARD ({STARBOARD}), got {det.side!r}. "
                f"A side-scan return with no side is not half a measurement, it is none.")

        if det.kind == ROPE:
            line = self.map.line(det.assoc_id)
            target_depth = line.depth_m
            mean = line.midpoint
            cov = rope_prior_covariance(line, self.cfg)
        else:
            b = self.map.buoy(det.assoc_id)
            target_depth = b.depth_m
            mean = b.xy
            cov = ((b.sigma_m ** 2, 0.0), (0.0, b.sigma_m ** 2))

        r_ground = ground_range(det.slant_range_m, det.vehicle_depth_m, target_depth)

        key = f"l{self._landmark_seq}"
        self._landmark_seq += 1
        self.graph.landmark_priors.append(PriorFactor(
            key=key, mean=tuple(mean), cov=cov,
            what=f"{det.kind} prior instantiated from {det.assoc_id}"))
        self.graph.landmark_origin[key] = det.assoc_id
        self.graph.landmark_kind[key] = det.kind

        bearing = det.bearing_rad if det.bearing_rad is not None \
            else SIDE_BEARING_RAD[det.side]
        self.graph.measurements.append(BearingRangeFactor(
            pose_key=self.current_pose_key, landmark_key=key,
            bearing_rad=bearing, range_m=r_ground,
            sigma_bearing_rad=det.sigma_bearing_rad, sigma_range_m=det.sigma_range_m,
            assoc_id=det.assoc_id, kind=det.kind))

        px, py, pth = self._values[self.current_pose_key]
        th = pth + bearing
        self._values[key] = (px + r_ground * math.cos(th), py + r_ground * math.sin(th))
        self._solved = False
        return key

    # ---- solving ----------------------------------------------------------------------

    def update(self) -> None:
        if self.backend == ISAM2:
            self._solve_isam2()
        else:
            self._solve_batch()
        self._solved = True

    def pose(self, key: Optional[str] = None) -> Tuple[float, float, float]:
        return self._values[key or self.current_pose_key]        # type: ignore[return-value]

    def landmark(self, key: str) -> Tuple[float, float]:
        return self._values[key]                                  # type: ignore[return-value]

    def line_estimate(self, line_id: str) -> Tuple[Tuple[float, float], float, int]:
        """(point on the line, heading, how many landmarks it was fitted from).

        Total-least-squares through this line's own landmark instances — the estimate P8.4's
        rope following servos on. REFUSES on fewer than two: one point is a position, not a line,
        and returning the PRIOR's heading when the fit is underdetermined would be a measurement
        that is secretly a configuration file (SETTLED §3k's near-collinear refusal, same rule).
        """
        pts = [self._values[k] for k, origin in self.graph.landmark_origin.items()
               if origin == line_id and self.graph.landmark_kind[k] == ROPE]
        if len(pts) < 2:
            raise SlamRefusal(
                f"line {line_id!r} has {len(pts)} rope landmark(s); a line cannot be fitted from "
                f"fewer than two, and reporting the prior's own heading instead would be a "
                f"configuration file wearing a measurement's name.")
        n = float(len(pts))
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        syy = sum((p[1] - my) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        heading = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        return ((mx, my), _wrap(heading), len(pts))

    def correction(self) -> Tuple[float, float, float]:
        """The SE(2) taking a DR-frame point to where the graph says it belongs.

        THIS IS FOR GOALS, NOT FOR CONTROLLERS (P8.3, invariant 12). Applied to sub-goals by
        whoever already makes them, so the commanded track sits where the observed ropes are.
        The alternative — re-pointing controllers at a `farm_slam/odom` — is a second position
        source for every consumer, and this project has already paid for what two writers cost.

        It is a DELTA between two poses in ONE frame, which is why it is safe where the gross
        prior->observed transform (SETTLED §3l) is not: nothing here crosses a frame boundary,
        so the estimator's 126.92 m origin offset cancels instead of being smuggled in.
        """
        if not self._solved:
            raise SlamRefusal("correction() before update(): there is nothing solved to report")
        dx, dy, dth = self._dr_pose_of(self.current_pose_key)
        sx, sy, sth = self._values[self.current_pose_key]         # type: ignore[misc]
        dth_corr = _wrap(sth - dth)
        c, s = math.cos(dth_corr), math.sin(dth_corr)
        return (sx - (c * dx - s * dy), sy - (s * dx + c * dy), dth_corr)

    def _dr_pose_of(self, key: str) -> Tuple[float, float, float]:
        """Where pure dead reckoning alone would put this pose — the odometry chain, unoptimised.
        Recomputed rather than cached, because a cached copy is a second source of one truth."""
        x, y, th = self._origin
        for f in self.graph.odometry:
            c, s = math.cos(th), math.sin(th)
            x += c * f.delta[0] - s * f.delta[1]
            y += s * f.delta[0] + c * f.delta[1]
            th = _wrap(th + f.delta[2])
            if f.key_to == key:
                break
        return (x, y, th)

    # ---- backends ---------------------------------------------------------------------

    def _solve_isam2(self) -> None:  # pragma: no cover - exercised only where gtsam exists
        g = _gtsam
        graph = g.NonlinearFactorGraph()
        initial = g.Values()
        sym = {}

        def pkey(k: str) -> int:
            return g.symbol('x', int(k[1:]))

        def lkey(k: str) -> int:
            return g.symbol('l', int(k[1:]))

        for pf in self.graph.pose_priors:
            graph.add(g.PriorFactorPose2(
                pkey(pf.key), g.Pose2(*pf.mean),
                g.noiseModel.Gaussian.Covariance(_np().array(pf.cov))))
        for f in self.graph.odometry:
            graph.add(g.BetweenFactorPose2(
                pkey(f.key_from), pkey(f.key_to), g.Pose2(*f.delta),
                g.noiseModel.Diagonal.Sigmas(_np().array(f.sigmas))))
        for pf in self.graph.landmark_priors:
            graph.add(g.PriorFactorPoint2(
                lkey(pf.key), _np().array(pf.mean),
                g.noiseModel.Gaussian.Covariance(_np().array(pf.cov))))
        for m in self.graph.measurements:
            graph.add(g.BearingRangeFactor2D(
                pkey(m.pose_key), lkey(m.landmark_key),
                g.Rot2(m.bearing_rad), m.range_m,
                g.noiseModel.Diagonal.Sigmas(
                    _np().array([m.sigma_bearing_rad, m.sigma_range_m]))))
        for k, v in self._values.items():
            if k.startswith("x"):
                initial.insert(pkey(k), g.Pose2(*v))
            else:
                initial.insert(lkey(k), _np().array(v))
        isam = g.ISAM2(g.ISAM2Params())
        isam.update(graph, initial)
        isam.update()
        est = isam.calculateEstimate()
        for k in list(self._values):
            if k.startswith("x"):
                p = est.atPose2(pkey(k))
                self._values[k] = (p.x(), p.y(), p.theta())
            else:
                p = est.atPoint2(lkey(k))
                self._values[k] = (float(p[0]), float(p[1]))
        sym.clear()

    def _residuals(self, values):
        """(list of (keys, jacobians, residual, covariance)) at `values`. One place, so the cost
        a step is judged by and the system that proposes the step cannot drift apart."""
        np = _np()
        out = []
        for pf in self.graph.pose_priors:
            v = np.array(values[pf.key], dtype=float)
            r = np.array([v[0] - pf.mean[0], v[1] - pf.mean[1], _wrap(v[2] - pf.mean[2])])
            out.append(([pf.key], [np.eye(3)], r, np.array(pf.cov, dtype=float)))
        for pf in self.graph.landmark_priors:
            v = np.array(values[pf.key], dtype=float)
            out.append(([pf.key], [np.eye(2)], v - np.array(pf.mean, dtype=float),
                        np.array(pf.cov, dtype=float)))
        for f in self.graph.odometry:
            x0 = np.array(values[f.key_from], dtype=float)
            x1 = np.array(values[f.key_to], dtype=float)
            c, s = math.cos(x0[2]), math.sin(x0[2])
            R = np.array([[c, s], [-s, c]])
            d = x1[:2] - x0[:2]
            r = np.array([R[0] @ d - f.delta[0], R[1] @ d - f.delta[1],
                          _wrap(_wrap(x1[2] - x0[2]) - f.delta[2])])
            J0 = np.zeros((3, 3))
            J0[:2, :2] = -R
            J0[0, 2] = -s * d[0] + c * d[1]
            J0[1, 2] = -c * d[0] - s * d[1]
            J0[2, 2] = -1.0
            J1 = np.zeros((3, 3))
            J1[:2, :2] = R
            J1[2, 2] = 1.0
            out.append(([f.key_from, f.key_to], [J0, J1], r,
                        np.diag(np.array(f.sigmas, dtype=float) ** 2)))
        for m in self.graph.measurements:
            x = np.array(values[m.pose_key], dtype=float)
            lm = np.array(values[m.landmark_key], dtype=float)
            d = lm - x[:2]
            q = float(d @ d)
            q = q if q > 1e-9 else 1e-9
            rho = math.sqrt(q)
            r = np.array([_wrap(_wrap(math.atan2(d[1], d[0]) - x[2]) - m.bearing_rad),
                          rho - m.range_m])
            Jx = np.array([[d[1] / q, -d[0] / q, -1.0],
                           [-d[0] / rho, -d[1] / rho, 0.0]])
            Jl = np.array([[-d[1] / q, d[0] / q],
                           [d[0] / rho, d[1] / rho]])
            out.append(([m.pose_key, m.landmark_key], [Jx, Jl], r,
                        np.diag(np.array([m.sigma_bearing_rad, m.sigma_range_m],
                                         dtype=float) ** 2)))
        return out

    @staticmethod
    def _cost(blocks) -> float:
        np = _np()
        return float(sum(r @ np.linalg.inv(cov) @ r for _k, _J, r, cov in blocks))

    def _solve_batch(self) -> None:
        """Levenberg-Marquardt over the same factors. NOT iSAM2 — see the module docstring.

        Written out rather than pulled in so the off-rig answer can be compared with the
        vehicle's, and so nothing in this file pretends to be incremental when it is not: every
        call re-linearises the whole graph.

        WHY LM AND NOT PLAIN GAUSS-NEWTON, WHICH IS WHAT THIS WAS FIRST. The rope priors are
        deliberately enormous ALONG the line, which is the whole method — and it leaves the
        system very nearly singular in that direction. Plain Gauss-Newton with token damping
        (`1e-9 * I`) then takes an unbounded step down the null direction: on the synthetic
        corridor survey it settled 65 m from truth in `y` while being right in `x` to a
        centimetre. **Every existing test still passed**, because they all asked about the
        across-line direction the ropes actually constrain. The cross-check against iSAM2 on the
        identical graph is what found it — which is the reason the graph is data and the reason
        two backends exist at all. Damping is scaled to the diagonal (not absolute), and a step
        is only accepted if it lowers the cost.
        """
        np = _np()
        index: Dict[str, Tuple[int, int]] = {}
        n = 0
        for k in self._values:
            dim = 3 if k.startswith("x") else 2
            index[k] = (n, dim)
            n += dim

        lam = 1e-4
        blocks = self._residuals(self._values)
        cost = self._cost(blocks)
        for _ in range(self.cfg.max_iterations):
            H = np.zeros((n, n))
            b = np.zeros(n)
            for keys, Js, r, cov in blocks:
                W = np.linalg.inv(cov)
                for ka, Ja in zip(keys, Js):
                    ia, da = index[ka]
                    b[ia:ia + da] -= Ja.T @ W @ r
                    for kb, Jb in zip(keys, Js):
                        ib, db = index[kb]
                        H[ia:ia + da, ib:ib + db] += Ja.T @ W @ Jb
            diag = np.clip(np.diag(H).copy(), 1e-12, None)
            accepted = False
            for _try in range(12):
                try:
                    dx = np.linalg.solve(H + np.diag(diag * lam), b)
                except np.linalg.LinAlgError:      # pragma: no cover - damping normally prevents
                    lam *= 10.0
                    continue
                cand = {}
                for k, (i, dim) in index.items():
                    v = list(self._values[k])
                    for j in range(dim):
                        v[j] += dx[i + j]
                    if dim == 3:
                        v[2] = _wrap(v[2])
                    cand[k] = tuple(v)
                cand_blocks = self._residuals(cand)
                cand_cost = self._cost(cand_blocks)
                if cand_cost <= cost:
                    self._values = cand
                    blocks, cost = cand_blocks, cand_cost
                    lam = max(lam * 0.1, 1e-9)
                    accepted = True
                    break
                lam *= 10.0
            if not accepted or float(np.max(np.abs(dx))) < self.cfg.convergence_delta:
                break

    def report(self) -> dict:
        """What a health line and MC's T3 row read. Says which solver, always.

        Invariant 5c, finally satisfiable in one mode: the estimate carries its own uncertainty.
        A report that does not name its backend is a report that cannot be compared with the one
        from the machine next to it.
        """
        return {
            "backend": self.backend,
            "solved": self._solved,
            "poses": self.graph.n_poses,
            "landmarks": self.graph.n_landmarks,
            "rope_landmarks": sum(1 for v in self.graph.landmark_kind.values() if v == ROPE),
            "buoy_landmarks": sum(1 for v in self.graph.landmark_kind.values() if v == BUOY),
            "lines_seen": sorted({self.graph.landmark_origin[k]
                                  for k, v in self.graph.landmark_kind.items() if v == ROPE}),
            "map_source": self.map.source,
        }


def apply_correction(correction: Tuple[float, float, float],
                     point: Sequence[float]) -> Tuple[float, ...]:
    """Move a DR-frame goal to where the observed farm says it belongs.

    THE ONE IMPLEMENTATION (P8.3). Exported so the planner does not write its own: a correction
    applied two ways is two corrections, and the second one is always the stale one.

    Accepts a 2-tuple (a waypoint) or a 3-tuple (a pose); returns the same shape. Note that the
    TRANSLATION PART OF THIS SE(2) IS NOT "how far off the vehicle was" — the rotation acts about
    the graph's origin, so at 38 m out a 0.03 rad correction moves a point more than a metre
    while the stored translation reads centimetres. Judge a correction by applying it, never by
    reading its `x` and `y`. That mistake cost a red test the day this was written.
    """
    cx, cy, cth = correction
    c, s = math.cos(cth), math.sin(cth)
    x, y = float(point[0]), float(point[1])
    out = (cx + c * x - s * y, cy + s * x + c * y)
    if len(point) >= 3:
        return out + (_wrap(float(point[2]) + cth),)
    return out


_NP = None


def _np():
    global _NP
    if _NP is None:
        import numpy                      # local: keeps the import cost off a pure-geometry path
        _NP = numpy
    return _NP


# --------------------------------------------------------- the T2 verdicts are the gate (P8.2)

#: Verdicts a buoy may carry into farm-relative navigation. `moved` counts: the encircle SAW it
#: and measured where it now is, which is a measurement, not a doubt. `missing` and
#: `not_surveyed` do not — and they are different facts (SETTLED §3k's four verdicts), so a map
#: built from them must not quietly merge them.
ARMABLE_VERDICTS = ("confirmed", "moved")


def verified_map_to_slam_map(farm_map, rope_depth_m: float,
                             sigma_confirmed_m: float = 0.5,
                             sigma_moved_m: float = 1.5) -> FarmSlamMap:
    """`farm_mission.FarmMap` (the T2-verified map) -> the map P8 may plant priors from.

    Only lines that FITTED and only buoys with an armable verdict come through. A line that did
    not fit is absent, never straightened onto its prior; a buoy reported `moved` comes through
    with a WIDER sigma, because "we found it somewhere else" is a weaker statement about where it
    is than "we found it where it was" — and giving them the same weight is how one displaced
    buoy drags a whole graph.
    """
    lines = tuple(RopeLine(line_id=name, a=tuple(a), b=tuple(b), depth_m=float(rope_depth_m))
                  for name, (a, b) in sorted(getattr(farm_map, "lines_xz", {}).items()))
    verdicts = getattr(farm_map, "verdicts", {}) or {}
    buoys = []
    for name, xz in sorted((getattr(farm_map, "buoys_xz", {}) or {}).items()):
        verdict = verdicts.get(name)
        if verdict not in ARMABLE_VERDICTS:
            continue
        buoys.append(Buoy(buoy_id=name, xy=tuple(xz),
                          sigma_m=sigma_confirmed_m if verdict == "confirmed" else sigma_moved_m))
    return FarmSlamMap(lines=lines, buoys=tuple(buoys),
                       source=getattr(farm_map, "source", "") or "T2-verified map")
