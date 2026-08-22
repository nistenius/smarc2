#!/usr/bin/env python3
"""Side-scan geometry — what the sonar can and cannot see, in metres.

THE SINGLE SOURCE for the sonar-geometry arithmetic in this mission. The farm-prior
generator (`data-cube/scripts/kristineberg-site/make_farm_prior.py`) imports this file
by explicit path and bakes its answers into `farm_prior.yaml`; the planner and the
in-editor verifier both read those. Nothing re-derives the numbers; if this file is
wrong, everything downstream is wrong in the same direction, which is the point.

WHY THIS EXISTS AT ALL (measured 2026-08-16, and it changed the mission profile):
a side-scan sonar looks DOWN and OUT. It cannot see anything at its own depth, and it
cannot see anything above it. The Kristineberg culture ropes sit at -2.0 m, so
"fly at the rope depth" — the naive reading of the mission spec's requirement 7 — puts
every rope at exactly 90 deg off nadir, which is outside every side-scan beam ever
built. The rope depth is the number the geometry is BUILT FROM, not the depth the
vehicle flies at. Both papers agree: the IROS 2025 survey of this very farm was flown
FROM THE SURFACE, with the ropes 1.5 m below.

Beam convention, read straight out of `Sonar.cs` (`SetupSonarRaycastJob`, SSS branch)
so the model and the simulator cannot drift apart:

    rayAngle  = rayNum * (breadth / (rays-1)) - breadth/2       in [-B/2, +B/2]
    rayAngle += side * (90 - tilt - breadth/2)                  side = -1 port, +1 stbd
    direction = rotate(-up, rayAngle, about forward)

so a ray's angle FROM NADIR runs over

    theta in [90 - tilt - breadth, 90 - tilt]

and `TiltAngleDeg` is measured from the horizontal: tilt is the grazing angle of the
beam's OUTERMOST (shallowest-looking) ray. A nadir gap of 2*(90 - tilt - breadth)
degrees opens up when tilt + breadth < 90, which is what a real side-scan has and is
why the 2022 detector runs a nadir window before an object window.

Everything here is pure arithmetic: no ROS, no numpy, no I/O. It is the piece that
must be testable on a laptop, because a geometry error looks exactly like a detector
that "does not work".
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import math


@dataclass(frozen=True)
class BeamGeometry:
    """One side-scan transducer pair, in the simulator's own parameter names.

    `tilt_deg` and `breadth_deg` are `Sonar.TiltAngleDeg` / `Sonar.BeamBreadthDeg`
    verbatim, so a value here can be compared against a prefab without translation.
    """

    tilt_deg: float
    breadth_deg: float
    max_range_m: float
    num_buckets: int
    #: Free-text provenance. Never let a beam config travel without saying where it
    #: came from — "wild guess" and "datasheet" must not look alike (SETTLED §3f).
    source: str = "unstated"

    # ---------------------------------------------------------------- beam angles
    @property
    def theta_min_deg(self) -> float:
        """Smallest angle from nadir the beam reaches. > 0 means there is a nadir gap."""
        return max(0.0, 90.0 - self.tilt_deg - self.breadth_deg)

    @property
    def theta_max_deg(self) -> float:
        """Largest angle from nadir. 90 deg would be horizontal, which no SSS reaches."""
        return min(90.0, 90.0 - self.tilt_deg)

    @property
    def nadir_gap_deg(self) -> float:
        """Total unensonified cone straight below the vehicle (both sides)."""
        return 2.0 * self.theta_min_deg

    @property
    def bucket_m(self) -> float:
        """Slant-range resolution: one intensity sample per this many metres."""
        return self.max_range_m / float(self.num_buckets)

    # ------------------------------------------------------------------ visibility
    def lateral_window_m(self, depth_below_sonar_m: float) -> Optional[Tuple[float, float]]:
        """Horizontal offsets at which a target `depth_below_sonar_m` BELOW the sonar
        falls inside the beam.

        Returns None — not an empty range, not a zero — when the target is at or above
        the sonar's own depth. That case is not "a narrow window", it is "no geometry
        at all", and a caller that treats it as a number will plan a lane that can
        never see anything. Same rule as ABSENT-IS-NOT-EMPTY (SETTLED §3e).
        """
        if depth_below_sonar_m <= 0.0:
            return None
        lo = depth_below_sonar_m * math.tan(math.radians(self.theta_min_deg))
        hi = depth_below_sonar_m * math.tan(math.radians(self.theta_max_deg))
        # Slant range also has to fit inside the recorded ping.
        hi = min(hi, self._max_lateral_in_range(depth_below_sonar_m))
        if hi < lo:
            return None
        return (lo, hi)

    def _max_lateral_in_range(self, dz: float) -> float:
        if self.max_range_m <= dz:
            return 0.0
        return math.sqrt(self.max_range_m ** 2 - dz ** 2)

    def sees(self, depth_below_sonar_m: float, lateral_m: float) -> bool:
        w = self.lateral_window_m(depth_below_sonar_m)
        return w is not None and w[0] <= abs(lateral_m) <= w[1]


def slant_to_ground_range(slant_m: float, depth_below_sonar_m: float) -> Optional[float]:
    """IROS 2025 Eq. 1: d_2D = sqrt(d_3D^2 - dz^2), the flat-bottom slant correction.

    Returns None when the slant range is shorter than the depth difference — that is a
    geometrically impossible return, not a zero-offset one, and rounding it to 0.0 puts
    a phantom detection directly under the vehicle. Seen once already in this project as
    the class of bug where an impossible measurement is silently made possible.
    """
    if slant_m < 0.0 or depth_below_sonar_m < 0.0:
        return None
    if slant_m < depth_below_sonar_m:
        return None
    return math.sqrt(slant_m * slant_m - depth_below_sonar_m * depth_below_sonar_m)


def first_bottom_return_slant_m(seabed_depth_m: float, sonar_depth_m: float,
                                beam: BeamGeometry) -> Optional[float]:
    """Slant range of the nadir return — the boundary of the water-column window.

    The 2022 detector's whole structure (nadir window first, object window second)
    depends on knowing where the bottom return starts: everything before it is water
    column, and a rope at 2 m depth lives there. With a nadir gap the first bottom
    return is NOT at the vertical distance but at that distance over cos(theta_min).

    Depths are positive-down metres. Returns None if the seabed is above the sonar,
    which means the caller's depths disagree with each other.
    """
    dz = seabed_depth_m - sonar_depth_m
    if dz <= 0.0:
        return None
    return dz / math.cos(math.radians(beam.theta_min_deg))


@dataclass(frozen=True)
class LaneGeometry:
    """A flyable scan line, or a refusal that says why."""

    ok: bool
    reason: str
    scan_depth_m: float = 0.0        # positive-down, the depth the VEHICLE flies at
    standoff_m: float = 0.0          # horizontal distance from the rope line
    depth_below_sonar_m: float = 0.0
    window_m: Tuple[float, float] = (0.0, 0.0)
    #: How far inside the beam edges the chosen standoff sits. Small = grazing, and a
    #: grazing lane degrades to nothing the moment the vehicle's depth control wanders.
    beam_margin_m: float = 0.0


def choose_lane_geometry(rope_depth_m: float,
                         beam: BeamGeometry,
                         preferred_standoff_m: float = 2.0,
                         min_standoff_m: float = 1.0,
                         min_scan_depth_m: float = 0.5,
                         max_scan_depth_m: float = 1.5,
                         depth_control_error_m: float = 0.3,
                         min_beam_margin_m: float = 0.5,
                         scan_depth_step_m: float = 0.1) -> LaneGeometry:
    """Pick the vehicle depth and lateral standoff for a rope-following lane.

    `rope_depth_m` is positive-down and comes from the prior (mission requirement 7 —
    the rope depth is carried by the mission, never hardcoded). Everything else is a
    sensor/safety constraint, and the answer is DERIVED. That is the whole design:
    change the prior and the lanes move.

    The chooser prefers the standoff nearest `preferred_standoff_m` (2 m — the IROS
    survey's swath spacing) among depths that leave the target comfortably inside the
    beam even after `depth_control_error_m` of depth-holding error, and it REFUSES,
    by name, when no such pair exists. It never silently returns the beam edge: a lane
    flown on the last ray of the beam sees the ropes on a good day and nothing on a
    normal one, and "we scanned it and found nothing" is the most expensive possible
    output of an inspection mission.

    The two error budgets are deliberately separate and are NOT double-counting:
    `depth_control_error_m` is vertical (the dive controller's depth-holding error, which
    swings the target toward the horizon and out through the OUTER beam edge), while
    `min_beam_margin_m` is horizontal (the line-following controller's cross-track error,
    which can push the target out through EITHER edge). Both are provisional rig numbers
    in the ADR-004 sense — nothing has flown them.
    """
    if rope_depth_m <= 0.0:
        return LaneGeometry(False, f"rope depth {rope_depth_m:.2f} m is not below the "
                                   "surface — the prior is positive-down metres")
    if min_scan_depth_m > max_scan_depth_m:
        return LaneGeometry(False, "min_scan_depth_m is deeper than max_scan_depth_m")

    best: Optional[LaneGeometry] = None
    tried = 0
    depth = min_scan_depth_m
    while depth <= max_scan_depth_m + 1e-9:
        tried += 1
        # Worst case for the beam is the depth-hold error that makes the vehicle
        # DEEPEST, because that shrinks the vertical separation and swings the target
        # toward the horizon, i.e. out through the outer beam edge.
        dz_worst = rope_depth_m - (depth + depth_control_error_m)
        dz_nominal = rope_depth_m - depth
        w = beam.lateral_window_m(dz_worst)
        if w is not None:
            # The usable band is the beam window pulled in by the cross-track margin on
            # BOTH sides, then floored by the safety standoff. Pulling in on both sides
            # is what stops the chooser handing back the last ray of the beam.
            lo = max(w[0] + min_beam_margin_m, min_standoff_m)
            hi = w[1] - min_beam_margin_m
            if hi >= lo:
                s = min(max(preferred_standoff_m, lo), hi)
                margin = min(s - w[0], w[1] - s)
                cand = LaneGeometry(
                    True,
                    ("standoff at the preferred %.1f m" % preferred_standoff_m)
                    if abs(s - preferred_standoff_m) < 1e-6 else
                    ("preferred %.1f m is outside the beam; nearest usable is %.2f m"
                     % (preferred_standoff_m, s)),
                    scan_depth_m=round(depth, 3),
                    standoff_m=round(s, 3),
                    depth_below_sonar_m=round(dz_nominal, 3),
                    window_m=(round(w[0], 3), round(w[1], 3)),
                    beam_margin_m=round(margin, 3))
                # Prefer the candidate closest to the preferred standoff; break ties on
                # the larger beam margin. Shallower is not automatically better.
                if best is None or (
                        abs(cand.standoff_m - preferred_standoff_m),
                        -cand.beam_margin_m) < (
                        abs(best.standoff_m - preferred_standoff_m),
                        -best.beam_margin_m):
                    best = cand
        depth += scan_depth_step_m

    if best is not None:
        return best

    # No feasible pair. Say precisely what blocked it — an operator can act on
    # "the beam stops at 45 deg off nadir" and cannot act on "planning failed".
    dz = rope_depth_m - min_scan_depth_m
    widest = beam.lateral_window_m(dz - depth_control_error_m)
    if widest is None:
        detail = (f"at the shallowest allowed depth {min_scan_depth_m:.2f} m the ropes are "
                  f"{max(0.0, dz - depth_control_error_m):.2f} m below the sonar in the worst "
                  f"case — a side scan sees nothing at or above its own depth")
    else:
        detail = (f"the widest beam band is {widest[0]:.2f}..{widest[1]:.2f} m, which after "
                  f"{min_beam_margin_m:.2f} m of cross-track margin on each side and a "
                  f"{min_standoff_m:.2f} m minimum safe standoff leaves nothing")
    return LaneGeometry(False,
                        f"no scan depth in {min_scan_depth_m:.2f}..{max_scan_depth_m:.2f} m puts a "
                        f"rope at {rope_depth_m:.2f} m inside the beam "
                        f"(off-nadir {beam.theta_min_deg:.0f}..{beam.theta_max_deg:.0f} deg, "
                        f"beam source: {beam.source}): {detail}")


@dataclass(frozen=True)
class EncircleGeometry:
    """The T2 perimeter loop's depth, or a refusal that names the buoys it would miss."""

    ok: bool
    reason: str
    depth_m: float = 0.0             # positive-down, the depth the VEHICLE flies at
    standoff_m: float = 0.0          # horizontal distance from the buoy hull
    visible: Tuple[str, ...] = ()
    invisible: Tuple[str, ...] = ()
    #: Vertical separation, in metres, between the sonar and the SHALLOWEST-bottomed buoy
    #: that is still visible. This is the whole safety margin of the encircle: it is tens
    #: of centimetres, not metres, and it is the number to watch.
    worst_separation_m: float = 0.0


def buoy_visibility(depth_m: float,
                    buoy_extents_m: Dict[str, float],
                    beam: BeamGeometry,
                    standoff_m: float) -> Tuple[Tuple[str, ...], Tuple[str, ...], float]:
    """Which buoys a side scan at `depth_m` can see from `standoff_m` away.

    A FLOATING BUOY IS ALMOST ENTIRELY ABOVE THE WATER LINE, AND A SIDE SCAN SEES NOTHING
    AT OR ABOVE ITS OWN DEPTH. So the only part of a buoy that can ever return an echo is
    the part BELOW the transducer, and `buoy_extents_m[name]` is how far below the surface
    each buoy reaches (its radius — the colliders are spheres centred on the water line).

    That makes the T2 encircle depth a derived quantity with a very small budget, and it
    is why the IROS 2025 survey of this farm was flown FROM THE SURFACE. Returns
    (visible, invisible, worst_separation_m) so a caller can report which buoys it would
    miss by name rather than discovering an incomplete map after the fact.
    """
    vis, invis, worst = [], [], None
    for name in sorted(buoy_extents_m):
        dz = float(buoy_extents_m[name]) - depth_m      # below the sonar = positive
        if dz > 0.0 and beam.sees(dz, standoff_m):
            vis.append(name)
            worst = dz if worst is None else min(worst, dz)
        else:
            invis.append(name)
    return tuple(vis), tuple(invis), (worst or 0.0)


def choose_encircle_geometry(buoy_extents_m: Dict[str, float],
                             beam: BeamGeometry,
                             standoff_m: float,
                             candidate_depths_m: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
                             requested_depth_m: Optional[float] = None) -> EncircleGeometry:
    """Pick the encircle depth, or refuse and say which buoys would be invisible.

    Visibility is monotonically *worse* with depth (a deeper sonar has less of the buoy
    below it), so the shallowest candidate is always the best one and this is a search
    only so that the refusal can quote what it tried.

    `requested_depth_m` overrides the search: it exists so that a mission configured to
    encircle at, say, the rope depth gets a REFUSAL NAMING THE BUOYS instead of a survey
    that quietly returns nothing. That failure mode — "we looked and found nothing" — is
    the most expensive output an inspection mission has, and it is indistinguishable from
    a real empty farm unless something checks the geometry first.
    """
    if not buoy_extents_m:
        return EncircleGeometry(False, "the prior lists no buoy extents, so nothing can be "
                                       "said about what the encircle would see")
    if standoff_m <= 0.0:
        return EncircleGeometry(False, f"encircle standoff {standoff_m:.2f} m is not outside "
                                       "the farm")

    def _at(depth):
        vis, invis, worst = buoy_visibility(depth, buoy_extents_m, beam, standoff_m)
        return EncircleGeometry(
            ok=not invis,
            reason=("all %d buoys are below the sonar at %.2f m and inside the beam at "
                    "%.1f m standoff" % (len(vis), depth, standoff_m)) if not invis else
                   ("at %.2f m depth the side scan cannot see %s: a buoy only reaches "
                    "%.2f m below the surface, and a side scan sees nothing at or above "
                    "its own depth (beam %s)"
                    % (depth, ", ".join(invis),
                       min(buoy_extents_m[b] for b in invis), beam.source)),
            depth_m=round(depth, 3), standoff_m=round(standoff_m, 3),
            visible=vis, invisible=invis, worst_separation_m=round(worst, 3))

    if requested_depth_m is not None:
        return _at(float(requested_depth_m))

    tried = []
    for d in candidate_depths_m:
        cand = _at(d)
        if cand.ok:
            return cand
        tried.append(d)
    deepest_possible = min(buoy_extents_m.values())
    return EncircleGeometry(
        False,
        "no encircle depth in %s puts every buoy below the sonar: the shallowest-bottomed "
        "buoy reaches only %.2f m below the surface, so the loop must be flown above that "
        "or those buoys cannot be detected at all"
        % (", ".join("%.2f" % d for d in tried), deepest_possible))


#: Must match `SSS_Pub.SoundSpeedMS` in SMARCAssets. The publisher expresses the sonar's
#: geometric range as a two-way travel time with this constant; converting back with the
#: same one recovers the range exactly, which is the only property that matters.
SOUND_SPEED_MS = 1500.0


def range_per_bin(max_duration_s: float, n_bins: int,
                  fallback_max_range_m: float = 0.0,
                  sound_speed_ms: float = SOUND_SPEED_MS
                  ) -> Tuple[Optional[float], str, Optional[str]]:
    """Metres per range bin, and — always — WHERE that number came from.

    Returns (metres_per_bin or None, source, complaint or None) where source is
    "message", "parameter" or a reason string when there is no scale at all.

    This is a single function rather than four lines in the node because the range scale
    multiplies EVERY detection range the mission produces. Getting it from a stale
    parameter does not make the detector fail; it makes it produce a farm of the wrong
    size, which reads as a displaced farm — the exact finding the mission exists to
    report. So the source is returned alongside the number and the node puts it on every
    health line.

    The message's own `max_duration` always wins when it is present: the publisher knows
    its range and the parameter is a guess about it. A disagreement of more than 5 % is
    returned as a complaint for the caller to log loudly, not silently reconciled.
    """
    if n_bins <= 0:
        return None, "no bins in the ping", None
    dur = float(max_duration_s or 0.0)
    if dur > 0.0:
        rng = sound_speed_ms * dur / 2.0
        complaint = None
        if fallback_max_range_m > 0.0 and \
                abs(rng - fallback_max_range_m) > 0.05 * fallback_max_range_m:
            complaint = (f"the ping says its range is {rng:.1f} m (max_duration {dur:.4f} s "
                         f"at {sound_speed_ms:.0f} m/s) but the configured max_range_m is "
                         f"{fallback_max_range_m:.1f} m. Using the PING. One of them is "
                         f"wrong and every detection range scales with it.")
        return rng / n_bins, "message", complaint
    if fallback_max_range_m <= 0.0:
        return None, ("the ping carries no max_duration and no max_range_m is configured, "
                      "so there is no range scale at all"), None
    return fallback_max_range_m / n_bins, "parameter", None


def required_off_nadir_deg(rope_depth_m: float,
                           standoff_m: float = 2.0,
                           min_scan_depth_m: float = 0.5,
                           depth_control_error_m: float = 0.3,
                           min_beam_margin_m: float = 0.5) -> Tuple[float, float]:
    """The beam a lane at `standoff_m` needs, as two angles: (max allowed theta_min,
    min required theta_max), both in degrees from nadir.

    This exists so the in-editor verifier can check a live prefab against a PROPERTY
    rather than against a pair of constants somebody chose. Unity reads these two
    numbers out of the site manifest and compares them with the Sonar component's own
    tilt/breadth, so if a future session picks different beam parameters the check goes
    on meaning the same thing: "can this sonar see a rope at the prior's depth from a
    lane the vehicle can actually fly?"

    The two worst cases are opposite and both are taken:
      * theta_max is set by the vehicle being DEEPEST (least vertical separation) and
        the cross-track error pushing it FURTHEST from the line;
      * theta_min is set by the vehicle being SHALLOWEST and the cross-track error
        pushing it CLOSEST, i.e. toward the nadir gap.
    """
    dz_min = max(1e-6, rope_depth_m - (min_scan_depth_m + depth_control_error_m))
    dz_max = max(1e-6, rope_depth_m - max(0.0, min_scan_depth_m - depth_control_error_m))
    theta_max_required = math.degrees(math.atan2(standoff_m + min_beam_margin_m, dz_min))
    theta_min_allowed = math.degrees(math.atan2(max(0.0, standoff_m - min_beam_margin_m), dz_max))
    return (theta_min_allowed, theta_max_required)


#: The beam the simulator ships today, taken from the COMMITTED
#: SMARCAssets/Runtime/Prefabs/Components/SAMSensorsV2.prefab — tilt 0 / breadth 60,
#: set by commit 7226e43 ("sidescan mount fixed") on 2026-08-12 to the DE680D vertical
#: opening: fan centre 30 deg below horizontal per side, off-nadir 30..90 deg, with a
#: proper nadir gap. HF680 preset: DeepVisionSSS.Apply() overwrites MaxRange/buckets/rays
#: at Awake, so those three are the preset's values and tilt/breadth are the prefab's.
#:
#: READ THIS BEFORE YOU UPDATE THESE NUMBERS. The 2026-08-16 version of this constant
#: said tilt 45 / breadth 45, and it was wrong: it had been read out of the WORKING TREE,
#: where Unity had silently re-serialized the prefab back to its pre-7226e43 values
#: without anyone committing that. One bad read propagated into this constant, into its
#: guard test, into the farm prior, and into a PROPOSED_BEAM invented to solve a problem
#: the repo had already solved four days earlier. A prefab value is only a fact once
#: `git diff -- <prefab>` is clean. SETTLED section 3k carries the correction.
SIM_BEAM_AS_SHIPPED = BeamGeometry(
    tilt_deg=0.0, breadth_deg=60.0, max_range_m=100.0, num_buckets=2000,
    source="SAMSensorsV2.prefab at commit 7226e43 (HF680 preset); breadth is the DE680D "
           "vertical opening, tilt is the sim's mount, NOT a full DeepVision datasheet "
           "figure — nobody in this repo has read the datasheet")

#: NOT A TARGET — the known-bad beam Unity wrote over the prefab on 2026-08-16 (the
#: pre-7226e43 mount: off-nadir 0..45 deg, so the visible lateral band never exceeds the
#: vertical separation and no 2 m lane exists at any rope depth). Kept because it is the
#: SIGNATURE of that regression: if a future session measures the prefab and gets this
#: back, the prefab has been re-serialized, not redesigned. Also gives the refusal guards
#: a beam that genuinely cannot reach, so they test the refusal path rather than whatever
#: the shipped beam happens to be this month.
REGRESSED_BEAM_2026_08_16 = BeamGeometry(
    tilt_deg=45.0, breadth_deg=45.0, max_range_m=100.0, num_buckets=2000,
    source="REGRESSION SIGNATURE, not a configuration: Unity's uncommitted re-serialization "
           "of SAMSensorsV2.prefab on 2026-08-16, reverting commit 7226e43")
