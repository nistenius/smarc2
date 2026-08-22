#!/usr/bin/env python3
"""The farm-inspection mission plan: four phases, one sub-goal at a time.

P5's core. Pure python — no ROS, no numpy, no I/O — because a mission plan is arithmetic
and geometry, and everything that decides where the vehicle goes must be testable on a
laptop. The ROS node is a thin shell around this (W3), and the behaviour tree owns the
phase transitions: this object only ever answers *"what next?"*.

    T1 approach   surface transit from the launch point to a standoff south of the
                  PRIOR farm centre, then (only if the encircle depth is not the
                  surface) a dive.
    T2 encircle   a closed loop `encircle_standoff_m` outside the prior buoy hull, at
                  the depth the prior's own geometry says can SEE the buoys.
    T3 lanes      computed from the UPDATED map: one pass in each gap between culture
                  lines plus one outside each outer line (IROS 2025 V.B, scaled to a
                  two-line farm = corridor + 2 external), at the prior's lane depth and
                  standoff, at scan speed.
    T4 return     surface where it is, transit home on the surface, report done.

THREE THINGS THAT ARE NOT OBVIOUS AND ARE THE POINT OF THIS FILE
===============================================================

1. NO CONSTANT ABOUT THE FARM LIVES HERE. Depths, standoffs, speeds, the hull, the
   buoys and the metres->lat/lon map all come from `kristineberg_farm_prior.yaml`
   through `farm_prior.load_farm_prior`. Change the prior and the mission moves. The
   numbers that DO live here are mission-shape parameters (how far south to approach
   from, how long a leg may be), and each one says why it is what it is.

2. THE ENCIRCLE DEPTH IS DERIVED, AND AT THIS FARM IT IS THE SURFACE. A side scan sees
   nothing at or above its own depth. A buoy is a sphere centred on the water line, so
   the only part of it that can ever echo is the 0.30 m (0.18 m for the intermediate
   buoys) that hangs below. Fly the encircle at the rope depth — the naive reading of
   "the ropes are at 2 m, that is the working depth" — and the loop returns NO buoys at
   all while looking straight at the farm. The prior carries the derived depth and the
   list of buoys visible from it; this planner refuses to start T2 if that block says
   the loop is not flyable, naming the buoys.

3. THE GROSS PRIOR->OBSERVED TRANSFORM IS NOT A FARM DISPLACEMENT AND IS NOT USED TO
   NAVIGATE. The localizer fits the prior onto detections placed with `dr/odom`, whose
   origin is wherever the vehicle happened to start (SETTLED §3e measured that offset at
   126.92 m). So the fitted translation is dominated by the frame offset and CANNOT be
   separated from a real, rigid movement of the whole farm by any amount of cleverness —
   the information is simply not in the fit. What IS frame-independent is the SHAPE
   correction: each culture line's fitted bearing and its perpendicular offset from the
   transformed prior line, and each buoy's residual vector rotated back through the
   fit's own rotation. So `update_map_from_report` applies those to the prior IN THE
   GEOREFERENCED FRAME and reports the gross transform as a finding, labelled for what
   it is. Using the raw fitted translation as a waypoint would fly the vehicle to the
   odom origin's offset — which is how a 127 m error gets into a mission plan.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

# ---------------------------------------------------------------- phase vocabulary
T1_APPROACH = "T1_approach"
T2_ENCIRCLE = "T2_encircle"
T3_LANES = "T3_lanes"
T4_RETURN = "T4_return"
PHASES = (T1_APPROACH, T2_ENCIRCLE, T3_LANES, T4_RETURN)

#: Answer kinds. `wait` is not a failure: it means the phase has run out of sub-goals and
#: is waiting for evidence that has not arrived (the map, after the encircle). A planner
#: that answered `phase_done` there would hand the tree a map it does not have.
GOAL = "goal"
PHASE_DONE = "phase_done"
MISSION_DONE = "mission_done"
REFUSED = "refused"
WAIT = "wait"

#: How a sub-goal is flown (P8.2, 2026-08-19 — Ivan's farm-relative requirement).
#:
#:   dr_waypoints    a lat/lon flown on dead reckoning. Everything before P8, and every phase
#:                   except T3, forever: an approach across open water has no farm to be
#:                   relative to, and the encircle is what MEASURES the farm.
#:   farm_relative   the same lat/lon, but the commanded track is re-expressed through the
#:                   SLAM correction so it sits where the OBSERVED ropes are (P8.3). Only T3,
#:                   and only after `arm_farm_relative()` has said yes by name.
#:
#: THERE IS NO THIRD VALUE AND THERE IS NO DEFAULT-TO-THE-NEAREST-THING. An unknown mode is
#: refused (SETTLED §3d), and a gate that cannot be satisfied falls back to surface-and-report,
#: never silently to DR lanes — flying pre-assigned waypoints while REPORTING farm-relative
#: navigation is the failure this whole work package exists to prevent.
DR_WAYPOINTS = "dr_waypoints"
FARM_RELATIVE = "farm_relative"
NAV_MODES = (DR_WAYPOINTS, FARM_RELATIVE)

#: Buoy verdicts that count towards arming farm-relative navigation. `moved` counts: the
#: encircle SAW it and measured where it now is. `missing` and `not_surveyed` do not, and they
#: are different facts (SETTLED §3k's four verdicts). Kept here AND in `farm_slam` because the
#: two answer different questions — "may this arm?" and "may this be a landmark?" — and a
#: shared constant would hide the day they diverge.
ARMING_VERDICTS = ("confirmed", "moved")


@dataclass(frozen=True)
class SubGoal:
    """One waypoint, with the reason it exists.

    `why` is not decoration. Every sub-goal this planner emits is derived, so when a
    rehearsal produces a track nobody expected, the answer to "why did it go there" has
    to be readable off the goal itself rather than reconstructed from the code.
    """

    phase: str
    name: str
    lat: float
    lon: float
    depth_m: float          # POSITIVE-DOWN metres; 0.0 = the surface
    speed_ms: float
    tolerance_m: float
    why: str
    unity_xz: Tuple[float, float] = (0.0, 0.0)
    #: HOW THIS GOAL IS TO BE FLOWN (P8.2, 2026-08-19). `dr_waypoints` is everything this
    #: planner has ever emitted: a lat/lon the vehicle flies to on dead reckoning. Only T3 can
    #: ever be `farm_relative`, and only when `arm_farm_relative()` has said so by name — the
    #: mode is a PROPERTY OF THE GOAL rather than a mode flag on the planner, because a mode
    #: flag is a second piece of state that can disagree with the goal actually in flight, and
    #: this project has already paid for what two writers of one truth cost (invariant 12).
    nav_mode: str = "dr_waypoints"

    def to_waypoint_params(self, rpm: float, timeout_s: float) -> Dict[str, object]:
        """The `auv-depth-move-to` params dict, exactly as the action server parses it.

        `ActionServerDiveSub.goal_callback` reads latitude/longitude/rpm/target_depth/
        tolerance unconditionally and `speed` only if present. So speed is included (this
        mission commands metres per second — ADR-004) and rpm travels ALONGSIDE it, never
        instead of it: a hull that ignores speed must still get a flyable waypoint.

        `depth_mode` and `target_altitude` are deliberately OMITTED. ADR-004 D2's presence
        rule: a mission that does not ask for them stays byte-identical to one from QGIS,
        and this mission holds depth throughout — the lane depth is what the sonar geometry
        requires, and bottom-following would move it.
        """
        return {
            "waypoint": {
                "latitude": self.lat,
                "longitude": self.lon,
                "target_depth": self.depth_m,
                "rpm": float(rpm),
                "speed": self.speed_ms,
                "tolerance": self.tolerance_m,
            },
            "name": self.name,
            "timeout": float(timeout_s),
        }


@dataclass(frozen=True)
class Answer:
    kind: str
    phase: str
    reason: str
    goal: Optional[SubGoal] = None
    #: Set on REFUSED. What the MISSION should do about it, not what went wrong — a
    #: refusal an operator cannot act on is a log line.
    response: str = ""


@dataclass
class FarmMap:
    """The farm in the GEOREFERENCED (prior) frame after the fix's shape corrections.

    See module docstring point 3 for why this is not simply the localizer's output.
    """

    ok: bool
    reason: str
    buoys_xz: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    verdicts: Dict[str, str] = field(default_factory=dict)
    #: name -> (end_a_xz, end_b_xz) for each culture line that FITTED. A line that did
    #: not fit is absent, not straightened onto the prior — losing the east line is not
    #: losing the farm, and a lane flown down a line nobody measured is a guess.
    lines_xz: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = \
        field(default_factory=dict)
    line_refusals: Dict[str, str] = field(default_factory=dict)
    #: The gross prior->observed transform, reported and NOT navigated on.
    gross_rotation_deg: float = 0.0
    gross_translation_m: Tuple[float, float] = (0.0, 0.0)
    rms_m: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class MissionConfig:
    """Mission-shape parameters. Farm geometry is NOT here — it is in the prior."""

    #: How far south of the prior farm centre T1 aims, before the encircle starts. The
    #: instructions say "~20 m south of the prior farm centre" (§3 D3); it must be at
    #: least the encircle standoff plus the prior's position uncertainty or the approach
    #: leg can end up inside the loop it is about to fly.
    approach_offset_m: float = 20.0
    #: Bearing (grid) from the farm centre to the approach point. South, because the
    #: launch point is 237 m to the SOUTH-WEST — approaching from the far side would fly
    #: the transit straight over the farm.
    approach_bearing_deg: float = 180.0
    #: Longest straight leg. Not a control parameter: `auv_depth_move_to` flies straight
    #: lines, so this only densifies the loop for the operator's per-leg view and gives
    #: the vehicle more chances to re-converge. 20 m keeps a 4-corner loop at ~10 legs.
    max_leg_m: float = 20.0
    #: How far each lane runs past the end of its rope line, so the ends are ensonified
    #: rather than clipped. One rope-line half-length is far too much; 5 m is about two
    #: vehicle lengths and two seconds of turn-in.
    lane_overrun_m: float = 5.0
    #: Arrival tolerance. Transit legs may be sloppy; a lane leg may not, because the
    #: standoff budget is metres and the beam margin is what pays for the error.
    transit_tolerance_m: float = 3.0
    lane_tolerance_m: float = 1.5
    #: Per-waypoint timeout handed to the action client. Generous: this is a runaway
    #: guard, not a schedule. 0 disables it.
    goal_timeout_s: float = 900.0
    #: Thruster fallback for hulls that ignore `speed`. Never used as the primary
    #: command — see `to_waypoint_params`.
    fallback_rpm: float = 500.0
    #: How far the geo map may be wrong before this planner refuses to emit lat/lon at
    #: all. The generator measures the real figure (0.037 m at Kristineberg); the limit
    #: is one lane standoff, because beyond that the linearisation is eating the margin
    #: the whole lane geometry was computed to protect.
    max_geo_error_m: float = 1.0
    #: ---- P8.2, the gate that arms farm-relative navigation (2026-08-19) --------------------
    #: How many culture lines T2 must have FITTED before T3 may be flown farm-relative. Two,
    #: because this farm has two and the lanes are planned between them: a corridor lane derived
    #: from one measured line and one prior line is half a measurement, and half a measurement
    #: presented as farm-relative is worse than an honest DR lane.
    min_fitted_lines_for_farm_relative: int = 2
    #: How many buoys must carry an arming verdict. Three, because two is a line and three is
    #: the fewest that fixes a rigid body in the plane — the same reasoning that makes the
    #: localizer refuse a fit from two correspondences (SETTLED §3k).
    min_verified_buoys_for_farm_relative: int = 3
    #: The map's own fit residual. Beyond this the encircle measured SOMETHING, but not this
    #: farm well enough to navigate relative to. Not a tuning knob: it is one lane standoff's
    #: worth of error, i.e. the point at which the correction could push a lane through a rope.
    max_map_rms_for_farm_relative_m: float = 2.0


class FarmMissionPlanner:
    """The phase machine. Ask `next()`; tell it `goal_reached()` and `map_update()`.

    Deliberately NOT a ROS node and deliberately not in charge: the behaviour tree owns
    transitions and this object owns geometry. It holds no action client, publishes
    nothing and writes no actuator — same shape as ADR-004's bottom-following setpoint
    source (invariant 12).
    """

    def __init__(self, prior, config: Optional[MissionConfig] = None,
                 encircle_depth_override_m: Optional[float] = None):
        self.prior = prior
        self.cfg = config or MissionConfig()
        self.map: Optional[FarmMap] = None
        self._phase_idx = 0
        self._goals: List[SubGoal] = []
        self._i = 0
        self._built_for_phase: Optional[str] = None
        self._refusal: Optional[Answer] = None
        self._last_xz: Tuple[float, float] = tuple(prior.launch_xz)
        self._encircle_override = encircle_depth_override_m
        # P8.2. Starts as DR and is only ever moved by arm_farm_relative(), which is the
        # ONE writer. A planner that could be put into farm-relative from two places is a
        # planner whose reported mode and flown mode can disagree.
        self._nav_mode: str = DR_WAYPOINTS
        self._log: List[str] = []

    # -------------------------------------------------------------- public surface
    @property
    def phase(self) -> str:
        return PHASES[self._phase_idx] if self._phase_idx < len(PHASES) else T4_RETURN

    @property
    def finished(self) -> bool:
        return self._phase_idx >= len(PHASES)

    def map_update(self, report: dict) -> None:
        """Hand in the localizer's latest report. Accepted at any time; only T2 waits."""
        self.map = update_map_from_report(self.prior, report)

    def goal_reached(self) -> None:
        """The vehicle arrived. Advance within the phase."""
        if self._i < len(self._goals):
            self._last_xz = self._goals[self._i].unity_xz
            self._i += 1

    def next(self) -> Answer:
        if self._refusal is not None:
            # A refusal latches. The tree may keep ticking; the answer may not quietly
            # become "carry on" because a later map arrived.
            return self._refusal
        if self.finished:
            return Answer(MISSION_DONE, T4_RETURN, "all four phases complete")

        phase = self.phase
        if self._built_for_phase != phase:
            ans = self._build(phase)
            if ans is not None:
                return ans
            self._built_for_phase = phase
            self._i = 0

        if self._i < len(self._goals):
            return Answer(GOAL, phase, self._goals[self._i].why, goal=self._goals[self._i])

        # Out of sub-goals for this phase.
        if phase == T2_ENCIRCLE and (self.map is None or not self.map.ok):
            return self._encircle_exhausted()

        self._phase_idx += 1
        if self.finished:
            return Answer(MISSION_DONE, phase, "all four phases complete")
        return Answer(PHASE_DONE, phase, f"{phase} complete, next is {self.phase}")

    def report(self) -> dict:
        """The mission's own view, for the status field and the operator."""
        return {
            "phase": self.phase if not self.finished else "done",
            "goal_index": self._i,
            "goals_in_phase": len(self._goals),
            "nav_mode": self._nav_mode,
            "farm_relative_gate": self.farm_relative_gate()[1],
            "refused": self._refusal is not None,
            "refusal": self._refusal.reason if self._refusal else None,
            "refusal_response": self._refusal.response if self._refusal else None,
            "map": None if self.map is None else {
                "ok": self.map.ok,
                "reason": self.map.reason,
                "gross_rotation_deg": round(self.map.gross_rotation_deg, 3),
                "gross_translation_m": [round(v, 2) for v in self.map.gross_translation_m],
                "gross_transform_note": (
                    "prior -> the localizer's pose frame. It contains the estimator's own "
                    "frame offset and cannot be separated from a real farm displacement; "
                    "the per-buoy verdicts are the displacement finding."),
                "rms_m": round(self.map.rms_m, 3),
                "verdicts": dict(self.map.verdicts),
                "lines_fitted": sorted(self.map.lines_xz),
                "line_refusals": dict(self.map.line_refusals),
            },
            "log": list(self._log),
            "caveats": list(self.prior.caveats),
        }

    # -------------------------------------------------------------- phase building
    def _build(self, phase: str) -> Optional[Answer]:
        """Fill `self._goals` for `phase`, or return the refusal that stops the mission."""
        geo_err = float(getattr(self.prior.geo, "max_error_m", 0.0))
        if geo_err > self.cfg.max_geo_error_m:
            return self._refuse(
                phase, f"the prior's metres->lat/lon map is measured accurate to only "
                       f"{geo_err:.2f} m over the mission box, and this planner will not "
                       f"emit waypoints beyond {self.cfg.max_geo_error_m:.2f} m",
                "regenerate the prior, or re-fit the geo map for this site")

        if phase == T1_APPROACH:
            self._goals = self._plan_approach()
        elif phase == T2_ENCIRCLE:
            ok, why, resp = self._encircle_is_flyable()
            if not ok:
                return self._refuse(phase, why, resp)
            self._goals = self._plan_encircle()
        elif phase == T3_LANES:
            goals, refusal = self._plan_lanes()
            if refusal is not None:
                return self._refuse(phase, refusal[0], refusal[1])
            self._goals = goals
        else:
            self._goals = self._plan_return()
        self._log.append(f"{phase}: {len(self._goals)} sub-goal(s)")
        return None

    def _refuse(self, phase: str, reason: str, response: str) -> Answer:
        """Latch a named refusal AND leave the vehicle a way home.

        The mission's answer to "the farm is not where the prior says" is never a guess
        and never a silent stop: hold off the farm, surface, report (instructions §3 D3).
        So the refusal replaces the remaining plan with T4's surface-and-go-home legs,
        and the tree still gets a REFUSED answer so nothing records a completed survey.
        """
        self._refusal = Answer(REFUSED, phase, reason, response=response)
        self._log.append(f"{phase} REFUSED: {reason}")
        return self._refusal

    # ------------------------------------------------ P8.2: arming farm-relative navigation
    def farm_relative_gate(self) -> Tuple[bool, str]:
        """May T3 be flown relative to the farm the vehicle just measured? (ok, sentence)

        Ivan, 2026-08-19: once the farm has been encircled and the buoys, ropes and anchors
        identified, the vehicle follows the ropes and plans from where they actually are,
        instead of flying DR-based pre-assigned waypoints.

        **THE T2 VERDICTS ARE THE GATE.** Farm-relative navigation is only ever as good as the
        map it is relative to, and the encircle is the only thing that measures that map. So
        this asks three questions of the VERIFIED map and nothing else:

          * did enough culture lines actually FIT? A line that did not fit is absent from
            `lines_xz`, never straightened onto its prior — a lane derived from a prior line is
            a guess wearing a measurement's name;
          * did enough buoys come back with a verdict that means "we saw it"? `confirmed` and
            `moved` both do. `missing` and `not_surveyed` do not, and treating them alike is
            precisely the error SETTLED §3k's four verdicts exist to prevent;
          * is the fit's own residual small enough that a correction derived from it cannot push
            a lane through a rope?

        RETURNS A SENTENCE, ALWAYS — including on success, because "farm-relative, from 2 fitted
        lines and 5 confirmed buoys, RMS 0.31 m" is what MC's T3 row shows and what a debrief
        reads back. A gate that only speaks when it refuses leaves the operator unable to tell
        *armed* from *nobody asked*.

        This method DECIDES NOTHING. The caller (`_plan_lanes`) applies it, and a failure is a
        fall back to `dr_waypoints`... no: **a failure is a REFUSAL**, see `arm_farm_relative`.
        """
        if self.map is None or not self.map.ok:
            return False, ("there is no verified map at all — the encircle produced nothing to "
                           "be relative to")
        fitted = sorted(self.map.lines_xz)
        if len(fitted) < self.cfg.min_fitted_lines_for_farm_relative:
            missing = sorted(self.map.line_refusals)
            return False, (
                f"only {len(fitted)} culture line(s) fitted "
                f"({', '.join(fitted) or 'none'}); farm-relative navigation needs "
                f"{self.cfg.min_fitted_lines_for_farm_relative}"
                + (f". Did not fit: {', '.join(missing)}" if missing else ""))
        seen = sorted(name for name, v in self.map.verdicts.items() if v in ARMING_VERDICTS)
        if len(seen) < self.cfg.min_verified_buoys_for_farm_relative:
            return False, (
                f"only {len(seen)} buoy(s) came back seen ({', '.join(seen) or 'none'}); "
                f"farm-relative navigation needs "
                f"{self.cfg.min_verified_buoys_for_farm_relative}. Verdicts: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.map.verdicts.items())))
        if self.map.rms_m > self.cfg.max_map_rms_for_farm_relative_m:
            return False, (
                f"the map fitted with RMS {self.map.rms_m:.2f} m, past the "
                f"{self.cfg.max_map_rms_for_farm_relative_m:.2f} m a lane standoff can absorb; "
                f"a correction from this fit could push a lane through a rope")
        return True, (
            f"farm-relative: {len(fitted)} line(s) fitted ({', '.join(fitted)}), "
            f"{len(seen)} buoy(s) seen, map RMS {self.map.rms_m:.2f} m")

    def arm_farm_relative(self) -> Answer:
        """Arm it, or refuse by name and go home. Never silently fly DR lanes instead.

        THE FALLBACK IS SURFACE-AND-REPORT, NOT DR LANES, AND THAT IS THE WHOLE POINT. A quiet
        downgrade to dead-reckoned waypoints is exactly the behaviour Ivan's requirement removes,
        and it is the more dangerous of the two failures: DR lanes THROUGH a farm whose position
        was not established are flown blind past ropes with a 2 m standoff budget, while the
        report says the survey happened. Same rule as SETTLED §3d — an unknown or unsatisfied
        mode is refused by name, never downgraded to the nearest thing that will still run.

        Called by the tree at T2 exit. Idempotent: asking twice does not arm twice.
        """
        ok, why = self.farm_relative_gate()
        if not ok:
            return self._refuse(
                T3_LANES, f"farm-relative navigation cannot be armed: {why}",
                "surface and report — do NOT fly the lanes on dead reckoning; the farm's "
                "position was never established well enough to scan between its ropes")
        self._nav_mode = FARM_RELATIVE
        self._log.append(f"T3 armed {FARM_RELATIVE}: {why}")
        return Answer(PHASE_DONE, T2_ENCIRCLE, why)

    @property
    def nav_mode(self) -> str:
        """The mode T3's goals will carry. Read-only: `arm_farm_relative` is the one writer."""
        return self._nav_mode

    def recovery_goals(self) -> List[SubGoal]:
        """The legs to fly after a refusal: surface where we are, then home on the surface.

        Exposed separately from `next()` so the tree, which owns transitions, decides
        when to fly them. A planner that started issuing recovery waypoints inside the
        same answer stream would be making that decision on the tree's behalf.
        """
        here = self._last_xz
        lat, lon = self.prior.geo.to_latlon(*here)
        hlat, hlon = self.prior.geo.to_latlon(*self.prior.launch_xz)
        why = "refused: " + (self._refusal.reason if self._refusal else "no refusal recorded")
        return [
            SubGoal(T4_RETURN, "R1_surface_here", lat, lon, 0.0,
                    self.prior.transit_speed_ms, self.cfg.transit_tolerance_m,
                    "surface where we are, off the farm — " + why, here),
            SubGoal(T4_RETURN, "R2_home", hlat, hlon, 0.0,
                    self.prior.transit_speed_ms, self.cfg.transit_tolerance_m,
                    "return to the launch point on the surface — " + why,
                    tuple(self.prior.launch_xz)),
        ]

    # ------------------------------------------------------------------ T1
    def _plan_approach(self) -> List[SubGoal]:
        cx, cz = self.prior.approx_xz
        b = math.radians(self.cfg.approach_bearing_deg)
        # Grid bearing: 0 = +Z (north), 90 = +X (east) — the same convention the prior's
        # own line bearings use, stated because the other convention is one sign away.
        ax = cx + self.cfg.approach_offset_m * math.sin(b)
        az = cz + self.cfg.approach_offset_m * math.cos(b)
        lat, lon = self.prior.geo.to_latlon(ax, az)
        goals = [SubGoal(
            T1_APPROACH, "T1_standoff", lat, lon, 0.0,
            self.prior.transit_speed_ms, self.cfg.transit_tolerance_m,
            f"surface transit to {self.cfg.approach_offset_m:.0f} m on bearing "
            f"{self.cfg.approach_bearing_deg:.0f} from the PRIOR farm centre, under GPS; "
            f"the farm's position is only known to "
            f"{self.prior.approx_uncertainty_m:.0f} m until T2 measures it",
            (ax, az))]

        depth = self._encircle_depth()
        if depth > 0.0:
            goals.append(SubGoal(
                T1_APPROACH, "T1_dive", lat, lon, depth,
                self.prior.transit_speed_ms, self.cfg.transit_tolerance_m,
                f"dive to the encircle depth {depth:.2f} m before the loop starts",
                (ax, az)))
        return goals

    # ------------------------------------------------------------------ T2
    def _encircle_depth(self) -> float:
        if self._encircle_override is not None:
            return float(self._encircle_override)
        return float(self.prior.encircle.get("depth_m", 0.0))

    def _encircle_is_flyable(self) -> Tuple[bool, str, str]:
        """The prior already asked the sonar geometry this question. Read the answer.

        Recomputing it here would be a second implementation of the check, and the two
        would drift — the failure this whole generated-prior arrangement exists to stop.
        The one case worth recomputing is an OVERRIDE, because then nothing has checked.
        """
        enc = self.prior.encircle
        if self._encircle_override is not None:
            from sam_farm_inspection.sss_geometry import (SIM_BEAM_AS_SHIPPED,
                                                          choose_encircle_geometry)
            g = choose_encircle_geometry(
                self.prior.buoy_extents_m, SIM_BEAM_AS_SHIPPED,
                self.prior.encircle_standoff_m,
                requested_depth_m=float(self._encircle_override))
            if not g.ok:
                return (False, g.reason,
                        "fly the encircle shallower, or accept that those buoys will read "
                        "`not_surveyed` — do not fly it and call the result an empty farm")
            return True, g.reason, ""
        if not enc.get("flyable", False):
            return (False,
                    str(enc.get("reason", "the prior does not say why")),
                    "regenerate the prior after fixing the geometry; flying it anyway "
                    "returns no buoys and reads exactly like a farm that is not there")
        return True, str(enc.get("reason", "")), ""

    def _plan_encircle(self) -> List[SubGoal]:
        depth = self._encircle_depth()
        speed = self.prior.scan_speed_ms
        loop = offset_convex_polygon(self.prior.hull_xz, self.prior.encircle_standoff_m)
        loop = _rotate_to_nearest(loop, self._last_xz)
        loop = _densify_closed(loop, self.cfg.max_leg_m)
        n_vis = len(self.prior.encircle.get("visible_buoys", []) or [])
        sep = self.prior.encircle.get("worst_separation_m", 0.0)
        goals = []
        for i, (x, z) in enumerate(loop):
            lat, lon = self.prior.geo.to_latlon(x, z)
            goals.append(SubGoal(
                T2_ENCIRCLE, f"T2_loop_{i + 1:02d}", lat, lon, depth, speed,
                self.cfg.transit_tolerance_m,
                f"encircle leg {i + 1}/{len(loop)}, {self.prior.encircle_standoff_m:.0f} m "
                f"outside the prior buoy hull at {depth:.2f} m — the depth that keeps all "
                f"{n_vis} buoys below the sonar (worst separation {sep:.2f} m)",
                (x, z)))
        # Close the loop: the localizer refuses a fit whose look-bearings do not span
        # enough of the farm, so the last leg back to the first waypoint is not a
        # flourish, it is the coverage the fit is gated on.
        if goals:
            f = goals[0]
            goals.append(SubGoal(T2_ENCIRCLE, "T2_loop_close", f.lat, f.lon, depth, speed,
                                 self.cfg.transit_tolerance_m,
                                 "close the loop — the fit is refused unless the look "
                                 "bearings span the farm", f.unity_xz))
        return goals

    def _encircle_exhausted(self) -> Answer:
        why = "the encircle finished and no usable farm fix arrived"
        if self.map is not None and not self.map.ok:
            why = f"the encircle finished and the localizer refused: {self.map.reason}"
        return self._refuse(
            T2_ENCIRCLE, why,
            "hold off the farm, surface and report. Do NOT scan lanes computed from the "
            "prior alone: an unverified map is the prior wearing a measurement's label, "
            "and the lanes it produces are flown blind past real structures")

    # ------------------------------------------------------------------ T3
    def _plan_lanes(self):
        lane = self.prior.lane
        if not lane.get("flyable", False):
            return [], (f"the prior's lane geometry is not flyable: "
                        f"{lane.get('reason', 'no reason given')}",
                        "regenerate the prior; do not fly a lane the sonar cannot see from")
        if self.map is None or not self.map.ok:
            return [], ("T3 was reached with no verified map",
                        "surface and report — lanes must come from the measured farm")
        if not self.map.lines_xz:
            return [], ("no culture line fitted: " +
                        "; ".join(f"{k}: {v}" for k, v in sorted(
                            self.map.line_refusals.items())),
                        "fly the encircle again from the missing side, then re-fit")

        depth = float(lane["scan_depth_m"])
        standoff = float(lane["standoff_m"])
        speed = self.prior.scan_speed_ms
        r_stop = self.prior.r_stop_at_scan_speed_m

        lines = [(name, self.map.lines_xz[name]) for name in sorted(self.map.lines_xz)]
        tracks, notes = plan_lane_tracks(lines, standoff, r_stop, self.cfg.lane_overrun_m)
        for n in notes:
            self._log.append("T3: " + n)
        if not tracks:
            return [], ("no lane clears the protective stop at the scan speed "
                        f"(R_stop {r_stop:.2f} m): " + "; ".join(notes),
                        "reduce the scan speed and re-plan, or report the farm as too "
                        "tight to scan — NEVER shrink the envelope to make a lane fit")

        goals: List[SubGoal] = []
        here = self._last_xz
        remaining = list(tracks)
        while remaining:
            # Nearest end first, and fly the track away from it: a boustrophedon that is
            # chosen rather than assumed, because the lanes are not parallel at this farm
            # (the two culture lines are 20.3 deg apart — the storm skew).
            best_i, best_flip, best_d = 0, False, None
            for i, (nm, a, b, why) in enumerate(remaining):
                for flip in (False, True):
                    start = b if flip else a
                    d = math.dist(here, start)
                    if best_d is None or d < best_d:
                        best_i, best_flip, best_d = i, flip, d
            nm, a, b, why = remaining.pop(best_i)
            p0, p1 = (b, a) if best_flip else (a, b)
            for k, pt in enumerate((p0, p1)):
                lat, lon = self.prior.geo.to_latlon(*pt)
                goals.append(SubGoal(
                    T3_LANES, f"T3_{nm}_{'end' if k else 'start'}", lat, lon, depth, speed,
                    self.cfg.lane_tolerance_m, why, pt, nav_mode=self._nav_mode))
            here = p1
        return goals, None

    # ------------------------------------------------------------------ T4
    def _plan_return(self) -> List[SubGoal]:
        here = self._last_xz
        lat, lon = self.prior.geo.to_latlon(*here)
        hlat, hlon = self.prior.geo.to_latlon(*self.prior.launch_xz)
        return [
            SubGoal(T4_RETURN, "T4_surface", lat, lon, 0.0, self.prior.scan_speed_ms,
                    self.cfg.transit_tolerance_m,
                    "surface at the end of the scan, clear of the farm, to re-anchor the "
                    "dead reckoning on a GPS fix before the transit home",
                    here),
            SubGoal(T4_RETURN, "T4_home", hlat, hlon, 0.0, self.prior.transit_speed_ms,
                    self.cfg.transit_tolerance_m,
                    "surface transit back to the launch point under GPS",
                    tuple(self.prior.launch_xz)),
        ]


# ==================================================================== map correction
def update_map_from_report(prior, report: dict) -> FarmMap:
    """Turn the localizer's report into a GEOREFERENCED map (module docstring, point 3).

    Only frame-independent quantities cross: each line's fitted bearing and perpendicular
    offset, and each buoy's residual vector rotated back through the fit's own rotation.
    The gross translation is carried as a finding and never as a coordinate.
    """
    if not isinstance(report, dict):
        return FarmMap(False, "the localizer report is not a dict")
    if not report.get("ok"):
        return FarmMap(False, str(report.get("reason", "the localizer refused, unstated")),
                       source=str(report.get("pose_topic", "")))

    rot = math.radians(float(report.get("rotation_deg", 0.0)))
    # R inverse: the residual is measured in the observed frame and the prior frame is
    # rotated from it by `rot`.
    c, s = math.cos(-rot), math.sin(-rot)

    buoys = dict(prior.buoys)
    verdicts: Dict[str, str] = {}
    for b in report.get("buoys", []) or []:
        name = b.get("name")
        if name not in buoys:
            continue
        verdicts[name] = str(b.get("status", "unknown"))
        obs, pred = b.get("observed_xz"), b.get("predicted_xz")
        if obs and pred:
            dx, dz = float(obs[0]) - float(pred[0]), float(obs[1]) - float(pred[1])
            px, pz = buoys[name]
            buoys[name] = (px + c * dx - s * dz, pz + s * dx + c * dz)

    prior_lines = {name: (a, b) for name, a, b, _ in prior.culture_lines}
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    refusals: Dict[str, str] = {}
    for ln in report.get("lines", []) or []:
        name = ln.get("name")
        if name not in prior_lines:
            refusals[str(name)] = "the report names a line the prior does not have"
            continue
        if not ln.get("ok"):
            refusals[name] = str(ln.get("reason", "refused, unstated"))
            continue
        a, b = prior_lines[name]
        lines[name] = _shift_segment(a, b, float(ln.get("offset_m", 0.0)))

    return FarmMap(
        ok=True, reason="ok",
        buoys_xz=buoys, verdicts=verdicts, lines_xz=lines, line_refusals=refusals,
        gross_rotation_deg=float(report.get("rotation_deg", 0.0)),
        gross_translation_m=tuple(float(v) for v in
                                  (report.get("translation_m") or (0.0, 0.0))),
        rms_m=float(report.get("rms_m", 0.0)),
        source=str(report.get("pose_topic", "")))


def _shift_segment(a, b, offset_m: float):
    """Move a segment `offset_m` to the RIGHT of its own bearing.

    Right-of-bearing is `fit_culture_lines`' convention for `offset_m` (`nvec = [u_z,
    -u_x]` there, in its (x, y) = our (x, z)). Restating the convention rather than
    importing it would be the copy that drifts, so the sign is spelled out here and
    pinned by a test that drives both.
    """
    ax, az = a
    bx, bz = b
    ux, uz = bx - ax, bz - az
    L = math.hypot(ux, uz)
    if L < 1e-9:
        return (a, b)
    ux, uz = ux / L, uz / L
    nx, nz = uz, -ux
    return ((ax + nx * offset_m, az + nz * offset_m),
            (bx + nx * offset_m, bz + nz * offset_m))


# ==================================================================== plain geometry
def offset_convex_polygon(pts: Sequence[Tuple[float, float]],
                          d: float) -> List[Tuple[float, float]]:
    """Push every edge of a convex polygon outward by `d` and re-intersect the edges.

    The naive alternative — move each VERTEX radially away from the centroid — leaves the
    edge MIDPOINTS closer than `d`, by up to a factor of the polygon's aspect ratio. At
    this farm (26 x 33 m hull, 9 m standoff) that would put the encircle 2.9 m closer to
    the south edge than the plan says, which is inside the protective stop's trigger at
    the transit speed. So the edges are offset properly and only a degenerate (nearly
    parallel) pair falls back to the radial construction, saying so is not needed because
    the fallback is never closer than `d` either — it is only further out than necessary.
    """
    pts = list(pts)
    n = len(pts)
    if n < 3 or d == 0.0:
        return pts
    if _signed_area(pts) < 0:            # normalise to counter-clockwise
        pts = pts[::-1]
    cx = sum(p[0] for p in pts) / n
    cz = sum(p[1] for p in pts) / n

    lines = []
    for i in range(n):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % n]
        ux, uz = bx - ax, bz - az
        L = math.hypot(ux, uz)
        if L < 1e-9:
            continue
        ux, uz = ux / L, uz / L
        nx, nz = uz, -ux                  # right of the edge = outward for CCW
        if (ax + nx - cx) * nx + (az + nz - cz) * nz < 0:
            nx, nz = -nx, -nz             # belt and braces: always point away
        lines.append(((ax + nx * d, az + nz * d), (ux, uz)))

    out = []
    m = len(lines)
    for i in range(m):
        p, u = lines[(i - 1) % m]
        q, v = lines[i]
        hit = _line_intersection(p, u, q, v)
        if hit is None:                   # parallel edges: fall back to the radial point
            vx, vz = pts[i][0] - cx, pts[i][1] - cz
            L = math.hypot(vx, vz) or 1.0
            hit = (pts[i][0] + d * vx / L, pts[i][1] + d * vz / L)
        out.append(hit)
    return out


def _signed_area(pts) -> float:
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                     - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))


def _line_intersection(p, u, q, v):
    den = u[0] * v[1] - u[1] * v[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * v[1] - (q[1] - p[1]) * v[0]) / den
    return (p[0] + t * u[0], p[1] + t * u[1])


def _rotate_to_nearest(loop, here):
    if not loop:
        return loop
    i = min(range(len(loop)), key=lambda k: math.dist(loop[k], here))
    return loop[i:] + loop[:i]


def _densify_closed(loop, max_leg_m: float):
    """Insert points so no leg exceeds `max_leg_m`, walking the closed loop."""
    if len(loop) < 2 or max_leg_m <= 0:
        return list(loop)
    out = []
    n = len(loop)
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        out.append(a)
        L = math.dist(a, b)
        k = int(math.ceil(L / max_leg_m)) - 1
        for j in range(1, k + 1):
            f = j / (k + 1.0)
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return out


def plan_lane_tracks(lines, standoff_m: float, r_stop_m: float, overrun_m: float):
    """Lane centre-lines for a set of fitted culture lines: one per GAP plus one outside
    each outer line — the IROS 2025 §V.B swath pattern, scaled (3 lanes for 2 lines).

    Returns (tracks, notes) where a track is (name, start_xz, end_xz, why). A lane that
    cannot hold `r_stop_m` from the structure it is scanning is DROPPED with a note, never
    moved closer (design decision D7: the envelope is not weakened to make a lane fly, and
    a lane that cannot be flown safely is a report).

    Lanes are placed by ordering the lines across their common normal, so this does not
    assume two lines, does not assume they are parallel (at this farm they are 20.3 deg
    apart) and does not assume which one is east.
    """
    notes: List[str] = []
    if not lines:
        return [], ["no fitted lines"]

    # A common reference direction: the mean line bearing as an AXIS (a line has no
    # direction, so averaging the bearings directly would cancel two anti-parallel fits).
    sx = sum(math.sin(2 * _bearing(a, b)) for _, (a, b) in lines)
    cxx = sum(math.cos(2 * _bearing(a, b)) for _, (a, b) in lines)
    axis = 0.5 * math.atan2(sx, cxx)
    ux, uz = math.sin(axis), math.cos(axis)          # along the lines
    nx, nz = uz, -ux                                 # across them

    # A line's offset across the axis is NOT one number: this farm's two culture lines
    # are 20.3 deg apart, so each line's projection onto the normal spans several metres
    # along its own length. Placing a lane against the line's MIDPOINT would put it
    # metres closer at one end than the plan claims — the same class of error as the
    # radial polygon offset above. Each line is therefore carried as a BAND [s_lo, s_hi],
    # and the final clearance is measured segment-to-segment, not from these numbers.
    spans = []
    for name, (a, b) in lines:
        sa, sb = _dot(a, (nx, nz)), _dot(b, (nx, nz))
        spans.append((min(sa, sb), max(sa, sb), name, a, b))
    spans.sort()

    positions: List[Tuple[str, float]] = [
        (f"outer_{spans[0][2]}", spans[0][0] - standoff_m),
    ]
    for i in range(1, len(spans)):
        lo_hi, n0 = spans[i - 1][1], spans[i - 1][2]
        hi_lo, n1 = spans[i][0], spans[i][2]
        gap = hi_lo - lo_hi
        # Inside a gap the ideal 2 m standoff is only reachable if the gap is wide enough
        # for BOTH lines to keep it. Otherwise the pass goes down the middle and the
        # resulting standoff is REPORTED — a corridor pass at 7 m still sees the ropes, it
        # just resolves them less well, and the operator should know which they got.
        if gap < 2.0 * (r_stop_m + 0.1):
            notes.append(f"corridor {n0}|{n1}: the lines come within {gap:.1f} m of each "
                         f"other, under twice the protective-stop trigger {r_stop_m:.2f} m "
                         f"— dropped, not squeezed")
            continue
        if gap > 2.0 * standoff_m + 1.0:
            notes.append(f"corridor {n0}|{n1}: the gap is {gap:.1f} m at its narrowest, so "
                         f"the pass runs down the middle at {gap / 2.0:.1f} m from each "
                         f"line rather than the preferred {standoff_m:.1f} m")
        positions.append((f"corridor_{n0}_{n1}", 0.5 * (lo_hi + hi_lo)))
    positions.append((f"outer_{spans[-1][2]}", spans[-1][1] + standoff_m))

    # Along-track extent: the union of every line's projection, plus the overrun.
    lo = min(min(_dot(a, (ux, uz)), _dot(b, (ux, uz))) for _, (a, b) in lines) - overrun_m
    hi = max(max(_dot(a, (ux, uz)), _dot(b, (ux, uz))) for _, (a, b) in lines) + overrun_m

    tracks = []
    for name, across in positions:
        start = (ux * lo + nx * across, uz * lo + nz * across)
        end = (ux * hi + nx * across, uz * hi + nz * across)
        # THE clearance number: true distance between the lane the vehicle will fly and
        # each rope line it will fly past. Derived from the tracks themselves, so no
        # placement mistake above can pass this check silently.
        clear = min(segment_distance(start, end, a, b) for _, (a, b) in lines)
        if clear < r_stop_m:
            notes.append(f"{name}: {clear:.2f} m from the nearest fitted rope line is "
                         f"inside the protective stop {r_stop_m:.2f} m — dropped")
            continue
        tracks.append((name, start, end,
                       f"lane {name}: {clear:.2f} m from the nearest FITTED rope line "
                       f"(protective stop {r_stop_m:.2f} m at scan speed), "
                       f"{hi - lo:.0f} m long including {overrun_m:.0f} m of overrun at "
                       f"each end"))
    return tracks, notes


def segment_distance(p0, p1, q0, q1) -> float:
    """Shortest distance between two 2D segments. No shapely on the vehicle."""
    def pt_seg(p, a, b):
        ax, az = a
        vx, vz = b[0] - ax, b[1] - az
        L2 = vx * vx + vz * vz
        if L2 < 1e-12:
            return math.dist(p, a)
        t = max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - az) * vz) / L2))
        return math.dist(p, (ax + t * vx, az + t * vz))

    if _segments_cross(p0, p1, q0, q1):
        return 0.0
    return min(pt_seg(p0, q0, q1), pt_seg(p1, q0, q1),
               pt_seg(q0, p0, p1), pt_seg(q1, p0, p1))


def _segments_cross(a, b, c, d) -> bool:
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = side(c, d, a), side(c, d, b)
    d3, d4 = side(a, b, c), side(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _bearing(a, b) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


def _mid(a, b):
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _dot(p, u) -> float:
    return p[0] * u[0] + p[1] * u[1]
