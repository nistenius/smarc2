"""Bottom following: turn "stay N m above the seabed" into a DEPTH the existing controller flies.

Data Cube 2026-08-15, Ivan: "as a complement to holding a certain depth at each leg, I'd also
like to add the possibility for bottom tracking (this should then also be looking ahead using the
3D sonar to plan better the path to follow to track a varying bottom contour)".

WHY THIS IS A SETPOINT SOURCE AND NOT A NEW CONTROL MODE
--------------------------------------------------------
The obvious implementation is a second controller that closes the loop on altitude. This is
deliberately not that. The blend PID depth controller is the one piece of this stack that has
been tuned and flown and is the bringup default; a parallel altitude controller would need its
own gains, its own limit-cycle work (see 2026-08-09's dive limit cycle, fixed by structure), and
would be a second writer racing the first for the same actuators -- spec invariant 12, which cost
two hours of a healthy vehicle sitting at 0.00 m/s.

So bottom following computes a DEPTH SETPOINT and hands it to the controller that already works:

    seabed depth ahead (sonar)  ->  desired depth = seabed - target_altitude  ->  existing PID

Everything novel lives in choosing that number, which is a geometry and estimation problem, not a
control one. It also degrades honestly: with no bottom in view the node stops overriding and the
leg's own `target_depth` stands, which is why the wire format keeps sending it (spec §, tst.py).

WHY LOOKAHEAD, RATHER THAN JUST THE ALTIMETER
---------------------------------------------
A DVL/altimeter reads the seabed DIRECTLY BELOW. By the time a rising slope is under the vehicle,
the vehicle is already too low over it, and the pitch authority needed to recover grows with the
slope -- at 1 m/s over a 30 degrees rise, an altimeter-only loop is permanently behind. The
forward 3D sonar sees the bottom several seconds ahead, which converts a reaction into a plan.

The rule used here is deliberately simple and conservative: over the lookahead window, take the
SHALLOWEST SEABED (the highest point of the bottom) rather than the nearest or the mean, and
command the depth that clears it by target_altitude. Rising terrain is therefore anticipated,
while a hole or a trench does NOT pull the vehicle down into it -- an asymmetry that is the whole
safety argument. Descending to follow a drop-off has no deadline; failing to climb does.

NOT FLOWN. Built off-rig 2026-08-15 against recorded geometry and unit tests. The lookahead
window and climb-rate limit are the numbers a rig session must set (ADR-004).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


# Sentinel: "no usable bottom information". Never 0.0 -- a zero altitude reads as "the seabed is
# exactly at the vehicle", which is the most dangerous possible misreading of missing data.
NO_BOTTOM = None


@dataclass
class BottomFollowConfig:
    """Every number the policy needs, in one place, so a rig session can tune it visibly."""

    # How far ahead to look, in seconds of travel. Seconds rather than metres because the useful
    # horizon is "how long do I have to react", which scales with speed.
    lookahead_s: float = 8.0
    # Never look further than the sonar can actually resolve, whatever the speed says.
    max_lookahead_m: float = 30.0
    # Never look less far than this even when crawling; at 0.1 m/s an 8 s window is 0.8 m, which
    # is no lookahead at all and would silently degrade to altimeter-only behaviour.
    min_lookahead_m: float = 5.0
    # Rate limit on the commanded depth, m/s. The vehicle cannot climb arbitrarily fast, and a
    # step change in the setpoint is how the 2026-08-09 dive limit cycle was provoked.
    max_climb_rate_mps: float = 0.3
    max_descend_rate_mps: float = 0.2
    # Refuse to command shallower than this (surfacing mid-mission) or deeper than the leg's own
    # target_depth ceiling; both are hard clamps, not suggestions.
    min_depth_m: float = 0.5
    # A sonar return older than this is not evidence about what is ahead NOW.
    max_sample_age_s: float = 3.0
    # Below this many usable returns the window is not describing terrain, it is describing noise.
    min_samples: int = 3


@dataclass
class BottomSample:
    """One seabed observation: how far ahead along track, and how deep the seabed is there.

    `seabed_depth_m` is depth below the surface, positive down -- the same convention as
    `target_depth` on the waypoint, deliberately, so the two can be compared without a sign
    argument. Converting from a sonar range + vehicle attitude is the caller's job; this module
    is given geometry, not beams.
    """
    range_ahead_m: float
    seabed_depth_m: float
    age_s: float = 0.0


@dataclass
class BottomFollowResult:
    depth_setpoint_m: Optional[float]
    reason: str
    """Why this setpoint, or why there is none. NEVER empty and never a guess.

    A "no link"/"no bottom" output with no reason is the output this whole component exists to
    prevent -- dived out of range, silted water, a beam blocked by the hull, and a genuinely flat
    seabed all read identically otherwise, and they need different responses. Same rule the
    comms modems follow (SETTLED §3).
    """
    seabed_ahead_m: Optional[float] = None
    used_samples: int = 0
    clamped: bool = False


def lookahead_distance(speed_mps: float, cfg: BottomFollowConfig) -> float:
    """How far ahead to care about, given how fast we are actually moving."""
    d = max(0.0, float(speed_mps)) * cfg.lookahead_s
    return max(cfg.min_lookahead_m, min(cfg.max_lookahead_m, d))


def plan_depth(
    samples: Sequence[BottomSample],
    *,
    target_altitude_m: float,
    speed_mps: float,
    current_depth_m: float,
    fallback_depth_m: float,
    dt_s: float,
    cfg: Optional[BottomFollowConfig] = None,
    previous_setpoint_m: Optional[float] = None,
) -> BottomFollowResult:
    """The whole policy, as one pure function.

    Pure on purpose: this is the part that decides where the vehicle goes, and it must be
    testable without a ROS graph, a sonar, or water. The node below is a thin adapter.

    `fallback_depth_m` is the leg's own `target_depth` -- what to hold when the bottom cannot be
    seen. Returning None instead would leave the vehicle with no z command at all.
    """
    cfg = cfg or BottomFollowConfig()
    horizon = lookahead_distance(speed_mps, cfg)

    fresh = [s for s in samples
             if s.age_s <= cfg.max_sample_age_s
             and 0.0 <= s.range_ahead_m <= horizon
             and s.seabed_depth_m is not None]

    if len(fresh) < cfg.min_samples:
        return BottomFollowResult(
            depth_setpoint_m=fallback_depth_m,
            reason=(f"no usable bottom within {horizon:.0f} m ahead "
                    f"({len(fresh)} fresh return(s), need {cfg.min_samples}) — "
                    f"holding the leg's target depth {fallback_depth_m:.1f} m"),
            used_samples=len(fresh))

    # THE SHALLOWEST SEABED IN THE WINDOW, not the nearest and not the mean. Anticipates a rise;
    # refuses to be pulled into a hole. See the module docstring -- this asymmetry is the safety
    # argument for lookahead, and averaging would throw it away by design.
    shallowest = min(s.seabed_depth_m for s in fresh)
    desired = shallowest - float(target_altitude_m)

    clamped = False
    if desired < cfg.min_depth_m:
        desired = cfg.min_depth_m
        clamped = True
    # Never deeper than the leg asked for. `target_depth` is a ceiling in bottom-follow mode, and
    # a seabed reading that would drive the vehicle below it is more likely a bad return than an
    # instruction.
    if desired > fallback_depth_m > 0:
        desired = fallback_depth_m
        clamped = True

    # Rate limit from where we actually are, not from the last setpoint: if the vehicle has not
    # kept up, the setpoint must not run away from it and hand the PID a growing error.
    base = current_depth_m if previous_setpoint_m is None else previous_setpoint_m
    up = cfg.max_climb_rate_mps * max(0.0, dt_s)
    down = cfg.max_descend_rate_mps * max(0.0, dt_s)
    limited = max(base - up, min(base + down, desired))
    if abs(limited - desired) > 1e-9:
        clamped = True

    return BottomFollowResult(
        depth_setpoint_m=limited,
        reason=(f"seabed {shallowest:.1f} m at up to {horizon:.0f} m ahead "
                f"({len(fresh)} returns); holding {target_altitude_m:.1f} m above it"
                + (" (rate/limit clamped)" if clamped else "")),
        seabed_ahead_m=shallowest,
        used_samples=len(fresh),
        clamped=clamped)
