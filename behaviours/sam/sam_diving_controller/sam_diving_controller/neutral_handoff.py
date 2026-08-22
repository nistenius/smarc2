"""When is it safe for the dive controller to LET GO of the actuators?

THE BUG THIS EXISTS TO FIX  (Ivan, several rig sessions, most recently 2026-08-16)

    "Unity HUD is still showing depth hold and VBS at 43% after mission finished (it starts with
     vbs 0 and surfacing, then pops back to the hold stuff)"
    "Unity HID still shows VBS 45% when done"

The chain, read end to end rather than guessed at:

  1. The last waypoint completes.  ActionServerDiveSub sets MissionStates.COMPLETED.
  2. DiveControllerBlendPID.update() sees COMPLETED and calls _set_actuators_neutral(), which
     commands `vbs_u_neutral` — 0.0 — and flips DivePub to ActuatorStates.NEUTRAL.  Correct.  This
     is the "starts with vbs 0 and surfacing" Ivan sees.
  3. DivePub.update() then publishes the neutral command **twenty times** and switches itself to
     DISENGAGED, after which it publishes nothing at all.

Step 3 is the defect, and it is a familiar shape: **a fixed tick count standing in for an
outcome.**  Twenty publishes at the controller rate is a second or two.  A VBS tank does not empty
in a second or two.  So the command to empty is withdrawn while the tank is still draining, the
tank stops wherever it had got to — around 45% — and the vehicle is left part-buoyant with nobody
commanding it.  Nothing "pops back": the surfacing simply stops halfway and the last reported
value sits there looking like a held setpoint.

Same family as SETTLED §3e's "a stopwatch may not certify a flight nobody saw take off", and as
the health gate that counted attempts instead of readings.  A hand-off is a claim that the vehicle
HAS REACHED a safe resting state; only the vehicle's own depth and actuator feedback can support
that claim, and a counter never can.

WHAT THIS MODULE DOES.  It is pure — no ROS, no node, no clock — so it can be driven exhaustively
in a test rather than inferred from a rig run.  `neutral_handoff()` answers one question: given
what we have published and what the vehicle reports back, may we now stop publishing?

THREE PROPERTIES WORTH STATING, because they are the ones that were wrong before:

  * A hand-off is CONFIRMED only by feedback.  Reaching the surface and reaching neutral VBS are
    both required; either alone is not the resting state.
  * A hand-off on a TIMEOUT is still a hand-off — a controller that clings to the actuators
    forever is worse than one that lets go — but it is reported as a timeout, never as a
    confirmation.  The caller logs the difference so a vehicle that never manages to surface is
    visible instead of silently identical to one that did.
  * ABSENT FEEDBACK IS NOT ARRIVAL.  A vehicle that reports no depth cannot be certified as
    surfaced.  It falls through to the timeout branch and says so.  (The same absent-versus-empty
    rule as `bags_onboard` and `dr_track` — this project keeps paying for the other reading.)

SIGN CONVENTION: none required, deliberately.  `depth_m` is compared as |depth| against the
surface tolerance, because "at the surface" is "near zero" under BOTH conventions in this tree —
`DiveSub.get_depth()` returns an odom/DR `position.z` whose sign depends on the frame, while
`get_sensor_depth()` is a pressure reading.  This project has already lost a session to a sign
inversion (SETTLED, rebuilt-VM yaw divergence), and a hand-off that silently inverts would release
the actuators at depth.  Taking the magnitude costs nothing — a vehicle 5 m ABOVE the water is not
a case — and removes the whole class of bug.
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class HandoffVerdict(NamedTuple):
    """`release` is the decision; `confirmed` says whether the vehicle EARNED it."""
    release: bool
    confirmed: bool
    reason: str


# Defaults are deliberately conservative and are all overridable from the caller's params.
DEFAULT_MIN_TICKS = 20          # what the old code used as its ONLY rule; kept as the floor
DEFAULT_MAX_TICKS = 600         # ~60 s at 10 Hz — long enough for a real VBS purge
DEFAULT_SURFACE_DEPTH_M = 0.35  # "at the surface" for a hull whose sensor sits below the waterline
DEFAULT_VBS_TOL_PCT = 5.0


def at_surface(depth_m: Optional[float],
               surface_depth_m: float = DEFAULT_SURFACE_DEPTH_M) -> Optional[bool]:
    """Is the hull at the surface, according to its OWN depth reading?

    Extracted from `neutral_handoff()` on 2026-08-19 so there is exactly ONE definition of "at
    the surface" in this tree, and `neutral_handoff` below now calls it rather than repeating
    the comparison. The second caller is the recorder in `data-cube/services/unity_bridge/
    bridge_node.py`: the post-mission stop policy (stop and save the bag once the vehicle has
    been surfaced and idle for five minutes) needs the same fact, and a recorder carrying its
    own private threshold is the two-implementations-of-one-safety-decision shape SETTLED 3l
    already charges this project for.

    Returns None -- never False -- when the vehicle has not reported a depth. ABSENT FEEDBACK IS
    NOT ARRIVAL, and it is equally not a refutation: a caller has to be able to tell "it says it
    is down" from "it has said nothing", because those need different words in front of an
    operator. `neutral_handoff`'s own timeout branch is where "it never said" becomes a decision.

    Magnitude, not sign -- see this module's SIGN CONVENTION note.
    """
    if depth_m is None:
        return None
    return abs(depth_m) <= surface_depth_m


def neutral_handoff(
    ticks: int,
    depth_m: Optional[float],
    vbs_pct: Optional[float],
    vbs_target_pct: float,
    *,
    min_ticks: int = DEFAULT_MIN_TICKS,
    max_ticks: int = DEFAULT_MAX_TICKS,
    surface_depth_m: float = DEFAULT_SURFACE_DEPTH_M,
    vbs_tol_pct: float = DEFAULT_VBS_TOL_PCT,
) -> HandoffVerdict:
    """May the controller stop publishing neutral commands?

    Args:
        ticks:          how many neutral commands have been published so far.
        depth_m:        the vehicle's own depth, positive down.  None = it has not said.
        vbs_pct:        the vehicle's own VBS feedback.  None = it has not said.
        vbs_target_pct: the neutral VBS the controller is commanding (`vbs_u_neutral`).
    """
    # The command has to actually go out before any of this means anything.  A single publish can
    # be lost; the floor is the old behaviour's whole rule, kept as a floor and nothing more.
    if ticks < min_ticks:
        return HandoffVerdict(False, False, f"holding neutral — {ticks}/{min_ticks} commands sent")

    # One definition of "at the surface", shared with the recorder -- see at_surface() above.
    # `is True` because at_surface returns None for "has not said", which is not arrival.
    surfaced = at_surface(depth_m, surface_depth_m) is True
    at_neutral = vbs_pct is not None and abs(vbs_pct - vbs_target_pct) <= vbs_tol_pct

    if surfaced and at_neutral:
        return HandoffVerdict(
            True, True,
            f"surfaced ({abs(depth_m):.2f} m) with VBS at neutral ({vbs_pct:.0f}%) — releasing actuators")

    if ticks >= max_ticks:
        # Let go anyway, and say exactly which half never arrived. A controller that holds the
        # actuators forever cannot be taken over by an operator, which is the worse failure.
        missing = []
        if depth_m is None:
            missing.append("no depth feedback")
        elif not surfaced:
            missing.append(f"still {abs(depth_m):.2f} m down")
        if vbs_pct is None:
            missing.append("no VBS feedback")
        elif not at_neutral:
            missing.append(f"VBS at {vbs_pct:.0f}%, wanted {vbs_target_pct:.0f}%")
        return HandoffVerdict(
            True, False,
            "releasing actuators on TIMEOUT after "
            f"{ticks} commands — never confirmed surfaced: {'; '.join(missing)}")

    # Still trying, and still commanding. This is the branch the old code never had.
    waiting = []
    if depth_m is None:
        waiting.append("depth not reported")
    elif not surfaced:
        waiting.append(f"{abs(depth_m):.2f} m down")
    if vbs_pct is None:
        waiting.append("VBS not reported")
    elif not at_neutral:
        waiting.append(f"VBS {vbs_pct:.0f}%")
    return HandoffVerdict(False, False, f"surfacing — {', '.join(waiting)}")
