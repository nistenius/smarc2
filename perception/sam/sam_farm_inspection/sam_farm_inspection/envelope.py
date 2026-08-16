#!/usr/bin/env python3
"""The protective-stop envelope, as a function — so the planner and the detector
cannot disagree about how much room a lane needs.

`sam_perception/obstacle_detector.py` owns this envelope and is the only thing that
can trigger a stop. Everything here is a READ of that node's rule, kept honest by
`test_envelope_matches_the_detector.py`, which parses the detector's own
`declare_parameter` defaults and fails if they drift from the numbers below. That
test exists because copying safety constants into a second file is precisely how
`STOP_M` inherited a suppression value and the protective stop became a report of the
collision (spec invariant 8, SETTLED §3).

Do NOT weaken any of these to make a lane fit. If a lane cannot be flown at a safe
distance, that is a finding to report, not a parameter to change (mission design
decision D7).
"""

#: Defaults as declared in sam_perception/obstacle_detector.py on 2026-08-16.
#: Names match the ROS parameter names exactly so a param dump can be diffed against them.
STOP_MARGIN_M = 0.5
STOP_T_REACT_S = 1.0
STOP_A_STOP_MS2 = 0.35


def r_stop_m(u_ms: float,
             margin_m: float = STOP_MARGIN_M,
             t_react_s: float = STOP_T_REACT_S,
             a_stop_ms2: float = STOP_A_STOP_MS2) -> float:
    """R_stop(u) = margin + u*t_react + u^2 / (2*a_stop) — QUADRATIC in u.

    Measured consequences at the shipped defaults, worth having in front of you when
    picking a scan speed:

        0.3 m/s -> 0.93 m      0.5 m/s -> 1.36 m
        0.7 m/s -> 1.90 m      1.0 m/s -> 2.93 m

    Note that the mission instructions quote "R_stop ~ 0.9 m at 0.5 m/s"; that is the
    figure with the reaction term dropped. The shipped `stop_t_react` is 1.0 s, so the
    real trigger at 0.5 m/s is 1.36 m and at the IROS scan speed of 0.7 m/s it is 1.90 m.
    A 2 m lane standoff clears the 0.7 m/s trigger by 0.10 m.
    """
    u = abs(float(u_ms))
    return margin_m + u * t_react_s + u * u / (2.0 * a_stop_ms2)


def stop_cone_applies(bearing_deg: float, cone_half_deg: float = 35.0) -> bool:
    """Only obstacles inside the FORWARD cone can trigger the stop.

    This is why a 2 m lane standoff is flyable at all: a rope line running parallel to
    the lane sits abeam, outside the cone, and never triggers. It comes back INSIDE the
    cone at the end of every lane, when the vehicle turns and the cross-line is ahead —
    which is where a farm-inspection mission will actually meet the protective stop.
    """
    b = abs(((float(bearing_deg) + 180.0) % 360.0) - 180.0)
    return b <= cone_half_deg
