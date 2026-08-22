"""The farm-inspection mission plan, driven rather than described.

Every test here runs the REAL planner against the REAL generated prior and asserts on
what it produced — a waypoint's coordinates, a refusal's text, a measured clearance. None
of them assert on the shape of the source (SETTLED §1c: a structural test cannot see
whether a branch runs), and none re-implement the geometry they check (SETTLED §3d: a
test that re-implements the code proves only that the copy is self-consistent). Where a
number is asserted, it is measured from the planner's own output against an INDEPENDENT
computation — distance from a lane to a rope line, for instance, is measured with
`math.dist` over the line's endpoints, not with the planner's own placement arithmetic.

Run:
    PYTHONPYCACHEPREFIX=/tmp/pyc python3 -m pytest -p no:cacheprovider \
        smarc2/perception/sam/sam_farm_inspection/test/test_farm_mission.py -q
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam_farm_inspection.farm_mission import (  # noqa: E402
    GOAL, MISSION_DONE, PHASE_DONE, REFUSED, T1_APPROACH, T2_ENCIRCLE, T3_LANES, T4_RETURN,
    FarmMissionPlanner, MissionConfig, offset_convex_polygon, plan_lane_tracks,
    segment_distance, update_map_from_report)
from sam_farm_inspection.farm_prior import load_farm_prior  # noqa: E402


@pytest.fixture(scope="module")
def prior():
    return load_farm_prior()


# ------------------------------------------------------------------ helpers (test-side)
def _fake_report(prior, rot_deg=0.0, tx=0.0, tz=0.0, line_offsets=None,
                 refuse_lines=(), buoy_shift=None):
    """A localizer report as `farm_localizer_node.recompute` publishes it.

    The transform is applied HERE, in the test, exactly as the localizer would: the
    planner must be able to undo it. `line_offsets` and `buoy_shift` are the frame-free
    findings that must survive.
    """
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))

    def T(p):
        return (c * p[0] - s * p[1] + tx, s * p[0] + c * p[1] + tz)

    buoys = []
    for name, xz in prior.buoys.items():
        pred = T(xz)
        obs = pred
        if buoy_shift and name in buoy_shift:
            dx, dz = buoy_shift[name]
            obs = (pred[0] + dx, pred[1] + dz)
        buoys.append({"name": name, "status": "confirmed",
                      "residual_m": math.dist(pred, obs),
                      "predicted_xz": list(pred), "observed_xz": list(obs)})
    lines = []
    for name, a, b, _bearing in prior.culture_lines:
        if name in refuse_lines:
            lines.append({"name": name, "ok": False, "reason": "only 3 rope detections",
                          "bearing_deg": 0.0, "offset_m": 0.0, "rms_m": 0.0,
                          "n_points": 3})
            continue
        lines.append({"name": name, "ok": True, "reason": "ok", "bearing_deg": 0.0,
                      "offset_m": (line_offsets or {}).get(name, 0.0),
                      "rms_m": 0.2, "n_points": 50})
    return {"ok": True, "reason": "ok", "pose_topic": "dr/odom",
            "rotation_deg": rot_deg, "translation_m": [tx, tz], "rms_m": 0.4,
            "n_clusters": len(buoys), "n_matched": len(buoys),
            "buoys": buoys, "lines": lines, "caveats": list(prior.caveats)}


def _run(planner, stop_after=400):
    """Drive the planner to completion, collecting goals and the final answer."""
    goals, answers = [], []
    for _ in range(stop_after):
        a = planner.next()
        answers.append(a)
        if a.kind == GOAL:
            goals.append(a.goal)
            planner.goal_reached()
        elif a.kind in (REFUSED, MISSION_DONE):
            return goals, a
    raise AssertionError("planner did not terminate")


def _fitted_segments(prior, report):
    """The rope lines as the MAP says they are, computed independently of the planner."""
    m = update_map_from_report(prior, report)
    return list(m.lines_xz.values())


# ============================================================================ T1
def test_t1_is_a_surface_transit_to_a_standoff_south_of_the_prior_centre(prior):
    pl = FarmMissionPlanner(prior)
    a = pl.next()
    assert a.kind == GOAL and a.goal.phase == T1_APPROACH
    g = a.goal
    assert g.depth_m == 0.0, "the approach runs on the surface, under GPS"
    cx, cz = prior.approx_xz
    assert math.isclose(math.dist(g.unity_xz, (cx, cz)), 20.0, abs_tol=1e-6)
    assert g.unity_xz[1] < cz, "south of the centre, i.e. on the launch side"
    assert g.speed_ms == prior.transit_speed_ms


def test_t1_does_not_dive_when_the_encircle_is_at_the_surface(prior):
    """The dive leg is DERIVED. At this farm the encircle depth is the surface (a side
    scan cannot see a buoy from below it), so there is nothing to dive to and the plan
    must not contain a leg that does nothing."""
    assert prior.encircle["depth_m"] == 0.0
    pl = FarmMissionPlanner(prior)
    goals, _ = _run(pl)
    assert [g.name for g in goals if g.phase == T1_APPROACH] == ["T1_standoff"]


def test_t1_does_dive_when_the_encircle_depth_is_below_the_surface(prior):
    pl = FarmMissionPlanner(prior, encircle_depth_override_m=0.15)
    goals, _ = _run(pl)
    t1 = [g for g in goals if g.phase == T1_APPROACH]
    assert [g.name for g in t1] == ["T1_standoff", "T1_dive"]
    assert t1[1].depth_m == pytest.approx(0.15)


def test_the_approach_point_is_outside_the_encircle_loop(prior):
    """Approaching to a point inside the loop makes the first encircle leg cross the farm."""
    pl = FarmMissionPlanner(prior)
    goals, _ = _run(pl)
    approach = [g for g in goals if g.name == "T1_standoff"][0]
    hull = prior.hull_xz
    inside = _point_in_polygon(approach.unity_xz, hull)
    assert not inside
    assert min(_dist_point_to_polygon(approach.unity_xz, hull), 1e9) > 0.0


def _point_in_polygon(p, poly):
    x, z = p
    hit = False
    for i in range(len(poly)):
        ax, az = poly[i]
        bx, bz = poly[(i + 1) % len(poly)]
        if (az > z) != (bz > z):
            xx = ax + (z - az) * (bx - ax) / (bz - az)
            if xx > x:
                hit = not hit
    return hit


def _dist_point_to_polygon(p, poly):
    return min(segment_distance(p, p, poly[i], poly[(i + 1) % len(poly)])
               for i in range(len(poly)))


# ============================================================================ T2
def test_the_encircle_stays_at_least_the_standoff_off_the_hull_everywhere(prior):
    """Not just at the corners. Offsetting a polygon by moving its vertices away from the
    centroid leaves the EDGE MIDPOINTS closer than the standoff — on this 26x33 m hull by
    metres — so the check is measured against every hull edge from every loop leg."""
    pl = FarmMissionPlanner(prior)
    goals, _ = _run(pl)
    loop = [g.unity_xz for g in goals if g.phase == T2_ENCIRCLE]
    hull = prior.hull_xz
    worst = min(
        segment_distance(loop[i], loop[i + 1], hull[j], hull[(j + 1) % len(hull)])
        for i in range(len(loop) - 1) for j in range(len(hull)))
    assert worst >= prior.encircle_standoff_m - 1e-6, (
        f"the encircle comes within {worst:.2f} m of the buoy hull, but the plan claims "
        f"{prior.encircle_standoff_m:.2f} m")


def test_the_encircle_loop_closes(prior):
    pl = FarmMissionPlanner(prior)
    goals, _ = _run(pl)
    loop = [g for g in goals if g.phase == T2_ENCIRCLE]
    assert loop[-1].name == "T2_loop_close"
    assert loop[-1].unity_xz == loop[0].unity_xz


def test_no_encircle_leg_is_longer_than_the_configured_maximum(prior):
    cfg = MissionConfig(max_leg_m=8.0)
    pl = FarmMissionPlanner(prior, cfg)
    goals, _ = _run(pl)
    loop = [g.unity_xz for g in goals if g.phase == T2_ENCIRCLE]
    longest = max(math.dist(loop[i], loop[i + 1]) for i in range(len(loop) - 1))
    assert longest <= 8.0 + 1e-6


def test_the_encircle_depth_comes_from_the_prior_and_not_from_this_file(prior):
    """Move the prior's derived encircle depth and the plan must move with it. A hardcoded
    depth passes every other test in this file."""
    import copy
    p2 = copy.deepcopy(prior)
    object.__setattr__(p2, "encircle", dict(prior.encircle, depth_m=0.12))
    goals, _ = _run(FarmMissionPlanner(p2))
    assert {g.depth_m for g in goals if g.phase == T2_ENCIRCLE} == {0.12}


def test_encircling_at_the_rope_depth_is_refused_and_names_the_buoys(prior):
    """The naive reading of requirement 7 — 'the ropes are at 2 m, that is the working
    depth' — makes the encircle return NO buoys, because a side scan sees nothing at or
    above its own depth and a buoy only hangs 0.18-0.30 m below the water line. That has
    to fail loudly: an empty survey and an absent farm are indistinguishable afterwards."""
    pl = FarmMissionPlanner(prior, encircle_depth_override_m=prior.rope_depth_m)
    _, final = _run(pl)
    assert final.kind == REFUSED and final.phase == T2_ENCIRCLE
    assert "M1_west_mid" in final.reason and "C2_corner_SW" in final.reason
    assert "at or above its own depth" in final.reason
    assert final.response, "a refusal without a response is a log line"


def test_a_shallow_override_that_still_sees_every_buoy_is_allowed(prior):
    pl = FarmMissionPlanner(prior, encircle_depth_override_m=0.1)
    goals, final = _run(pl)
    assert final.kind == MISSION_DONE or final.kind == REFUSED
    assert {g.depth_m for g in goals if g.phase == T2_ENCIRCLE} == {0.1}


def test_the_encircle_runs_at_the_scan_speed_not_the_transit_speed(prior):
    goals, _ = _run(FarmMissionPlanner(prior))
    assert {g.speed_ms for g in goals if g.phase == T2_ENCIRCLE} == {prior.scan_speed_ms}


# ============================================================ T2 -> T3 refusals
def test_no_map_after_the_encircle_is_a_named_refusal_not_a_guessed_lane_plan(prior):
    pl = FarmMissionPlanner(prior)
    _, final = _run(pl)
    assert final.kind == REFUSED and final.phase == T2_ENCIRCLE
    assert "no usable farm fix" in final.reason
    assert "surface" in final.response and "report" in final.response


def test_a_refused_localizer_report_is_quoted_verbatim_in_the_refusal(prior):
    pl = FarmMissionPlanner(prior)
    pl.map_update({"ok": False, "reason": "coverage refusal: look bearings span 96 deg"})
    _, final = _run(pl)
    assert final.kind == REFUSED
    assert "look bearings span 96 deg" in final.reason


def test_a_refusal_latches_even_if_a_map_turns_up_afterwards(prior):
    """A late map does not retroactively make the survey valid, and the tree must not see
    the answer change from REFUSED back to GOAL under it."""
    pl = FarmMissionPlanner(prior)
    _, final = _run(pl)
    assert final.kind == REFUSED
    pl.map_update(_fake_report(prior))
    assert pl.next().kind == REFUSED


def test_after_a_refusal_the_recovery_legs_surface_here_then_go_home(prior):
    pl = FarmMissionPlanner(prior)
    _, final = _run(pl)
    assert final.kind == REFUSED
    rec = pl.recovery_goals()
    assert [g.name for g in rec] == ["R1_surface_here", "R2_home"]
    assert all(g.depth_m == 0.0 for g in rec)
    assert rec[-1].unity_xz == tuple(prior.launch_xz)
    assert final.reason in rec[0].why


# ============================================================================ T3
def test_a_two_line_farm_gets_three_lanes_corridor_plus_two_external(prior):
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior))
    goals, final = _run(pl)
    lanes = [g for g in goals if g.phase == T3_LANES]
    assert len(lanes) == 6, "three lanes, two waypoints each"
    names = {g.name.rsplit("_", 1)[0] for g in lanes}
    assert len(names) == 3
    assert sum(1 for n in names if n.startswith("T3_corridor")) == 1
    assert sum(1 for n in names if n.startswith("T3_outer")) == 2
    assert final.kind == MISSION_DONE


def test_every_lane_clears_the_protective_stop_measured_against_the_fitted_lines(prior):
    """D7: never weaken the envelope to make a lane fly. The check is on the LANE THE
    VEHICLE WILL FLY against the lines the MAP reports, measured here independently."""
    rep = _fake_report(prior)
    pl = FarmMissionPlanner(prior)
    pl.map_update(rep)
    goals, _ = _run(pl)
    lanes = [g for g in goals if g.phase == T3_LANES]
    segs = _fitted_segments(prior, rep)
    for i in range(0, len(lanes), 2):
        a, b = lanes[i].unity_xz, lanes[i + 1].unity_xz
        clear = min(segment_distance(a, b, s[0], s[1]) for s in segs)
        assert clear >= prior.r_stop_at_scan_speed_m, (
            f"{lanes[i].name} passes {clear:.2f} m from a rope line, inside the "
            f"{prior.r_stop_at_scan_speed_m:.2f} m protective stop")


def test_the_lane_depth_and_speed_are_the_priors(prior):
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior))
    goals, _ = _run(pl)
    lanes = [g for g in goals if g.phase == T3_LANES]
    assert {g.depth_m for g in lanes} == {prior.lane["scan_depth_m"]}
    assert {g.speed_ms for g in lanes} == {prior.scan_speed_ms}
    assert prior.lane["scan_depth_m"] < prior.rope_depth_m, \
        "the vehicle must fly ABOVE the ropes to see them"


def test_the_lanes_follow_the_measured_line_not_the_prior_line(prior):
    """The whole point of T2. Shift one fitted line 3 m and its outer lane must move by
    the same 3 m — a planner that quietly used the prior geometry would not move at all."""
    name = prior.culture_lines[0][0]
    base = _lane_positions(prior, _fake_report(prior))
    moved = _lane_positions(prior, _fake_report(prior, line_offsets={name: 3.0}))
    shifts = {k: math.dist(base[k], moved[k]) for k in base if k in moved}
    assert max(shifts.values()) > 2.5, (
        f"no lane moved when a fitted line moved 3 m: {shifts}")


def _lane_positions(prior, report):
    pl = FarmMissionPlanner(prior)
    pl.map_update(report)
    goals, _ = _run(pl)
    out = {}
    lanes = [g for g in goals if g.phase == T3_LANES]
    for i in range(0, len(lanes), 2):
        key = lanes[i].name.rsplit("_", 1)[0]
        a, b = lanes[i].unity_xz, lanes[i + 1].unity_xz
        out[key] = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    return out


def test_the_offset_sign_agrees_with_the_localizers_own_line_fit(prior):
    """The one place P4 and P5 share a CONVENTION rather than a value: `offset_m` is
    signed, and both sides have to mean the same 'right'. So this drives the real
    `fit_culture_lines` on detections deliberately placed 1.5 m to one side, and asserts
    that the planner's corrected line ends up on THAT side. Writing the convention down in
    two places and checking neither is how a farm gets scanned 3 m off, on the wrong side,
    with every test green."""
    import numpy as np
    from sam_farm_inspection.localizer import WorldDetection, fit_culture_lines

    name, a, b, _ = prior.culture_lines[0]
    ux, uz = b[0] - a[0], b[1] - a[1]
    L = math.hypot(ux, uz)
    ux, uz = ux / L, uz / L
    nx, nz = uz, -ux                      # the localizer's "right of the bearing"
    shift = 1.5
    dets = []
    for k in range(40):
        f = k / 39.0
        px = a[0] + ux * L * f + nx * shift
        pz = a[1] + uz * L * f + nz * shift
        dets.append(WorldDetection(x=px, y=pz, target="rope", confidence=1.0,
                                   look_bearing_deg=0.0))
    fits = fit_culture_lines(dets, [(name, a, b)], np.eye(2), np.zeros(2))
    fit = [f for f in fits if f.name == name][0]
    assert fit.ok, fit.reason
    assert fit.offset_m == pytest.approx(shift, abs=0.05)

    m = update_map_from_report(prior, {
        "ok": True, "rotation_deg": 0.0, "translation_m": [0.0, 0.0], "buoys": [],
        "lines": [{"name": name, "ok": True, "offset_m": fit.offset_m}]})
    ca, cb = m.lines_xz[name]
    # The corrected line must sit where the detections were, not 1.5 m the other way.
    assert math.dist(ca, (a[0] + nx * shift, a[1] + nz * shift)) < 0.05
    assert math.dist(cb, (b[0] + nx * shift, b[1] + nz * shift)) < 0.05


def test_the_gross_fitted_translation_is_never_used_as_a_coordinate(prior):
    """THE frame test. The localizer fits the prior onto detections placed with `dr/odom`,
    whose origin is wherever the vehicle started — measured at 126.92 m on 2026-08-16. That
    offset is inside the fitted translation and cannot be separated from a real farm
    displacement. Feeding it into a waypoint would fly the vehicle 127 m away from the farm
    it just mapped, which is exactly the shape of the bug SETTLED §3e records.

    So: the same farm, reported in two different odom frames, must produce the SAME lanes.
    """
    here = _lane_positions(prior, _fake_report(prior))
    far = _lane_positions(prior, _fake_report(prior, rot_deg=7.0, tx=126.92, tz=-61.6))
    assert set(here) == set(far)
    for k in here:
        assert math.dist(here[k], far[k]) < 0.05, (
            f"lane {k} moved {math.dist(here[k], far[k]):.1f} m when only the localizer's "
            f"own frame changed")


def test_a_buoy_residual_survives_the_frame_change_but_a_frame_offset_does_not(prior):
    """The corrected buoy positions are the frame-free finding, so a 4 m displacement of
    one buoy must appear in the georeferenced map at 4 m whatever frame it was seen in."""
    name = sorted(prior.buoys)[0]
    rep = _fake_report(prior, rot_deg=25.0, tx=-300.0, tz=88.0,
                       buoy_shift={name: (3.0, -2.0)})
    m = update_map_from_report(prior, rep)
    moved = math.dist(m.buoys_xz[name], prior.buoys[name])
    assert moved == pytest.approx(math.hypot(3.0, 2.0), abs=0.01)
    for other in prior.buoys:
        if other != name:
            assert math.dist(m.buoys_xz[other], prior.buoys[other]) < 0.01


def test_one_refused_line_costs_that_line_and_not_the_mission(prior):
    """'Losing the east line is not losing the farm' — the localizer refuses per line for
    exactly this reason, and the planner has to honour it rather than collapsing to one
    global failure."""
    name = prior.culture_lines[1][0]
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior, refuse_lines=(name,)))
    goals, final = _run(pl)
    lanes = [g for g in goals if g.phase == T3_LANES]
    assert final.kind == MISSION_DONE
    assert len(lanes) == 4, "one line -> two lanes, one either side"
    assert pl.map.line_refusals[name].startswith("only 3 rope detections")


def test_every_line_refused_is_a_named_refusal(prior):
    names = tuple(ln[0] for ln in prior.culture_lines)
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior, refuse_lines=names))
    _, final = _run(pl)
    assert final.kind == REFUSED and final.phase == T3_LANES
    assert "only 3 rope detections" in final.reason
    assert "encircle again" in final.response


def test_a_lane_that_cannot_clear_the_envelope_is_dropped_and_not_squeezed(prior):
    """Two rope lines 1.5 m apart: the corridor cannot exist at the scan speed's 1.90 m
    trigger. It must disappear WITH A NOTE, and the outer lanes must survive."""
    lines = [("A", ((0.0, 0.0), (0.0, 30.0))), ("B", ((1.5, 0.0), (1.5, 30.0)))]
    tracks, notes = plan_lane_tracks(lines, standoff_m=2.0, r_stop_m=1.90, overrun_m=5.0)
    assert len(tracks) == 2, [t[0] for t in tracks]
    assert all("corridor" not in t[0] for t in tracks)
    assert any("dropped, not squeezed" in n for n in notes)


def test_the_envelope_beats_the_requested_standoff_and_the_lane_is_dropped(prior):
    """D7 stated as an inequality the code has to honour: if the standoff asked for is
    INSIDE the protective stop at the scan speed, the lane does not get flown at the
    smaller distance and the stop does not get relaxed — the lane is dropped and said so.

    Added 2026-08-17 because mutation testing found the measured clearance check
    unreachable: with a sane standoff the band placement already guarantees it, so
    disabling the check changed nothing and the guard was proving only the arithmetic
    above it. This is the case where the two disagree.
    """
    lines = [("A", ((0.0, 0.0), (0.0, 30.0))), ("B", ((14.0, 0.0), (14.0, 30.0)))]
    tracks, notes = plan_lane_tracks(lines, standoff_m=1.0, r_stop_m=1.90, overrun_m=5.0)
    assert [t[0] for t in tracks] == ["corridor_A_B"], \
        "both outer lanes were 1.0 m from a rope line, inside the 1.90 m stop"
    assert sum("inside the protective stop" in n for n in notes) == 2


def test_a_wide_gap_reports_the_standoff_it_actually_got(prior):
    """At this farm the lines are 15 m apart at the narrowest, so the corridor pass runs
    at 7.5 m, not the preferred 2 m. That is a fine lane and a bad silence: the resolution
    of the corridor pass is four times worse than the plan's nominal figure."""
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior))
    _run(pl)
    notes = " ".join(pl.report()["log"])
    assert "rather than the preferred" in notes and "7.5 m from each line" in notes


# ============================================================================ T4
def test_t4_surfaces_before_it_transits_home(prior):
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior))
    goals, final = _run(pl)
    t4 = [g for g in goals if g.phase == T4_RETURN]
    assert [g.name for g in t4] == ["T4_surface", "T4_home"]
    assert all(g.depth_m == 0.0 for g in t4)
    assert t4[-1].unity_xz == tuple(prior.launch_xz)
    assert final.kind == MISSION_DONE


def test_the_phases_run_in_order_and_none_is_skipped(prior):
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior))
    seen = []
    for _ in range(400):
        a = pl.next()
        if a.kind == GOAL:
            if not seen or seen[-1] != a.goal.phase:
                seen.append(a.goal.phase)
            pl.goal_reached()
        elif a.kind == MISSION_DONE:
            break
        elif a.kind == REFUSED:
            raise AssertionError(a.reason)
    assert seen == [T1_APPROACH, T2_ENCIRCLE, T3_LANES, T4_RETURN]


def test_asking_twice_without_arriving_returns_the_same_goal(prior):
    """`next()` answers a question; it does not advance the mission. If it did, a tree that
    ticks faster than the vehicle flies would skip most of the plan."""
    pl = FarmMissionPlanner(prior)
    a, b = pl.next(), pl.next()
    assert a.goal == b.goal


# ============================================================== waypoints and the wire
def test_the_waypoint_params_are_exactly_what_the_action_server_parses(prior):
    """`ActionServerDiveSub.goal_callback` reads latitude/longitude/rpm/target_depth/
    tolerance unconditionally — a missing one is a KeyError inside a goal callback, i.e. a
    silent rejection — and `speed` only if present."""
    pl = FarmMissionPlanner(prior)
    g = pl.next().goal
    p = g.to_waypoint_params(rpm=500.0, timeout_s=900.0)
    wp = p["waypoint"]
    assert set(wp) >= {"latitude", "longitude", "rpm", "target_depth", "tolerance"}
    assert wp["speed"] == g.speed_ms
    assert wp["rpm"] == 500.0, "speed travels ALONGSIDE rpm, never instead of it"
    assert p["timeout"] == 900.0 and p["name"] == g.name


def test_depth_mode_and_target_altitude_are_omitted_not_defaulted(prior):
    """ADR-004 D2's presence rule: a mission that does not ask for them stays
    byte-identical to one from QGIS, and `seed_reference_mission.py`'s field-for-field
    check depends on that."""
    pl = FarmMissionPlanner(prior)
    wp = pl.next().goal.to_waypoint_params(500.0, 900.0)["waypoint"]
    assert "depth_mode" not in wp and "target_altitude" not in wp


def test_no_waypoint_is_commanded_at_a_negative_depth(prior):
    """`goal_callback` REJECTS target_depth < 0 outright ('SAM can't fly')."""
    pl = FarmMissionPlanner(prior)
    pl.map_update(_fake_report(prior))
    goals, _ = _run(pl)
    assert all(g.depth_m >= 0.0 for g in goals)


def test_the_geo_map_reproduces_the_priors_own_buoy_latlon(prior):
    """The planner turns metres into lat/lon with a linear map fitted by pyproj in the
    generator. Checked against the prior's OWN per-buoy lat/lon, which the generator
    computed the honest way — point-wise pyproj — so this catches a wrong Jacobian, a
    swapped axis or a lost microdegree scaling."""
    import yaml
    doc = yaml.safe_load(open(prior.path).read())
    worst = 0.0
    for b in doc["farm"]["buoys"]:
        lat, lon = prior.geo.to_latlon(b["unity_x"], b["unity_z"])
        dn = (lat - b["lat"]) * 111320.0
        de = (lon - b["lon"]) * 111320.0 * math.cos(math.radians(lat))
        worst = max(worst, math.hypot(dn, de))
    assert worst <= prior.geo.max_error_m + 0.02, (
        f"the linear geo map is {worst:.3f} m from the prior's point-wise lat/lon, but "
        f"claims {prior.geo.max_error_m:.3f} m")


def test_a_geo_map_that_is_too_coarse_stops_the_mission_before_it_moves(prior):
    import copy
    from sam_farm_inspection.farm_prior import GeoMap
    p2 = copy.deepcopy(prior)
    object.__setattr__(p2, "geo", GeoMap(
        prior.geo.origin_lat, prior.geo.origin_lon, prior.geo.dlat_dx, prior.geo.dlat_dz,
        prior.geo.dlon_dx, prior.geo.dlon_dz, max_error_m=5.0, basis="deliberately bad"))
    a = FarmMissionPlanner(p2).next()
    assert a.kind == REFUSED and "5.00 m" in a.reason


# ============================================================== plain geometry guards
def test_offsetting_a_rectangle_keeps_the_distance_on_the_EDGES_not_just_the_corners():
    """The radial-from-centroid shortcut passes a corner check and fails here, by the
    polygon's aspect ratio. A 40x10 rectangle offset by 5 m makes that obvious."""
    rect = [(0.0, 0.0), (40.0, 0.0), (40.0, 10.0), (0.0, 10.0)]
    out = offset_convex_polygon(rect, 5.0)
    for i in range(len(out)):
        for j in range(len(rect)):
            d = segment_distance(out[i], out[(i + 1) % len(out)],
                                 rect[j], rect[(j + 1) % len(rect)])
            assert d >= 5.0 - 1e-6, f"offset polygon comes within {d:.2f} m of the source"


def test_segment_distance_is_zero_for_crossing_segments():
    assert segment_distance((0, 0), (10, 10), (0, 10), (10, 0)) == 0.0
    assert segment_distance((0, 0), (1, 0), (0, 3), (1, 3)) == pytest.approx(3.0)
