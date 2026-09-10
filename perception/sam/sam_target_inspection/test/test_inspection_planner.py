"""The inspection planner — letter B, strategy §5.2, §5.3, §3 (refusals), §8 (resume).

The properties:

  * THE WHOLE SEQUENCE IS WALKED, in order, and the resume goals are at P0 — not at the next
    waypoint. The remaining swath from P0 to its waypoint is otherwise a hole in the coverage
    nobody sees until Debrief (strategy §8);
  * EVERY REFUSAL NAMES ITS REASON (rule 12) and the five the strategy lists are each driven;
  * THE RING RADIUS FLOOR CANNOT BE UNDERCUT BY A MISSION. A number that disables a safety
    layer may not become a default (invariant 8);
  * THE PLANNER HOLDS NO CLIENT AND WRITES NO ACTUATOR (ADR-004, invariant 12) — checked
    structurally, on the module's parsed syntax rather than on its prose;
  * THE BUDGET IS ACCOUNTED, and what the mission timeout is extended by is the time the
    diversion actually CONSUMED, not the budget it was allowed (SETTLED §3p's rule).

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_inspection_planner.py
"""
import ast
import math
import pathlib

import pytest

from sam_target_inspection import inspection_planner as P
from sam_target_inspection.inspection_planner import (BURST, DiversionPoint, GOAL, MISSION_DONE,
                                                      InspectionPlanner, PHASE_DONE, Policy,
                                                      PlannerRefusal, REFUSED, SONAR_MODE,
                                                      TargetSite, VehicleState, WAIT)

SITE = TargetSite(lat=58.82150, lon=17.63480, sigma_m=1.5, seabed_depth_m=8.0, id="C1")
P0 = DiversionPoint(lat=58.82153, lon=17.63515, leg_id=2, leg_fraction=0.4,
                    leg_heading_deg=270.0, depth_m=2.0)
OK_VEHICLE = VehicleState(lat=58.82153, lon=17.63515, estimator_age_s=0.5,
                          remaining_mission_s=3000.0, diversions_used=0)


def _planner(policy=None, vehicle=None, **kw):
    return InspectionPlanner(policy or Policy.from_dict({"n_stations": 4}), SITE, P0,
                             vehicle or OK_VEHICLE, **kw)


def _walk(pl, max_steps=200, clock=None):
    """Drive the planner exactly as the behaviour tree does: start, then reached, reached, ..."""
    out = []
    ev = {"event": "start", "seq": 1}
    for i in range(max_steps):
        a = pl.what_next(ev, now=None if clock is None else clock(i))
        out.append(a)
        if a["kind"] in (MISSION_DONE, REFUSED):
            break
        ev = {"event": "reached", "seq": i + 2}
    return out


# ------------------------------------------------------------------ the sequence
def test_the_sequence_is_approach_mode_ring_mode_surface_leadin_p0():
    steps = _walk(_planner())
    kinds = [s["kind"] for s in steps]
    names = [(s.get("params") or {}).get("name") or s.get("mode") for s in steps]
    assert kinds[0] == GOAL and names[0] == "approach_abeam"
    assert kinds[1] == SONAR_MODE and names[1] == "inspection"
    assert names[-4:-1] == ["surface_and_report", "resume_lead_in", "resume_at_diversion_point"]
    assert kinds[-1] == MISSION_DONE
    # the navigation mode is commanded back BEFORE the vehicle moves at scan speed again
    i_nav = names.index("navigation")
    assert i_nav < names.index("resume_lead_in")


def test_every_station_gets_a_goal_and_a_burst_in_that_order():
    pol = Policy.from_dict({"n_stations": 4, "altitudes_m": [2.0, 3.0]})
    steps = _walk(_planner(pol))
    stations = [s for s in steps if s["kind"] == GOAL and "station" in s]
    bursts = [s for s in steps if s["kind"] == BURST]
    assert len(stations) == 8 and len(bursts) == 8
    order = [s["kind"] for s in steps if s["kind"] in (GOAL, BURST) and "station" in s]
    assert order == [GOAL, BURST] * 8


def test_every_station_faces_the_candidate_and_sits_on_the_ring():
    pol = Policy.from_dict({"n_stations": 12})
    ring = P.build_ring(SITE, pol)
    for st in ring:
        d = P.distance_m(SITE.lat, SITE.lon, st.lat, st.lon)
        assert d == pytest.approx(pol.ring_radius_m, abs=0.05)
        b = P.bearing_deg(st.lat, st.lon, SITE.lat, SITE.lon)
        assert abs(((b - st.heading_deg + 180) % 360) - 180) < 1.0


def test_the_two_rings_are_offset_by_half_a_step_so_the_bearings_differ():
    """Two rings at the same bearings give the bundle adjustment two views along the same line
    of sight and no extra convergence (strategy §6.2 asks for ≥ 10 distinct bearings)."""
    pol = Policy.from_dict({"n_stations": 12, "altitudes_m": [2.0, 3.0]})
    ring = P.build_ring(SITE, pol)
    b0 = sorted(round(s.bearing_from_target_deg, 3) for s in ring if s.ring == 0)
    b1 = sorted(round(s.bearing_from_target_deg, 3) for s in ring if s.ring == 1)
    assert set(b0) & set(b1) == set()
    assert len(set(b0) | set(b1)) == 24


def test_the_two_rings_are_at_the_two_configured_altitudes():
    pol = Policy.from_dict({"n_stations": 4, "altitudes_m": [2.0, 3.0]})
    ring = P.build_ring(SITE, pol)
    d0 = {round(s.depth_m, 3) for s in ring if s.ring == 0}
    d1 = {round(s.depth_m, 3) for s in ring if s.ring == 1}
    assert d0 == {round(SITE.seabed_depth_m - 2.0, 3)}
    assert d1 == {round(SITE.seabed_depth_m - 3.0, 3)}


def test_the_approach_is_on_the_side_the_vehicle_is_already_on():
    lat, lon, abeam = P.approach_point(SITE, P0, Policy())
    b_p0 = P.bearing_deg(SITE.lat, SITE.lon, P0.lat, P0.lon)
    b_a = P.bearing_deg(SITE.lat, SITE.lon, lat, lon)
    assert abs(((b_a - b_p0 + 180) % 360) - 180) < 1.0
    assert abeam == pytest.approx(Policy().ring_radius_m)


def test_every_goal_carries_the_five_keys_the_dive_server_reads_without_a_default():
    """A missing one raises inside the goal callback and rclpy turns that into a rejected goal
    with no explanation on the wire — a silent refusal (SETTLED §1b)."""
    for s in _walk(_planner()):
        if s["kind"] != GOAL:
            continue
        wp = s["params"]["waypoint"]
        for k in ("latitude", "longitude", "rpm", "target_depth", "tolerance"):
            assert k in wp, f"{s['params']['name']} is missing {k}"


def test_the_surface_goal_is_at_depth_zero_and_only_when_the_policy_says_so():
    names = [(s.get("params") or {}).get("name") for s in _walk(_planner())]
    assert "surface_and_report" in names
    pol = Policy.from_dict({"n_stations": 4, "surface_between": False})
    names2 = [(s.get("params") or {}).get("name") for s in _walk(_planner(pol))]
    assert "surface_and_report" not in names2
    surf = next(s for s in _walk(_planner())
                if (s.get("params") or {}).get("name") == "surface_and_report")
    assert surf["params"]["waypoint"]["target_depth"] == 0.0


# ------------------------------------------------------------------ the resume
def test_the_resume_goes_to_p0_and_not_to_the_next_waypoint():
    steps = _walk(_planner())
    last = steps[-2]
    assert last.get("resume") is True and last["leg_id"] == P0.leg_id
    wp = last["params"]["waypoint"]
    assert wp["latitude"] == pytest.approx(P0.lat)
    assert wp["longitude"] == pytest.approx(P0.lon)
    assert wp["target_depth"] == pytest.approx(P0.depth_m)


def test_the_lead_in_is_behind_p0_along_the_legs_own_heading():
    steps = _walk(_planner())
    lead = next(s for s in steps if (s.get("params") or {}).get("name") == "resume_lead_in")
    wp = lead["params"]["waypoint"]
    d = P.distance_m(wp["latitude"], wp["longitude"], P0.lat, P0.lon)
    assert d == pytest.approx(Policy().lead_in_m, abs=0.5)
    b = P.bearing_deg(wp["latitude"], wp["longitude"], P0.lat, P0.lon)
    assert abs(((b - P0.leg_heading_deg + 180) % 360) - 180) < 1.0
    assert wp["heading"] == pytest.approx(P0.leg_heading_deg)


def test_the_resume_goals_carry_the_legs_heading_so_the_swath_re_enters_straight():
    steps = _walk(_planner())
    for s in steps[-3:-1]:
        assert s["params"]["waypoint"]["heading"] == pytest.approx(P0.leg_heading_deg)


# ------------------------------------------------------------------ the refusals
def test_a_dead_estimator_refuses_and_names_it():
    v = VehicleState(lat=58.8, lon=17.6, estimator_age_s=30.0, remaining_mission_s=3000.0)
    a = _walk(_planner(vehicle=v))[0]
    assert a["kind"] == REFUSED and "estimate" in a["reason"] and a["response"]


def test_no_position_at_all_refuses_and_names_it():
    v = VehicleState(lat=None, lon=None, estimator_age_s=0.1, remaining_mission_s=3000.0)
    a = _walk(_planner(vehicle=v))[0]
    assert a["kind"] == REFUSED and "no position estimate" in a["reason"]


def test_a_candidate_beyond_the_distance_limit_refuses_with_the_distance():
    far = TargetSite(lat=58.8300, lon=17.6348, sigma_m=1.5, seabed_depth_m=8.0)
    pl = InspectionPlanner(Policy.from_dict({"n_stations": 4}), far, P0, OK_VEHICLE)
    a = _walk(pl)[0]
    assert a["kind"] == REFUSED and "diversion limit" in a["reason"]
    assert "m away" in a["reason"]


def test_a_spent_diversion_budget_refuses_and_offers_the_follow_up():
    v = VehicleState(lat=58.82153, lon=17.63515, estimator_age_s=0.5,
                     remaining_mission_s=3000.0, diversions_used=1)
    a = _walk(_planner(vehicle=v))[0]
    assert a["kind"] == REFUSED and "diversion budget" in a["reason"]
    assert "follow-up" in a["response"]


def test_a_duplicate_refuses_and_names_the_target_it_duplicates():
    pl = _planner(duplicate_of="T1")
    a = _walk(pl)[0]
    assert a["kind"] == REFUSED and "T1" in a["reason"]


def test_a_mission_timeout_that_cannot_fit_the_inspection_refuses_before_anything_moves():
    """Refused rather than started and abandoned: the mission timeout is extended only by a
    diversion that actually happened."""
    v = VehicleState(lat=58.82153, lon=17.63515, estimator_age_s=0.5, remaining_mission_s=300.0)
    a = _walk(_planner(vehicle=v))[0]
    assert a["kind"] == REFUSED
    assert "s left" in a["reason"] and "to get back" in a["reason"]


def test_the_timeout_check_counts_the_way_HOME_and_not_only_the_inspection():
    """A mission with more time left than the inspection budget but less than the budget plus
    the return leg must still refuse. Otherwise the vehicle starts a diversion it can finish
    and then cannot get back to the line it left."""
    pl = _planner()
    need_return = pl._return_estimate_s()
    assert need_return > 60.0, "the surfacing and the transit home are not free"
    budget = Policy.from_dict({"n_stations": 4}).inspection_budget_s
    v = VehicleState(lat=58.82153, lon=17.63515, estimator_age_s=0.5,
                     remaining_mission_s=budget + 0.5 * need_return)
    a = _walk(_planner(vehicle=v))[0]
    assert a["kind"] == REFUSED and "to get back" in a["reason"]
    # ... and with room for both it proceeds
    v2 = VehicleState(lat=58.82153, lon=17.63515, estimator_age_s=0.5,
                      remaining_mission_s=budget + 2.0 * need_return)
    assert _walk(_planner(vehicle=v2))[0]["kind"] == GOAL


def test_a_refusal_is_sticky_and_never_becomes_a_goal_later():
    pl = _planner(vehicle=VehicleState(None, None, 0.1, 3000.0))
    for _ in range(5):
        assert pl.what_next({"event": "reached", "seq": 9})["kind"] == REFUSED


def test_every_refusal_carries_a_response_saying_what_happens_next():
    cases = [
        VehicleState(None, None, 0.1, 3000.0),
        VehicleState(58.82153, 17.63515, 0.5, 300.0),
        VehicleState(58.82153, 17.63515, 0.5, 3000.0, diversions_used=5),
    ]
    for v in cases:
        a = _walk(_planner(vehicle=v))[0]
        assert a["kind"] == REFUSED
        assert a["response"], f"a refusal with no response: {a['reason']}"


# ------------------------------------------------------------------ safety floors
def test_a_ring_radius_below_the_modelled_floor_is_refused_at_policy_load():
    with pytest.raises(PlannerRefusal) as e:
        Policy.from_dict({"ring_radius_m": 1.0})
    assert "protective stop" in str(e.value)
    assert "invariant 8" in str(e.value)


def test_a_two_station_ring_is_refused():
    with pytest.raises(PlannerRefusal):
        Policy.from_dict({"n_stations": 2})


def test_a_mission_may_widen_the_ring_but_not_narrow_it():
    assert Policy.from_dict({"ring_radius_m": 6.0}).ring_radius_m == 6.0


# ------------------------------------------------------------------ the budget
def test_running_out_of_budget_skips_to_the_resume_and_calls_it_a_planned_end():
    pol = Policy.from_dict({"n_stations": 12, "inspection_budget_s": 30.0})
    pl = _planner(pol)
    steps = _walk(pl, clock=lambda i: i * 20.0)
    kinds = [s["kind"] for s in steps]
    assert PHASE_DONE in kinds
    done = next(s for s in steps if s["kind"] == PHASE_DONE)
    assert done["verdict_hint"] == "inconclusive"
    assert "not an emergency" in done["reason"]
    names = [(s.get("params") or {}).get("name") for s in steps]
    assert "resume_at_diversion_point" in names, \
        "a spent budget must still fly the resume legs; the vehicle does not stop where it is"


def test_the_budget_report_extends_the_timeout_by_what_was_consumed_not_by_the_budget():
    pl = _planner()
    _walk(pl, clock=lambda i: i * 3.0)
    rep = pl.budget_report()
    assert rep["extend_mission_timeout_by_s"] == rep["consumed_s"]
    assert rep["consumed_s"] < rep["budget_s"]
    assert rep["stations_planned"] == 8 and rep["stations_reached"] == 8


def test_stations_reached_is_a_count_and_not_a_zero_based_index():
    """Off by one here would report 11 of 12 on a complete ring — which is exactly the coverage
    number the verdict ladder's rung 2 thresholds on."""
    pol = Policy.from_dict({"n_stations": 6, "altitudes_m": [2.0]})
    pl = _planner(pol)
    _walk(pl)
    assert pl.budget_report() == pytest.approx(pl.budget_report())
    assert pl.budget_report()["stations_reached"] == 6


def test_every_answer_echoes_the_sequence_number_it_was_asked_with():
    """A stale answer to a question we are no longer asking is how a plan skips a leg."""
    pl = _planner()
    a = pl.what_next({"event": "start", "seq": 41})
    assert a["seq"] == 41
    b = pl.what_next({"event": "reached", "seq": 42})
    assert b["seq"] == 42


def test_every_answer_carries_the_phase_and_the_target_id():
    for s in _walk(_planner()):
        assert s["phase"] in ("approach", "close_ops", "verify", "resume", "done", "divert")
        assert s["target_id"] == "C1"


# ------------------------------------------------------------------ the sonar mode
def test_the_sonar_mode_is_commanded_at_close_ops_entry_and_reverted_at_its_exit():
    modes = [s for s in _walk(_planner()) if s["kind"] == SONAR_MODE]
    assert [m["mode"] for m in modes] == ["inspection", "navigation"]
    assert "mission knows when it is inspecting" in modes[0]["reason"]
    assert "protective stop" in modes[0]["reason"], \
        "the mode change moves the protective stop's horizon and the reason must say so"


# ------------------------------------------------------------------ structure
def _module_source():
    return pathlib.Path(P.__file__).read_text()


def test_the_planner_holds_no_action_client_and_writes_no_actuator():
    """ADR-004, invariant 12 — checked on the parsed syntax, with comments and docstrings gone,
    so the test cannot pass by matching prose."""
    tree = ast.parse(_module_source())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            for a in node.names:
                names.add((mod + "." + a.name).strip("."))
    forbidden = {"ActionClient", "ActionServer", "send_goal", "cancel_goal", "create_publisher",
                 "create_subscription", "rclpy", "Node"}
    assert not (names & forbidden), f"the planner reaches for {sorted(names & forbidden)}"


def test_the_planner_imports_nothing_from_ros_or_numpy():
    """Pure python and no numpy: the whole point is that a diversion can be walked through
    deterministically in a test on a laptop."""
    tree = ast.parse(_module_source())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods <= {"__future__", "math", "dataclasses", "typing"}, sorted(mods)
