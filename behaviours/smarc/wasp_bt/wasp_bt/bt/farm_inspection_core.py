"""The decision logic behind the `auv-farm-inspection` task and `A_SurfaceAndReport`.

PURE PYTHON ON PURPOSE — no py_trees, no rclpy, no message types. The py_trees behaviours in
`farm_inspection.py` are thin shells over these two classes, so everything that decides what
the vehicle does can be driven in a test on a laptop. The alternative, logic inside
`update()`, is only testable with a live tree and a live graph, which is how a branch ends up
shipped having never run (SETTLED §1c: a structural test cannot see whether a branch runs).

THE TWO RULES THIS FILE IS BUILT AROUND
=======================================

**ONE ACTION CLIENT.** The farm inspection does not get an action server or a client of its
own. It streams the planner's sub-goals through the *existing* `auv_depth_move_to` client
that the ordinary `auv-depth-move-to` task already uses. On 2026-08-15 two action servers on
one action name made the BT's goal response come back from the wrong one, rclpy discarded it
("there may be more than one action server"), and every mission stopped after waypoint 1 —
for a week. `send_goal` here is a callable handed in by the shell, and the shell is given the
tree's cached client.

**THE TREE OWNS THE PHASES, THE PLANNER OWNS THE GEOMETRY.** The planner node answers exactly
three things: here is the next sub-goal, this phase is done, or I refuse and here is why. It
never commands anything. This runner turns those answers into goals for the one client and
decides when to stop — which is the behaviour tree's job, and the reason the end-of-mission
surfacing belongs here rather than in the controller (SETTLED §3j).
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

# Answer kinds, mirrored from `sam_farm_inspection.farm_mission`. Deliberately DUPLICATED as
# strings rather than imported: wasp_bt must not depend on the perception package to build
# (they are separate colcon packages and the BT has to come up on a hull with no farm code
# installed at all). The wire is the contract; `test_farm_inspection_task.py` pins these
# against the planner's own constants so the two cannot drift silently.
GOAL = "goal"
PHASE_DONE = "phase_done"
MISSION_DONE = "mission_done"
REFUSED = "refused"
WAIT = "wait"

#: What the shell should do this tick.
SEND_GOAL = "send_goal"
RUNNING = "running"
SUCCESS = "success"
FAILURE = "failure"


class Step:
    """One tick's instruction to the shell, and the feedback line the operator sees."""

    __slots__ = ("action", "params", "feedback", "detail")

    def __init__(self, action: str, feedback: str, params: Optional[dict] = None,
                 detail: str = ""):
        self.action = action
        self.params = params
        self.feedback = feedback
        self.detail = detail

    def __repr__(self):                       # pragma: no cover - debugging aid
        return f"Step({self.action}, {self.feedback!r})"


class FarmInspectionRunner:
    """Drives one `auv-farm-inspection` task from start to finish.

    `link` is anything with `request(dict)` and `latest() -> Optional[dict]`; in flight it is
    a pair of ROS topics, and in tests it is a fake that answers immediately. Every request
    carries a sequence number and an answer with the wrong one is IGNORED rather than used:
    a stale answer to a question we are no longer asking is how a plan skips a leg.
    """

    def __init__(self,
                 goal_state: Callable[[], str],
                 link: Any,
                 now: Callable[[], float],
                 params: Optional[dict] = None,
                 log: Optional[Callable[[str, str], None]] = None):
        # The core RETURNS a goal to send and never sends one itself: exactly one place in
        # the process touches the action client, and it is the shell. A core that could send
        # would be a second caller of `send_goal` the moment anyone adds a retry.
        self.goal_state = goal_state          # "idle" | "running" | "done" | "failed"
        self.link = link
        self.now = now
        self.log = log or (lambda level, msg: None)
        p = params or {}
        self.rpm = float(p.get("rpm", 500.0))
        self.goal_timeout_s = float(p.get("goal_timeout_s", 900.0))
        #: How long to wait for the planner node before calling it dead. A component that
        #: cannot do its job must SAY so rather than leave the tree ticking forever
        #: (SETTLED §1b) — so this is a named refusal, not a silent stall.
        self.planner_timeout_s = float(p.get("planner_timeout_s", 20.0))

        self._seq = 0
        self._asked_at: Optional[float] = None
        self._awaiting_goal = False
        self._sent: Optional[dict] = None
        self.refusal: Optional[Tuple[str, str, str]] = None    # (phase, reason, response)
        self.finished_ok = False
        self.goals_sent = 0
        self.phase = "?"
        self.history: List[str] = []

    # ------------------------------------------------------------------ the tick
    def tick(self) -> Step:
        if self.refusal is not None:
            phase, reason, response = self.refusal
            return Step(FAILURE, f"REFUSED in {phase}: {reason}", detail=response)
        if self.finished_ok:
            return Step(SUCCESS, "farm inspection complete")

        state = self.goal_state()
        if self._awaiting_goal:
            if state == "done":
                self._awaiting_goal = False
                self._ask({"event": "reached", "name": (self._sent or {}).get("name")})
                return Step(RUNNING, f"{self.phase}: reached "
                                     f"{(self._sent or {}).get('name', 'a sub-goal')}")
            if state == "failed":
                # The action client failed. That is not the planner's problem to solve and it
                # is not something to retry blindly: the vehicle did not get where the plan
                # needs it, so the survey from here on would be of somewhere else.
                self._refuse(self.phase,
                             f"the move to {(self._sent or {}).get('name', '?')} failed at the "
                             f"action client",
                             "check the diving controller and the action server, then re-run "
                             "the inspection — do not treat the partial scan as a survey")
                return self.tick()
            return Step(RUNNING, f"{self.phase}: flying "
                                 f"{(self._sent or {}).get('name', 'a sub-goal')}")

        ans = self._answer()
        if ans is None:
            if self._asked_at is None:
                self._ask({"event": "start"})
                return Step(RUNNING, "asking the farm planner for the first sub-goal")
            waited = self.now() - self._asked_at
            if waited > self.planner_timeout_s:
                self._refuse(self.phase,
                             f"the farm planner did not answer for {waited:.0f} s "
                             f"(> {self.planner_timeout_s:.0f} s)",
                             "check that the farm planner node is running and that its prior "
                             "loaded; the inspection cannot be flown without it")
                return self.tick()
            return Step(RUNNING, f"waiting for the farm planner ({waited:.0f} s)")

        kind = ans.get("kind")
        self.phase = ans.get("phase", self.phase)
        if kind == GOAL:
            params = ans.get("params") or {}
            if not _looks_like_a_waypoint(params):
                self._refuse(self.phase,
                             "the planner answered with a sub-goal that is not a waypoint "
                             f"({sorted(params.get('waypoint', {}) or {})})",
                             "this is a planner defect — do not fly it")
                return self.tick()
            self._sent = {"name": params.get("name", "sub-goal"),
                          "why": ans.get("reason", "")}
            self._awaiting_goal = True
            self.goals_sent += 1
            self.history.append(f"{self.phase}:{self._sent['name']}")
            self.log("info", f"farm inspection {self.phase}: {self._sent['name']} — "
                             f"{ans.get('reason', '')}")
            return Step(SEND_GOAL, f"{self.phase}: {self._sent['name']}", params=params,
                        detail=ans.get("reason", ""))
        if kind == PHASE_DONE:
            self.log("info", f"farm inspection: {ans.get('reason', '')}")
            self._ask({"event": "next"})
            return Step(RUNNING, str(ans.get("reason", "phase complete")))
        if kind == MISSION_DONE:
            self.finished_ok = True
            return Step(SUCCESS, "farm inspection complete — all four phases flown")
        if kind == REFUSED:
            self._refuse(ans.get("phase", self.phase), str(ans.get("reason", "unstated")),
                         str(ans.get("response", "")))
            return self.tick()
        if kind == WAIT:
            self._ask({"event": "poll"})
            return Step(RUNNING, str(ans.get("reason", "waiting for the planner")))

        self._refuse(self.phase, f"the planner answered with an unknown kind {kind!r}",
                     "this is a planner defect — do not fly it")
        return self.tick()

    # ------------------------------------------------------------------ internals
    def _ask(self, payload: dict) -> None:
        self._seq += 1
        self._asked_at = self.now()
        out = dict(payload)
        out["seq"] = self._seq
        self.link.request(out)

    def _answer(self) -> Optional[dict]:
        ans = self.link.latest()
        if not isinstance(ans, dict):
            return None
        if ans.get("seq") != self._seq:
            # A stale answer is not an answer. Acting on the previous question's reply is
            # exactly how a mission skips a leg and nobody can say why afterwards.
            return None
        return ans

    def _refuse(self, phase: str, reason: str, response: str) -> None:
        self.refusal = (phase, reason, response)
        self.log("error", f"farm inspection REFUSED in {phase}: {reason} | {response}")


def _looks_like_a_waypoint(params: dict) -> bool:
    """The keys `ActionServerDiveSub.goal_callback` reads WITHOUT a default.

    A missing one raises inside the goal callback, which rclpy turns into a rejected goal
    with no explanation on the wire — a silent refusal, and the hardest kind to diagnose
    (SETTLED §1b). Checking here means the planner's defect is named by the vehicle instead.
    """
    wp = params.get("waypoint")
    if not isinstance(wp, dict):
        return False
    return all(k in wp for k in ("latitude", "longitude", "rpm", "target_depth", "tolerance"))


class SurfaceAndReportCore:
    """`A_SurfaceAndReport`'s decision logic — invariant 5b's missing behaviour-tree half.

    The controller half already exists (`sam_diving_controller.neutral_handoff`, unflown): it
    knows when SURFACING HAS FINISHED. The tree is the only thing that knows a MISSION IS
    OVER. This is that half, and it does three things:

      1. gate on the protective stop before commanding a surface;
      2. command the surface through the ONE action client, as an ordinary depth-0 waypoint
         at the vehicle's current position — no new writer, no new topic (invariant 12);
      3. wait for the CONTROLLER'S OWN word that it has let go, and report which kind of
         word it was.

    ON THE GATE, HONESTLY. Invariant 5b says surfacing under an overhang damages the hull,
    and this vehicle has no upward-looking sensor: the protective stop is fed by a
    FORWARD-looking sonar. So `stop_active` is a proxy, and a real overhead-clearance check
    does not exist on this hull. What the gate genuinely buys is the case that has actually
    happened — the vehicle is stopped against something and blowing the tank would drive it
    up along that something. Holding and reporting is right there. Anything stronger would
    be a claim the sensors do not support, and this file will not make one.

    ON THE CONFIRMATION. `neutral_handoff` releases on confirmed-surfaced-and-neutral OR on a
    long timeout, and it reports which. A timeout release is still a release (a controller
    that never lets go cannot be taken over) but it is NOT evidence the vehicle surfaced, and
    this class keeps the two apart all the way out to the mission record: `confirmed` is what
    MC may record as a completed recovery; `timeout` is reported as a release that was never
    confirmed. Same rule as SETTLED §3e — completion is the vehicle's word.
    """

    HOLDING = "holding"

    def __init__(self,
                 goal_state: Callable[[], str],
                 position: Callable[[], Optional[Tuple[float, float]]],
                 handoff: Callable[[], Optional[Dict[str, Any]]],
                 stop_active: Callable[[], bool],
                 now: Callable[[], float],
                 params: Optional[dict] = None,
                 log: Optional[Callable[[str, str], None]] = None):
        self.goal_state = goal_state
        self.position = position
        self.handoff = handoff
        self.stop_active = stop_active
        self.now = now
        self.log = log or (lambda level, msg: None)
        p = params or {}
        self.rpm = float(p.get("rpm", 500.0))
        self.tolerance_m = float(p.get("tolerance_m", 3.0))
        self.speed_ms = float(p.get("speed_ms", 0.5))
        #: How long to wait for the controller to say it has let go before reporting that it
        #: never did. Longer than `neutral_handoff`'s own max (600 ticks ~ 60 s at 10 Hz), so
        #: this fires only when the controller is not speaking at all.
        self.handoff_timeout_s = float(p.get("handoff_timeout_s", 120.0))
        self.goal_timeout_s = float(p.get("goal_timeout_s", 300.0))

        self._commanded = False
        self._commanded_at: Optional[float] = None
        self.outcome: Optional[str] = None    # confirmed | timeout | held | no_position
        self.detail = ""

    def tick(self) -> Step:
        if self.outcome is not None:
            ok = self.outcome == "confirmed"
            return Step(SUCCESS if ok else FAILURE,
                        f"surface and report: {self.outcome} — {self.detail}",
                        detail=self.detail)

        if not self._commanded:
            if self.stop_active():
                # Do not blow the tank into whatever the vehicle is stopped against. Hold and
                # say so: an operator can act on "held at depth, protective stop active" and
                # cannot act on a vehicle that quietly did nothing.
                self.outcome = "held"
                self.detail = ("the protective stop is active, so the end-of-mission surface "
                               "was NOT commanded (forward-looking proxy: this hull has no "
                               "overhead clearance sensor). Holding depth and reporting")
                self.log("error", "A_SurfaceAndReport: " + self.detail)
                return self.tick()
            pos = self.position()
            if pos is None:
                # No position means no waypoint. Refusing is right: a surface command to
                # (0, 0) would be a transit across the sea, not a surfacing.
                self.outcome = "no_position"
                self.detail = ("no position available, so the surfacing waypoint cannot be "
                               "placed. The vehicle has not been commanded to surface")
                self.log("error", "A_SurfaceAndReport: " + self.detail)
                return self.tick()
            self._commanded = True
            self._commanded_at = self.now()
            params = {
                "waypoint": {"latitude": pos[0], "longitude": pos[1], "target_depth": 0.0,
                             "rpm": self.rpm, "speed": self.speed_ms,
                             "tolerance": self.tolerance_m},
                "name": "mission_complete_surface",
                "timeout": self.goal_timeout_s,
            }
            self.log("info", "A_SurfaceAndReport: mission complete, commanding the surface "
                             "at the vehicle's own position")
            return Step(SEND_GOAL, "mission complete — surfacing", params=params)

        if self.goal_state() == "failed":
            # The surface command itself was rejected or errored. Say so: the vehicle is
            # still down, and a mission recorded as complete here would be a fiction.
            self.outcome = "goal_failed"
            self.detail = ("the surfacing waypoint was rejected or failed at the action "
                           "client, so the vehicle has NOT been commanded to the surface")
            self.log("error", "A_SurfaceAndReport: " + self.detail)
            return self.tick()

        h = self.handoff()
        if isinstance(h, dict) and h.get("released"):
            self.outcome = "confirmed" if h.get("confirmed") else "timeout"
            self.detail = str(h.get("reason", "the controller did not say why"))
            level = "info" if self.outcome == "confirmed" else "error"
            self.log(level, f"A_SurfaceAndReport: controller released the actuators "
                            f"({self.outcome}): {self.detail}")
            return self.tick()

        # `is None`, NOT `or`: a clock that starts at 0.0 makes `self._commanded_at or now()`
        # evaluate to now() every tick, so the wait never grows and the timeout never fires.
        # Found by the test below, which sets the clock to 0 at the command and 500 after.
        started = self._commanded_at if self._commanded_at is not None else self.now()
        waited = self.now() - started
        if waited > self.handoff_timeout_s:
            self.outcome = "timeout"
            self.detail = (f"the diving controller never reported letting go "
                           f"({waited:.0f} s). The vehicle may still be holding depth with "
                           f"the tank part full — this is NOT a confirmed surfacing")
            self.log("error", "A_SurfaceAndReport: " + self.detail)
            return self.tick()
        return Step(RUNNING, f"surfacing — waiting for the controller to let go "
                             f"({waited:.0f} s)")
