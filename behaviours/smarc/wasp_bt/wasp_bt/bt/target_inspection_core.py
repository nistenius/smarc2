"""The decision logic behind the adaptive close inspection — the CLOSE-OPS runner and the latch.

PURE PYTHON, exactly as `farm_inspection_core.py` is and for exactly the same reason: everything
that decides what the vehicle does has to be drivable in a test on a laptop. `target_inspection.py`
is a thin shell of py_trees behaviours over these classes.

THE FOUR RULES THIS FILE IS BUILT AROUND
========================================

**ONE ACTION CLIENT.** The inspection streams the planner's sub-goals through the *existing*
`auv_depth_move_to` client. Two servers on one action name is what stopped every mission at
waypoint 1 for a week (SETTLED §1c). `send_goal` here is a callable handed in by the shell.

**PREEMPT BY TREE PRIORITY, NEVER BY `cancel_goal` FROM ANOTHER BEHAVIOUR.** Measured
2026-09-09: `wasp_bt/bt/actions.py:322-534` lists `ActionClientState.CANCELLED` among
`A_ActionClient`'s FAILURE states, and a failure calls `clear_current_task()` — so a goal
cancelled from the side would DELETE THE INTERRUPTED LEG from the queue. The path that works is
the one py_trees gives for free: the inspection subtree sits as a HIGHER-PRIORITY SIBLING in the
task-handler Fallback, py_trees terminates the running `A_ActionClient` with `Status.INVALID`,
`smarc_action_base/bt_action_client_action.py:110-136` cancels the goal and calls `get_ready()`,
and the next tick re-sends the same goal from `get_current_task_params()`. This module therefore
never calls anything named `cancel`, and `test_target_inspection_task.py` asserts that on the
parsed syntax of both this file and `target_inspection.py`.

**THE LATCH LIVES ON THE TASK HANDLER, NEVER ON A NODE.** `_update_task_handler_tree` rebuilds
the whole subtree whenever an action server's heartbeat changes (SETTLED §3f0p), and a flag on a
behaviour is erased when it does. `DiversionLatch` is attached to the task-handler OBJECT, which
outlives every rebuild — the same mechanism `end_of_mission_core.get_or_create` uses.

**A DIVERSION IS NOT AN EMERGENCY.** Nothing here touches `emergency_flag`, `A_Chilling`,
`A_EmergencyParked`, the abort origins or `A_Abort`. It is a mission PHASE and is reported as
one, on an additive `adaptive_phase` word.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

# Answer kinds, mirrored from `sam_target_inspection.inspection_planner`. DUPLICATED as plain
# strings rather than imported, exactly as `farm_inspection_core` duplicates the farm planner's:
# wasp_bt must build on a hull with no perception package installed at all. The wire is the
# contract, and `test_target_inspection_task.py` pins these against the planner's own constants.
GOAL = "goal"
PHASE_DONE = "phase_done"
MISSION_DONE = "mission_done"
REFUSED = "refused"
WAIT = "wait"
SONAR_MODE = "sonar_mode"
BURST = "burst"

#: What the shell should do this tick.
SEND_GOAL = "send_goal"
SET_SONAR_MODE = "set_sonar_mode"
START_BURST = "start_burst"
RUNNING = "running"
SUCCESS = "success"
FAILURE = "failure"

#: The phase word that crosses the link (strategy §7.2). One word, five values, additive.
PHASES = ("scan", "divert", "inspect", "verify", "resume")


class Step:
    """One tick's instruction to the shell, and the feedback line the operator sees."""

    __slots__ = ("action", "params", "feedback", "detail", "mode", "station")

    def __init__(self, action: str, feedback: str, params: Optional[dict] = None,
                 detail: str = "", mode: str = "", station: Optional[int] = None):
        self.action = action
        self.params = params
        self.feedback = feedback
        self.detail = detail
        self.mode = mode
        self.station = station

    def __repr__(self):                       # pragma: no cover - debugging aid
        return f"Step({self.action}, {self.feedback!r})"


# --------------------------------------------------------------------------------------
# the latch
# --------------------------------------------------------------------------------------
class DiversionLatch:
    """Everything about the current diversion that must survive a subtree rebuild.

    ATTACHED TO THE TASK HANDLER, never to a behaviour (SETTLED §3f0p). `get_or_create` is the
    same pattern `end_of_mission_core` uses, and it is the reason this is a plain object with no
    py_trees or rclpy in sight.

    It holds: whether a candidate is pending, the diversion point P0 (pose + leg id + fraction),
    how many diversions this mission has spent, how many seconds the sanctioned diversion
    consumed, the phase word, and the verdict. Nothing else — and in particular NOT the
    emergency flag, which this feature may not touch.
    """

    ATTR = "_adaptive_diversion_latch"

    def __init__(self):
        self.candidate: Optional[dict] = None
        self.p0: Optional[dict] = None
        self.leg_id: Optional[int] = None
        self.active: bool = False
        self.diversions_used: int = 0
        self.consumed_s: float = 0.0
        self.phase: str = "scan"
        self.verdict: Optional[str] = None
        self.last_refusal: Optional[str] = None
        self.history: List[str] = []

    # -- the record -------------------------------------------------------------------
    def offer(self, candidate: dict, p0: dict, leg_id: Optional[int]) -> bool:
        """A detector has a candidate. Returns True if it was latched.

        Refuses a second offer while one is active: the tree is already flying a diversion and
        a candidate that arrives mid-inspection is a re-observation or the next one, not a
        reason to abandon this one.
        """
        if self.active:
            self.last_refusal = ("a diversion is already running; the new candidate is queued "
                                 "for the ledger, not acted on")
            return False
        self.candidate = dict(candidate)
        self.p0 = dict(p0)
        self.leg_id = leg_id
        self.active = True
        self.phase = "divert"
        self.verdict = None
        self.history.append(f"latched {candidate.get('id', '?')} at leg {leg_id}")
        return True

    def note_result(self, verdict: str, consumed_s: float) -> None:
        """The inspection is over; the RESUME has not happened yet.

        `active` deliberately stays True. The subtree that holds the diversion is a Sequence
        under a condition that reads `active`, so clearing it here would make the condition fail
        on the very next tick and the resume leg would never run — the vehicle would be left at
        the ring and the ordinary task tree would re-send the interrupted waypoint from there,
        which is precisely the hole in the swath this whole feature exists to avoid
        (strategy §8). `finish()` is what clears it, after the resume.
        """
        self.phase = "resume"
        self.verdict = verdict
        self.consumed_s += float(consumed_s)
        self.history.append(f"inspection ended: {verdict} after {consumed_s:.0f} s")

    def finish(self) -> None:
        """The resume is over. NOW the latch is cleared and the diversion is counted."""
        self.active = False
        self.candidate = None
        self.phase = "scan"
        self.diversions_used += 1
        self.history.append(f"resumed; diversions used {self.diversions_used}")

    def refuse(self, reason: str) -> None:
        """A candidate we did not go to. The reason is kept: "we did not go" is a different
        fact from "we went and it was nothing" (SETTLED §3e)."""
        self.active = False
        self.candidate = None
        self.phase = "scan"
        self.verdict = "not_inspected"
        self.last_refusal = reason
        self.history.append(f"refused: {reason}")

    def as_status(self) -> dict:
        """The additive fields that cross the link (strategy §7.2)."""
        return {"adaptive_phase": self.phase,
                "diversions_used": self.diversions_used,
                "diversion_consumed_s": round(self.consumed_s, 1),
                "last_verdict": self.verdict}


def get_or_create(task_handler) -> DiversionLatch:
    """The latch for this task handler, created on first use.

    On the HANDLER, because `_update_task_handler_tree` rebuilds subtrees on any heartbeat
    change and a flag on a node is erased by that rebuild (SETTLED §3f0p). The handler is the
    object that outlives it.
    """
    latch = getattr(task_handler, DiversionLatch.ATTR, None)
    if latch is None:
        latch = DiversionLatch()
        setattr(task_handler, DiversionLatch.ATTR, latch)
    return latch


# --------------------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------------------
class CloseInspectionRunner:
    """Drives one diversion through the ONE client, from the planner's answers.

    `link` is anything with `request(dict)` and `latest() -> Optional[dict]` — two ROS topics in
    flight, a fake in tests. Every request carries a sequence number and an answer with the wrong
    one is IGNORED: a stale answer to a question we are no longer asking is how a plan skips a
    leg (`farm_inspection_core._answer`'s rule, kept).

    THE SONAR MODE IS A SENSOR MODE, NOT AN ACTUATOR. It goes out as a plain `String` on
    `payload/sonar3d/set_mode` and the runner WAITS for the sonar's own announcement on
    `payload/sonar3d/mode` before declaring the mode set. Waiting is not politeness: the
    protective stop reads the same cloud and bounds itself by the announced range, so continuing
    before the announcement would fly the ring against a horizon nobody has confirmed changed.
    """

    def __init__(self,
                 goal_state: Callable[[], str],
                 link: Any,
                 now: Callable[[], float],
                 *,
                 sonar_mode: Optional[Callable[[], Optional[str]]] = None,
                 coverage: Optional[Callable[[int], Tuple[bool, str]]] = None,
                 params: Optional[dict] = None,
                 log: Optional[Callable[[str, str], None]] = None):
        self.goal_state = goal_state
        self.link = link
        self.now = now
        self.sonar_mode = sonar_mode or (lambda: None)
        self.coverage = coverage or (lambda station: (True, "no recorder; station not gated"))
        self.log = log or (lambda level, msg: None)
        p = params or {}
        self.planner_timeout_s = float(p.get("planner_timeout_s", 20.0))
        self.mode_timeout_s = float(p.get("mode_timeout_s", 10.0))
        self.burst_timeout_s = float(p.get("burst_timeout_s", 60.0))

        self._seq = 0
        self._asked_at: Optional[float] = None
        self._awaiting_goal = False
        self._sent: Optional[dict] = None
        self._mode_wanted: Optional[str] = None
        self._mode_at: Optional[float] = None
        self._burst_station: Optional[int] = None
        self._burst_at: Optional[float] = None
        self.started_at: Optional[float] = None
        self.refusal: Optional[Tuple[str, str, str]] = None
        self.finished_ok = False
        self.goals_sent = 0
        self.stations_done = 0
        self.phase = "divert"
        self.verdict_hint: Optional[str] = None
        self.history: List[str] = []

    # ------------------------------------------------------------------ the tick
    def tick(self) -> Step:
        if self.started_at is None:
            self.started_at = self.now()
        if self.refusal is not None:
            phase, reason, response = self.refusal
            return Step(FAILURE, f"REFUSED in {phase}: {reason}", detail=response)
        if self.finished_ok:
            return Step(SUCCESS, "close inspection complete")

        # 1. a goal is in flight
        if self._awaiting_goal:
            state = self.goal_state()
            if state == "done":
                self._awaiting_goal = False
                self._ask({"event": "reached", "name": (self._sent or {}).get("name")})
                return Step(RUNNING, f"{self.phase}: reached "
                                     f"{(self._sent or {}).get('name', 'a sub-goal')}")
            if state == "failed":
                self._refuse(self.phase,
                             f"the move to {(self._sent or {}).get('name', '?')} failed at the "
                             f"action client",
                             "the vehicle is not where the plan needs it, so the rest of the "
                             "ring would be of somewhere else; the diversion ends and the "
                             "mission resumes")
                return self.tick()
            return Step(RUNNING, f"{self.phase}: flying "
                                 f"{(self._sent or {}).get('name', 'a sub-goal')}")

        # 2. a sonar mode change is in flight — WAIT for the sonar's own announcement
        if self._mode_wanted is not None:
            announced = self.sonar_mode()
            if announced and announced.split("|")[0].strip().lower() == self._mode_wanted.lower():
                self.log("info", f"sonar announced {announced}")
                self._mode_wanted = None
                self._mode_at = None
                self._ask({"event": "next"})
                return Step(RUNNING, f"{self.phase}: sonar mode {announced}")
            waited = self.now() - (self._mode_at if self._mode_at is not None else self.now())
            if waited > self.mode_timeout_s:
                self._refuse(self.phase,
                             f"the sonar did not announce the {self._mode_wanted} mode within "
                             f"{self.mode_timeout_s:.0f} s (last heard: {announced!r})",
                             "the protective stop bounds itself by the ANNOUNCED range, so "
                             "orbiting on an unconfirmed mode would fly the ring against a "
                             "horizon nobody has confirmed; the diversion ends and the mission "
                             "resumes")
                return self.tick()
            return Step(RUNNING, f"{self.phase}: waiting for the sonar to announce "
                                 f"{self._mode_wanted} ({waited:.0f} s)")

        # 3. a capture burst is open — the STATION IS DONE ON A COUNT, not a stopwatch
        if self._burst_station is not None:
            done, why = self.coverage(self._burst_station)
            if done:
                self.stations_done += 1
                st = self._burst_station
                self._burst_station = None
                self._burst_at = None
                self._ask({"event": "next"})
                return Step(RUNNING, f"{self.phase}: station {st} done — {why}")
            waited = self.now() - (self._burst_at if self._burst_at is not None else self.now())
            if waited > self.burst_timeout_s:
                # NOT a refusal. A station that could not fill its frame quota is a fact about
                # the water, and the ring goes on; the ledger's rung 2 counts stations that
                # ACCEPTED, so an unfilled one simply does not count towards it.
                st = self._burst_station
                self._burst_station = None
                self._burst_at = None
                self.log("error", f"station {st} produced too little in "
                                  f"{self.burst_timeout_s:.0f} s: {why}")
                self.history.append(f"station {st}: not accepted ({why})")
                self._ask({"event": "next"})
                return Step(RUNNING, f"{self.phase}: station {st} NOT accepted — {why}")
            return Step(RUNNING, f"{self.phase}: capturing at station {self._burst_station} "
                                 f"({waited:.0f} s) — {why}")

        # 4. otherwise, read the planner
        ans = self._answer()
        if ans is None:
            if self._asked_at is None:
                self._ask({"event": "start"})
                return Step(RUNNING, "asking the inspection planner for the approach")
            waited = self.now() - self._asked_at
            if waited > self.planner_timeout_s:
                self._refuse(self.phase,
                             f"the inspection planner did not answer for {waited:.0f} s "
                             f"(> {self.planner_timeout_s:.0f} s)",
                             "check that the inspection planner node is running; the diversion "
                             "cannot be flown without it and the mission resumes unchanged")
                return self.tick()
            return Step(RUNNING, f"waiting for the inspection planner ({waited:.0f} s)")

        kind = ans.get("kind")
        self.phase = _phase_word(ans.get("phase", self.phase))
        if kind == GOAL:
            params = ans.get("params") or {}
            if not _looks_like_a_waypoint(params):
                self._refuse(self.phase,
                             "the planner answered with a sub-goal that is not a waypoint "
                             f"({sorted((params.get('waypoint') or {}))})",
                             "this is a planner defect — do not fly it")
                return self.tick()
            self._sent = {"name": params.get("name", "sub-goal"), "why": ans.get("reason", "")}
            self._awaiting_goal = True
            self.goals_sent += 1
            self.history.append(f"{self.phase}:{self._sent['name']}")
            self.log("info", f"close inspection {self.phase}: {self._sent['name']} — "
                             f"{ans.get('reason', '')}")
            return Step(SEND_GOAL, f"{self.phase}: {self._sent['name']}", params=params,
                        detail=ans.get("reason", ""))
        if kind == SONAR_MODE:
            self._mode_wanted = str(ans.get("mode", ""))
            self._mode_at = self.now()
            self.history.append(f"{self.phase}:mode={self._mode_wanted}")
            return Step(SET_SONAR_MODE, f"{self.phase}: commanding sonar mode "
                                        f"{self._mode_wanted}", mode=self._mode_wanted,
                        detail=ans.get("reason", ""))
        if kind == BURST:
            self._burst_station = int(ans.get("station", 0))
            self._burst_at = self.now()
            return Step(START_BURST, f"{self.phase}: capture burst at station "
                                     f"{self._burst_station}",
                        station=self._burst_station, detail=ans.get("reason", ""))
        if kind == PHASE_DONE:
            hint = ans.get("verdict_hint")
            if hint:
                self.verdict_hint = str(hint)
            self.log("info", f"close inspection: {ans.get('reason', '')}")
            self._ask({"event": "next"})
            return Step(RUNNING, str(ans.get("reason", "phase complete")))
        if kind == MISSION_DONE:
            self.finished_ok = True
            return Step(SUCCESS, str(ans.get("reason", "close inspection complete")))
        if kind == REFUSED:
            self._refuse(_phase_word(ans.get("phase", self.phase)),
                         str(ans.get("reason", "unstated")), str(ans.get("response", "")))
            return self.tick()
        if kind == WAIT:
            self._ask({"event": "poll"})
            return Step(RUNNING, str(ans.get("reason", "waiting for the planner")))

        self._refuse(self.phase, f"the planner answered with an unknown kind {kind!r}",
                     "this is a planner defect — do not fly it")
        return self.tick()

    # ------------------------------------------------------------------ accounting
    def consumed_s(self) -> float:
        """What the mission timeout is to be extended by: what the diversion ACTUALLY used."""
        if self.started_at is None:
            return 0.0
        return max(0.0, self.now() - self.started_at)

    def verdict(self) -> str:
        """The onboard word for this diversion. NEVER `confirmed`: that rung needs a model, and
        a model is built at the base station in RECOVER (strategy §7.1 rung 3)."""
        if self.refusal is not None:
            return "not_inspected"
        if self.verdict_hint:
            return self.verdict_hint
        if self.finished_ok:
            return "provisional"
        return "inconclusive"

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
            return None
        return ans

    def _refuse(self, phase: str, reason: str, response: str) -> None:
        self.refusal = (phase, reason, response)
        self.log("error", f"close inspection REFUSED in {phase}: {reason} | {response}")


def _phase_word(planner_phase: str) -> str:
    """Map the planner's own phase names onto the five words that cross the link (§7.2).

    Two vocabularies on purpose: the planner's phases are about GEOMETRY (approach, close_ops,
    verify, resume) and the link's word is about what an operator needs to know is happening.
    Mapping here rather than renaming the planner's keeps the acoustic vocabulary fixed while
    the geometry is free to grow more phases.
    """
    return {"approach": "divert", "divert": "divert", "close_ops": "inspect",
            "verify": "verify", "resume": "resume", "done": "scan"}.get(planner_phase, "divert")


def _looks_like_a_waypoint(params: dict) -> bool:
    """The keys `ActionServerDiveSub.goal_callback` reads WITHOUT a default.

    A missing one raises inside the goal callback, which rclpy turns into a rejected goal with
    no explanation on the wire — a silent refusal, and the hardest kind to diagnose
    (SETTLED §1b). Same check as `farm_inspection_core`, deliberately duplicated with it for the
    same build-independence reason the answer kinds are.
    """
    wp = params.get("waypoint")
    if not isinstance(wp, dict):
        return False
    return all(k in wp for k in ("latitude", "longitude", "rpm", "target_depth", "tolerance"))


# --------------------------------------------------------------------------------------
# the resume
# --------------------------------------------------------------------------------------
class ResumeCore:
    """Fly back to P0 on the interrupted leg's heading, then hand the leg back to the task tree.

    THE POINT OF THIS CLASS IS THE ONE FACT IT ENFORCES: the vehicle returns to the DIVERSION
    POINT, not to the next waypoint. The remaining swath from P0 to its waypoint is otherwise a
    hole in the coverage nobody sees until Debrief (strategy §8). The lead-in and P0 both come
    from the latch, which recorded them at the moment of the diversion — not recomputed here,
    because a recomputed P0 is a P0 that moved while the vehicle was away.
    """

    def __init__(self, goal_state: Callable[[], str], latch: DiversionLatch,
                 now: Callable[[], float],
                 params: Optional[dict] = None,
                 log: Optional[Callable[[str, str], None]] = None):
        self.goal_state = goal_state
        self.latch = latch
        self.now = now
        self.log = log or (lambda level, msg: None)
        p = params or {}
        self.rpm = float(p.get("rpm", 500.0))
        self.tolerance_m = float(p.get("tolerance_m", 2.0))
        self.goal_timeout_s = float(p.get("goal_timeout_s", 300.0))
        self._sent = False
        self.done = False
        self.failed_reason: Optional[str] = None

    def tick(self) -> Step:
        if self.failed_reason:
            return Step(FAILURE, f"resume failed: {self.failed_reason}")
        if self.done:
            return Step(SUCCESS, "back at the diversion point; the interrupted leg resumes")
        p0 = self.latch.p0
        if not p0:
            self.failed_reason = ("no diversion point was recorded, so there is nowhere to "
                                  "resume to; the task tree re-sends the interrupted leg from "
                                  "wherever the vehicle is and the swath has a hole")
            self.log("error", "A_ResumeAtDiversionPoint: " + self.failed_reason)
            return self.tick()
        if not self._sent:
            self._sent = True
            params = {
                "waypoint": {"latitude": p0["lat"], "longitude": p0["lon"],
                             "target_depth": p0.get("depth_m", 0.0), "rpm": self.rpm,
                             "tolerance": self.tolerance_m,
                             "heading": p0.get("leg_heading_deg", 0.0)},
                "name": "resume_at_diversion_point",
                "timeout": self.goal_timeout_s,
            }
            self.log("info", f"resuming at the diversion point on leg {self.latch.leg_id}")
            return Step(SEND_GOAL, "returning to the diversion point", params=params)
        state = self.goal_state()
        if state == "done":
            self.done = True
            return self.tick()
        if state == "failed":
            self.failed_reason = "the return goal failed at the action client"
            return self.tick()
        return Step(RUNNING, "returning to the diversion point")
