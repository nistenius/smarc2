#!/usr/bin/env python3
"""ONE WRITER, TWO SERVERS — the arbiter option M1 needs. Strategy §5.3, ADR-004 invariant 12.

WHAT THIS IS FOR. `diving_node` runs ONE controller family per bringup (`entrypoints.py`): the
PID/blend waypoint server on `auv_depth_move_to` and the MPC trajectory server on
`auv_trajectory_tracking` have never shared a process. The inspection orbit (strategy §5.3, M1)
wants both — ordinary legs through the waypoint server, the turbo-turn ring through the MPC —
over ONE `DivePub`. That is a one-writer question, and an arbiter is the answer: not two nodes,
not a mutex nobody can see, but a named object that REFUSES the second goal and says whose goal
it is refusing for.

PURE PYTHON, NO ROS. Everything that decides whether a goal may be accepted is here and is
driven exhaustively in a test; the servers' `goal_callback`s ask it and do nothing else.

THE THREE THINGS IT MUST NEVER DO, each one a rule this project already paid for:

1. **It must never let both servers hold an active goal.** That is invariant 12, and the reason
   is not tidiness: two writers on one actuator path is how a controller ends up fighting itself
   at 10 Hz with no log line saying so.
2. **It must never refuse silently.** A refusal names the holder, the goal, and how long it has
   held — because "goal rejected" with no explanation on the wire is the hardest kind of failure
   to diagnose (SETTLED §1b), and rclpy turns an exception in a goal callback into exactly that.
3. **It must never release a hold nobody asked it to release.** A timeout that quietly frees the
   writer would produce the two-writer state it exists to prevent, at the worst possible moment.
   `stale_holder()` REPORTS a suspiciously long hold; it does not act on it.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


class ArbiterRefusal(RuntimeError):
    """Raised only for programming errors (an unknown server name). A goal being refused is a
    RETURN VALUE with a sentence, never an exception: an exception inside a goal callback is
    what rclpy turns into a silent rejection."""


@dataclass(frozen=True)
class Hold:
    server: str
    goal_id: str
    since: float
    detail: str = ""


@dataclass
class AuditEntry:
    t: float
    server: str
    goal_id: str
    outcome: str          # accepted | refused | released
    reason: str = ""


class OneWriterArbiter:
    """Exactly one of the registered servers may hold a goal at a time.

    `servers` is the closed set of names. Closed on purpose: a typo'd name would otherwise
    create a third writer nobody registered, which is the failure mode inverted.
    """

    def __init__(self, servers: Tuple[str, ...], *, now: Optional[Callable[[], float]] = None):
        if len(set(servers)) < 2:
            raise ArbiterRefusal(
                "an arbiter over fewer than two servers arbitrates nothing; if there is only one "
                "writer there is no question to answer and this class should not be in the graph")
        self.servers = tuple(servers)
        self.now = now or (lambda: 0.0)
        self._hold: Optional[Hold] = None
        self.audit: List[AuditEntry] = []

    # ---------------------------------------------------------------- state
    @property
    def holder(self) -> Optional[str]:
        return self._hold.server if self._hold else None

    @property
    def hold(self) -> Optional[Hold]:
        return self._hold

    def _check(self, server: str) -> None:
        if server not in self.servers:
            raise ArbiterRefusal(
                f"{server!r} is not one of this arbiter's servers ({', '.join(self.servers)}). "
                f"An unregistered name would be a third writer nobody arbitrates.")

    # ---------------------------------------------------------------- the decision
    def request(self, server: str, goal_id: str, detail: str = "") -> Tuple[bool, str]:
        """(accepted, sentence). THE ONE PLACE a goal is admitted or refused.

        A server that already holds the writer may replace its OWN goal — that is an ordinary
        re-send, and the alternative (refusing it) would make a preempted-and-re-sent waypoint
        unflyable, which is the behaviour the whole diversion design depends on.
        """
        self._check(server)
        t = self.now()
        if self._hold is None:
            self._hold = Hold(server, str(goal_id), t, detail)
            self.audit.append(AuditEntry(t, server, str(goal_id), "accepted"))
            return True, f"{server} holds the writer (goal {goal_id})"
        if self._hold.server == server:
            prev = self._hold.goal_id
            self._hold = Hold(server, str(goal_id), t, detail)
            self.audit.append(AuditEntry(t, server, str(goal_id), "accepted",
                                         f"replaces its own goal {prev}"))
            return True, f"{server} replaces its own goal {prev} with {goal_id}"
        held = t - self._hold.since
        why = (f"REFUSED: {server} asked for the writer while {self._hold.server} has held it "
               f"for {held:.1f} s (goal {self._hold.goal_id}"
               f"{'; ' + self._hold.detail if self._hold.detail else ''}). Exactly one writer "
               f"may command the actuators at a time (ADR-004, invariant 12) — cancel or finish "
               f"that goal first.")
        self.audit.append(AuditEntry(t, server, str(goal_id), "refused", why))
        return False, why

    def release(self, server: str, goal_id: Optional[str] = None,
                reason: str = "") -> Tuple[bool, str]:
        """Give the writer back. Returns (released, sentence).

        A release from a server that does NOT hold it is refused and recorded: it is either a
        late result from a goal that was already superseded, or a bug, and treating it as a
        release would free the CURRENT holder's writer under it.
        """
        self._check(server)
        t = self.now()
        if self._hold is None:
            return False, f"{server} released a writer nobody was holding"
        if self._hold.server != server:
            why = (f"{server} tried to release the writer, but {self._hold.server} is holding it. "
                   f"Ignored — releasing it here would free the current holder's writer under it.")
            self.audit.append(AuditEntry(t, server, str(goal_id or ""), "refused", why))
            return False, why
        if goal_id is not None and str(goal_id) != self._hold.goal_id:
            why = (f"{server} released goal {goal_id} but is holding {self._hold.goal_id}; a late "
                   f"result from a superseded goal does not end the current one")
            self.audit.append(AuditEntry(t, server, str(goal_id), "refused", why))
            return False, why
        held = t - self._hold.since
        gid = self._hold.goal_id
        self._hold = None
        self.audit.append(AuditEntry(t, server, gid, "released", reason))
        return True, f"{server} released the writer after {held:.1f} s (goal {gid}){' — ' + reason if reason else ''}"

    # ---------------------------------------------------------------- reporting
    def stale_holder(self, max_hold_s: float) -> Optional[str]:
        """A sentence when the current hold is suspiciously long, else None.

        IT REPORTS AND DOES NOT ACT. A timeout that freed the writer would create the two-writer
        state this class exists to prevent, at the worst possible moment — a controller that has
        stopped answering is exactly when a second one must NOT start writing. What to do about
        a stuck holder is the behaviour tree's decision (cancel the goal), and the tree can only
        make it if this line reaches it.
        """
        if self._hold is None:
            return None
        held = self.now() - self._hold.since
        if held <= max_hold_s:
            return None
        return (f"{self._hold.server} has held the writer for {held:.0f} s (goal "
                f"{self._hold.goal_id}, limit {max_hold_s:.0f} s). The writer is NOT taken away "
                f"— cancel that goal if it is stuck.")

    def health_line(self) -> str:
        """`HOLDER|server|goal|held_s|n_refused` — one line for the node's health topic."""
        n_ref = sum(1 for a in self.audit if a.outcome == "refused")
        if self._hold is None:
            return f"IDLE|none|-|0.0|{n_ref}"
        held = self.now() - self._hold.since
        return f"HOLDER|{self._hold.server}|{self._hold.goal_id}|{held:.1f}|{n_ref}"
