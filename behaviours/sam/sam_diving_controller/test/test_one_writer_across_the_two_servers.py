"""ONE WRITER, TWO SERVERS — letter D, strategy §5.3 (M1) and §9's named guard.

THE INVARIANT: at every instant exactly one of `auv_depth_move_to` and `auv_trajectory_tracking`
may hold the writer. Two writers on one actuator path is how a controller ends up fighting itself
at 10 Hz with nothing in the log saying so (ADR-004, invariant 12) — and this is the option that
makes both servers live in one `diving_node`, which they never have.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider \\
        smarc2/behaviours/sam/sam_diving_controller/test/test_one_writer_across_the_two_servers.py
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sam_diving_controller.one_writer_arbiter import (ArbiterRefusal, OneWriterArbiter)

WP = "auv_depth_move_to"
MPC = "auv_trajectory_tracking"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _arb():
    c = Clock()
    return OneWriterArbiter((WP, MPC), now=c), c


# ------------------------------------------------------------------ THE invariant
def test_both_servers_active_at_once_is_impossible():
    """THE guard the strategy names. Driven over every interleaving of two goals."""
    a, _c = _arb()
    ok, _ = a.request(WP, "g1")
    assert ok and a.holder == WP
    ok, why = a.request(MPC, "g2")
    assert ok is False, "both servers were allowed to hold the writer"
    assert a.holder == WP
    assert "invariant 12" in why


def test_no_interleaving_of_requests_and_releases_ever_yields_two_holders():
    """Exhaustive over a short alphabet: every sequence of (request|release) x (WP|MPC) up to
    length 6. `holder` is a single value by construction, so what is checked is that an ACCEPTED
    request never lands while the other server holds."""
    import itertools
    ops = [(kind, srv) for kind in ("request", "release") for srv in (WP, MPC)]
    for seq in itertools.product(ops, repeat=5):
        a, c = _arb()
        for i, (kind, srv) in enumerate(seq):
            c.t += 1.0
            before = a.holder
            if kind == "request":
                ok, _ = a.request(srv, f"g{i}")
                if ok and before is not None:
                    assert before == srv, (
                        f"{srv} was accepted while {before} was holding, in {seq}")
            else:
                a.release(srv, f"g{i}")
            assert a.holder in (None, WP, MPC)


def test_a_server_may_replace_its_OWN_goal():
    """An ordinary re-send. Refusing it would make a preempted-and-re-sent waypoint unflyable,
    which is exactly what the diversion design depends on (letter C)."""
    a, c = _arb()
    a.request(WP, "g1")
    c.t = 5.0
    ok, why = a.request(WP, "g2")
    assert ok and a.holder == WP and "replaces its own goal g1" in why


def test_the_refusal_names_the_holder_the_goal_and_how_long():
    """A rejected goal with no explanation on the wire is the hardest kind of failure to
    diagnose (SETTLED §1b), and rclpy turns an exception in a goal callback into exactly that."""
    a, c = _arb()
    a.request(WP, "leg_7", detail="waypoint 7 of the lawnmower")
    c.t = 12.0
    ok, why = a.request(MPC, "ring_0")
    assert not ok
    assert WP in why and "leg_7" in why and "12.0 s" in why
    assert "waypoint 7 of the lawnmower" in why
    assert "cancel or finish" in why


# ------------------------------------------------------------------ releasing
def test_the_writer_is_free_after_a_release_and_the_other_server_may_take_it():
    a, c = _arb()
    a.request(WP, "g1")
    c.t = 3.0
    released, why = a.release(WP, "g1", reason="goal succeeded")
    assert released and a.holder is None and "3.0 s" in why
    ok, _ = a.request(MPC, "g2")
    assert ok and a.holder == MPC


def test_a_release_from_the_server_that_is_NOT_holding_is_refused():
    """A late result from a superseded goal must not free the current holder's writer under it."""
    a, _c = _arb()
    a.request(WP, "g1")
    released, why = a.release(MPC, "g0")
    assert released is False and a.holder == WP
    assert "would free the current holder's writer under it" in why


def test_a_release_of_the_WRONG_goal_id_is_refused():
    a, _c = _arb()
    a.request(WP, "g1")
    a.request(WP, "g2")
    released, why = a.release(WP, "g1")
    assert released is False and a.holder == WP
    assert "superseded goal" in why


def test_releasing_when_nobody_holds_is_a_no_op_that_says_so():
    a, _c = _arb()
    released, why = a.release(WP)
    assert released is False and "nobody was holding" in why


# ------------------------------------------------------------------ refusals of the class itself
def test_an_unregistered_server_name_is_a_programming_error_and_raises():
    a, _c = _arb()
    with pytest.raises(ArbiterRefusal) as e:
        a.request("auv_something_else", "g1")
    assert "third writer" in str(e.value)


def test_an_arbiter_over_one_server_is_refused():
    with pytest.raises(ArbiterRefusal):
        OneWriterArbiter((WP,))
    with pytest.raises(ArbiterRefusal):
        OneWriterArbiter((WP, WP))


# ------------------------------------------------------------------ the stale hold
def test_a_stuck_holder_is_REPORTED_and_the_writer_is_NOT_taken_away():
    """A timeout that freed the writer would create the two-writer state this class exists to
    prevent, at the worst possible moment: a controller that has stopped answering is exactly
    when a second one must NOT start writing."""
    a, c = _arb()
    a.request(WP, "g1")
    c.t = 5.0
    assert a.stale_holder(60.0) is None
    c.t = 600.0
    line = a.stale_holder(60.0)
    assert line and "NOT taken away" in line and "g1" in line
    assert a.holder == WP, "the arbiter freed a stuck holder by itself"
    ok, _ = a.request(MPC, "g2")
    assert ok is False, "a stale hold let the other server in"


# ------------------------------------------------------------------ the audit
def test_every_decision_is_audited_with_its_outcome():
    a, c = _arb()
    a.request(WP, "g1")
    c.t = 1.0
    a.request(MPC, "g2")
    c.t = 2.0
    a.release(WP, "g1")
    outcomes = [(e.server, e.outcome) for e in a.audit]
    assert outcomes == [(WP, "accepted"), (MPC, "refused"), (WP, "released")]
    assert all(e.reason for e in a.audit if e.outcome == "refused")


def test_the_health_line_says_who_holds_and_how_many_were_refused():
    a, c = _arb()
    assert a.health_line().startswith("IDLE|none|")
    a.request(WP, "g1")
    c.t = 4.0
    a.request(MPC, "g2")
    line = a.health_line()
    assert line.startswith(f"HOLDER|{WP}|g1|4.0|")
    assert line.endswith("|1")
