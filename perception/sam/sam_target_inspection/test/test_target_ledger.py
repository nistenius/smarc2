"""The candidate ledger — letter B, strategy §4b (fusion) and §7.1 (the verdict ladder).

The scenarios the work order names, plus the rules the vocabulary exists to protect:

  * both sensors agree      -> `leader`, then `provisional`, then `confirmed` WITH a model
  * SSS only, outside the FLS footprint -> stays a `candidate`; the sweep says nothing about it
  * SSS only, INSIDE a swept-and-empty FLS footprint -> `discarded`, with the sentence
  * a duplicate on the next lane -> recognised, and NOT re-diverted to
  * the budget spent before rung 2 -> `inconclusive`
  * never diverted -> `not_inspected` WITH A REASON, which is a different fact from `discarded`
  * `confirmed` is unreachable without a model reference

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_target_ledger.py
"""
import math

import pytest

from sam_farm_inspection.farm_ledger import Falsifier, LedgerRefusal
from sam_target_inspection.target_ledger import (ModelRef, Observation, POSITION_DRIFT_M_PER_DAY,
                                                 RUNGS, TargetLedger, TargetRefusal, VERDICTS)

SSS = dict(sensor="sss", sigma_m=2.0, extent_m={"across": 1.56, "shadow_len": 5.3})
FLS = dict(sensor="fls", sigma_m=0.5, extent_m={"length": 2.1, "width": 1.45, "height": 1.3})


def _sss(t=0.0, xy=(10.0, 20.0), aspect=90.0, **kw):
    d = dict(SSS, t=t, xy=xy, aspect_deg=aspect)
    d.update(kw)
    return Observation(**d)


def _fls(t=10.0, xy=(10.4, 20.3), aspect=95.0, **kw):
    d = dict(FLS, t=t, xy=xy, aspect_deg=aspect)
    d.update(kw)
    return Observation(**d)


# ------------------------------------------------------------------ the vocabulary
def test_the_vocabulary_is_the_eight_words_and_the_rungs_are_the_first_four():
    assert VERDICTS == ("candidate", "leader", "provisional", "confirmed", "discarded",
                        "inconclusive", "not_inspected", "ambiguous")
    assert RUNGS == VERDICTS[:4]


def test_an_unknown_sensor_is_refused_by_name():
    with pytest.raises(TargetRefusal):
        Observation(sensor="lidar", t=0.0, xy=(0.0, 0.0), sigma_m=1.0)


def test_a_zero_sigma_observation_is_refused():
    with pytest.raises(TargetRefusal):
        Observation(sensor="sss", t=0.0, xy=(0.0, 0.0), sigma_m=0.0)


# ------------------------------------------------------------------ association
def test_a_first_observation_creates_a_target_at_rung_zero():
    L = TargetLedger()
    t, status = L.observe(_sss())
    assert status == "new" and t.verdict() == "candidate"
    assert t.falsifiers, "a target with no falsifier is a preference, not a hypothesis"


def test_both_sensors_agreeing_promotes_to_leader():
    L = TargetLedger()
    a, _ = L.observe(_sss())
    b, status = L.observe(_fls())
    assert status == "committed" and b.id == a.id
    assert b.sensors() == ("fls", "sss")
    assert b.verdict() == "leader"


def test_a_second_sss_aspect_alone_promotes_to_leader():
    """The approach leg's abeam pass is a free falsifier and is what promotes a candidate before
    any camera frame exists (strategy §5.2)."""
    L = TargetLedger()
    L.observe(_sss(t=0.0, aspect=90.0))
    t, _ = L.observe(_sss(t=200.0, xy=(10.3, 20.2), aspect=200.0))
    assert t.has_second_aspect() and t.verdict() == "leader"


def test_two_looks_from_nearly_the_same_bearing_are_one_look():
    L = TargetLedger()
    L.observe(_sss(t=0.0, aspect=90.0))
    t, _ = L.observe(_sss(t=20.0, xy=(10.1, 20.1), aspect=95.0))
    assert not t.has_second_aspect()
    assert t.verdict() == "candidate"


def test_an_observation_far_away_makes_a_second_target():
    L = TargetLedger()
    L.observe(_sss())
    t, status = L.observe(_sss(t=5.0, xy=(200.0, 300.0)))
    assert status == "new" and len(L.targets) == 2


def test_ambiguity_is_reported_and_not_resolved_by_taking_the_nearest():
    """farm_ledger rule 2: an association is a SET until something kills the alternatives."""
    L = TargetLedger()
    tight = dict(sensor="fls", sigma_m=0.5, extent_m={"width": 1.4})
    L.observe(Observation(t=0.0, xy=(0.0, 0.0), **tight))
    L.observe(Observation(t=1.0, xy=(5.0, 0.0), **tight))
    assert len(L.targets) == 2, "5 m apart at 0.5 m sigma is two targets, not one"
    t, status = L.observe(Observation(t=2.0, xy=(2.5, 0.0), **tight))
    assert status == "ambiguous"
    assert any("ambiguous" in line for line in L.log)


# ------------------------------------------------------------------ the falsifier
def test_a_swept_and_empty_fls_footprint_kills_an_sss_candidate_with_its_sentence():
    L = TargetLedger()
    t, _ = L.observe(_sss())
    killed = L.note_fls_swept((10.0, 20.0), radius_m=5.0, saw_object=False)
    assert len(killed) == 1
    assert t.verdict() == "discarded"
    assert t.dead_because and "proud object" in t.dead_because
    assert "bare seabed" in t.dead_because


def test_a_candidate_outside_the_swept_footprint_is_neither_promoted_nor_killed():
    """Most SSS candidates, at 15-40 m abeam, are here. The sweep says nothing about them and
    the ledger must not pretend otherwise."""
    L = TargetLedger()
    t, _ = L.observe(_sss(xy=(10.0, 20.0)))
    killed = L.note_fls_swept((60.0, 20.0), radius_m=5.0, saw_object=False)
    assert killed == []
    assert t.verdict() == "candidate"


def test_a_sweep_that_saw_something_somewhere_promotes_nothing():
    """Promotion happens by ASSOCIATION, when the FLS's own observation lands within sigma. A
    sweep that saw an object somewhere is not evidence about THIS position."""
    L = TargetLedger()
    t, _ = L.observe(_sss())
    L.note_fls_swept((10.0, 20.0), radius_m=5.0, saw_object=True)
    assert t.verdict() == "candidate"


def test_a_dead_target_is_kept_not_deleted():
    L = TargetLedger()
    t, _ = L.observe(_sss())
    L.note_fls_swept((10.0, 20.0), 5.0, saw_object=False)
    assert t.id in L.targets
    assert L.verdicts()[t.id] == "discarded"


def test_a_kill_without_a_sentence_is_refused():
    L = TargetLedger()
    t, _ = L.observe(_sss())
    with pytest.raises(TargetRefusal):
        t.kill("")


def test_a_dead_target_does_not_accept_new_observations():
    L = TargetLedger()
    L.observe(_sss())
    L.note_fls_swept((10.0, 20.0), 5.0, saw_object=False)
    t2, status = L.observe(_fls())
    assert status == "new", "a ruled-out target must not silently absorb later observations"


# ------------------------------------------------------------------ the ladder
def test_provisional_needs_both_extents_agreeing_and_the_station_count():
    L = TargetLedger(min_stations_accepted=8)
    L.observe(_sss())
    t, _ = L.observe(_fls())
    assert t.verdict() == "leader"
    L.note_inspection(t.id, stations_planned=12, stations_accepted=7, budget_spent=False)
    assert t.verdict() == "leader", "7 of 12 stations is below the rung-2 bar"
    L.note_inspection(t.id, stations_planned=12, stations_accepted=8, budget_spent=False)
    assert t.verdict() == "provisional"


def test_extents_that_disagree_hold_the_target_at_leader():
    L = TargetLedger()
    L.observe(_sss())
    t, _ = L.observe(_fls(extent_m={"length": 9.0, "width": 6.0, "height": 1.0}))
    L.note_inspection(t.id, stations_planned=12, stations_accepted=12, budget_spent=False)
    assert t.extents_agree() is False
    assert t.verdict() == "leader"


def test_a_missing_second_sensor_makes_extents_agree_return_none_not_false():
    """ABSENT IS NOT EMPTY. "The FLS never looked" and "the FLS looked and disagreed" are
    different facts and only one of them should stop a promotion."""
    L = TargetLedger()
    t, _ = L.observe(_sss())
    assert t.extents_agree() is None


def test_confirmed_is_unreachable_without_a_model():
    L = TargetLedger()
    L.observe(_sss())
    t, _ = L.observe(_fls())
    L.note_inspection(t.id, stations_planned=12, stations_accepted=12, budget_spent=False)
    assert t.verdict() == "provisional"
    assert L.confirmed() == []
    t.set_model(ModelRef("sonar3d", "/campaigns/x/model.obj", {"width": 1.5}, 0.04))
    assert t.verdict() == "confirmed"
    assert L.confirmed() == [t.id]


def test_a_model_whose_extent_disagrees_does_not_confirm():
    L = TargetLedger()
    L.observe(_sss())
    t, _ = L.observe(_fls())
    L.note_inspection(t.id, stations_planned=12, stations_accepted=12, budget_spent=False)
    t.set_model(ModelRef("photogrammetry", "/c/m.obj", {"width": 6.0}, 0.02))
    assert t.verdict() == "provisional"


def test_a_model_with_no_residual_is_not_a_model():
    with pytest.raises(TargetRefusal):
        ModelRef("sonar3d", "/c/m.obj", {"width": 1.4}, -1.0)
    with pytest.raises(TargetRefusal):
        ModelRef("sonar3d", "", {"width": 1.4}, 0.0)
    with pytest.raises(TargetRefusal):
        ModelRef("guesswork", "/c/m.obj", {"width": 1.4}, 0.0)


def test_a_model_cannot_be_attached_to_a_ruled_out_target():
    L = TargetLedger()
    t, _ = L.observe(_sss())
    L.note_fls_swept((10.0, 20.0), 5.0, saw_object=False)
    with pytest.raises(TargetRefusal):
        t.set_model(ModelRef("sonar3d", "/c/m.obj", {"width": 1.4}, 0.0))


# ------------------------------------------------------------------ the other three words
def test_the_budget_spent_before_rung_two_is_inconclusive():
    L = TargetLedger()
    t, _ = L.observe(_sss())
    L.note_inspection(t.id, stations_planned=12, stations_accepted=2, budget_spent=True)
    assert t.verdict() == "inconclusive"


def test_a_refused_diversion_is_not_inspected_with_its_reason():
    L = TargetLedger()
    t, _ = L.observe(_sss())
    L.note_not_inspected(t.id, "the mission's diversion budget is spent (1/1)")
    assert t.verdict() == "not_inspected"
    assert "diversion budget" in t.describe(0.0)


def test_not_inspected_requires_a_reason():
    L = TargetLedger()
    t, _ = L.observe(_sss())
    with pytest.raises(TargetRefusal):
        L.note_not_inspected(t.id, "")


def test_not_inspected_is_not_reachable_by_falling_through_from_inconclusive():
    """"We did not go" is not a weaker "we went and it was nothing" (SETTLED §3e). A target that
    was refused on one lane and reached on the next must stop reading `not_inspected`."""
    L = TargetLedger()
    t, _ = L.observe(_sss())
    L.note_not_inspected(t.id, "beyond the 60 m diversion limit")
    assert t.verdict() == "not_inspected"
    L.note_inspection(t.id, stations_planned=12, stations_accepted=1, budget_spent=True)
    assert t.verdict() == "inconclusive"


# ------------------------------------------------------------------ aging and duplicates
def test_not_inspected_beats_every_rung_the_evidence_would_otherwise_support():
    """It is checked FIRST and independently. A candidate both sensors saw, on the far side of
    a spent diversion budget, is `not_inspected` — we did not go and look at it, whatever the
    scan-leg evidence says. Letting the rungs win here is how "we did not go" quietly becomes
    a claim about the object (SETTLED §3e)."""
    L = TargetLedger()
    L.observe(_sss())
    t, _ = L.observe(_fls())
    assert t.rung() >= 1
    L.note_not_inspected(t.id, "the mission's diversion budget is spent (1/1)")
    assert t.verdict() == "not_inspected"


def test_a_target_offered_an_empty_falsifier_list_is_refused():
    """farm_ledger rule 3, at this layer's own door. The default falsifier makes the guard
    unreachable on the normal path, so it is driven directly — otherwise the refusal is
    untested code that reads as protection."""
    L = TargetLedger()
    with pytest.raises(TargetRefusal) as e:
        L.observe(_sss(), falsifiers=[])
    assert "preference" in str(e.value)


def test_the_position_sigma_grows_with_time_since_the_last_look():
    L = TargetLedger()
    t, _ = L.observe(_sss(t=0.0))
    now = 0.0
    assert t.sigma_at(now) == pytest.approx(t.sigma0_m())
    a_day = t.sigma_at(86400.0)
    assert a_day > t.sigma_at(now)
    assert a_day == pytest.approx(math.hypot(t.sigma0_m(), POSITION_DRIFT_M_PER_DAY))


def test_two_observations_tighten_the_position_more_than_either_alone():
    L = TargetLedger()
    a, _ = L.observe(_sss())
    s1 = a.sigma0_m()
    b, _ = L.observe(_fls())
    assert b.sigma0_m() < min(s1, 0.5)


def test_the_position_is_inverse_variance_weighted_not_the_latest_look():
    """The FLS at 7 m has a far tighter sigma than the SSS at 25 m; taking the most recent look
    would throw that away, and taking the mean would throw away half of it."""
    L = TargetLedger()
    L.observe(_sss(xy=(0.0, 0.0)))
    t, _ = L.observe(_fls(xy=(4.0, 0.0)))
    n, _e = t.position()
    assert n > 3.5, "the tighter observation must dominate"
    assert n < 4.0


def test_an_inspected_target_is_recognised_as_a_duplicate_on_the_next_lane():
    L = TargetLedger()
    t, _ = L.observe(_sss(t=0.0))
    L.note_inspection(t.id, stations_planned=12, stations_accepted=12, budget_spent=False)
    dup = L.duplicate_of((10.5, 20.5), sigma_m=2.0, now=300.0)
    assert dup is not None and dup.id == t.id


def test_an_uninspected_target_is_not_a_duplicate():
    L = TargetLedger()
    L.observe(_sss(t=0.0))
    assert L.duplicate_of((10.5, 20.5), sigma_m=2.0, now=300.0) is None


def test_a_far_away_candidate_is_not_a_duplicate():
    L = TargetLedger()
    t, _ = L.observe(_sss(t=0.0))
    L.note_inspection(t.id, stations_planned=12, stations_accepted=12, budget_spent=False)
    assert L.duplicate_of((300.0, 20.0), sigma_m=2.0, now=300.0) is None


# ------------------------------------------------------------------ reporting
def test_the_report_names_every_target_and_its_verdict():
    L = TargetLedger()
    a, _ = L.observe(_sss(xy=(0.0, 0.0)))
    b, _ = L.observe(_sss(t=1.0, xy=(400.0, 0.0)))
    L.note_not_inspected(b.id, "the mission's diversion budget is spent")
    lines = L.report(now=100.0)
    assert any(a.id in ln and "candidate" in ln for ln in lines)
    assert any(b.id in ln and "not_inspected" in ln for ln in lines)


def test_the_ledger_reuses_the_farm_ledgers_falsifier_type():
    """Imported, not forked. If this ever becomes a local class, the aging and the "no
    falsifier, no hypothesis" refusal have been duplicated somewhere."""
    L = TargetLedger()
    t, _ = L.observe(_sss())
    assert all(isinstance(f, Falsifier) for f in t.falsifiers)
    assert issubclass(TargetRefusal, LedgerRefusal)
