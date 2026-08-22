#!/usr/bin/env python3
"""Guards for P8.0 — the expectation ledger and the association layer.

The headline case is `test_the_thirteen_metre_corridor_shift_*`: the farm (or the vehicle's
belief about it) displaced by exactly one corridor width. Every detection is then
self-consistent with the WRONG line, so no amount of local geometry resolves it — only a
global cue does. A system that cannot notice this case flies a confident, coherent, wrong
map, and that is the failure this whole layer exists to prevent.

Run:
    export PYTHONPYCACHEPREFIX=/tmp/pyc && rm -rf /tmp/pyc
    python3 -m pytest -p no:cacheprovider -q test/test_farm_ledger.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sam_farm_inspection.farm_ledger import (        # noqa: E402
    BELIEF_DRIFT_M_PER_DAY, DESIGN_SIGMA_M, Belief, Candidate, Evidence, Falsifier,
    FarmLedger, LedgerRefusal,
)

DAY = 86400.0
NOW = 1_787_000_000.0

#: Kristineberg: three culture lines ~26 m long, two corridors 13 m wide (paper §V.B).
CORRIDOR_M = 13.0


def line_spacing_falsifier(predicted_offset_m: float, sentence: str) -> Falsifier:
    return Falsifier("line_spacing", predicted_offset_m, 1.0, sentence)


def global_fix_falsifier(predicted_east_m: float, name: str) -> Falsifier:
    return Falsifier("global_fix", predicted_east_m, 1.0,
                     f"a surface GPS fix would put {name} here")


def buoys(*specs) -> list:
    """(name, x, z, source, age_days, sigma0) -> Beliefs."""
    out = []
    for name, x, z, source, age_days, sigma0 in specs:
        out.append(Belief(name, "surface_buoy", (x, z), source,
                          None if age_days is None else NOW - age_days * DAY, sigma0))
    return out


# ---------------------------------------------------------------------------- beliefs age

def test_a_belief_grows_less_certain_with_age():
    fresh = Belief("B", "surface_buoy", (0.0, 0.0), "as_observed", NOW, 0.2)
    assert fresh.sigma_at(NOW) == pytest.approx(0.2)
    week = fresh.sigma_at(NOW + 7 * DAY)
    assert week > fresh.sigma_at(NOW + 1 * DAY) > 0.2
    # The number is the aged one, not the survey one — this is the whole point.
    assert week == pytest.approx((0.2 ** 2 + (BELIEF_DRIFT_M_PER_DAY * 7) ** 2) ** 0.5)


def test_an_as_designed_belief_is_worse_than_a_survey_and_does_not_age():
    d = Belief("D", "surface_buoy", (0.0, 0.0), "as_designed")
    assert d.sigma_at(NOW) == pytest.approx(DESIGN_SIGMA_M)
    assert d.sigma_at(NOW + 30 * DAY) == pytest.approx(DESIGN_SIGMA_M)


def test_an_observation_without_a_date_is_refused():
    with pytest.raises(LedgerRefusal) as e:
        Belief("B", "surface_buoy", (0, 0), "as_observed", None, 0.2)
    assert "cannot be aged" in str(e.value)


def test_an_observed_position_with_no_stated_accuracy_is_not_offered_as_a_belief():
    b = Belief("B", "surface_buoy", (0, 0), "as_observed", NOW, None)
    with pytest.raises(LedgerRefusal) as e:
        b.sigma_at(NOW)
    assert "no stated accuracy" in str(e.value)


def test_weather_can_only_widen_a_belief():
    b = Belief("B", "surface_buoy", (0, 0), "as_observed", NOW - DAY, 0.2)
    assert b.sigma_at(NOW, 3.0) > b.sigma_at(NOW, 1.0)
    with pytest.raises(LedgerRefusal):
        b.sigma_at(NOW, 0.5)


def test_provenance_says_which_kind_of_position_it_is():
    obs = Belief("O", "surface_buoy", (0, 0), "as_observed", NOW - 2 * DAY, 0.2)
    des = Belief("D", "surface_buoy", (0, 0), "as_designed")
    assert "last observed 2.0 d ago" in obs.provenance(NOW)
    assert "never observed" in des.provenance(NOW)


# ------------------------------------------------------------------- a candidate must risk

def test_a_candidate_with_no_falsifier_is_refused():
    led = FarmLedger(buoys(("B1", 0, 0, "as_observed", 1, 0.3)))
    with pytest.raises(LedgerRefusal) as e:
        led.observe("d1", (0.2, 0.0), NOW)          # no falsifiers_for supplied
    assert "preference" in str(e.value)


def test_a_falsifier_that_cannot_be_contradicted_is_refused():
    with pytest.raises(LedgerRefusal) as e:
        Falsifier("line_spacing", 0.0, 0.0, "never wrong")
    assert "not a falsifier" in str(e.value)


# ------------------------------------------------------------- THE 13 m CORRIDOR SHIFT

def _shifted_farm():
    """Three lines 13 m apart, believed from a week-old survey. The vehicle is flying
    beside ONE of them; its DR has drifted by about a corridor width, so the detection sits
    almost exactly on top of the neighbouring line's believed position."""
    beliefs = buoys(
        ("line1_end", 0.0, 0.0, "as_observed", 7, 0.3),
        ("line2_end", CORRIDOR_M, 0.0, "as_observed", 7, 0.3),
        ("line3_end", 2 * CORRIDOR_M, 0.0, "as_observed", 7, 0.3),
    )
    return FarmLedger(beliefs)


def _falsifiers(b):
    """Each line-end predicts where a global fix would put it, and how far it should be
    from the farm's western edge. Both are real discriminators at this farm."""
    return [global_fix_falsifier(b.xz[0], b.name),
            line_spacing_falsifier(b.xz[0], f"{b.name} sits {b.xz[0]:.0f} m from the west edge")]


def test_the_thirteen_metre_corridor_shift_does_not_commit():
    """The nastiest realistic case: the detection is consistent with two lines at once."""
    led = _shifted_farm()
    # Detection sits midway-ish but well inside the gate of both line1 and line2, because a
    # week-old belief is worth sigma ~3.5 m and the gate is 4 sigma.
    a = led.observe("d1", (CORRIDOR_M / 2.0, 0.0), NOW, falsifiers_for=_falsifiers)
    assert a.status == "ambiguous"
    assert len(a.alive) >= 2
    assert a.committed_part is None, "committed while an alternative was still alive"
    assert "ambiguous(" in a.describe()


def test_the_leader_exists_but_is_not_a_commitment():
    """`leader()` is for display and for ranking what to look at next. It must never be
    mistaken for an identification — that is the difference between 'closest match' and
    'this is it', and the whole point of rule 2."""
    led = _shifted_farm()
    a = led.observe("d1", (CORRIDOR_M / 2.0 - 1.0, 0.0), NOW, falsifiers_for=_falsifiers)
    assert a.leader() is not None
    assert a.leader().part == "line1_end"          # genuinely the closest
    assert a.committed_part is None                # and still not committed


def test_a_global_fix_collapses_the_ambiguity_and_says_why():
    """Ivan's 'go a bit further and rule the others out' — here the discriminating
    observation is the surfacing fix, which is absolute and therefore breaks a tie that no
    amount of local geometry can."""
    led = _shifted_farm()
    a = led.observe("d1", (CORRIDOR_M / 2.0, 0.0), NOW, falsifiers_for=_falsifiers)
    assert a.status == "ambiguous"
    killed = led.apply_evidence(Evidence("global_fix", 0.0, 0.3, "RTK on surfacing"))
    assert killed, "the fix should have ruled something out"
    assert a.status == "committed"
    assert a.committed_part == "line1_end"
    assert any("ruled out" in k for k in killed)
    assert any("line2_end" in k for k in killed)


def test_evidence_that_is_too_sloppy_to_discriminate_kills_nothing():
    """A measurement whose own tolerance spans the alternatives is not evidence. Reporting
    it as decisive would be minting a certainty the sensor never had."""
    led = _shifted_farm()
    led.observe("d1", (CORRIDOR_M / 2.0, 0.0), NOW, falsifiers_for=_falsifiers)
    killed = led.apply_evidence(Evidence("global_fix", 0.0, 50.0, "a bad fix"))
    assert killed == []
    assert led.associations["d1"].status == "ambiguous"


def test_a_dead_candidate_is_kept_with_its_reason():
    """'We considered it and killed it' is a more useful record than never having
    considered it — and it is what lets a later contradiction be noticed."""
    led = _shifted_farm()
    a = led.observe("d1", (CORRIDOR_M / 2.0, 0.0), NOW, falsifiers_for=_falsifiers)
    n_before = len(a.candidates)
    led.apply_evidence(Evidence("global_fix", 0.0, 0.3, "RTK"))
    assert len(a.candidates) == n_before
    assert any(c.dead_because for c in a.candidates)


# --------------------------------------------------------------- active disambiguation

def test_what_would_discriminate_ranks_the_useful_look_first():
    led = _shifted_farm()
    led.observe("d1", (CORRIDOR_M / 2.0, 0.0), NOW, falsifiers_for=_falsifiers)
    ranked = led.what_would_discriminate()
    assert ranked, "an ambiguous ledger must be able to say what would settle it"
    kinds = [k for k, _ in ranked]
    assert "global_fix" in kinds and "line_spacing" in kinds
    assert all(n > 0 for _, n in ranked)


def test_a_settled_ledger_asks_no_questions():
    led = _shifted_farm()
    led.observe("d1", (CORRIDOR_M / 2.0, 0.0), NOW, falsifiers_for=_falsifiers)
    led.apply_evidence(Evidence("global_fix", 0.0, 0.3, "RTK"))
    assert led.what_would_discriminate() == []
    assert led.open_questions() == []


# ------------------------------------------------------------------------- the residual

def test_a_detection_matching_nothing_is_residual_not_an_error():
    """An unknown object is the mission's most interesting output, not a failure. It is
    also the measurement that decides whether a reasoning model earns its power budget."""
    led = _shifted_farm()
    a = led.observe("d9", (500.0, 500.0), NOW, falsifiers_for=_falsifiers)
    assert a.status == "unexplained"
    rep = led.report(NOW)
    assert rep.residual == 1
    assert "explained by nothing" in " ".join(rep.lines)


def test_missing_and_not_surveyed_stay_different_facts():
    led = _shifted_farm()
    rep_never = led.report(NOW, surveyed=lambda n: False)
    assert all("not_surveyed" in ln for ln in rep_never.lines if "line1_end" in ln)
    rep_looked = led.report(NOW, surveyed=lambda n: True)
    assert any("missing (looked, not found)" in ln for ln in rep_looked.lines)


def test_two_detections_cannot_both_be_the_same_part():
    """A farm with a duplicated part is a map nobody can act on. The ledger refuses rather
    than silently keeping the second commitment."""
    led = FarmLedger(buoys(("B1", 0, 0, "as_observed", 0, 0.3)))
    f = lambda b: [global_fix_falsifier(b.xz[0], b.name)]
    led.observe("d1", (0.1, 0.0), NOW, falsifiers_for=f)
    led.observe("d2", (0.2, 0.0), NOW, falsifiers_for=f)
    with pytest.raises(LedgerRefusal) as e:
        led.report(NOW)
    assert "both committed" in str(e.value)


# ----------------------------------------------------------------------------- verdicts

def test_confirmed_requires_no_surviving_alternative():
    led = _shifted_farm()
    led.observe("d1", (CORRIDOR_M / 2.0, 0.0), NOW, falsifiers_for=_falsifiers)
    assert led.verdict("line1_end", NOW) == "ambiguous"
    led.apply_evidence(Evidence("global_fix", 0.0, 0.3, "RTK"))
    assert led.verdict("line1_end", NOW) in ("confirmed", "moved")


def test_moved_is_a_normal_verdict_not_an_error():
    """Identified, but not where we believed it was — the case the mission exists for."""
    led = FarmLedger(buoys(("B1", 0, 0, "as_observed", 0, 0.3)))
    f = lambda b: [global_fix_falsifier(b.xz[0], b.name)]
    led.observe("d1", (1.0, 0.0), NOW, falsifiers_for=f)     # 1.0 m off a 0.3 m belief
    assert led.verdict("B1", NOW) == "moved"


def test_expected_counts_come_from_the_manifest_not_from_what_was_found():
    led = FarmLedger([
        Belief("b1", "surface_buoy", (0, 0), "as_designed"),
        Belief("m1", "mooring", (8, 0), "as_designed"),
        Belief("a1", "anchor_block", (8, 0), "as_designed"),
    ])
    exp = led.expected_counts()
    assert exp["surface_buoy"] == 1 and exp["mooring"] == 1 and exp["anchor_block"] == 1
    assert exp["culture_line"] == 0


def test_a_kind_outside_the_vocabulary_is_refused():
    with pytest.raises(LedgerRefusal) as e:
        Belief("x", "kelp_monster", (0, 0), "as_designed")
    assert "unknown part kind" in str(e.value)


def test_duplicate_part_names_are_refused():
    with pytest.raises(LedgerRefusal) as e:
        FarmLedger([Belief("b", "surface_buoy", (0, 0), "as_designed"),
                    Belief("b", "mooring", (1, 1), "as_designed")])
    assert "duplicate part name" in str(e.value)


def test_an_empty_manifest_is_refused():
    with pytest.raises(LedgerRefusal) as e:
        FarmLedger([])
    assert "no expectations" in str(e.value)


def test_the_same_detection_id_twice_is_refused():
    led = _shifted_farm()
    led.observe("d1", (0.0, 0.0), NOW, falsifiers_for=_falsifiers)
    with pytest.raises(LedgerRefusal) as e:
        led.observe("d1", (1.0, 0.0), NOW, falsifiers_for=_falsifiers)
    assert "twice" in str(e.value)


def test_kinds_filter_stops_a_buoy_matching_a_mooring():
    led = FarmLedger([
        Belief("b1", "surface_buoy", (0, 0), "as_observed", NOW, 0.3),
        Belief("m1", "mooring", (0.2, 0.0), "as_observed", NOW, 0.3),
    ])
    f = lambda b: [global_fix_falsifier(b.xz[0], b.name)]
    a = led.observe("d1", (0.05, 0.0), NOW, kinds=("surface_buoy",), falsifiers_for=f)
    assert [c.part for c in a.candidates] == ["b1"]


def test_zero_sigma_is_refused_rather_than_reading_as_a_perfect_match():
    c = Candidate("p", 0.0, 0.0, (Falsifier("k", 0.0, 1.0, "s"),))
    with pytest.raises(LedgerRefusal) as e:
        _ = c.normalised_distance
    assert "perfect match" in str(e.value)
