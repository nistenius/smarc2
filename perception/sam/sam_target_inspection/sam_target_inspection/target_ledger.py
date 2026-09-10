#!/usr/bin/env python3
"""The candidate ledger for the adaptive inspection — strategy §4b (fusion) and §7.1 (verdicts).

BUILT ON `sam_farm_inspection.farm_ledger`, NOT BESIDE IT. `Belief`, `Falsifier`, `Evidence`,
`Candidate` and `Association` are IMPORTED and used as they are; the aging, the "a candidate
with no falsifier is a preference" refusal, and the "an association is a set until something
kills the alternatives" rule all come from there unchanged. What this module adds is the part
the farm did not need: two sensors whose observations must be fused, a rung ladder, and a
verdict word that is only allowed to say `confirmed` when a model exists.

Forking `farm_ledger` would have been faster and would have meant two implementations of the
one rule this project has paid most for — that `confirmed` means "no surviving alternative"
(SETTLED §3s7).

THE VOCABULARY (strategy §7.1). Eight words, and each is a different fact:

    candidate       rung 0 — an SSS highlight-and-shadow, one sensor, one aspect
    leader          rung 1 — a second sensor or a second aspect agrees, in position and extent
    provisional     rung 2 — the FLS box agrees with the SSS extent AND the capture gate
                             accepted enough stations
    confirmed       rung 3 — a REGISTERED MODEL exists and its extent agrees. The only rung
                             that may say so, and it cannot be reached onboard without a model
                             reference being handed in
    discarded       a rung contradicted rung 0 — and the SENTENCE that killed it is kept
    inconclusive    the inspection budget was spent before rung 2
    not_inspected   we never went — with the reason. NOT the same as `discarded`
                             (ABSENT IS NOT EMPTY, SETTLED §3e)
    ambiguous       more than one target survives as an explanation of an observation

NOTHING HERE MINTS A CONFIDENCE. Every number that ranks anything is a distance divided by an
uncertainty whose parts are named (§3s7).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sam_farm_inspection.farm_ledger import (Association, Candidate, Evidence, Falsifier,
                                             LedgerRefusal, Belief, FALSIFY_SIGMA, GATE_SIGMA)

#: The rung ladder, in order. The index IS the rung number in strategy §7.1.
RUNGS = ("candidate", "leader", "provisional", "confirmed")

#: Every word a target may carry.
VERDICTS = ("candidate", "leader", "provisional", "confirmed", "discarded", "inconclusive",
            "not_inspected", "ambiguous")

#: Drift of a target's positional belief, metres of 1-sigma per day since it was last observed.
#: A car on the seabed does not move; what moves is OUR ESTIMATE OF WHERE IT IS, because the
#: position was written in a dead-reckoned frame that has drifted since. This is therefore an
#: estimator property, not an object property, and it is PROVISIONAL until measured against two
#: surveys of the same site. `farm_ledger.BELIEF_DRIFT_M_PER_DAY` (0.5) is a farm's mooring
#: spread and is the wrong quantity here, so it is deliberately NOT reused.
POSITION_DRIFT_M_PER_DAY = 2.0


class TargetRefusal(LedgerRefusal):
    """Raised with an operator-readable reason. Never caught and defaulted."""


# --------------------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Observation:
    """One look at one place by one sensor.

    `xy` is (north, east) in the site frame — metres, not degrees, because every comparison in
    this file is a distance and doing them in degrees is how a latitude difference becomes 1.9
    times a longitude one without anyone noticing.

    `extent_m` carries whatever that sensor measured: the SSS gives `across` and `shadow_len`,
    the FLS gives `length`, `width` and `height`. Both are kept as measured; nothing is
    converted into a common "size" number, because the two sensors measure different things and
    a single number would hide which one disagreed.
    """

    sensor: str                       # sss | fls | camera | model
    t: float
    xy: Tuple[float, float]
    sigma_m: float
    extent_m: Dict[str, float] = field(default_factory=dict)
    aspect_deg: Optional[float] = None
    detail: str = ""
    score: Optional[float] = None

    def __post_init__(self):
        if self.sensor not in ("sss", "fls", "camera", "model"):
            raise TargetRefusal(
                f"unknown sensor {self.sensor!r}. A sensor nothing recognises would contribute "
                f"to no rung and be missed by every check.")
        if self.sigma_m <= 0:
            raise TargetRefusal(
                f"{self.sensor}: an observation with zero sigma reads as a perfect measurement "
                f"on the least trustworthy input there is (farm_ledger's own rule).")


@dataclass(frozen=True)
class ModelRef:
    """A model built at the station, and the only thing that may raise a target to `confirmed`.

    `source` is `sonar3d` or `photogrammetry` — the ladder accepts either (strategy §7.1 rung
    3), which is what turns D4 from a blocker into a preference. `residual_m` is the
    registration residual and it is MANDATORY: a model whose fit error is unknown cannot be
    said to agree with anything (§3c's consensus rule — a residual is reported, never absorbed).
    """

    source: str
    path: str
    extent_m: Dict[str, float]
    residual_m: float

    def __post_init__(self):
        if self.source not in ("sonar3d", "photogrammetry"):
            raise TargetRefusal(f"unknown model source {self.source!r}")
        if not self.path:
            raise TargetRefusal("a model reference with no path is not a model reference")
        if self.residual_m < 0:
            raise TargetRefusal("a negative registration residual is not a residual")


# --------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------
@dataclass
class Target:
    """One object we think is there, everything ever seen of it, and what would rule it out."""

    id: str
    observations: List[Observation] = field(default_factory=list)
    falsifiers: List[Falsifier] = field(default_factory=list)
    dead_because: Optional[str] = None
    model: Optional[ModelRef] = None
    #: Stations whose capture gate ACCEPTED — evidence, not a stopwatch (SETTLED §3e).
    stations_accepted: int = 0
    stations_planned: int = 0
    budget_spent: bool = False
    not_inspected_reason: Optional[str] = None
    #: Set when this target was reached by an inspection at all. `not_inspected` is about
    #: whether we WENT, and it must not be inferable from how the inspection turned out.
    inspected: bool = False

    # ---------------------------------------------------------------- geometry
    @property
    def alive(self) -> bool:
        return self.dead_because is None

    @property
    def last_t(self) -> float:
        return max(o.t for o in self.observations)

    def position(self) -> Tuple[float, float]:
        """Inverse-variance weighted mean of every observation. The obvious estimator, and the
        reason it is written out rather than 'the latest': the FLS at 7 m has a far tighter
        sigma than the SSS at 25 m, and taking the most recent look would throw that away."""
        wsum = sum(1.0 / (o.sigma_m ** 2) for o in self.observations)
        n = sum(o.xy[0] / (o.sigma_m ** 2) for o in self.observations) / wsum
        e = sum(o.xy[1] / (o.sigma_m ** 2) for o in self.observations) / wsum
        return n, e

    def sigma0_m(self) -> float:
        """Uncertainty AT the moment of the last observation: the combined inverse variance."""
        return 1.0 / math.sqrt(sum(1.0 / (o.sigma_m ** 2) for o in self.observations))

    def sigma_at(self, now: float) -> float:
        """Uncertainty NOW — what it was, grown by how long ago that was.

        Delegated to `farm_ledger.Belief` so there is exactly one aging rule in the codebase,
        with this module's own drift constant. `Belief` refuses an observed position with no
        stated accuracy and refuses a weather factor below 1, and both refusals are wanted here.
        """
        b = Belief(name=self.id, kind="anchor_block", xz=self.position(),
                   source="as_observed", observed_at=self.last_t, sigma0_m=self.sigma0_m())
        age_days = max(0.0, (now - self.last_t) / 86400.0)
        drift = POSITION_DRIFT_M_PER_DAY * age_days
        return math.sqrt(b.base_sigma() ** 2 + drift ** 2)

    # ---------------------------------------------------------------- evidence
    def sensors(self) -> Tuple[str, ...]:
        return tuple(sorted({o.sensor for o in self.observations}))

    def aspects(self) -> List[float]:
        return [o.aspect_deg for o in self.observations if o.aspect_deg is not None]

    def has_second_aspect(self, min_separation_deg: float = 30.0) -> bool:
        """Two looks from bearings far enough apart to be a different aspect.

        30° because that is the convergence the inspection ring is built on (strategy §5.3);
        two looks 5° apart are one look with a longer baseline and must not promote anything.
        """
        a = sorted(self.aspects())
        return any(b - x >= min_separation_deg for x, b in zip(a, a[1:]))

    def extents_agree(self, tol_m: float = 1.0) -> Optional[bool]:
        """Does the FLS box agree with the SSS across-track extent? None when one is missing.

        None, not False. "The FLS never looked" and "the FLS looked and disagreed" are
        different facts and only one of them should stop a promotion (ABSENT IS NOT EMPTY).
        """
        sss = [o for o in self.observations if o.sensor == "sss" and "across" in o.extent_m]
        fls = [o for o in self.observations if o.sensor == "fls" and "width" in o.extent_m]
        if not sss or not fls:
            return None
        a = sorted(o.extent_m["across"] for o in sss)[len(sss) // 2]
        w = sorted(o.extent_m["width"] for o in fls)[len(fls) // 2]
        return abs(a - w) <= tol_m

    # ---------------------------------------------------------------- the ladder
    def rung(self, *, min_stations_accepted: int = 8) -> int:
        """The highest rung the EVIDENCE supports. Pure — it reads, it does not decide.

        Every step up needs something that could have failed:
          1 a second SENSOR or a second ASPECT agreeing in position (the association already
            enforced the position part: an observation that did not fall within sigma is not
            on this target at all);
          2 the two sensors' extents agreeing AND at least `min_stations_accepted` stations
            whose capture gate accepted — a coverage COUNT, never a stopwatch;
          3 a model reference whose extent agrees with the measured extent.
        """
        if not self.observations:
            return 0
        r = 0
        if len(self.sensors()) > 1 or self.has_second_aspect():
            r = 1
        if r >= 1 and self.extents_agree() and self.stations_accepted >= min_stations_accepted:
            r = 2
        if r >= 2 and self.model is not None and self._model_extent_agrees():
            r = 3
        return r

    def _model_extent_agrees(self, tol_m: float = 1.0) -> bool:
        if self.model is None:
            return False
        meas = [o.extent_m.get("width") for o in self.observations
                if o.sensor == "fls" and "width" in o.extent_m]
        meas += [o.extent_m.get("across") for o in self.observations
                 if o.sensor == "sss" and "across" in o.extent_m]
        meas = [m for m in meas if m is not None]
        mw = self.model.extent_m.get("width")
        if not meas or mw is None:
            return False
        return abs(sorted(meas)[len(meas) // 2] - mw) <= tol_m

    def verdict(self, *, min_stations_accepted: int = 8) -> str:
        """The one word. Order matters and each branch is a different fact.

        `not_inspected` is checked FIRST and independently of everything else: "we did not go"
        is not a weaker version of "we went and it was nothing" (SETTLED §3e), and letting a
        refusal fall through to `inconclusive` is exactly the collapse the fourth verdict
        exists to prevent.
        """
        if self.not_inspected_reason is not None:
            return "not_inspected"
        if not self.alive:
            return "discarded"
        r = self.rung(min_stations_accepted=min_stations_accepted)
        if self.budget_spent and r < 2:
            return "inconclusive"
        return RUNGS[r]

    def kill(self, sentence: str) -> None:
        """Rule this target out, keeping the sentence that did it.

        A dead target is KEPT. "We considered the bright patch at 25 m and killed it with the
        FLS sweep" is a different and more useful record than never having considered it.
        """
        if not sentence:
            raise TargetRefusal("a target may not be killed without a sentence saying what "
                                "killed it; an unexplained kill is unauditable")
        self.dead_because = sentence

    def set_model(self, model: ModelRef) -> None:
        if not self.alive:
            raise TargetRefusal(f"{self.id} was ruled out ({self.dead_because}); attaching a "
                                f"model to it would resurrect it silently")
        self.model = model

    def describe(self, now: float) -> str:
        v = self.verdict()
        n, e = self.position()
        if v == "not_inspected":
            return f"{self.id}: not_inspected — {self.not_inspected_reason}"
        if v == "discarded":
            return f"{self.id}: discarded — {self.dead_because}"
        return (f"{self.id}: {v} at ({n:.1f}, {e:.1f}) m, sigma {self.sigma_at(now):.2f} m, "
                f"{len(self.observations)} observation(s) from {'+'.join(self.sensors())}, "
                f"stations {self.stations_accepted}/{self.stations_planned}")


# --------------------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------------------
class TargetLedger:
    """Associates observations to targets, applies falsifiers, and reports verdicts."""

    def __init__(self, *, gate_sigma: float = GATE_SIGMA,
                 min_stations_accepted: int = 8):
        self.gate_sigma = float(gate_sigma)
        self.min_stations_accepted = int(min_stations_accepted)
        self.targets: Dict[str, Target] = {}
        self.associations: List[Association] = []
        self.log: List[str] = []
        self._n = 0

    # ---------------------------------------------------------------- association
    def _default_falsifiers(self, obs: Observation) -> List[Falsifier]:
        """Every target must carry at least one falsifier (farm_ledger rule 3).

        The one every candidate gets for free is the FLS sweep: if the vehicle's forward sonar
        passes over this position at a range where a car-sized object would be resolved and
        reports nothing standing above the seabed, the candidate is bottom texture. It is
        `predicts = 1` ("a proud object is there"), and evidence of `0` at a tolerance smaller
        than the prediction contradicts it.
        """
        # THE TOLERANCES ARE SET BY farm_ledger's OWN KILL RULE and are derived, not chosen.
        # `farm_ledger` kills a candidate only when the evidence differs from the prediction by
        # more than FALSIFY_SIGMA (5) times the quadrature of the two tolerances — deliberately
        # a higher bar than the bar to admit, because killing a true candidate is the one error
        # this layer cannot recover from. For a BINARY indicator (predicts 1 = "a proud object
        # is there", observed 0 = "the seabed here is bare") the kill therefore needs
        #     |1 - 0| > 5 * sqrt(tol_f^2 + tol_e^2)   =>   sqrt(tol_f^2 + tol_e^2) < 0.2
        # and the pair below (0.10, 0.02) gives 0.102, i.e. a kill at 9.8 sigma. Read the other
        # way round: the sweep must be at least 90 % reliable at seeing a car-sized object it
        # passes over before its silence is allowed to kill anything.
        return [Falsifier(
            kind="fls_swept",
            predicts=1.0,
            tolerance_m=0.10,
            sentence=("a highlight-and-shadow at this position implies a proud object; the "
                      "forward sonar sweeping the same position and finding a bare seabed "
                      "contradicts it"))]

    def observe(self, obs: Observation, *, now: Optional[float] = None,
                falsifiers: Optional[Sequence[Falsifier]] = None) -> Tuple[Target, str]:
        """Offer one observation. Returns (target, status) where status is the ASSOCIATION's.

        `status` is `committed` when exactly one existing target survives the gate,
        `ambiguous` when several do, and `new` when none does. Ambiguity is NOT resolved by
        taking the nearest: an association is a set until something kills the alternatives
        (farm_ledger rule 2), so the observation is attached to the leader for bookkeeping and
        the ambiguity is recorded and reported.
        """
        now = obs.t if now is None else now
        assoc = Association(det_id=f"{obs.sensor}@{obs.t:.3f}", xz=obs.xy)
        for tid in sorted(self.targets):
            tgt = self.targets[tid]
            if not tgt.alive:
                continue
            sig = math.sqrt(tgt.sigma_at(now) ** 2 + obs.sigma_m ** 2)
            n, e = tgt.position()
            d = math.hypot(obs.xy[0] - n, obs.xy[1] - e)
            if d > self.gate_sigma * sig:
                continue
            assoc.add_candidate(Candidate(tid, d, sig, tuple(tgt.falsifiers)))
        self.associations.append(assoc)

        if assoc.status == "unexplained":
            self._n += 1
            tid = f"T{self._n}"
            # `is None`, NOT `or`: an explicitly EMPTY list must reach the refusal below rather
            # than fall back to the default. Found by the test that drives the refusal — with
            # `or`, a caller who deliberately offered no falsifier was silently given one, and
            # the guard that is the whole point of rule 3 was unreachable.
            fs = self._default_falsifiers(obs) if falsifiers is None else list(falsifiers)
            tgt = Target(id=tid, falsifiers=fs)
            if not tgt.falsifiers:
                raise TargetRefusal(f"{tid}: a target with no falsifier is a preference, not a "
                                    f"hypothesis. State what observation would rule it out.")
            tgt.observations.append(obs)
            self.targets[tid] = tgt
            self.log.append(f"{tid}: new target from {obs.sensor} at "
                            f"({obs.xy[0]:.1f}, {obs.xy[1]:.1f}) sigma {obs.sigma_m:.2f} m")
            return tgt, "new"

        lead = assoc.leader()
        tgt = self.targets[lead.part]
        tgt.observations.append(obs)
        status = assoc.status
        if status == "ambiguous":
            self.log.append(
                f"{assoc.det_id}: ambiguous — one of "
                f"[{', '.join(sorted(c.part for c in assoc.alive))}]; attached to "
                f"{tgt.id} (nearest) for bookkeeping, NOT committed")
        else:
            self.log.append(f"{tgt.id}: {obs.sensor} observation associated "
                            f"({lead.normalised_distance:.2f} sigma)")
        return tgt, status

    # ---------------------------------------------------------------- falsifiers
    def note_fls_swept(self, xy: Tuple[float, float], radius_m: float, *, saw_object: bool,
                       detail: str = "") -> List[str]:
        """The FLS swept a patch of seabed. Apply that fact to every target inside it.

        THE FALSIFIER THE TWO SENSORS BUY EACH OTHER (strategy §4b). An SSS candidate that lies
        inside a footprint the FLS actually swept, at a range where a car-sized object would be
        resolved, and that the FLS did NOT see, is bottom texture at a slope — and the kill is
        recorded with its sentence.

        `saw_object=True` does NOT promote anything here: promotion happens by ASSOCIATION,
        when the FLS's own observation lands within sigma of the target. A sweep that saw
        *something* somewhere is not evidence about *this* position.
        """
        killed: List[str] = []
        for tid in sorted(self.targets):
            tgt = self.targets[tid]
            if not tgt.alive:
                continue
            n, e = tgt.position()
            if math.hypot(n - xy[0], e - xy[1]) > radius_m:
                continue
            if saw_object:
                continue
            ev = Evidence(kind="fls_swept", value=0.0, tolerance_m=0.02,
                          detail=detail or f"swept within {radius_m:.1f} m and found a bare seabed")
            # The farm ledger's own contradiction test, so there is one rule for "does this
            # evidence kill this falsifier" in the codebase.
            for f in tgt.falsifiers:
                if f.kind != ev.kind:
                    continue
                margin = math.sqrt(f.tolerance_m ** 2 + ev.tolerance_m ** 2)
                if abs(ev.value - f.predicts) > FALSIFY_SIGMA * margin:
                    why = (f"{tgt.id} ruled out: {f.sentence} — predicted {f.predicts:.1f}, "
                           f"observed {ev.value:.1f} (±{margin:.2f}) [{ev.detail}]")
                    tgt.kill(why)
                    killed.append(why)
                    self.log.append(why)
                    break
        return killed

    def note_not_inspected(self, target_id: str, reason: str) -> None:
        """We did not go, and this is why. A refusal names its reason (rule 12)."""
        if not reason:
            raise TargetRefusal("`not_inspected` without a reason is the verdict this "
                                "vocabulary exists to prevent")
        self.targets[target_id].not_inspected_reason = reason
        self.log.append(f"{target_id}: not_inspected — {reason}")

    def note_inspection(self, target_id: str, *, stations_planned: int,
                        stations_accepted: int, budget_spent: bool) -> None:
        t = self.targets[target_id]
        t.inspected = True
        t.stations_planned = int(stations_planned)
        t.stations_accepted = int(stations_accepted)
        t.budget_spent = bool(budget_spent)
        # Going somewhere ANSWERS "did we go", so the refusal is cleared: a target that was
        # refused on one lane and reached on the next must not still read `not_inspected`.
        t.not_inspected_reason = None

    def duplicate_of(self, xy: Tuple[float, float], sigma_m: float,
                     now: float) -> Optional[Target]:
        """An already-inspected target at this position, within the AGED sigma of both.

        Strategy §8: a candidate re-detected on the next lane at an inspected position is the
        same object and is not re-diverted to. Aged, because the answer depends on how long ago
        we were there — `farm_ledger.sigma_at`'s rule applied to a decision instead of a report.
        """
        for tid in sorted(self.targets):
            t = self.targets[tid]
            if not t.inspected or not t.alive:
                continue
            n, e = t.position()
            sig = math.sqrt(t.sigma_at(now) ** 2 + sigma_m ** 2)
            if math.hypot(n - xy[0], e - xy[1]) <= self.gate_sigma * sig:
                return t
        return None

    # ---------------------------------------------------------------- reporting
    def verdicts(self) -> Dict[str, str]:
        return {tid: t.verdict(min_stations_accepted=self.min_stations_accepted)
                for tid, t in sorted(self.targets.items())}

    def report(self, now: float) -> List[str]:
        out = [t.describe(now) for _, t in sorted(self.targets.items())]
        amb = [a for a in self.associations if a.status == "ambiguous"]
        if amb:
            out.append(f"{len(amb)} observation(s) could not be committed to one target")
        return out

    def confirmed(self) -> List[str]:
        """The only place `confirmed` is produced, and it cannot be produced without a model."""
        out = []
        for tid, t in sorted(self.targets.items()):
            if t.verdict(min_stations_accepted=self.min_stations_accepted) == "confirmed":
                if t.model is None:                       # pragma: no cover - belt and braces
                    raise TargetRefusal(
                        f"{tid} reported `confirmed` with no model reference. That is the one "
                        f"thing this vocabulary may never do (strategy §7.1 rung 3).")
                out.append(tid)
        return out
