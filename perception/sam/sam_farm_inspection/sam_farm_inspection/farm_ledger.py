#!/usr/bin/env python3
"""P8.0 — the expectation ledger and the association layer.

Pure python: no ROS, no numpy, no I/O. Same shape as `farm_mission.py`, and for the same
reason — the part of this system that decides *what we believe we are looking at* has to be
drivable exhaustively from a test, on a laptop, with no rig.

WHY THIS EXISTS (Ivan, 2026-08-20): "nothing will be exactly where we think it is or even
where we know we left it last time, like e.g. yesterday. So whatever we do it must contain
some sort of ability to judge *this is the closest match*, or *this is one of 4 instances,
let me go a bit further and depending on what I see I can rule out the other options
gradually*."

The SSS-SLAM paper this mission is built on (Valdez/Torroba/Folkesson/Stenius) deliberately
does NOT solve that. It reduces data association to matching each detection to the prior of
the rope it came from, which it calls trivial *given the AUV pose estimate and the prior map
of the farm*. That is fair for one survey of a known farm with modest drift. It is exactly
the assumption above attacks. The paper also records what over-trusting a prior costs: with
a single prior per rope, the corrections "diverge rapidly with the first detections of a new
line". So we keep the paper's back-end and add the front-end gate it does not have.

FOUR RULES, and every one of them is a rule this project has already paid for once:

1. A BELIEF HAS AN AGE. `Belief.sigma_at(now)` grows with time since the part was last
   observed. A buoy fixed yesterday is not survey-grade today. Same family as the staleness
   rules in spec §5: age it, never blank it, and never read a stale fact as current.

2. AN ASSOCIATION IS A SET UNTIL SOMETHING KILLS THE ALTERNATIVES. `commit()` happens when
   exactly one candidate survives -- never because one candidate scored best. `confirmed`
   therefore means "no surviving alternative"; the fifth verdict `ambiguous` carries the k
   candidates by name. "One of four" is a different fact from `missing` and from
   `not_surveyed`, and reporting it as either is the error the original four verdicts were
   invented to prevent (SETTLED §3k).

3. EVERY CANDIDATE CARRIES ITS OWN FALSIFIER -- the observation that would rule it OUT.
   Ivan's acoustic-range rule generalised (SETTLED §3s2: "positions agreeing is not
   evidence, the range is the falsifier"). A candidate with no falsifier is not a
   hypothesis, it is a preference, and `add_candidate` refuses one.

4. NOTHING HERE MINTS A CONFIDENCE. Weights come from geometry -- a distance divided by an
   uncertainty that says where it came from. No score is invented, and no sentence in this
   module states a number that was not measured. That is the standing lesson of the
   station's "MEASURED +/-0.014 m", which was a hardcoded constant reading as a measurement
   (SETTLED §3s2).

WHAT THIS MODULE DELIBERATELY DOES NOT DO: it does not optimise a graph, does not hold an
action client, does not write an actuator, and does not decide where to go. It answers
"what do we think we are seeing, and what would settle it" -- `farm_slam.py` (P8.1) takes
only its COMMITTED associations as hard factors, and the planner (P8.4e) turns
`what_would_discriminate()` into the next sub-goal.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Policy constants. ALL PROVISIONAL, none measured on a rig, and named loudly so that
# nobody later reads them as findings. The drift figure in particular is the PI's call:
# how far a moored buoy wanders in a day is a property of this farm, this mooring spread
# and this weather, and no line of python knows it.
# --------------------------------------------------------------------------------------

#: Growth of positional uncertainty per day since a part was last OBSERVED (metres, 1 sigma).
#: PROVISIONAL. Chosen so a week-old fix is worth ~3.5 m, which is under a corridor width and
#: over a buoy radius -- i.e. it degrades to "useful but not decisive", which is the honest
#: shape. Measure it against two surveys of the same farm before believing it.
#:
#: KNOWN WRONG IN SHAPE, NOT ONLY IN VALUE (Ivan, 2026-08-20): "as the buoys in the farm are
#: tied together with the ropes they are not going to drift away from each other but rather
#: move with the current and get skewed", and the magnitude is not hundreds of metres absent
#: something drastic. Ageing each belief INDEPENDENTLY, as below, lets this ledger imagine a
#: farm whose buoys scatter -- the one thing a tied structure cannot do. It is conservative
#: in the wrong direction twice over: it over-widens each part, and it cannot use the
#: structure, so one confirmed buoy does not tighten any other.
#: The correct model is a shared farm-frame pose (translation + small rotation) carrying the
#: large uncertainty, a bounded shear/skew term, and a small per-part residual -- with scale
#: never fitted (a farm does not resize; SETTLED §3l). Planned as P8.0c; see the plan doc's
#: "(a2) THE FARM MOVES AS ONE TIED STRUCTURE". Until then this constant is the interim, and
#: it is documented as interim so nobody reads it as a finding.
BELIEF_DRIFT_M_PER_DAY = 0.5

#: Uncertainty of an AS-DESIGNED position -- a farm is built to a drawing by people in boats.
#: PROVISIONAL.
DESIGN_SIGMA_M = 3.0

#: How many sigma a detection may be from a belief and still be a candidate at all. Wider
#: than a textbook 3 on purpose: excluding the true part is unrecoverable here, whereas
#: carrying one extra candidate merely costs an `ambiguous` verdict that further looking
#: resolves.
GATE_SIGMA = 4.0

#: A candidate is DEAD when the evidence contradicts its prediction by more than this many
#: sigma of the evidence's own stated tolerance. Killing a true candidate is the one error
#: this layer cannot recover from, so the bar to kill is deliberately higher than the bar to
#: admit.
FALSIFY_SIGMA = 5.0


class LedgerRefusal(RuntimeError):
    """Raised with an operator-readable reason. Never caught and defaulted."""


# --------------------------------------------------------------------------------------
# Beliefs -- what we expect, where, and how old that expectation is
# --------------------------------------------------------------------------------------

#: The part vocabulary. Kept explicit rather than free-form strings so a typo is a KeyError
#: at build time instead of a part class that silently never matches anything.
KINDS = ("surface_buoy", "intermediate_buoy", "culture_line", "cross_line",
         "mooring", "anchor_block")


@dataclass(frozen=True)
class Belief:
    """Where we believe one part is, and -- inseparably -- how old that belief is.

    `source` is `as_designed` or `as_observed`. The distinction is not bookkeeping: a
    position from a drawing and a position from last week's survey fail differently, and an
    operator asked to trust one of them deserves to know which it is.
    """

    name: str
    kind: str
    xz: Tuple[float, float]
    source: str                          # as_designed | as_observed
    observed_at: Optional[float] = None  # epoch seconds; None => never observed
    sigma0_m: Optional[float] = None     # uncertainty AT the moment of observation

    def __post_init__(self):
        if self.kind not in KINDS:
            raise LedgerRefusal(
                f"{self.name}: unknown part kind {self.kind!r}. Known kinds: "
                f"{', '.join(KINDS)}. A part class nothing recognises would be counted "
                f"in no total and missed by every check.")
        if self.source == "as_observed" and self.observed_at is None:
            raise LedgerRefusal(
                f"{self.name}: source is 'as_observed' but no observation time was given. "
                f"An observation without a date cannot be aged, and an un-aged observation "
                f"is indistinguishable from a fresh one -- which is the whole failure this "
                f"class exists to prevent.")

    def base_sigma(self) -> float:
        if self.source == "as_designed":
            return DESIGN_SIGMA_M if self.sigma0_m is None else self.sigma0_m
        if self.sigma0_m is None:
            raise LedgerRefusal(
                f"{self.name}: an observed position with no stated accuracy. It is not "
                f"offered as a belief at all -- see the all-zero-covariance rule in "
                f"station_report.py, same reasoning one layer up.")
        return self.sigma0_m

    def age_days(self, now: float) -> Optional[float]:
        if self.observed_at is None:
            return None
        return max(0.0, (now - self.observed_at) / 86400.0)

    def sigma_at(self, now: float, weather_factor: float = 1.0) -> float:
        """Uncertainty NOW: what it was when measured, grown by how long ago that was.

        `weather_factor` multiplies the drift term only -- a storm moves things, it does not
        retroactively worsen the survey that measured them.
        """
        if weather_factor < 1.0:
            raise LedgerRefusal(
                "weather_factor < 1.0 would make a belief more certain because time passed. "
                "Weather can only widen a belief.")
        s0 = self.base_sigma()
        age = self.age_days(now)
        if age is None:
            return s0
        drift = BELIEF_DRIFT_M_PER_DAY * age * weather_factor
        return (s0 * s0 + drift * drift) ** 0.5

    def provenance(self, now: float, weather_factor: float = 1.0) -> str:
        """One sentence an operator can act on. Every number in it was measured or aged."""
        s = self.sigma_at(now, weather_factor)
        if self.source == "as_designed":
            return (f"{self.name}: as-designed position, never observed, "
                    f"sigma {s:.2f} m")
        age = self.age_days(now) or 0.0
        return (f"{self.name}: last observed {age:.1f} d ago at sigma "
                f"{self.base_sigma():.2f} m, now sigma {s:.2f} m")


# --------------------------------------------------------------------------------------
# Falsifiers -- what would rule a candidate OUT
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Falsifier:
    """A prediction that, if contradicted, kills the candidate carrying it.

    `kind` is machine-readable so the planner can ask "which manoeuvre produces evidence of
    this kind" without parsing prose. `sentence` is the same fact for a human, because an
    operator reading a refusal needs to know what the vehicle was trying to settle.

    Kinds this farm hands us cheaply, all geometric:
      * `line_spacing`      -- 13 m corridors against 26 m lines
      * `end_buoy_count`    -- how many buoys stand at the end of the line we are beside
      * `intermediate_at_half` -- the intermediate buoy sits at exactly t=0.5 of its side
      * `mooring_geometry`  -- moorings splay ~8 m outboard and are steeply inclined, so a
                               near-horizontal return at rope depth is not one
      * `cross_line_present`-- cross lines exist only at the ends
      * `global_fix`        -- an absolute position from a GPS fix on surfacing
    """

    kind: str
    predicts: float
    tolerance_m: float
    sentence: str

    def __post_init__(self):
        if self.tolerance_m <= 0.0:
            raise LedgerRefusal(
                f"falsifier {self.kind}: a tolerance of {self.tolerance_m} can never be "
                f"contradicted, so it is not a falsifier.")


@dataclass(frozen=True)
class Evidence:
    """One observation offered against the surviving candidates.

    `value` is compared with each candidate's falsifier of the same `kind`. `tolerance_m` is
    the EVIDENCE's own accuracy; the check uses both, so a sloppy measurement kills nothing
    and says so.
    """

    kind: str
    value: float
    tolerance_m: float
    detail: str = ""


@dataclass(frozen=True)
class Candidate:
    part: str
    distance_m: float
    sigma_m: float
    falsifiers: Tuple[Falsifier, ...]
    #: Set when this candidate has been ruled out, to the sentence that ruled it out. A dead
    #: candidate is KEPT, not deleted: "we considered the NE mooring and killed it with the
    #: surfacing fix" is a different and more useful record than never having considered it.
    dead_because: Optional[str] = None

    @property
    def alive(self) -> bool:
        return self.dead_because is None

    @property
    def normalised_distance(self) -> float:
        """Distance in units of its own uncertainty. This is the only score in this module,
        and it is a ratio of two measured quantities rather than an invented confidence."""
        if self.sigma_m <= 0.0:
            raise LedgerRefusal(f"{self.part}: zero sigma would divide by zero and read as "
                                f"a perfect match on the least trustworthy input there is.")
        return self.distance_m / self.sigma_m


# --------------------------------------------------------------------------------------
# Associations
# --------------------------------------------------------------------------------------

@dataclass
class Association:
    """One detection and every part it might be.

    Verdicts: `committed` once exactly one candidate survives; `ambiguous` while more than
    one does; `unexplained` when none does -- which is NOT an error, it is the residual, and
    the size of that residual is the honest input to whether this mission needs a reasoning
    model at all.
    """

    det_id: str
    xz: Tuple[float, float]
    candidates: List[Candidate] = field(default_factory=list)

    def add_candidate(self, cand: Candidate) -> None:
        if not cand.falsifiers:
            raise LedgerRefusal(
                f"{self.det_id}->{cand.part}: a candidate with no falsifier is a preference, "
                f"not a hypothesis. State what observation would rule it out.")
        self.candidates.append(cand)

    @property
    def alive(self) -> List[Candidate]:
        return [c for c in self.candidates if c.alive]

    @property
    def status(self) -> str:
        n = len(self.alive)
        if n == 1:
            return "committed"
        if n == 0:
            return "unexplained"
        return "ambiguous"

    @property
    def committed_part(self) -> Optional[str]:
        """The part this detection IS -- or None while anything else is still possible.

        Note what this deliberately does not do: it does not return the best candidate when
        several survive. `confirmed` means no surviving alternative (rule 2). A caller that
        wants the leader must ask for `leader()` and will get it labelled as a leader.
        """
        return self.alive[0].part if self.status == "committed" else None

    def leader(self) -> Optional[Candidate]:
        """The closest surviving candidate, for DISPLAY and for ranking what to look at
        next. Never for committing -- see `committed_part`."""
        return min(self.alive, key=lambda c: c.normalised_distance) if self.alive else None

    def apply(self, ev: Evidence) -> List[str]:
        """Rule out every surviving candidate this evidence contradicts.

        Returns the sentences of the candidates killed, so the caller can log WHY the world
        got simpler -- a hypothesis set that shrinks without saying why is unauditable.
        """
        killed: List[str] = []
        for i, cand in enumerate(self.candidates):
            if not cand.alive:
                continue
            for f in cand.falsifiers:
                if f.kind != ev.kind:
                    continue
                margin = (f.tolerance_m ** 2 + ev.tolerance_m ** 2) ** 0.5
                if abs(ev.value - f.predicts) > FALSIFY_SIGMA * margin:
                    why = (f"{cand.part} ruled out: {f.sentence} -- predicted "
                           f"{f.predicts:.2f}, observed {ev.value:.2f} "
                           f"(+/-{margin:.2f}){(' [' + ev.detail + ']') if ev.detail else ''}")
                    self.candidates[i] = Candidate(
                        cand.part, cand.distance_m, cand.sigma_m, cand.falsifiers, why)
                    killed.append(why)
                    break
        return killed

    def describe(self) -> str:
        if self.status == "committed":
            return f"{self.det_id}: {self.committed_part} (no surviving alternative)"
        if self.status == "unexplained":
            return (f"{self.det_id}: explained by nothing in the prior -- "
                    f"{len(self.candidates)} candidate(s) all ruled out"
                    if self.candidates else
                    f"{self.det_id}: matches no expected part within the gate")
        names = ", ".join(sorted(c.part for c in self.alive))
        return f"{self.det_id}: ambiguous({len(self.alive)}) -- one of [{names}]"


# --------------------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------------------

@dataclass
class LedgerReport:
    expected: Dict[str, int]
    committed: Dict[str, str]          # part -> det_id
    ambiguous: Dict[str, Sequence[str]]  # det_id -> candidate names
    unexplained: List[str]             # det_ids explained by nothing
    unseen: List[str]                  # expected parts no detection is committed to
    lines: List[str]                   # operator-readable, one per notable fact

    @property
    def residual(self) -> int:
        """The number of detections the prior cannot explain. This is the measurement that
        decides whether a reasoning model is worth its share of a 30 W budget (Ivan,
        2026-08-20) -- it is reported rather than argued about."""
        return len(self.unexplained)


class FarmLedger:
    """The expectation manifest, and the association layer that fills it in.

    Built from beliefs rather than from the prior file directly, so it can be driven from a
    test with four buoys and no YAML. `from_prior()` is the adapter.
    """

    def __init__(self, beliefs: Sequence[Belief], *, weather_factor: float = 1.0):
        if not beliefs:
            raise LedgerRefusal(
                "a ledger with no expectations would report every detection as unexplained "
                "and an empty farm as fully verified.")
        names = [b.name for b in beliefs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise LedgerRefusal(
                f"duplicate part name(s): {', '.join(dupes)}. Two parts sharing a name "
                f"cannot be told apart in any verdict this ledger produces.")
        self.beliefs: Dict[str, Belief] = {b.name: b for b in beliefs}
        self.weather_factor = weather_factor
        self.associations: Dict[str, Association] = {}

    # -- expectations ------------------------------------------------------------------
    def expected_counts(self) -> Dict[str, int]:
        out = {k: 0 for k in KINDS}
        for b in self.beliefs.values():
            out[b.kind] += 1
        return out

    def provenance(self, now: float) -> List[str]:
        return [self.beliefs[n].provenance(now, self.weather_factor)
                for n in sorted(self.beliefs)]

    # -- association -------------------------------------------------------------------
    def observe(self, det_id: str, xz: Tuple[float, float], now: float, *,
                kinds: Optional[Sequence[str]] = None,
                det_sigma_m: float = 0.5,
                falsifiers_for=None) -> Association:
        """Offer one detection and get back every part it might be.

        `kinds` restricts which part classes may match -- a buoy detection should not be
        allowed to match a mooring merely because it is close. `falsifiers_for(belief)` is
        supplied by the caller because the falsifiers are geometry this module does not own:
        the farm's line spacing, its buoy counts, its mooring splay. Every candidate must
        have at least one (rule 3), and `add_candidate` enforces it.
        """
        if det_id in self.associations:
            raise LedgerRefusal(f"detection id {det_id!r} offered twice; ids must be unique "
                                f"or evidence applied to one will silently move the other.")
        assoc = Association(det_id, xz)
        for name in sorted(self.beliefs):
            b = self.beliefs[name]
            if kinds is not None and b.kind not in kinds:
                continue
            sigma = (b.sigma_at(now, self.weather_factor) ** 2 + det_sigma_m ** 2) ** 0.5
            d = ((xz[0] - b.xz[0]) ** 2 + (xz[1] - b.xz[1]) ** 2) ** 0.5
            if d > GATE_SIGMA * sigma:
                continue
            fs = tuple(falsifiers_for(b)) if falsifiers_for else ()
            assoc.add_candidate(Candidate(name, d, sigma, fs))
        self.associations[det_id] = assoc
        return assoc

    def apply_evidence(self, ev: Evidence) -> List[str]:
        """Apply one observation to every open association. Returns what it killed."""
        killed: List[str] = []
        for assoc in self.associations.values():
            killed.extend(assoc.apply(ev))
        return killed

    # -- active disambiguation ---------------------------------------------------------
    def what_would_discriminate(self) -> List[Tuple[str, int]]:
        """Which kinds of evidence would collapse the most ambiguity, most first.

        This is the hook for Ivan's "let me go a bit further and see": the planner turns a
        falsifier kind into the manoeuvre that produces it, and prefers the one that kills
        the most candidates per metre. Returned as counts of *candidates that could die*,
        never as a promise that they will -- whether the evidence contradicts them is a fact
        about the world, discovered by going and looking.
        """
        tally: Dict[str, int] = {}
        for assoc in self.associations.values():
            if assoc.status != "ambiguous":
                continue
            for cand in assoc.alive:
                for f in cand.falsifiers:
                    tally[f.kind] = tally.get(f.kind, 0) + 1
        return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))

    def open_questions(self) -> List[str]:
        return [a.describe() for a in self.associations.values() if a.status == "ambiguous"]

    # -- the report --------------------------------------------------------------------
    def report(self, now: float, *, surveyed=None) -> LedgerReport:
        """The checklist. `surveyed(part_name) -> bool` says whether the vehicle actually
        looked where that part is believed to be; without it every unseen part would be
        reported as `missing`, which is the "we never went there" error the fourth verdict
        exists to prevent (SETTLED §3k)."""
        committed: Dict[str, str] = {}
        ambiguous: Dict[str, Sequence[str]] = {}
        unexplained: List[str] = []
        for det_id, a in self.associations.items():
            if a.status == "committed":
                part = a.committed_part
                if part in committed:
                    raise LedgerRefusal(
                        f"two detections ({committed[part]}, {det_id}) both committed to "
                        f"{part}. One of them is wrong and the ledger cannot tell which; "
                        f"re-open both rather than recording a farm with a duplicated part.")
                committed[part] = det_id
            elif a.status == "ambiguous":
                ambiguous[det_id] = sorted(c.part for c in a.alive)
            else:
                unexplained.append(det_id)

        unseen = [n for n in sorted(self.beliefs) if n not in committed]
        lines: List[str] = []
        exp = self.expected_counts()
        for kind in KINDS:
            if not exp[kind]:
                continue
            got = sum(1 for p in committed if self.beliefs[p].kind == kind)
            lines.append(f"{kind}: {got}/{exp[kind]} identified")
        for det_id, names in sorted(ambiguous.items()):
            lines.append(f"{det_id}: one of {len(names)} -- {', '.join(names)}")
        for n in unseen:
            if n in {c for names in ambiguous.values() for c in names}:
                continue        # still in play; not missing and not un-surveyed
            looked = surveyed(n) if surveyed else False
            lines.append(f"{n}: {'missing (looked, not found)' if looked else 'not_surveyed'}")
        if unexplained:
            lines.append(f"{len(unexplained)} detection(s) explained by nothing in the prior")
        return LedgerReport(exp, committed, ambiguous, unexplained, unseen, lines)

    # -- verdicts, in the vocabulary the rest of the package already speaks -------------
    def verdict(self, part: str, now: float, *, surveyed=None) -> str:
        """`confirmed | moved | missing | not_surveyed | ambiguous`.

        `confirmed` requires that some detection COMMITTED to this part -- i.e. that no
        alternative survived. `moved` is a commitment whose distance exceeds the belief's
        own aged sigma: identified, but not where we believed it was, which is a normal and
        expected outcome at a farm and must not read as an error.
        """
        if part not in self.beliefs:
            raise LedgerRefusal(f"no such part {part!r} in the manifest.")
        for a in self.associations.values():
            if a.status == "ambiguous" and any(c.part == part for c in a.alive):
                return "ambiguous"
            if a.committed_part == part:
                cand = a.alive[0]
                return "moved" if cand.distance_m > cand.sigma_m else "confirmed"
        if surveyed and surveyed(part):
            return "missing"
        return "not_surveyed"
