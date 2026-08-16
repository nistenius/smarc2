#!/usr/bin/env python3
"""Reading `farm_prior.yaml` — the only way farm geometry enters this package.

No node in this package holds a buoy coordinate, a rope depth or a line bearing. They
come from the generated prior, which is produced from the site's own measurements by
`data-cube/scripts/kristineberg-site/make_farm_prior.py` and guarded by
`test_farm_prior_is_generated.py`. That is spec §7's single-sources rule applied to the
one thing this mission is entirely about.

The loader is strict on purpose:

  * a missing file is FATAL and names the path and the generator. It does not fall back
    to a built-in farm — an invented prior would produce a mission that flies confidently
    to the wrong place, which is exactly the class of failure `_required_env()` was
    added to `vehicle_services.py` to stop (SETTLED §3, no invented machine identity).
  * a prior without the generator's own `_generated` block is REFUSED, because that is
    what a hand-written or hand-edited file looks like.
  * `rope_depth_m` is read as positive-down and cross-checked against the Unity y in the
    same file. Half this project's sign bugs are a depth that was one convention in one
    file and the other convention in the next.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FarmPrior:
    path: str
    buoys: Dict[str, Tuple[float, float]]          # name -> Unity (x, z)
    moored: Sequence[str]
    intermediate: Sequence[str]
    culture_lines: List[Tuple[str, Tuple[float, float], Tuple[float, float], float]]
    rope_depth_m: float                            # POSITIVE-DOWN metres
    seabed_depth_range_m: Tuple[float, float]
    centre_xz: Tuple[float, float]
    approx_xz: Tuple[float, float]
    approx_uncertainty_m: float
    scan_speed_ms: float
    encircle_standoff_m: float
    lane: Dict[str, object]
    caveats: List[str]

    @property
    def n_buoys(self) -> int:
        return len(self.buoys)

    def line_endpoints(self) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
        """The shape `localizer.fit_culture_lines` wants."""
        return [(n, a, b) for n, a, b, _ in self.culture_lines]


class PriorRefusal(RuntimeError):
    """Raised with an operator-readable reason. Never caught and defaulted."""


def default_prior_path() -> Optional[str]:
    """Where the prior normally lives, from the environment or the repo layout.

    `FARM_PRIOR_PATH` wins. Otherwise the repo-relative location is computed from this
    file's own position, which works from a source checkout and from a colcon install
    only if the package was installed with the data-cube tree beside it — so a node that
    finds nothing says so rather than guessing.
    """
    env = os.environ.get("FARM_PRIOR_PATH")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # .../smarc2/perception/sam/sam_farm_inspection/sam_farm_inspection/ -> repo root
    repo = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    cand = os.path.join(repo, "data-cube", "asko_site_package", "site_registry",
                        "kristineberg_farm_prior.yaml")
    return cand if os.path.exists(cand) else None


def load_farm_prior(path: Optional[str] = None) -> FarmPrior:
    path = path or default_prior_path()
    if not path:
        raise PriorRefusal(
            "no farm prior found. Set FARM_PRIOR_PATH, or generate one with "
            "`python3 data-cube/scripts/kristineberg-site/make_farm_prior.py`. This node "
            "will not substitute a built-in farm: a mission flown against an invented "
            "prior goes confidently to the wrong place.")
    if not os.path.exists(path):
        raise PriorRefusal(f"farm prior not found at {path} — generate it with "
                           f"make_farm_prior.py; it is not hand-written.")
    try:
        import yaml
    except ImportError as e:      # pragma: no cover - deployment problem, not logic
        raise PriorRefusal(f"PyYAML is needed to read the farm prior ({e})")

    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "_generated" not in doc:
        raise PriorRefusal(
            f"{path} has no `_generated` block, so it was not produced by "
            f"make_farm_prior.py. A hand-written prior is exactly what the generator "
            f"exists to prevent; regenerate it.")

    farm = doc.get("farm") or {}
    buoys = {b["name"]: (float(b["unity_x"]), float(b["unity_z"]))
             for b in farm.get("buoys", [])}
    if not buoys:
        raise PriorRefusal(f"{path} contains no buoys")

    depth = float(farm["rope_depth_m"])
    unity_y = float(farm["rope_unity_y"])
    if depth <= 0 or abs(depth + unity_y) > 1e-6:
        raise PriorRefusal(
            f"rope depth conventions disagree in {path}: rope_depth_m = {depth} "
            f"(positive-down) and rope_unity_y = {unity_y}. They must be negatives of "
            f"each other; a sign error here puts every lane on the wrong side of the ropes.")

    lines = []
    for ln in farm.get("culture_lines", []):
        a, b = ln["nodes"][0], ln["nodes"][-1]
        if a not in buoys or b not in buoys:
            raise PriorRefusal(f"culture line references unknown buoy(s) {a}/{b}")
        lines.append((f"{a}->{b}", buoys[a], buoys[b], float(ln["bearing_grid_deg"])))

    ap = doc.get("approximate_position") or {}
    prof = doc.get("profile") or {}
    sonar = doc.get("sonar") or {}
    lanes = sonar.get("lanes") or {}
    # Prefer a lane the current beam can actually fly. `as_shipped` is the simulator's
    # own beam; if it is not flyable the caller is handed the refusal, not the
    # alternative — choosing `proposed` here would silently plan a mission the sensor in
    # the scene cannot execute.
    lane = lanes.get("as_shipped", {})

    seabed = farm.get("seabed_depth_range_m") or [0.0, 0.0]
    return FarmPrior(
        path=path,
        buoys=buoys,
        moored=[b["name"] for b in farm.get("buoys", []) if b.get("moored")],
        intermediate=[b["name"] for b in farm.get("buoys", []) if b.get("intermediate")],
        culture_lines=lines,
        rope_depth_m=depth,
        seabed_depth_range_m=(float(seabed[0]), float(seabed[-1])),
        centre_xz=tuple(float(v) for v in farm.get("centre_unity_xz", (0.0, 0.0))),
        approx_xz=tuple(float(v) for v in ap.get("unity_xz", farm.get("centre_unity_xz", (0.0, 0.0)))),
        approx_uncertainty_m=float(ap.get("uncertainty_radius_m", 0.0)),
        scan_speed_ms=float(prof.get("scan_speed_ms", 0.0)),
        encircle_standoff_m=float(prof.get("encircle_standoff_m", 0.0)),
        lane=dict(lane),
        caveats=list(doc.get("caveats") or []),
    )
