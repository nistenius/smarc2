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
class GeoMap:
    """The prior's own linear metres -> WGS84 map, and what it costs.

    Not a convenience: a planner that has to turn metres into waypoint lat/lon and has no
    projection library reaches for metres/cos(lat), which ignores the difference between
    Unity's +Z (UTM 32N GRID north) and true north — 2.081 deg at Kristineberg, i.e. 8.6 m
    across the 237 m transit. This map is the full 2x2 Jacobian, fitted by pyproj in the
    generator, and it carries its own measured worst-case error so a consumer can decide
    whether it is good enough instead of hoping.
    """

    origin_lat: float
    origin_lon: float
    dlat_dx: float          # degrees per metre (NOT microdegrees; converted on load)
    dlat_dz: float
    dlon_dx: float
    dlon_dz: float
    max_error_m: float
    basis: str

    def to_latlon(self, x: float, z: float) -> Tuple[float, float]:
        return (self.origin_lat + self.dlat_dx * x + self.dlat_dz * z,
                self.origin_lon + self.dlon_dx * x + self.dlon_dz * z)


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
    #: How far below the water line each buoy reaches — the only part of it a side scan
    #: can ever see. Decides the encircle depth; see `encircle`.
    buoy_extents_m: Dict[str, float]
    hull_xz: List[Tuple[float, float]]
    perimeter_order: List[str]
    launch_xz: Tuple[float, float]
    launch_latlon: Tuple[float, float]
    transit_speed_ms: float
    r_stop_at_scan_speed_m: float
    encircle: Dict[str, object]
    geo: GeoMap

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

    # Blocks added 2026-08-17 for the P5 planner. A prior that predates them is REFUSED
    # rather than defaulted: the planner would otherwise fall back to an invented geo map
    # or an invented encircle depth, and both fail silently — the mission flies and sees
    # nothing. Regenerating is one command; guessing is a wrong survey.
    geo_d = doc.get("geo")
    if not isinstance(geo_d, dict):
        raise PriorRefusal(
            f"{path} has no `geo` block. It was generated before 2026-08-17 and the planner "
            f"will not invent a metres->lat/lon conversion (the obvious one, metres/cos(lat), "
            f"ignores grid-vs-true north and is 8.6 m wrong over this transit). Regenerate "
            f"with make_farm_prior.py.")
    enc = (doc.get("sonar") or {}).get("encircle")
    if not isinstance(enc, dict):
        raise PriorRefusal(
            f"{path} has no `sonar.encircle` block, so nothing has checked whether the "
            f"encircle depth can see the buoys at all. Regenerate with make_farm_prior.py.")

    extents = {b["name"]: float(b["submerged_extent_m"])
               for b in farm.get("buoys", []) if "submerged_extent_m" in b}
    if len(extents) != len(buoys):
        raise PriorRefusal(
            f"{path} does not give every buoy a `submerged_extent_m`. That is how far below "
            f"the water line the buoy reaches, and it is the only part of it a side scan can "
            f"see; without it the encircle depth is a guess. Regenerate.")

    launch = doc.get("launch") or {}
    prof = doc.get("profile") or {}
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
        buoy_extents_m=extents,
        hull_xz=[(float(p[0]), float(p[1])) for p in farm.get("buoy_hull_unity_xz", [])],
        perimeter_order=list(farm.get("perimeter_order") or []),
        launch_xz=(float(launch.get("unity_xz", (0.0, 0.0))[0]),
                   float(launch.get("unity_xz", (0.0, 0.0))[1])),
        launch_latlon=(float(launch.get("lat", 0.0)), float(launch.get("lon", 0.0))),
        transit_speed_ms=float(prof.get("transit_speed_ms", 0.0)),
        r_stop_at_scan_speed_m=float(prof.get("r_stop_at_scan_speed_m", 0.0)),
        encircle=dict(enc),
        geo=GeoMap(
            origin_lat=float(geo_d["origin_lat"]),
            origin_lon=float(geo_d["origin_lon"]),
            # Stored as MICRODEGREES per metre so the generator's 6-decimal YAML rounding
            # does not throw away four significant figures. Converted once, here.
            dlat_dx=float(geo_d["dlat_dx_udeg_per_m"]) * 1e-6,
            dlat_dz=float(geo_d["dlat_dz_udeg_per_m"]) * 1e-6,
            dlon_dx=float(geo_d["dlon_dx_udeg_per_m"]) * 1e-6,
            dlon_dz=float(geo_d["dlon_dz_udeg_per_m"]) * 1e-6,
            max_error_m=float(geo_d.get("max_error_m", 0.0)),
            basis=str(geo_d.get("basis", "unstated")),
        ),
    )
