#!/usr/bin/env python3
"""The inspection orbit as a `TrajectoryMPC`-shaped table — letter D, strategy §5.3.

WHAT THIS BUILDS. The close-inspection ring is N stations on a circle around the target, each
faced by a TURBO-TURN PIVOT — the alternating-surge, thrust-vectored manoeuvre published in
Bhat, Stenius & Miao, IEEE JOE 46(4) 2021 §V.C — with a translate between them. This file turns
the planner's station list into the same columns `create_turbo_turn_path.py` writes, so the
existing MPC path client can fly it without learning a new format.

**IT IMPORTS `create_turbo_turn_path`'s OWN BUILDERS AND DOES NOT RE-IMPLEMENT THEM.** The
pivot's alternating ±surge and ±rudder and its linear yaw ramp are `build_on_spot_path`'s, the
between-station legs are `build_waypoints_path`'s shape, and the quaternion conversion is
`yaw_to_quaternion`. A second implementation of a published manoeuvre is a second thing that can
drift from the paper, and the paper is what the field test in §V.D validated.

**NOTHING HERE COMMANDS ANYTHING.** It returns a table. Whether the MPC flies it is option M1's
question and it is UNFLOWN: acados is on the real Orin and is UNMEASURED on vm1 (SETTLED §3ad),
and this session had neither.

`matplotlib`/`pandas` are optional exactly as in the source: the table is built as plain lists
and only turned into a DataFrame if pandas is importable, so a vehicle without it can still
produce the rows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sam_path_following.create_turbo_turn_path import build_on_spot_path, yaw_to_quaternion

#: The columns `create_turbo_turn_path.main()` writes, in its order. Imported in spirit rather
#: than in fact (they are built inline in that file's `main`), and pinned by
#: `test_inspection_trajectory.py` against the source so the two cannot drift.
COLUMNS = ["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r"]


@dataclass(frozen=True)
class _OnSpotArgs:
    """The subset of `create_turbo_turn_path`'s argparse namespace `build_on_spot_path` reads.

    A dataclass rather than a dict so a missing field is an AttributeError here instead of a
    KeyError inside the imported builder, where the traceback would point at code this file did
    not write.
    """

    n_waypoints: int
    total_yaw_deg: float
    center_x: float
    center_y: float
    center_z: float
    surge_speed: float
    rudder_angle_deg: float


@dataclass(frozen=True)
class Station:
    """One ring station, in the LOCAL frame the trajectory is expressed in (metres, x north)."""

    index: int
    x: float
    y: float
    z: float
    heading_deg: float


def stations_from_planner(rows: Sequence[Dict[str, Any]], *, origin_lat: float,
                          origin_lon: float) -> List[Station]:
    """Planner station dicts (lat/lon/depth/heading) -> local metres about `origin`.

    The planner works in geographic coordinates because that is what the action server reads;
    the MPC works in local metres because that is what a trajectory is. Converting HERE, once,
    with the constants named, is better than either side carrying both.
    """
    out: List[Station] = []
    for i, r in enumerate(rows):
        north = (float(r["lat"]) - origin_lat) * 110540.0
        east = ((float(r["lon"]) - origin_lon) * 111320.0
                * math.cos(math.radians(origin_lat)))
        out.append(Station(index=i, x=north, y=east, z=float(r.get("depth_m", 0.0)),
                           heading_deg=float(r.get("heading_deg", 0.0))))
    return out


def _pivot(start_yaw_deg: float, end_yaw_deg: float, st: Station, *, n_waypoints: int,
           surge_speed: float, rudder_angle_deg: float):
    """One turbo-turn pivot at a station, from `build_on_spot_path` — the published manoeuvre.

    The builder ramps yaw from 0 to `total_yaw_deg`; the ring needs it to ramp from the heading
    the vehicle arrives with to the heading that faces the target, so the SHORTEST signed turn is
    passed as the total and the start heading is added back. Shortest signed: turning 350° to
    avoid turning −10° would be ten times the pivot for the same result.
    """
    delta = ((float(end_yaw_deg) - float(start_yaw_deg) + 180.0) % 360.0) - 180.0
    args = _OnSpotArgs(n_waypoints=int(n_waypoints), total_yaw_deg=delta,
                       center_x=st.x, center_y=st.y, center_z=st.z,
                       surge_speed=float(surge_speed),
                       rudder_angle_deg=float(rudder_angle_deg))
    x, y, z, yaw, u, dr = build_on_spot_path(args)
    return x, y, z, yaw + math.radians(start_yaw_deg), u, dr


def build_inspection_ring(stations: Sequence[Station], *, pivot_waypoints: int = 8,
                          surge_speed: float = 0.2, rudder_angle_deg: float = 5.0,
                          translate_waypoints: int = 3,
                          start_heading_deg: Optional[float] = None):
    """The whole ring: pivot, translate, pivot, translate, ... as one trajectory.

    Returns `(x, y, z, yaw, u, dr)` as numpy arrays, in the same shape the turbo-turn builders
    return, so everything downstream of them applies unchanged.

    THE TRANSLATE BETWEEN STATIONS is a straight run of waypoints at the station's own heading,
    with u ramped to zero at arrival — the `waypoints` mode's rule, and the reason it matters is
    the same one `build_on_spot_path` gives for its own last row: the MPC brakes to a stop at a
    zero-velocity reference, and a station that is not a stop is not a station.
    """
    if len(stations) < 2:
        raise ValueError("a ring of fewer than two stations is not a ring")
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    yaws: List[float] = []
    us: List[float] = []
    drs: List[float] = []
    heading = (stations[0].heading_deg if start_heading_deg is None else float(start_heading_deg))

    for k, st in enumerate(stations):
        if k > 0:
            prev = stations[k - 1]
            # translate: straight from the previous station to this one, holding the heading the
            # vehicle will need on arrival, zero velocity at the end.
            n = max(2, int(translate_waypoints))
            for j in range(1, n + 1):
                t = j / float(n)
                xs.append(prev.x + t * (st.x - prev.x))
                ys.append(prev.y + t * (st.y - prev.y))
                zs.append(prev.z + t * (st.z - prev.z))
                yaws.append(math.radians(heading))
                us.append(0.0 if j == n else float(surge_speed))
                drs.append(0.0)
        px, py, pz, pyaw, pu, pdr = _pivot(heading, st.heading_deg, st,
                                           n_waypoints=pivot_waypoints,
                                           surge_speed=surge_speed,
                                           rudder_angle_deg=rudder_angle_deg)
        xs.extend(px.tolist())
        ys.extend(py.tolist())
        zs.extend(pz.tolist())
        yaws.extend(pyaw.tolist())
        us.extend(pu.tolist())
        drs.extend(pdr.tolist())
        heading = st.heading_deg

    return (np.asarray(xs), np.asarray(ys), np.asarray(zs),
            np.asarray(yaws), np.asarray(us), np.asarray(drs))


def to_rows(x, y, z, yaw, u) -> List[Dict[str, float]]:
    """The `TrajectoryMPC`-shaped rows, in `create_turbo_turn_path`'s own column order.

    Actuator states (VBS, LCG, stern, rudder, RPM) are omitted for the same reason that file
    gives: neutral defaults are applied downstream in `path_client` / `ActionServerDiveSub`, and
    a trajectory that carried them would be commanding actuators from a plan.
    """
    # PER SCALAR, exactly as `create_turbo_turn_path.main()` calls it
    # (`np.array([yaw_to_quaternion(yw) for yw in yaw])`). The function's `q1`/`q2` are python
    # floats, so handing it an ARRAY produces a ragged (4,) object array rather than an (N,4)
    # one — measured, and the reason this loop is not vectorised.
    q = np.array([yaw_to_quaternion(float(yw)) for yw in np.asarray(yaw)])
    rows = []
    for i in range(len(x)):
        row = {c: 0.0 for c in COLUMNS}
        row["x"], row["y"], row["z"] = float(x[i]), float(y[i]), float(z[i])
        row["q0"], row["q1"] = float(q[i, 0]), float(q[i, 1])
        row["q2"], row["q3"] = float(q[i, 2]), float(q[i, 3])
        row["u"] = float(u[i])
        rows.append(row)
    return rows


def to_dataframe(rows: Sequence[Dict[str, float]]):
    """A pandas DataFrame if pandas is here, else None with nothing pretended.

    Optional exactly as in the source: the rows above are the product, and a machine without
    pandas can still produce them.
    """
    try:
        import pandas as pd
    except ImportError:                                        # pragma: no cover
        return None
    return pd.DataFrame(list(rows), columns=COLUMNS)
