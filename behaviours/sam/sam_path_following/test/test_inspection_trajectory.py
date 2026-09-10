"""The inspection orbit as an MPC trajectory — letter D, strategy §5.3.

WHAT IS GUARDED, and why each one is a property rather than a decoration:

  * THE PIVOT IS THE PUBLISHED MANOEUVRE, not a re-implementation. `build_on_spot_path` is
    IMPORTED, and the test asserts its signature survives in the output: alternating surge sign
    inside each pivot, a linear yaw ramp, zero velocity at the end. A second implementation of a
    manoeuvre validated in a field test (Bhat, Stenius & Miao, JOE 46(4) 2021 §V.D) is a second
    thing that can drift from the paper.
  * EVERY STATION IS A STOP. The MPC brakes to a zero-velocity reference; a station that is not
    a stop is not a station, and the capture burst would be taken while moving.
  * THE RING REPRODUCES THE STATION POSES it was given — position and heading.
  * THE YAW RAMP TOTALS THE BEARING CHANGE, by the SHORTEST signed turn.
  * THE COLUMNS ARE `create_turbo_turn_path`'s own, in its order, read off that file.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider \\
        smarc2/behaviours/sam/sam_path_following/test/test_inspection_trajectory.py
"""
import ast
import math
import pathlib
import sys

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from sam_path_following import inspection_trajectory as IT      # noqa: E402
from sam_path_following import create_turbo_turn_path as TT      # noqa: E402

R = 3.5
N = 12


def ring(n=N, radius=R, depth=6.0):
    """N stations on a circle, each FACING the centre — the geometry the planner emits."""
    out = []
    for i in range(n):
        b = 360.0 * i / n
        out.append(IT.Station(index=i,
                              x=radius * math.cos(math.radians(b)),
                              y=radius * math.sin(math.radians(b)),
                              z=depth,
                              heading_deg=(b + 180.0) % 360.0))
    return out


# ------------------------------------------------------------------ the columns
def test_the_columns_are_the_turbo_turn_generators_own_in_its_order():
    """Read off `create_turbo_turn_path.main()`'s own list, not copied into the test."""
    src = ast.parse((PKG / "sam_path_following" / "create_turbo_turn_path.py").read_text())
    theirs = None
    for node in ast.walk(src):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "columns"
                and isinstance(node.value, ast.List)):
            theirs = [e.value for e in node.value.elts]
    assert theirs, "create_turbo_turn_path.main() no longer builds a `columns` list"
    assert IT.COLUMNS == theirs, f"columns drifted: {IT.COLUMNS} vs {theirs}"


def test_the_pivot_builder_is_IMPORTED_and_not_re_implemented():
    """The alternating-surge pivot is a PUBLISHED manoeuvre (JOE 2021 §V.C) validated in a field
    test. A local copy is a copy that can drift from the paper."""
    tree = ast.parse((PKG / "sam_path_following" / "inspection_trajectory.py").read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "create_turbo_turn_path" in n.module:
            imported |= {a.name for a in n.names}
    assert "build_on_spot_path" in imported
    assert "yaw_to_quaternion" in imported
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_on_spot_path" not in defined and "yaw_to_quaternion" not in defined


# ------------------------------------------------------------------ the ring
def test_a_twelve_station_ring_reproduces_the_station_poses():
    sts = ring()
    x, y, z, yaw, u, _dr = IT.build_inspection_ring(sts)
    for st in sts:
        d = np.hypot(x - st.x, y - st.y)
        assert d.min() < 0.01, f"station {st.index} never appears in the trajectory"
        i = int(np.argmin(d))
        # the heading at the END of that station's pivot must face the target
        j = i
        while j + 1 < len(x) and abs(x[j + 1] - st.x) < 1e-9 and abs(y[j + 1] - st.y) < 1e-9:
            j += 1
        got = math.degrees(yaw[j]) % 360.0
        assert abs(((got - st.heading_deg + 180.0) % 360.0) - 180.0) < 1.0, \
            f"station {st.index}: pivot ended at {got:.1f}°, wanted {st.heading_deg:.1f}°"


def test_every_station_is_a_STOP():
    """The MPC brakes to a zero-velocity reference; a station that is not a stop is not a
    station, and the capture burst would be taken while moving."""
    sts = ring()
    x, y, z, _yaw, u, _dr = IT.build_inspection_ring(sts)
    for st in sts:
        at = np.nonzero((np.hypot(x - st.x, y - st.y) < 0.01))[0]
        assert at.size, f"station {st.index} is not in the trajectory"
        assert abs(u[at[-1]]) < 1e-9, \
            f"station {st.index} ends at u = {u[at[-1]]}, not at a stop"


def test_the_surge_sign_alternates_inside_each_pivot():
    """THE SIGNATURE OF THE MANOEUVRE (JOE 2021 §V.C): the propellers are cycled between maximum
    positive and negative so the surge thrust cancels and the thrust vector does the turning. A
    pivot whose surge does not alternate is a forward-motion turn wearing its name."""
    sts = ring(n=4)
    x, y, _z, _yaw, u, _dr = IT.build_inspection_ring(sts, pivot_waypoints=8)
    for st in sts:
        at = np.nonzero(np.hypot(x - st.x, y - st.y) < 1e-9)[0]
        seg = u[at]
        nz = seg[np.abs(seg) > 1e-12]
        assert nz.size >= 4, f"station {st.index}: only {nz.size} moving rows in the pivot"
        signs = np.sign(nz)
        assert np.all(signs[1:] != signs[:-1]), \
            f"station {st.index}: surge does not alternate ({nz})"


def test_the_rudder_sign_alternates_too():
    sts = ring(n=3)
    x, y, _z, _yaw, _u, dr = IT.build_inspection_ring(sts, pivot_waypoints=8)
    at = np.nonzero(np.hypot(x - sts[0].x, y - sts[0].y) < 1e-9)[0]
    seg = dr[at]
    nz = seg[np.abs(seg) > 1e-12]
    assert nz.size >= 4
    assert np.all(np.sign(nz)[1:] != np.sign(nz)[:-1]), nz


def test_the_yaw_ramp_totals_the_bearing_change_and_is_linear():
    sts = ring(n=4)
    x, y, _z, yaw, _u, _dr = IT.build_inspection_ring(sts, pivot_waypoints=9)
    at = np.nonzero(np.hypot(x - sts[1].x, y - sts[1].y) < 1e-9)[0]
    seg = np.degrees(yaw[at])
    total = seg[-1] - seg[0]
    want = ((sts[1].heading_deg - sts[0].heading_deg + 180.0) % 360.0) - 180.0
    assert total == pytest.approx(want, abs=1e-6), (total, want)
    steps = np.diff(seg)
    # The FIRST step is zero by construction: the translate's arrival row sits exactly on the
    # station, so it is selected here too and it carries the arrival heading — the same heading
    # the pivot starts from. Every step inside the pivot itself must be equal.
    assert steps[0] == pytest.approx(0.0, abs=1e-9)
    ramp = steps[1:]
    assert np.allclose(ramp, ramp[0]), f"the yaw ramp is not linear: {ramp}"
    assert abs(ramp[0]) > 1e-6, "the pivot does not turn at all"


def test_the_shortest_signed_turn_is_taken():
    """Turning 350° to avoid turning −10° is ten times the pivot for the same result."""
    sts = [IT.Station(0, 0.0, 0.0, 5.0, 5.0), IT.Station(1, 1.0, 0.0, 5.0, 355.0)]
    x, y, _z, yaw, _u, _dr = IT.build_inspection_ring(sts, pivot_waypoints=5,
                                                      translate_waypoints=2)
    at = np.nonzero(np.hypot(x - 1.0, y - 0.0) < 1e-9)[0]
    total = math.degrees(yaw[at[-1]] - yaw[at[0]])
    assert total == pytest.approx(-10.0, abs=1e-6), total


def test_a_ring_of_one_station_is_refused():
    with pytest.raises(ValueError):
        IT.build_inspection_ring(ring(n=1))


# ------------------------------------------------------------------ the rows
def test_the_rows_carry_the_quaternion_of_the_yaw_and_nothing_actuator_shaped():
    sts = ring(n=4)
    x, y, z, yaw, u, _dr = IT.build_inspection_ring(sts)
    rows = IT.to_rows(x, y, z, yaw, u)
    assert len(rows) == len(x)
    assert list(rows[0]) == IT.COLUMNS
    for i in (0, len(rows) // 2, len(rows) - 1):
        q = TT.yaw_to_quaternion(float(yaw[i]))
        assert rows[i]["q0"] == pytest.approx(float(q[0]))
        assert rows[i]["q3"] == pytest.approx(float(q[3]))
        assert rows[i]["q1"] == 0.0 and rows[i]["q2"] == 0.0
    # no actuator columns: neutral defaults are applied downstream in path_client /
    # ActionServerDiveSub, and a trajectory that carried them would be commanding from a plan
    for bad in ("vbs", "lcg", "rpm", "rudder", "stern"):
        assert not any(bad in c for c in IT.COLUMNS)


def test_pandas_is_optional_and_the_rows_are_the_product():
    sts = ring(n=3)
    rows = IT.to_rows(*IT.build_inspection_ring(sts)[:5])
    df = IT.to_dataframe(rows)
    if df is None:
        pytest.skip("pandas is not on this machine — which is the point of the fallback")
    assert list(df.columns) == IT.COLUMNS
    assert len(df) == len(rows)


# ------------------------------------------------------------------ the frame
def test_the_planner_stations_convert_to_local_metres_about_the_origin():
    rows = [{"lat": 58.8215, "lon": 17.6348, "depth_m": 6.0, "heading_deg": 90.0},
            {"lat": 58.8215 + 10.0 / 110540.0, "lon": 17.6348, "depth_m": 6.0,
             "heading_deg": 270.0}]
    sts = IT.stations_from_planner(rows, origin_lat=58.8215, origin_lon=17.6348)
    assert sts[0].x == pytest.approx(0.0, abs=1e-6) and sts[0].y == pytest.approx(0.0, abs=1e-6)
    assert sts[1].x == pytest.approx(10.0, abs=0.01)
    assert sts[1].y == pytest.approx(0.0, abs=1e-6)
    assert sts[1].heading_deg == 270.0


def test_the_easting_carries_the_cos_lat_term():
    """At 58.8° north a degree of longitude is 111320 x cos(58.8) = 57.7 km, not 111.3 km.
    Dropping the term nearly DOUBLES every east offset — a 3.5 m ring becomes a 6.7 m one, and
    the stations end up outside the camera's usable standoff (the R0 model's whole point)."""
    dlon = 10.0 / (111320.0 * math.cos(math.radians(58.8215)))
    sts = IT.stations_from_planner(
        [{"lat": 58.8215, "lon": 17.6348 + dlon, "depth_m": 6.0, "heading_deg": 0.0}],
        origin_lat=58.8215, origin_lon=17.6348)
    assert sts[0].y == pytest.approx(10.0, abs=0.02), (
        f"10 m of easting came back as {sts[0].y:.2f} m — the cos(lat) term is missing")
