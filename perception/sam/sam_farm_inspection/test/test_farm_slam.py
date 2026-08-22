"""P8.1 — the farm-relative SLAM graph, against synthetic surveys with known truth.

WHAT IS BEING PROVEN, AND WHAT IS DELIBERATELY NOT
==================================================

Proven here: the paper's FORMULATION is what gets built (one landmark per detection, its prior
taken from the originating line, long along it and narrow across it), the geometry refuses
rather than clamps, and the solve pulls a drifting vehicle back onto the structure it can see.

NOT proven here, and it must not be claimed anywhere: that any of this works on the vehicle.
Nothing in this module has seen a sonar. The measurements are synthetic, generated from a truth
trajectory, and a synthetic survey can only ever falsify — never confirm.

THE STRUCTURAL CHECKS DRIVE THE REAL OBJECT. `graph` is exposed as data precisely so "one
landmark per detection" can be asserted about the graph rather than inferred from optimised
numbers, which cannot tell a right answer from a right answer for the wrong reason.

Run: PYTHONPYCACHEPREFIX=/tmp/pyc python3 -m pytest -p no:cacheprovider test/test_farm_slam.py -q
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam_farm_inspection.farm_slam import (            # noqa: E402
    ARMABLE_VERDICTS, BATCH, BUOY, GTSAM_AVAILABLE, ISAM2, PORT, ROPE, SONAR3D, STARBOARD,
    Buoy, Detection, FarmSlam, FarmSlamMap, Odometry, RopeLine, SlamConfig, SlamRefusal,
    apply_correction,
    UnavailableBackend, ground_range, rope_prior_covariance, verified_map_to_slam_map,
)

# Kristineberg's two culture lines, at their real spacing. Metres in the prior's own XZ frame,
# translated to start at the origin so the numbers in the test read.
LINE_W = RopeLine("line_west", a=(0.0, 0.0), b=(0.0, 40.0), depth_m=2.0)
LINE_E = RopeLine("line_east", a=(16.0, 0.0), b=(16.0, 40.0), depth_m=2.0)
MAP = FarmSlamMap(lines=(LINE_W, LINE_E),
                  buoys=(Buoy("C1", (0.0, 0.0), 0.5), Buoy("C2", (0.0, 40.0), 0.5),
                         Buoy("C3", (16.0, 40.0), 0.5), Buoy("C4", (16.0, 0.0), 0.5)),
                  source="synthetic")

BACKENDS_HERE = [BATCH] + ([ISAM2] if GTSAM_AVAILABLE else [])


def _slam(backend=BATCH, **kw):
    return FarmSlam(MAP, backend=backend, **kw)


# ---- Eq. 1: slant -> ground, and what it refuses ---------------------------------------------

def test_ground_range_is_the_flat_bottom_conversion():
    # 3-4-5: a vehicle 3 m above a rope, 5 m of slant, is 4 m away horizontally.
    assert ground_range(5.0, vehicle_depth_m=-1.0, target_depth_m=2.0) == pytest.approx(4.0)


def test_a_target_at_the_vehicles_own_depth_gives_the_slant_range_back():
    assert ground_range(7.5, 2.0, 2.0) == pytest.approx(7.5)


def test_an_impossible_geometry_is_refused_not_clamped():
    """`max(0, ...)` here plants a landmark at the vehicle's own feet and calls it a rope."""
    with pytest.raises(SlamRefusal) as e:
        ground_range(1.0, vehicle_depth_m=0.0, target_depth_m=5.0)
    assert "impossible geometry" in str(e.value)
    assert "5.000" in str(e.value) and "1.000" in str(e.value), "name both numbers"


def test_a_negative_slant_range_is_refused_separately():
    with pytest.raises(SlamRefusal) as e:
        ground_range(-1.0, 0.0, 2.0)
    assert "negative" in str(e.value)


# ---- the rope prior IS the method -------------------------------------------------------------

def test_the_rope_prior_is_long_along_the_line_and_narrow_across_it():
    cov = rope_prior_covariance(LINE_W, SlamConfig())      # runs north, i.e. along +y
    # Along (+y) must be enormous; across (+x) must be about the paper's own 1.0 m rope RMSE.
    assert math.sqrt(cov[1][1]) == pytest.approx(20.0, rel=1e-6)   # 0.5 * 40 m
    assert math.sqrt(cov[0][0]) == pytest.approx(1.0, rel=1e-6)


def test_the_prior_rotates_with_the_line_rather_than_with_the_axes():
    """A diagonal covariance would make every rope prior axis-aligned and stop it being about
    the rope at all. A 45-degree line must produce off-diagonal terms."""
    diag = RopeLine("d", a=(0.0, 0.0), b=(40.0, 40.0), depth_m=2.0)
    cov = rope_prior_covariance(diag, SlamConfig())
    assert abs(cov[0][1]) > 1.0, "a rotated line's prior has correlation; this one has none"
    assert cov[0][1] == pytest.approx(cov[1][0])


def test_the_prior_mean_is_the_midpoint_between_the_mooring_buoys():
    s = _slam()
    key = s.add_detection(Detection(t=0.0, kind=ROPE, assoc_id="line_west",
                                    slant_range_m=6.0, side=PORT, vehicle_depth_m=0.0,
                                    sigma_range_m=0.3))
    pf = [p for p in s.graph.landmark_priors if p.key == key][0]
    assert pf.mean == pytest.approx((0.0, 20.0))


def test_every_detection_gets_its_own_landmark():
    """The paper's central move. One landmark per rope would demand a correct data association
    along the rope's length — precisely the direction a side scan knows nothing about."""
    s = _slam()
    for i in range(5):
        s.add_odometry(Odometry(2.0, 0.0, 0.0, 0.1, 0.02))
        s.add_detection(Detection(t=float(i), kind=ROPE, assoc_id="line_west",
                                  slant_range_m=6.0, side=PORT, vehicle_depth_m=0.0,
                                  sigma_range_m=0.3))
    assert s.graph.n_landmarks == 5
    assert len(set(s.graph.landmark_origin.values())) == 1, "all five came from one line"
    assert len({p.key for p in s.graph.landmark_priors}) == 5


def test_each_landmarks_prior_comes_from_the_line_it_was_associated_with():
    s = _slam()
    kw = s.add_detection(Detection(0.0, ROPE, "line_west", 6.0, PORT, 0.0, 0.3))
    ke = s.add_detection(Detection(0.0, ROPE, "line_east", 6.0, STARBOARD, 0.0, 0.3))
    by_key = {p.key: p for p in s.graph.landmark_priors}
    assert by_key[kw].mean == pytest.approx((0.0, 20.0))
    assert by_key[ke].mean == pytest.approx((16.0, 20.0))


# ---- refusals: an unknown anything is named, never downgraded ----------------------------------

def test_a_detection_associated_with_a_line_the_map_does_not_have_is_refused():
    with pytest.raises(SlamRefusal) as e:
        _slam().add_detection(Detection(0.0, ROPE, "line_that_never_fitted", 6.0, PORT, 0.0, 0.3))
    assert "line_that_never_fitted" in str(e.value)
    assert "line_west" in str(e.value), "say what the map DOES have"


def test_an_unknown_detection_kind_is_refused_by_name():
    with pytest.raises(SlamRefusal) as e:
        _slam().add_detection(Detection(0.0, "anchor_chain", "line_west", 6.0, PORT, 0.0, 0.3))
    assert "anchor_chain" in str(e.value)


def test_a_detection_with_no_side_is_refused():
    with pytest.raises(SlamRefusal) as e:
        _slam().add_detection(Detection(0.0, ROPE, "line_west", 6.0, 0, 0.0, 0.3))
    assert "PORT" in str(e.value) and "STARBOARD" in str(e.value)


def test_an_unknown_backend_is_refused_by_name():
    with pytest.raises(SlamRefusal) as e:
        FarmSlam(MAP, backend="particle_filter")
    assert "particle_filter" in str(e.value)


def test_an_empty_map_cannot_arm_farm_relative_navigation():
    with pytest.raises(SlamRefusal) as e:
        FarmSlam(FarmSlamMap(lines=(), buoys=()), backend=BATCH)
    assert "surface-and-report" in str(e.value), \
        "the refusal must name the fallback, or somebody will invent a silent one"


@pytest.mark.skipif(GTSAM_AVAILABLE, reason="python-gtsam is importable here")
def test_isam2_refuses_rather_than_silently_solving_a_batch():
    """SETTLED §3d: an unknown mode is refused by name, never downgraded. A batch solve is not
    iSAM2 and must never be handed to a vehicle under the name of one."""
    with pytest.raises(UnavailableBackend) as e:
        FarmSlam(MAP, backend=ISAM2)
    assert "GTSAM_BUILD_PYTHON=OFF" in str(e.value), "name why, and what to do about it"


def test_the_install_flag_named_in_the_refusal_is_the_one_actually_in_the_installer():
    """The refusal tells an engineer to go and look at a specific line of a specific script. If
    that script changes, the instruction becomes a wild goose chase — so it is checked, not
    quoted from memory. This is the same rule as reading a prefab value from `git diff` rather
    than from the working tree (SETTLED §3k)."""
    here = os.path.dirname(os.path.abspath(__file__))
    installer = os.path.abspath(os.path.join(
        here, "..", "..", "..", "..", "scripts", "install_gtsam_for_hydrobatic.sh"))
    if not os.path.exists(installer):
        pytest.skip(f"installer not present at {installer}")
    assert "-DGTSAM_BUILD_PYTHON=OFF" in open(installer).read(), (
        "install_gtsam_for_hydrobatic.sh no longer builds with GTSAM_BUILD_PYTHON=OFF — if it "
        "now builds the bindings, farm_slam's refusal text is telling engineers to fix "
        "something that is already fixed")


# ---- the solve: does the structure actually pull the vehicle back? -----------------------------

def _survey(slam, *, n=24, step=1.5, drift_per_step=0.02, start=(8.0, 2.0, math.pi / 2)):
    """Fly north up the corridor between the two lines, seeing both.

    The vehicle is at the surface (depth 0.0) and the ropes are at 2.0 m, which is the geometry
    SETTLED §3l settled for this farm. `drift_per_step` is a constant lateral DR bias — the thing
    farm-relative navigation exists to absorb — injected into the ODOMETRY only, never into the
    ranges: the sonar is telling the truth and dead reckoning is not, which is the real case.
    """
    truth = [start]
    for i in range(n):
        x, y, th = truth[-1]
        truth.append((x, y + step, th))
    dets_per_pose = []
    for (x, y, th) in truth[1:]:
        d = []
        for line, side in ((LINE_W, PORT), (LINE_E, STARBOARD)):
            ground = abs(line.a[0] - x)
            slant = math.hypot(ground, 2.0 - 0.0)
            d.append(Detection(t=0.0, kind=ROPE, assoc_id=line.line_id, slant_range_m=slant,
                               side=side, vehicle_depth_m=0.0, sigma_range_m=0.3))
        dets_per_pose.append(d)
    for i, dets in enumerate(dets_per_pose):
        # The DR delta the vehicle BELIEVES: forward `step`, plus a lateral bias it cannot see.
        slam.add_odometry(Odometry(step, drift_per_step, 0.0, 0.2, 0.02))
        for det in dets:
            slam.add_detection(det)
    return truth


@pytest.mark.parametrize("backend", BACKENDS_HERE)
def test_the_ropes_pull_the_pose_back_against_a_drifting_dr(backend):
    """The whole point of P8. Dead reckoning walks sideways; the ropes do not move."""
    s = FarmSlam(MAP, backend=backend, origin=(8.0, 2.0, math.pi / 2))
    truth = _survey(s)
    dr_x = s._dr_pose_of(s.current_pose_key)[0]
    s.update()
    slam_x = s.pose()[0]
    truth_x = truth[-1][0]
    assert abs(dr_x - truth_x) > 0.4, \
        f"the fixture is not exercising anything: DR is already right ({dr_x:.3f})"
    assert abs(slam_x - truth_x) < abs(dr_x - truth_x), \
        f"SLAM ({slam_x:.3f}) is no closer to truth ({truth_x:.3f}) than DR ({dr_x:.3f}) is"


@pytest.mark.parametrize("backend", BACKENDS_HERE)
def test_a_vehicle_that_is_not_drifting_is_not_dragged_off(backend):
    """PASS-FIRST for the correction: run it against the state it must LEAVE ALONE. A filter that
    improves a bad estimate and ruins a good one has not been shown to work."""
    s = FarmSlam(MAP, backend=backend, origin=(8.0, 2.0, math.pi / 2))
    truth = _survey(s, drift_per_step=0.0)
    s.update()
    assert abs(s.pose()[0] - truth[-1][0]) < 0.5


@pytest.mark.parametrize("backend", BACKENDS_HERE)
def test_the_fitted_line_recovers_the_ropes_own_heading(backend):
    """What P8.4's rope following servos on. The lines run due north (+y) from a vehicle heading
    north, so the fitted heading is +/- pi/2 — a line has no direction, only an axis."""
    s = FarmSlam(MAP, backend=backend, origin=(8.0, 2.0, math.pi / 2))
    _survey(s)
    s.update()
    (_, heading, n) = s.line_estimate("line_west")
    assert n >= 20
    assert min(abs(abs(heading) - math.pi / 2), abs(abs(heading) - math.pi / 2)) < 0.15


def test_a_line_with_one_landmark_cannot_be_fitted_and_says_so():
    s = _slam()
    s.add_detection(Detection(0.0, ROPE, "line_west", 6.0, PORT, 0.0, 0.3))
    s.update()
    with pytest.raises(SlamRefusal) as e:
        s.line_estimate("line_west")
    assert "fewer than two" in str(e.value)
    assert "configuration file" in str(e.value), \
        "the refusal must say why returning the prior's heading would be worse than refusing"


def test_the_correction_barely_moves_a_goal_when_nothing_has_drifted():
    """PASS-FIRST. A correction that is never the identity is a correction that always moves the
    goals — and a lane standoff budget of metres cannot absorb that."""
    s = FarmSlam(MAP, backend=BATCH, origin=(8.0, 2.0, math.pi / 2))
    _survey(s, drift_per_step=0.0)
    s.update()
    goal = (8.0, 38.0)
    moved = apply_correction(s.correction(), goal)
    assert math.dist(moved, goal) < 0.6


def test_the_correction_is_refused_before_anything_is_solved():
    s = _slam()
    s.add_odometry(Odometry(1.0, 0.0, 0.0, 0.1, 0.02))
    with pytest.raises(SlamRefusal) as e:
        s.correction()
    assert "before update()" in str(e.value)


def test_the_correction_takes_the_dr_pose_onto_the_slam_pose():
    """WHAT A CORRECTION IS, ASSERTED BY APPLYING IT.

    The first version of this test read `hypot(dx, dy)` off the SE(2) and demanded it exceed
    0.2 m for a 1.2 m drift. It got 0.17 and looked like a broken correction. It was not: the
    rotation acts about the graph's origin, so at 38 m out a few hundredths of a radian carry
    most of the displacement and the stored translation reads centimetres. **Judge a transform
    by applying it.** Same family as reading a prefab value out of the working tree — the number
    was real and the thing it was taken to mean was not.
    """
    s = FarmSlam(MAP, backend=BATCH, origin=(8.0, 2.0, math.pi / 2))
    truth = _survey(s, drift_per_step=0.05)
    dr = s._dr_pose_of(s.current_pose_key)
    s.update()
    slam = s.pose()
    assert abs(dr[0] - truth[-1][0]) > 0.8, "the fixture must actually drift"
    landed = apply_correction(s.correction(), dr)
    assert math.dist(landed[:2], slam[:2]) < 1e-6, \
        "the correction must be exactly the transform from the DR pose to the SLAM pose"
    assert math.dist(landed[:2], truth[-1][:2]) < abs(dr[0] - truth[-1][0]), \
        "and applying it must move a goal TOWARDS truth, not away"


def test_a_goal_out_on_the_lane_is_moved_by_a_metre_scale_amount():
    """The number that matters operationally: how far does a T3 lane waypoint actually move?
    Nothing here claims a value — only that a real 1.2 m DR drift produces a real correction at
    the far end of a lane rather than a rounding error."""
    s = FarmSlam(MAP, backend=BATCH, origin=(8.0, 2.0, math.pi / 2))
    _survey(s, drift_per_step=0.05)
    s.update()
    far_goal = (8.0, 38.0)
    moved = apply_correction(s.correction(), far_goal)
    assert 0.3 < math.dist(moved, far_goal) < 5.0


def test_apply_correction_keeps_the_shape_it_was_given():
    c = (1.0, 2.0, 0.0)
    assert len(apply_correction(c, (0.0, 0.0))) == 2
    assert len(apply_correction(c, (0.0, 0.0, 0.5))) == 3


def test_the_identity_correction_moves_nothing():
    assert apply_correction((0.0, 0.0, 0.0), (3.0, 4.0)) == pytest.approx((3.0, 4.0))


# ---- the forward 3D sonar is a second source into the same graph (P8.5's hook) ------------------

def test_a_sonar3d_detection_carries_its_own_bearing_rather_than_assuming_abeam():
    """A side scan has no along-track resolution, so its bearing IS abeam. The forward 3D sonar
    measures one, and that is the only way a near-vertical mooring line is ever seen at all."""
    s = _slam()
    s.add_detection(Detection(0.0, ROPE, "line_west", 6.0, PORT, 0.0, 0.3,
                              source=SONAR3D, bearing_rad=-0.6, sigma_bearing_rad=0.05))
    m = s.graph.measurements[-1]
    assert m.bearing_rad == pytest.approx(-0.6)
    assert m.sigma_bearing_rad == pytest.approx(0.05)


def test_a_side_scan_detection_is_taken_as_exactly_abeam():
    s = _slam()
    s.add_detection(Detection(0.0, ROPE, "line_west", 6.0, PORT, 0.0, 0.3))
    assert s.graph.measurements[-1].bearing_rad == pytest.approx(math.pi / 2)
    s.add_detection(Detection(0.0, ROPE, "line_east", 6.0, STARBOARD, 0.0, 0.3))
    assert s.graph.measurements[-1].bearing_rad == pytest.approx(-math.pi / 2)


def test_port_is_to_the_west_of_a_vehicle_heading_north():
    """THE TEST THIS FILE MOST NEEDED AND DID NOT HAVE.

    The first implementation wrote the abeam bearing as `side * pi/2`, which puts every STARBOARD
    return to the WEST of a vehicle heading north — the sign backwards. Every check in this file
    passed anyway: the corridor survey is symmetric, so the pose still came out right in x. It
    surfaced only when the two backends disagreed and a cost breakdown showed one landmark 16 m
    from the prior it had been given.

    So the convention is asserted in WORLD TERMS, on the initialised landmark position, rather
    than as a number that can be mirrored without anybody noticing. A vehicle at the origin
    heading north (+y): port is west (-x), starboard is east (+x).
    """
    n = FarmSlam(MAP, backend=BATCH, origin=(8.0, 20.0, math.pi / 2))
    kp = n.add_detection(Detection(0.0, ROPE, "line_west", 8.0, PORT, 0.0, 0.3))
    ks = n.add_detection(Detection(0.0, ROPE, "line_east", 8.0, STARBOARD, 0.0, 0.3))
    assert n.landmark(kp)[0] < 8.0, "port of a northbound vehicle is to the WEST"
    assert n.landmark(ks)[0] > 8.0, "starboard of a northbound vehicle is to the EAST"
    assert n.landmark(kp)[1] == pytest.approx(20.0, abs=1e-6), "abeam means no along-track offset"


# ---- the T2 verdicts are the gate (P8.2's ground) ---------------------------------------------

class _FakeVerifiedMap:
    def __init__(self, lines, buoys, verdicts):
        self.lines_xz = lines
        self.buoys_xz = buoys
        self.verdicts = verdicts
        self.source = "T2"


def test_only_lines_that_fitted_reach_the_graph():
    """A line that did not fit is ABSENT, never straightened onto its prior. `FarmMap.lines_xz`
    already keeps this rule; the conversion must not undo it."""
    m = verified_map_to_slam_map(
        _FakeVerifiedMap({"line_west": ((0.0, 0.0), (0.0, 40.0))}, {}, {}), rope_depth_m=2.0)
    assert [l.line_id for l in m.lines] == ["line_west"]


def test_a_missing_buoy_is_not_a_landmark_and_a_moved_one_is():
    """Four verdicts, not three (SETTLED §3k). `moved` is a MEASUREMENT — we found it somewhere
    else — and `not_surveyed` is not; merging them is exactly the error that register names."""
    m = verified_map_to_slam_map(_FakeVerifiedMap(
        {}, {"C1": (0.0, 0.0), "C2": (0.0, 40.0), "C3": (16.0, 40.0), "C4": (16.0, 0.0)},
        {"C1": "confirmed", "C2": "moved", "C3": "missing", "C4": "not_surveyed"}),
        rope_depth_m=2.0)
    assert sorted(b.buoy_id for b in m.buoys) == ["C1", "C2"]


def test_a_moved_buoy_is_believed_less_than_a_confirmed_one():
    m = verified_map_to_slam_map(_FakeVerifiedMap(
        {}, {"C1": (0.0, 0.0), "C2": (0.0, 40.0)},
        {"C1": "confirmed", "C2": "moved"}), rope_depth_m=2.0)
    by_id = {b.buoy_id: b for b in m.buoys}
    assert by_id["C2"].sigma_m > by_id["C1"].sigma_m


def test_the_armable_verdicts_are_a_subset_of_the_localizers_own_four():
    """If the localizer ever grows a fifth verdict, this list must be revisited deliberately
    rather than by silently not matching it."""
    from sam_farm_inspection import localizer
    src = open(localizer.__file__).read()
    for v in ARMABLE_VERDICTS:
        assert f'"{v}"' in src or f"'{v}'" in src, \
            f"{v!r} is not a verdict the localizer produces"


# ---- the report ---------------------------------------------------------------------------------

def test_the_report_always_names_the_backend():
    """A report that does not say which solver produced it cannot be compared with the one from
    the machine next to it — and the two solvers here are deliberately not the same algorithm."""
    s = _slam()
    s.add_detection(Detection(0.0, ROPE, "line_west", 6.0, PORT, 0.0, 0.3))
    r = s.report()
    assert r["backend"] == BATCH
    assert r["solved"] is False, "an unsolved graph must not report as solved"
    s.update()
    assert s.report()["solved"] is True


def test_the_report_counts_rope_and_buoy_landmarks_separately():
    s = _slam()
    s.add_detection(Detection(0.0, ROPE, "line_west", 6.0, PORT, 0.0, 0.3))
    s.add_detection(Detection(0.0, BUOY, "C1", 6.0, PORT, 0.0, 0.3))
    r = s.report()
    assert r["rope_landmarks"] == 1 and r["buoy_landmarks"] == 1
    assert r["lines_seen"] == ["line_west"]


@pytest.mark.skipif(not GTSAM_AVAILABLE, reason="python-gtsam is not importable here")
def test_the_two_backends_agree_on_the_same_graph():
    """The reason the graph is data. iSAM2 and the batch Gauss-Newton are different algorithms
    over identical factors, so agreement is evidence about the FACTORS — and disagreement, when
    it comes, will point at whichever one is wrong instead of at the whole idea."""
    a = FarmSlam(MAP, backend=BATCH, origin=(8.0, 2.0, math.pi / 2))
    b = FarmSlam(MAP, backend=ISAM2, origin=(8.0, 2.0, math.pi / 2))
    _survey(a)
    _survey(b)
    a.update()
    b.update()
    assert abs(a.pose()[0] - b.pose()[0]) < 0.25
    assert abs(a.pose()[1] - b.pose()[1]) < 0.25
