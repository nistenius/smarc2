"""LETTING GO OF THE ACTUATORS IS A CLAIM, AND A TICK COUNT CANNOT SUPPORT IT.

Ivan, across several rig sessions, most recently 2026-08-16:

    "Unity HID still shows VBS 45% when done"
    "it starts with vbs 0 and surfacing, then pops back to the hold stuff"

Read end to end, nothing pops back. The last waypoint completes, the blend controller correctly
commands `vbs_u_neutral` (0.0) and hands DivePub the NEUTRAL state — and DivePub then published
that command exactly **twenty times** before switching itself to DISENGAGED and going silent. Two
seconds. A VBS tank does not empty in two seconds, so the purge was cut off mid-way and the tank
stopped at whatever it had reached. The ~45% on the HUD is an INTERRUPTED command, not a held one.

That is the same defect shape this project keeps finding: a fixed count standing in for an
outcome — SETTLED §3e's "a stopwatch may not certify a flight nobody saw take off", the health
gate that counted attempts instead of readings, the start banner that reported having asked.

    python3 -m pytest test/test_neutral_handoff.py
"""
import pytest

from sam_diving_controller.neutral_handoff import (
    neutral_handoff, DEFAULT_MIN_TICKS, DEFAULT_MAX_TICKS,
)

SURFACED = dict(depth_m=0.05, vbs_pct=0.0, vbs_target_pct=0.0)
DIVING = dict(depth_m=3.0, vbs_pct=45.0, vbs_target_pct=0.0)


# ---------------------------------------------------------------- the floor

def test_the_command_must_actually_go_out_first():
    """The old rule survives as a FLOOR and nothing more. A single publish can be dropped, so a
    vehicle that reports itself surfaced on tick one has not yet been commanded to stay there."""
    v = neutral_handoff(ticks=1, **SURFACED)
    assert not v.release
    assert "1/20" in v.reason


def test_reaching_the_floor_is_not_by_itself_permission():
    """THE BUG, stated as a test. Twenty ticks with the vehicle still 3 m down and the tank at 45%
    is exactly the state that produced the screenshots — and the old code released here."""
    v = neutral_handoff(ticks=DEFAULT_MIN_TICKS, **DIVING)
    assert not v.release, (
        "releasing at the tick floor is what cut the VBS purge off mid-way and left the vehicle "
        "part-buoyant with nobody commanding it")
    assert "3.00 m down" in v.reason and "45%" in v.reason


# ---------------------------------------------------------------- confirmation

def test_surfaced_and_neutral_releases_and_says_it_was_confirmed():
    v = neutral_handoff(ticks=DEFAULT_MIN_TICKS, **SURFACED)
    assert v.release and v.confirmed
    assert "surfaced" in v.reason


@pytest.mark.parametrize("depth,vbs,missing", [
    (0.05, 45.0, "VBS"),      # up, but the tank never emptied — this is the reported symptom
    (3.00, 0.0, "m down"),    # tank empty, but still under — surfacing not finished
])
def test_half_a_resting_state_is_not_a_resting_state(depth, vbs, missing):
    """Both conditions, not either. A vehicle at the surface with a half-full tank is not at rest,
    and neither is an empty tank still three metres down."""
    v = neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=depth, vbs_pct=vbs, vbs_target_pct=0.0)
    assert not v.release
    assert missing in v.reason


def test_it_keeps_commanding_while_it_waits():
    """The branch the old code did not have at all. Between the floor and the timeout the answer
    is 'not yet' — which means the neutral command keeps being published, which is the whole
    mechanism by which the tank actually empties."""
    for t in range(DEFAULT_MIN_TICKS, DEFAULT_MAX_TICKS):
        assert not neutral_handoff(ticks=t, **DIVING).release


# ---------------------------------------------------------------- absent is not arrived

@pytest.mark.parametrize("depth,vbs", [(None, 0.0), (0.05, None), (None, None)])
def test_a_vehicle_that_says_nothing_is_never_CONFIRMED_surfaced(depth, vbs):
    """ABSENT IS NOT EMPTY, again — the rule this project keeps paying to relearn (bags_onboard,
    dr_track, depth_m). Silence must not be read as arrival, or a vehicle whose depth sensor has
    died gets certified as safe."""
    v = neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=depth, vbs_pct=vbs, vbs_target_pct=0.0)
    assert not v.confirmed
    assert not v.release, "at the floor, missing feedback means keep commanding, not let go"


def test_missing_feedback_still_eventually_releases_but_never_claims_confirmation():
    """A controller that clings to the actuators forever cannot be taken over by an operator —
    that is the worse failure, so the timeout still fires. What must never happen is the timeout
    being reported as a success: the log has to distinguish 'it surfaced' from 'we gave up'."""
    v = neutral_handoff(ticks=DEFAULT_MAX_TICKS, depth_m=None, vbs_pct=None, vbs_target_pct=0.0)
    assert v.release and not v.confirmed
    assert "TIMEOUT" in v.reason
    assert "no depth feedback" in v.reason and "no VBS feedback" in v.reason


def test_the_timeout_names_which_half_never_arrived():
    """A give-up message with no diagnosis in it cannot be acted on, and this one is the only
    trace a failed surfacing leaves."""
    v = neutral_handoff(ticks=DEFAULT_MAX_TICKS, **DIVING)
    assert v.release and not v.confirmed
    assert "3.00 m down" in v.reason
    assert "wanted 0%" in v.reason


# ---------------------------------------------------------------- sign safety

@pytest.mark.parametrize("z", [0.05, -0.05, 0.0])
def test_at_the_surface_is_near_zero_under_EITHER_sign_convention(z):
    """`get_depth()` returns an odom/DR `position.z` whose sign depends on the frame, and this
    project has already lost a session to a sign inversion. A hand-off that inverted would release
    the actuators AT DEPTH. Comparing the magnitude costs nothing and removes the class."""
    assert neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=z, vbs_pct=0.0,
                           vbs_target_pct=0.0).confirmed


@pytest.mark.parametrize("z", [3.0, -3.0])
def test_and_being_DEEP_is_never_mistaken_for_the_surface_either_way(z):
    """The other half of the same guard: taking a magnitude must not make depth meaningless."""
    assert not neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=z, vbs_pct=0.0,
                               vbs_target_pct=0.0).release


# ---------------------------------------------------------------- a non-zero neutral

def test_neutral_is_whatever_the_config_says_it_is_not_hardcoded_zero():
    """`vbs_u_neutral` is a parameter and a trimmed hull may want a non-zero resting tank. The
    check is 'reached the commanded neutral', not 'reached zero' — otherwise a correctly trimmed
    vehicle would time out on every single mission."""
    v = neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=0.05, vbs_pct=30.0, vbs_target_pct=30.0)
    assert v.release and v.confirmed
    bad = neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=0.05, vbs_pct=0.0, vbs_target_pct=30.0)
    assert not bad.release


def test_the_tolerance_is_a_band_not_an_equality():
    """Floating-point actuator feedback never lands exactly on the setpoint."""
    assert neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=0.05, vbs_pct=4.0,
                           vbs_target_pct=0.0).confirmed
    assert not neutral_handoff(ticks=DEFAULT_MIN_TICKS, depth_m=0.05, vbs_pct=6.0,
                               vbs_target_pct=0.0).confirmed
