"""The prior loader refuses rather than substitutes.

2026-08-16. Every one of these is a case where returning *something* would produce a
mission that runs and goes to the wrong place — the most expensive failure available to
an autonomous inspection.

Run: python3 -m pytest smarc2/perception/sam/sam_farm_inspection/test/test_farm_prior_loader.py
"""
import pathlib
import sys

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from sam_farm_inspection.farm_prior import (  # noqa: E402
    PriorRefusal, default_prior_path, load_farm_prior)

yaml = pytest.importorskip("yaml")


def test_the_real_prior_loads_and_carries_the_farm():
    p = default_prior_path()
    if not p:
        pytest.skip("no generated farm prior in this checkout")
    prior = load_farm_prior(p)
    assert prior.n_buoys == 7
    assert len(prior.culture_lines) == 2
    assert prior.rope_depth_m == pytest.approx(2.0)
    assert prior.seabed_depth_range_m[0] > 0, "seabed range is positive-down metres"
    assert prior.caveats, "the qualitative-sonar caveat must travel with the prior"


def test_a_missing_prior_is_fatal_and_names_the_generator():
    with pytest.raises(PriorRefusal) as e:
        load_farm_prior("/nonexistent/farm_prior.yaml")
    assert "make_farm_prior.py" in str(e.value)


def test_a_hand_written_prior_is_refused(tmp_path):
    """No `_generated` block means nobody generated it. A prior that was edited by hand
    has already decoupled from the site it claims to describe."""
    p = tmp_path / "farm_prior.yaml"
    p.write_text(yaml.safe_dump({
        "farm": {"rope_depth_m": 2.0, "rope_unity_y": -2.0,
                 "buoys": [{"name": "a", "unity_x": 0.0, "unity_z": 0.0}]}}))
    with pytest.raises(PriorRefusal) as e:
        load_farm_prior(str(p))
    assert "_generated" in str(e.value)


def test_disagreeing_depth_conventions_are_refused(tmp_path):
    """rope_depth_m is positive-down and rope_unity_y is a Unity y. If they are not
    negatives of each other, one of them is wrong and every lane ends up on the wrong
    side of the ropes — which flies at a depth the sonar cannot see from."""
    p = tmp_path / "farm_prior.yaml"
    p.write_text(yaml.safe_dump({
        "_generated": {"by": "test"},
        "farm": {"rope_depth_m": 2.0, "rope_unity_y": 2.0,
                 "buoys": [{"name": "a", "unity_x": 0.0, "unity_z": 0.0}],
                 "culture_lines": []}}))
    with pytest.raises(PriorRefusal) as e:
        load_farm_prior(str(p))
    assert "conventions disagree" in str(e.value)


def test_a_line_referencing_an_unknown_buoy_is_refused(tmp_path):
    p = tmp_path / "farm_prior.yaml"
    p.write_text(yaml.safe_dump({
        "_generated": {"by": "test"},
        "farm": {"rope_depth_m": 2.0, "rope_unity_y": -2.0,
                 "buoys": [{"name": "a", "unity_x": 0.0, "unity_z": 0.0}],
                 "culture_lines": [{"nodes": ["a", "ghost"], "bearing_grid_deg": 0.0}]}}))
    with pytest.raises(PriorRefusal) as e:
        load_farm_prior(str(p))
    assert "unknown buoy" in str(e.value)


def test_the_loader_hands_back_the_SHIPPED_beams_lane_not_a_nicer_one():
    """The loader returns the lane belonging to the beam the vehicle ACTUALLY HAS,
    whatever that lane says. Handing back a nicer one would plan a mission the sensor in
    the scene cannot fly, and the mission would run — scanning, finding nothing, and
    reporting an empty farm. Same family as a refusal downgraded to a default (ADR-004).

    Updated 2026-08-16, deliberately, as the previous version of this test demanded: the
    prior no longer carries a second `proposed` lane, and the shipped lane is flyable. The
    refusal it used to assert was an artefact — the beam had been read off a working tree
    Unity had silently reverted (SETTLED 3k), not off the committed prefab.

    So this now asserts the INVARIANT rather than the verdict — the loader's lane IS the
    as-shipped lane, and there is no alternative lane in the document for it to prefer.
    A verdict-shaped assertion is what let a regression look like a design decision.
    """
    p = default_prior_path()
    if not p:
        pytest.skip("no generated farm prior in this checkout")
    prior = load_farm_prior(p)
    doc = yaml.safe_load(open(p).read())
    assert prior.lane == doc["sonar"]["lanes"]["as_shipped"]
    assert list(doc["sonar"]["lanes"]) == ["as_shipped"], (
        "the prior offers a choice of lanes again — the loader must not be in a position "
        "to prefer one, so this is a generator change that needs its own decision")
