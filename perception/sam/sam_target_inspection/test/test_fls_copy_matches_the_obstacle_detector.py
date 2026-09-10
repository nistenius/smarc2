"""The copied cloud parse must still BE the obstacle detector's cloud parse.

`sam_perception/obstacle_detector.py` owns the point-cloud layout. `fls_target_core` needs the
same parse, but the originals are methods on a `Node` subclass in a module that imports rclpy at
module scope, so they cannot be imported on any machine these cores are tested on. They are
therefore COPIED — and a copy of a format definition that nobody checks is how two readers of
the same bytes end up producing different point sets from the same cloud.

WHAT IS COMPARED, AND WHY IT IS NOT A TEXT DIFF. Both sides are parsed with `ast` and compared
as SYNTAX, via `ast.unparse` — so comments, blank lines and formatting are irrelevant by
construction and the test cannot pass by matching prose (work-order rule 8). What is compared is
the arithmetic that must not drift:

  * `quat_to_rot` — the whole function, exactly;
  * the `try:` block of `parse_cloud` — the dtype, the 13-byte fast path and the generic
    fallback. Its `except` branch and the node's own health counters are deliberately NOT
    compared: those belong to the node, and this copy raises instead of logging because it is a
    library function and swallowing the exception would hand the caller an empty cloud that
    reads exactly like a silent seabed;
  * the transform arithmetic of `to_body` — `quat_to_rot`, the translation, `xyz @ R.T + p`.
    The TF lookup is not copied and is not compared: there is no TF buffer here.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_fls_copy_matches_the_obstacle_detector.py
"""
import ast
import pathlib

import pytest

from sam_target_inspection import fls_target_core as F

PKG = pathlib.Path(__file__).resolve().parents[1]
SOURCE = PKG.parent / "sam_perception" / "sam_perception" / "obstacle_detector.py"


def _tree():
    if not SOURCE.exists():
        pytest.skip(f"obstacle_detector.py not found at {SOURCE}")
    return ast.parse(SOURCE.read_text())


def _find_function(tree, name, cls=None):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if cls is None:
                return node
            for c in ast.walk(tree):
                if isinstance(c, ast.ClassDef) and c.name == cls and node in ast.walk(c):
                    return node
    return None


def _norm(node, subs=()):
    """`ast.unparse` of a node, with documented textual substitutions applied afterwards.

    The substitutions are the ONLY licensed differences between the original and the copy, and
    listing them here is the point: each one is a place the copy deliberately differs, named, so
    an undeclared difference cannot hide behind a declared one.
    """
    src = ast.unparse(node)
    for a, b in subs:
        src = src.replace(a, b)
    return src


def test_the_quaternion_to_rotation_matrix_is_byte_for_byte_the_detectors():
    orig = _find_function(_tree(), "quat_to_rot")
    assert orig is not None, "obstacle_detector.py no longer defines quat_to_rot"
    ours = ast.parse(pathlib.Path(F.__file__).read_text())
    mine = _find_function(ours, "quat_to_rot")
    assert mine is not None
    assert _norm(mine) == _norm(orig), (
        "the copied quaternion-to-rotation conversion has drifted from the obstacle "
        "detector's. Both read the same clouds; two different rotations from the same "
        "quaternion is a bug that shows up as a target in the wrong place, not as an error.")


def test_the_cloud_parse_arithmetic_is_the_detectors():
    tree = _tree()
    orig = _find_function(tree, "parse_cloud", cls="ObstacleDetector")
    assert orig is not None, "ObstacleDetector no longer defines parse_cloud"
    orig_try = next(n for n in orig.body if isinstance(n, ast.Try))
    ours = ast.parse(pathlib.Path(F.__file__).read_text())
    mine = _find_function(ours, "parse_cloud")
    mine_try = next(n for n in mine.body if isinstance(n, ast.Try))
    a = "\n".join(_norm(s) for s in mine_try.body)
    b = "\n".join(_norm(s) for s in orig_try.body)
    assert a == b, ("the copied point-cloud parse has drifted from the obstacle detector's. "
                    "The 13-byte x/y/z/intensity layout is Unity's SonarPointCloud_Pub "
                    "contract; two readers disagreeing about it produce different point sets "
                    "from identical bytes.")


def test_the_intensity_gate_is_the_detectors_and_is_a_parameter_here():
    """The original reads `self.intensity_min`, declared with a default of 1. The copy takes it
    as an argument with the SAME default, because a library function may not read a node's
    attribute — but the default must not drift, or the copy silently keeps or drops the no-hit
    rays (intensity 0 at the world origin) that the original does not."""
    tree = _tree()
    declared = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "declare_parameter" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "intensity_min"):
            declared = ast.literal_eval(node.args[1])
    assert declared is not None, "obstacle_detector.py no longer declares intensity_min"
    import inspect
    sig = inspect.signature(F.parse_cloud)
    assert sig.parameters["intensity_min"].default == declared


def test_the_body_transform_is_the_detectors_last_three_lines():
    tree = _tree()
    orig = _find_function(tree, "to_body", cls="ObstacleDetector")
    assert orig is not None
    tail = [ast.unparse(s) for s in orig.body[-3:]]
    ours = ast.parse(pathlib.Path(F.__file__).read_text())
    mine = _find_function(ours, "apply_transform")
    mine_src = ast.unparse(mine)
    # The original's three lines, with its own attribute names mapped onto this function's
    # arguments. Each mapping is a declared difference.
    subs = [("t.transform.rotation", "rotation_xyzw"),
            ("q.x, q.y, q.z, q.w", "*rotation_xyzw"),
            ("t.transform.translation.x, t.transform.translation.y, t.transform.translation.z",
             "translation"),
            ("np.array([translation])", "np.asarray(translation, dtype=np.float64)"),
            ("xyz @ R.T + p", "np.asarray(xyz, dtype=np.float64) @ R.T + p")]
    for line in tail:
        want = line
        for a, b in subs:
            want = want.replace(a, b)
        if want.startswith("q = "):
            continue      # the quaternion is unpacked from the argument here, not from a TF
        assert want in mine_src, (
            f"the obstacle detector's transform line {line!r} has no counterpart in "
            f"apply_transform. The world->body transform is what reconstructs sensor-relative "
            f"ranges in sim (the cloud is world-frame ground truth), and a divergence here "
            f"moves every detection.")


def test_the_copy_names_its_source_in_the_module_text():
    """Not a prose test in disguise: what is asserted is that the module's own docstring/comment
    block CONTAINS the path of the file it copied from, so a reader of the copy can find the
    original. The comparison tests above are what guard the content."""
    src = pathlib.Path(F.__file__).read_text()
    assert "obstacle_detector.py" in src
