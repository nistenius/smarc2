"""The ROS shells, checked without a graph — letter B, second pass.

WHAT CAN BE GUARDED OFF-VEHICLE, and it is not nothing:

  * every shell IMPORTS on a machine with no ROS. That is not a nicety: it is what lets the
    cores be tested at all, and an unguarded `import rclpy` at module scope would take the whole
    package down on this machine;
  * every shell keeps its DECISIONS in a core. A shell with an `if` in it is a branch that only
    a live graph can exercise, and SETTLED §1c is the receipt for what that costs;
  * every node publishes a HEALTH line, and `NO_INPUT` names the side (nothing producing vs.
    nothing delivered) rather than guessing between them;
  * NOTHING in this package opens an action server or holds an action client (invariant 12);
  * every node cleans up with `if rclpy.ok(): rclpy.shutdown()` — a bare shutdown raises
    RCLError on a SIGTERM path, exits 1, and a systemd unit reads a clean stop as a crash
    (SETTLED §1c, the duplicate bringup);
  * the entry points in `setup.py` name modules that exist.

What CANNOT be guarded here is whether a subscription ever delivers anything. That is a rig
measurement (rung R3) and nothing in this file pretends otherwise.

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider smarc2/perception/sam/sam_target_inspection/test/test_ros_shells.py
"""
import ast
import importlib
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1]
SRC = PKG / "sam_target_inspection"
SHELLS = ["sss_target_detector", "fls_target_detector", "inspection_planner_node",
          "inspection_recorder"]
CORES = ["sss_target_core", "fls_target_core", "target_ledger", "inspection_planner",
         "inspection_recorder_core", "_health"]


def _tree(name):
    return ast.parse((SRC / f"{name}.py").read_text())


def _calls(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


@pytest.mark.parametrize("name", SHELLS + CORES)
def test_every_module_imports_without_ros(name):
    mod = importlib.import_module(f"sam_target_inspection.{name}")
    assert mod is not None


@pytest.mark.parametrize("name", SHELLS)
def test_every_shell_guards_its_ros_imports(name):
    """The guard must be a real try/except ImportError around the ROS imports, and the module
    must still define its node class afterwards."""
    tree = _tree(name)
    guarded = [n for n in tree.body if isinstance(n, ast.Try)]
    assert guarded, f"{name} imports ROS at module scope with no guard"
    ok = False
    for t in guarded:
        imported = {a.name.split(".")[0] for n in ast.walk(t)
                    if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(t)
                     if isinstance(n, ast.ImportFrom) and n.module}
        if "rclpy" in imported:
            handlers = [h for h in t.handlers
                        if isinstance(h.type, ast.Name) and h.type.id == "ImportError"]
            assert handlers, f"{name}'s ROS import is guarded against the wrong exception"
            ok = True
    assert ok, f"{name} does not guard `import rclpy`"


@pytest.mark.parametrize("name", SHELLS)
def test_a_shell_without_ros_refuses_BY_NAME_rather_than_raising_from_somewhere_odd(name):
    mod = importlib.import_module(f"sam_target_inspection.{name}")
    if getattr(mod, "_HAVE_ROS", False):
        pytest.skip("rclpy is present on this machine")
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "rclpy" in str(e.value)
    assert "sam_target_inspection." in str(e.value), \
        "the refusal must name the pure module that DOES run here"


@pytest.mark.parametrize("name", SHELLS)
def test_every_shell_publishes_a_health_topic(name):
    src = (SRC / f"{name}.py").read_text()
    tree = _tree(name)
    topics = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "create_publisher"):
            topics.add(ast.unparse(n))
    assert any("health" in t for t in topics), f"{name} publishes no health topic"
    assert "HealthLine" in src, f"{name} does not use the shared health-line state machine"


@pytest.mark.parametrize("name", SHELLS)
def test_no_shell_opens_an_action_server_or_holds_an_action_client(name):
    """Invariant 12 and SETTLED §1c: two servers on one action name stopped every mission at
    waypoint 1 for a week. This package answers questions; the tree flies."""
    names = _calls(_tree(name))
    forbidden = {"ActionServer", "ActionClient", "BTActionClient", "send_goal", "cancel_goal"}
    assert not (names & forbidden), f"{name} reaches for {sorted(names & forbidden)}"


@pytest.mark.parametrize("name", SHELLS)
def test_the_clean_stop_exits_zero(name):
    """A bare `rclpy.shutdown()` after rclpy's own SIGTERM handler raises RCLError and exits 1;
    a systemd unit with Restart=on-failure then reads a CLEAN STOP as a crash and relaunches —
    the duplicate-bringup chain (SETTLED §1c)."""
    tree = _tree(name)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = False
    for n in ast.walk(fn):
        if (isinstance(n, ast.If) and "rclpy.ok()" in ast.unparse(n.test)
                and "shutdown" in ast.unparse(n)):
            guarded = True
    assert guarded, f"{name}'s main() calls rclpy.shutdown() unguarded"


@pytest.mark.parametrize("name", SHELLS)
def test_a_shell_holds_no_decision_logic_of_its_own(name):
    """Not a line count: what is measured is that the shell IMPORTS from a core in this package
    and that the core is where the thresholds live. A shell that grew its own gate would import
    nothing and would still pass a length check."""
    tree = _tree(name)
    from_pkg = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module
                and n.module.startswith("sam_target_inspection")}
    assert from_pkg, f"{name} imports nothing from this package's cores — where is its logic?"


def test_the_entry_points_name_modules_that_exist():
    setup = ast.parse((PKG / "setup.py").read_text())
    eps = []
    for n in ast.walk(setup):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and " = " in n.value:
            eps.append(n.value)
    assert eps, "setup.py declares no console_scripts"
    for ep in eps:
        _name, _, target = ep.partition(" = ")
        module, _, func = target.partition(":")
        path = PKG / (module.replace(".", "/") + ".py")
        assert path.exists(), f"entry point {ep} names a module that does not exist"
        mod = importlib.import_module(module)
        assert hasattr(mod, func), f"entry point {ep} names a function that does not exist"


def test_the_health_line_names_which_side_is_silent():
    """"Nothing has arrived" is two different faults and only the publisher count tells them
    apart (SETTLED §3f0r: a Unity stop/play leaves ros_tcp_endpoint holding dead registrations,
    which looks exactly like nothing producing)."""
    from sam_target_inspection._health import HealthLine
    nothing = HealthLine(publisher_count=lambda: 0).compute(0.0)
    undelivered = HealthLine(publisher_count=lambda: 3).compute(0.0)
    unknown = HealthLine(publisher_count=lambda: -1).compute(0.0)
    assert nothing.startswith("NO_INPUT|")
    assert "nothing is producing" in nothing
    assert "subscriber-side" in undelivered
    assert "3 publisher" in undelivered
    assert "unavailable" in unknown
    assert nothing != undelivered


def test_the_health_line_tells_stale_from_blind_from_ok():
    from sam_target_inspection._health import HealthLine
    h = HealthLine(stale_timeout_s=2.0, publisher_count=lambda: 1)
    h.note_input(0.0)
    assert h.compute(0.5, ok_detail="fine").startswith("OK|")
    assert h.compute(10.0).startswith("STALE|")
    h.note_input(10.0)
    h.blind_streak = 9
    h.last_reason = "both channels: no bottom return"
    line = h.compute(10.1)
    assert line.startswith("BLIND|") and "no bottom return" in line


def test_the_health_line_counts_inputs_and_outputs_separately():
    """A detector with inputs and no outputs is a real and important state — "there is nothing
    there" — and it must be distinguishable from one that is not being fed at all."""
    from sam_target_inspection._health import HealthLine
    h = HealthLine(publisher_count=lambda: 1)
    for i in range(5):
        h.note_input(float(i))
    h.note_output(2)
    parts = h.compute(5.0, ok_detail="x").split("|")
    assert parts[3] == "5" and parts[4] == "2"
