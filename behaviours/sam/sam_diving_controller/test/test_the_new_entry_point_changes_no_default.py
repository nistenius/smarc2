"""The M1 entry point is ADDITIVE and NON-DEFAULT — letter D, work-order rule 10.

WHAT IS ACTUALLY AT STAKE. `diving_node` is the thing that commands the actuators. The team
flies `blend_pid_wp_following` (and `pid_wp_following` before it) and the hardware-change
discipline (`docs/hardware-affecting-changes.md`) governs anything that alters what comes up on
the hull. A new entry point that quietly became a default would be a bringup change nobody
agreed to — so the guard is not "the file compiles", it is:

  * the new entry point EXISTS and is registered, so it can be run by hand;
  * NO launch file references it;
  * the launch lines for the two flown modes are BYTE-IDENTICAL to HEAD;
  * `setup.py`'s existing console_scripts are byte-identical to HEAD;
  * the MPC import is inside the function, so the module still imports on a machine with no
    acados — which is this machine, and vm1 is unmeasured (SETTLED §3ad).

    export PYTHONPYCACHEPREFIX=/tmp/pyc
    python3 -m pytest -q -p no:cacheprovider \\
        smarc2/behaviours/sam/sam_diving_controller/test/test_the_new_entry_point_changes_no_default.py
"""
import ast
import pathlib
import subprocess

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1]
REPO = PKG.parents[2]
ENTRYPOINTS = PKG / "sam_diving_controller" / "entrypoints.py"
SETUP = PKG / "setup.py"
NEW = "mpc_and_pid_wp_following"


def _git_show(path):
    rel = path.relative_to(REPO)
    r = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel.as_posix()}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"{rel} is not in git HEAD here: {r.stderr.strip()[:200]}")
    return r.stdout


def _console_scripts(text):
    tree = ast.parse(text)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and " = " in n.value:
            out.append(n.value)
    return out


# ------------------------------------------------------------------ it exists
def test_the_entry_point_exists_and_is_registered():
    tree = ast.parse(ENTRYPOINTS.read_text())
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert NEW in names, f"{NEW}() is not defined"
    assert f"_build_{NEW}" in names, f"_build_{NEW}() is not defined"
    scripts = _console_scripts(SETUP.read_text())
    assert any(s.startswith(f"{NEW} = ") for s in scripts), \
        f"{NEW} is not in setup.py's console_scripts, so `ros2 run` cannot reach it"


def test_the_module_still_imports_on_a_machine_with_no_acados():
    """The MPC import must be INSIDE the builder, exactly as `mpc_wp_following` does it. acados
    is not installed here and is unmeasured on vm1 (SETTLED §3ad); a module-level import would
    take the whole entrypoints module — and every other mode in it — down with it."""
    tree = ast.parse(ENTRYPOINTS.read_text())
    module_level = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = getattr(n, "module", "") or ""
            module_level.add(mod)
            module_level |= {a.name for a in n.names}
    # `MPCPathServer` (from ActionServerDiveSub) is fine and pre-exists: it is an action server,
    # not the solver. What may not be at module scope is the CONTROLLER, which is what pulls in
    # acados and casadi through `smarc_modelling`.
    assert not any("DiveControllerMPC" in m or "acados" in m or "casadi" in m
                   for m in module_level), \
        f"an MPC controller / acados import is at module scope: {sorted(module_level)}"
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == f"_build_{NEW}")
    inner = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert any("DiveControllerMPC" in line for line in inner), \
        "the builder does not import the MPC controller at all"


# ------------------------------------------------------------------ it changes no default
def test_no_launch_file_references_the_new_entry_point():
    hits = []
    for f in PKG.rglob("*"):
        if f.is_file() and f.suffix in (".launch", ".xml", ".yaml", ".yml", ".sh", ".py"):
            if f.name in ("entrypoints.py", "setup.py") or "test" in f.parts:
                continue
            if NEW in f.read_text(errors="replace"):
                hits.append(str(f.relative_to(PKG)))
    assert hits == [], (
        f"the new entry point is referenced by {hits}. It is additive and non-default: making it "
        f"reachable from a launch file is a bringup change under the hardware-change discipline, "
        f"and not this round's to make.")


def test_the_two_flown_launch_files_are_byte_identical_to_HEAD():
    """`blend_pid_wp_following` is what the team flies. If this file changed at all, the change
    is a bringup change and belongs in a different conversation."""
    for name in ("blend_pid_wp_following.launch", "pid_wp_following.launch"):
        f = PKG / "launch" / name
        if not f.exists():
            pytest.skip(f"{name} is not in this checkout")
        assert f.read_text() == _git_show(f), f"{name} differs from HEAD"


def test_every_pre_existing_console_script_is_byte_identical_to_HEAD():
    """The new one is ADDED. Nothing else in the list may move, be renamed, or change target —
    that list is what `ros2 run` resolves and what every bringup script names."""
    before = _console_scripts(_git_show(SETUP))
    after = _console_scripts(SETUP.read_text())
    added = [s for s in after if s not in before]
    removed = [s for s in before if s not in after]
    assert removed == [], f"console scripts disappeared: {removed}"
    assert len(added) == 1 and added[0].startswith(f"{NEW} = "), f"unexpected additions: {added}"
    # ... and in the same ORDER, so nothing was reshuffled around the addition
    assert [s for s in after if s != added[0]] == before


def test_the_existing_builders_are_untouched():
    """A new mode must not have been bought by editing an old one."""
    before = ast.parse(_git_show(ENTRYPOINTS))
    after = ast.parse(ENTRYPOINTS.read_text())

    def bodies(tree):
        return {n.name: ast.unparse(n) for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)}

    b, a = bodies(before), bodies(after)
    for name, src in b.items():
        assert name in a, f"{name}() was removed from entrypoints.py"
        assert a[name] == src, f"{name}() was modified; the new mode must be purely additive"
    assert set(a) - set(b) == {NEW, f"_build_{NEW}"}, sorted(set(a) - set(b))


# ------------------------------------------------------------------ what it builds
def test_the_builder_uses_ONE_DivePub_and_puts_both_servers_behind_the_arbiter():
    """Two publishers on one actuator path is the state the arbiter exists to make impossible;
    building a second one would make the arbiter decorative."""
    tree = ast.parse(ENTRYPOINTS.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == f"_build_{NEW}")
    src = ast.unparse(fn)
    assert src.count("DivePub(") == 1, "more than one DivePub is constructed"
    assert "OneWriterArbiter(" in src
    assert "auv_depth_move_to" in src and "auv_trajectory_tracking" in src
    assert "one_writer_arbiter" in src, "the servers are not given the arbiter at all"


def test_the_builder_says_out_loud_that_it_is_unflown():
    """A bringup mode nobody has run must announce that on the vehicle's own log, not only in a
    doc — the log is what an operator reads when something behaves oddly."""
    tree = ast.parse(ENTRYPOINTS.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == f"_build_{NEW}")
    logged = " ".join(n.value for n in ast.walk(fn)
                      if isinstance(n, ast.Constant) and isinstance(n.value, str))
    assert "UNFLOWN" in logged
    assert "one-writer arbiter" in logged
