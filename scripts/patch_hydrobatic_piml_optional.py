#!/usr/bin/env python3
"""
Make hydrobatic_localization's piml_msgs dependency OPTIONAL instead of REQUIRED.
Idempotent; safe to re-run. Removes nothing — everything comes back by itself.

WHY (2026-08-11)
----------------
smarc2 pins hydrobatic_localization at 08394b51, whose CMakeLists does
`find_package(piml_msgs REQUIRED)`. piml_msgs is in no repo we can reach: not in smarc2
(upstream `humble` has no messages/piml_msgs), not in smarc_modelling, no public GitHub repo.
It belongs to the physics-informed motion-model line of work.

The ESTIMATOR does not use it. At the pinned commit piml_msgs appears exactly twice:
  * state_estimator.h — one include, with no member/parameter/callback of that type (every
    thruster/LCG/VBS callback that might have used it is commented out upstream);
  * the `motion_model` EXECUTABLE (src/motion_model.cpp), which we never launch because
    sam_bringup.sh passes use_motion_model:=false.

WHAT THIS DOES (and why it is not deletion)
-------------------------------------------
  1. `find_package(piml_msgs REQUIRED)` -> `QUIET`, so a missing package is not fatal.
  2. The motion_model target, its install() entry, and piml_msgs as a state_estimator
     dependency are wrapped in `if(piml_msgs_FOUND)`. Present the day the package is: the
     motion model then builds with NO further edits.
  3. The header include becomes `#if __has_include(<piml_msgs/...>)`, so the compiler decides.

So this is the upgrade path for the AUV motion model, not a fork of it: drop piml_msgs into the
workspace (and smarc_modelling for the CasADi SAM model that SamMotionModel.h imports through
pybind), rebuild, and the motion_model node appears. Nothing to revert.
"""
import glob
import os
import re
import sys
from pathlib import Path

pkg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
ws = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "colcon_ws"
if pkg is None or not pkg.is_dir():
    sys.exit("usage: patch_hydrobatic_piml_optional.py <hydrobatic_localization PACKAGE dir> [workspace]")

hdr = pkg / "include" / "hydrobatic_localization" / "state_estimator.h"
cml = pkg / "CMakeLists.txt"
for f in (hdr, cml):
    if not f.is_file():
        sys.exit(f"ERROR: expected {f} — is this the package directory?")

TAG = "[data-cube] piml_msgs optional"
changed = []


def fix_relocated_topic_constants():
    """Repoint `<pkg>::msg::Topics::CONST` at whichever message package actually defines CONST.

    2026-08-11: the estimator says `smarc_msgs::msg::Topics::GPS_TOPIC`, but in this tree the
    constant lives in sam_msgs — commit c1619ee on branch fix/sam-msgs-gps-topic registered
    GPS_TOPIC='core/gps' there precisely so "hydrobatic_localization's state_estimator can
    address GPS via sam_msgs::Topics like every other core sensor instead of reaching into
    smarc_msgs". data-cube/scripts/vm/sync_vm_gps_fix.sh step 3 made exactly this edit on the
    old VM; this generalises it so ANY constant that moves between message packages is followed
    automatically, and only when the answer is unambiguous.
    """
    defs = {}                     # CONST -> {package, ...}
    for msg in (ws / "src").rglob("msg/Topics.msg"):
        owner = msg.parent.parent.name            # <pkg>/msg/Topics.msg
        for line in msg.read_text(errors="replace").splitlines():
            m = re.match(r'\s*string\s+([A-Z0-9_]+)\s*=', line)
            if m:
                defs.setdefault(m.group(1), set()).add(owner)
    if not defs:
        return []                                  # no workspace messages visible; leave alone
    # Several vehicles define the same constant name (GPS_TOPIC is in BOTH sam_msgs and
    # lolo_msgs), so "exactly one definition" is too strict — it rejected the very fix we need.
    # Disambiguate by what THIS package actually depends on: the message packages it includes
    # or that its CMakeLists names. hydrobatic depends on sam_msgs, never on lolo_msgs.
    known = set(re.findall(r'\b([a-z0-9_]+_msgs)\b', cml.read_text()))
    for f in list(pkg.rglob("*.cpp")) + list(pkg.rglob("*.h")) + list(pkg.rglob("*.hpp")):
        known |= set(re.findall(r'#\s*include\s*<([a-z0-9_]+_msgs)/', f.read_text(errors="replace")))
    ref_re = re.compile(r'\b([a-z0-9_]+)::msg::Topics::([A-Z0-9_]+)\b')
    out = []
    for f in list(pkg.rglob("*.cpp")) + list(pkg.rglob("*.h")) + list(pkg.rglob("*.hpp")):
        txt = f.read_text(errors="replace")
        new = txt
        for used_pkg, const in set(ref_re.findall(txt)):
            owners = defs.get(const, set())
            if not owners or used_pkg in owners:
                continue                           # unknown constant, or already correct
            candidates = owners & known            # only packages this one actually depends on
            if len(candidates) != 1:
                continue                           # still ambiguous — leave it to a human
            owner = next(iter(candidates))
            new = new.replace(f"{used_pkg}::msg::Topics::{const}",
                              f"{owner}::msg::Topics::{const}")
            out.append(f"{f.relative_to(pkg)}: {const} -> {owner}::msg::Topics "
                       f"(was {used_pkg}, which does not define it)")
        if new != txt:
            f.write_text(new)
    return out


def ensure_geographiclib_target():
    """Guarantee GeographicLib::GeographicLib exists however find_package resolved.

    2026-08-11: the CMakeLists links the imported target, which only upstream's CONFIG package
    provides. Ubuntu ships just the legacy Find MODULE (variables, no target) — and if anything
    puts that module on CMAKE_MODULE_PATH, module mode wins, the config shim is never consulted,
    and generation dies with "target was not found". Defining the target from the variables when
    it is absent makes the package build under either resolution, with or without the shim.
    """
    txt = cml.read_text()
    if "GeographicLib::GeographicLib" not in txt or "_dc_geographiclib" in txt:
        return []
    m = re.search(r'^[ \t]*find_package\s*\(\s*GeographicLib[^)]*\)[ \t]*$', txt, flags=re.M)
    if not m:
        return []
    shim = ("\n# [data-cube] _dc_geographiclib: Ubuntu's Find module defines no imported target;\n"
            "# synthesise it from the variables so the link line works in module OR config mode.\n"
            "if(NOT TARGET GeographicLib::GeographicLib)\n"
            "  add_library(GeographicLib::GeographicLib INTERFACE IMPORTED)\n"
            "  set_target_properties(GeographicLib::GeographicLib PROPERTIES\n"
            '    INTERFACE_LINK_LIBRARIES "${GeographicLib_LIBRARIES}"\n'
            '    INTERFACE_INCLUDE_DIRECTORIES "${GeographicLib_INCLUDE_DIRS}")\n'
            "endif()\n")
    cml.write_text(txt[:m.end()] + "\n" + shim + txt[m.end():])
    return ["CMakeLists.txt: GeographicLib::GeographicLib synthesised when the Find module "
            "(no imported target) is what answered"]


def declare_missing_ros_deps():
    """Declare every ROS package whose headers the sources include but that CMakeLists never
    find_package()s.

    2026-08-11: state_estimator.h includes <ament_index_cpp/get_package_share_directory.hpp>.
    The header IS installed — but ament_index_cpp appears nowhere in CMakeLists, so the target
    is handed no -I for it and the compile dies on a header sitting right there on disk. An
    existence check can't see this; only the DECLARATION can. Same trap would fire for any
    other undeclared include, so fix the class: scan the includes that actually get compiled,
    map each to an installed ROS package, and declare whatever is missing.
    """
    inc_re = re.compile(r'^\s*#\s*include\s*<([^>]+)/([^>]+)>', re.M)
    # Where a ROS package's headers live: /opt/ros/<distro>/include/<pkg>/ and <ws>/install/<pkg>/
    ros_pkg_dirs = {}
    for d in glob.glob("/opt/ros/*/include/*") + glob.glob(str(ws / "install" / "*" / "include" / "*")):
        if os.path.isdir(d):
            ros_pkg_dirs.setdefault(os.path.basename(d), d)
    sources = [f for f in list(pkg.rglob("*.h")) + list(pkg.rglob("*.hpp")) + list(pkg.rglob("*.cpp"))
               if f.name != "motion_model.cpp" and "/test/" not in str(f)]
    needed = set()
    for f in sources:
        for first, _rest in inc_re.findall(f.read_text(errors="replace")):
            if first in ros_pkg_dirs and first != "piml_msgs":
                needed.add(first)
    txt = cml.read_text()
    missing = sorted(p for p in needed if not re.search(rf'\b{re.escape(p)}\b', txt))
    if not missing:
        return []
    block = [f"\n# {TAG.replace('piml_msgs optional', 'declared deps')}: headers these packages "
             f"provide are included by the sources, but CMakeLists never find_package()d them, so "
             f"the target got no -I and the compile failed on a header that was installed.",
             "foreach(_dc_dep " + " ".join(missing) + ")",
             "  find_package(${_dc_dep} REQUIRED)",
             "endforeach()",
             "ament_target_dependencies(state_estimator " + " ".join(missing) + ")",
             "if(TARGET logger)",
             "  ament_target_dependencies(logger " + " ".join(missing) + ")",
             "endif()\n"]
    anchor = re.search(r'^\s*ament_package\(\)\s*$', txt, flags=re.M)
    new = (txt[:anchor.start()] + "\n".join(block) + "\n" + txt[anchor.start():]) if anchor \
        else txt + "\n".join(block)
    cml.write_text(new)
    return [f"CMakeLists.txt: declared missing ROS dependencies: {', '.join(missing)}"]


def jazzy_include_compat():
    """ROS Jazzy removed the deprecated .h spellings of the tf2_*_msgs conversion headers
    (kept through Humble); only the .hpp forms exist now. The pinned commit mixes both —
    state_estimator.h says tf2_geometry_msgs.h while loggerNode.cpp already says .hpp — so
    rewrite the whole family across the package. Idempotent: writes only when a match exists."""
    renames = {
        "<tf2_geometry_msgs/tf2_geometry_msgs.h>": "<tf2_geometry_msgs/tf2_geometry_msgs.hpp>",
        "<tf2_eigen/tf2_eigen.h>": "<tf2_eigen/tf2_eigen.hpp>",
        "<tf2_sensor_msgs/tf2_sensor_msgs.h>": "<tf2_sensor_msgs/tf2_sensor_msgs.hpp>",
    }
    out = []
    for f in sorted(list(pkg.rglob("*.h")) + list(pkg.rglob("*.hpp")) + list(pkg.rglob("*.cpp"))):
        txt = f.read_text()
        new = txt
        for old, rep in renames.items():
            new = new.replace(old, rep)
        if new != txt:
            f.write_text(new)
            out.append(f"{f.relative_to(pkg)}: .h include(s) renamed to .hpp (removed in Jazzy)")
    return out

# A superseded version of this patch (2026-08-11 morning) DELETED the motion_model target instead
# of guarding it. A checkout carrying those edits no longer matches the patterns below, so reset
# our own handiwork to pristine upstream first — but only ours, and only if the file is dirty, so
# a genuine hand edit is never clobbered.
# Match on the "[data-cube]" prefix alone, NOT on a specific sentence: the superseded patch
# worded its header and CMake markers differently, the reset silently didn't fire, and the run
# died on text it had written itself (2026-08-11). Any file carrying any marker of ours is ours
# to reset.
OLD_MARKERS = ("[data-cube]",)
import subprocess

# Already in the desired state? Then touch nothing — rewriting identical content would bump the
# mtimes and make colcon rebuild the estimator on every install run.
if (TAG in cml.read_text() and "find_package(piml_msgs QUIET)" in cml.read_text()
        and TAG in hdr.read_text()):
    # piml already handled — but STILL run the Jazzy header compat (2026-08-11: this early
    # exit skipped it, so the .h include survived a "successful" patch run and the compile
    # failed after the patch had reported everything fine).
    later = (jazzy_include_compat() + fix_relocated_topic_constants()
             + ensure_geographiclib_target() + declare_missing_ros_deps())
    for c in later:
        print("  -", c)
    print("hydrobatic_localization: piml_msgs already optional — "
          + ("build compat applied." if later else "no change."))
    raise SystemExit(0)
for f in (hdr, cml):
    txt = f.read_text()
    if not any(m in txt for m in OLD_MARKERS):
        continue
    try:
        dirty = subprocess.run(["git", "-C", str(pkg), "status", "--porcelain", "--", str(f)],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        if dirty:
            subprocess.run(["git", "-C", str(pkg), "checkout", "--", str(f)],
                           check=True, capture_output=True, timeout=20)
            changed.append(f"{f.name}: reset to pristine upstream before patching")
    except Exception as e:                                   # not a git tree, git missing, ...
        print(f"note: could not git-reset {f.name} ({e}); patching in place")

# ---- 1. header: let the compiler decide whether the (unused) include resolves ---------------
src = hdr.read_text()
m = None
if "piml_msgs" in src and TAG not in src:
    m = re.search(r'^([ \t]*)#include\s*(<piml_msgs/[^>]+>).*$', src, flags=re.M)
if m is None:
    # No ACTIVE include: either only a comment mentions piml_msgs, or an earlier patch already
    # commented the include out and the git reset above could not undo it. Nothing can fail to
    # compile in that state — say so and carry on rather than abort the estimator build.
    if "piml_msgs" in src and TAG not in src:
        print("note: no active piml_msgs include in state_estimator.h — nothing to guard")
else:
    indent, inc = m.group(1), m.group(2)
    hdr.write_text(src[:m.start()] +
                   f"{indent}// {TAG}: unused here, and piml_msgs may be absent.\n"
                   f"{indent}#if __has_include({inc})\n"
                   f"{indent}#  include {inc}\n"
                   f"{indent}#endif" +
                   src[m.end():])
    changed.append("state_estimator.h: include guarded with __has_include")

# ---- 2. CMake: optional find_package + conditional motion_model -----------------------------
src = cml.read_text()
if "piml_msgs" in src and TAG not in src:
    out = src

    # 2a. not fatal when absent
    out = re.sub(r'^([ \t]*)find_package\(piml_msgs REQUIRED\).*$',
                 rf'\1find_package(piml_msgs QUIET)  # {TAG} (was REQUIRED)',
                 out, flags=re.M)

    # 2b. piml_msgs as a state_estimator dep -> re-added conditionally after the call.
    #     ament_target_dependencies() is additive, so a second call is legal.
    out = re.sub(r'^[ \t]*piml_msgs[ \t]*\n(?=(?:[ \t]*\w+[ \t]*\n)*\))', '', out, count=1, flags=re.M)

    # 2c. the motion_model executable + its deps, bounded by the next comment line
    # install(TARGETS motion_model) MUST come after add_executable(motion_model), so it lives
    # inside this same guard rather than with the other conditional bits below.
    def guard_motion_model(text):
        pat = re.compile(r'(add_executable\(motion_model.*?\n\)\s*\n)(?=# add logger node executable)',
                         re.S)
        return pat.sub(
            lambda m: (f"if(piml_msgs_FOUND)  # {TAG}\n{m.group(1)}"
                       "install(TARGETS motion_model DESTINATION lib/${PROJECT_NAME})\n"
                       "endif()\n\n"),
            text, count=1)
    out = guard_motion_model(out)

    # 2d. motion_model in install(TARGETS ...) -> its own conditional install
    out = re.sub(r'^[ \t]*motion_model[ \t]*\n', '', out, count=1, flags=re.M)

    # 2e. append the conditional bits once, after the state_estimator target block
    anchor = re.search(r'^target_link_libraries\(state_estimator.*?^\s*\)\s*$', out, flags=re.M | re.S)
    if not anchor:
        sys.exit("ERROR: could not find target_link_libraries(state_estimator ...) to anchor on")
    addition = (f"\n# {TAG}: these come back automatically once piml_msgs is in the workspace.\n"
                "if(piml_msgs_FOUND)\n"
                "  ament_target_dependencies(state_estimator piml_msgs)\n"
                "else()\n"
                '  message(WARNING "piml_msgs not found: building the estimator without the '
                'motion_model node (use_motion_model:=false). See scripts/'
                'patch_hydrobatic_piml_optional.py")\n'
                "endif()\n")
    out = out[:anchor.end()] + "\n" + addition + out[anchor.end():]

    if out == src:
        sys.exit("ERROR: CMakeLists mentions piml_msgs but nothing matched — the pinned commit "
                 "may have moved; patch by hand.")
    cml.write_text(out)
    changed.append("CMakeLists.txt: find_package QUIET; motion_model target, its install and the "
                   "state_estimator dep wrapped in if(piml_msgs_FOUND)")

# ---- 3. Jazzy header compat (after the git reset, so it survives) ---------------------------
changed += jazzy_include_compat()

# ---- 4. follow message constants that moved between packages --------------------------------
changed += fix_relocated_topic_constants()

# ---- 5. make the GeographicLib link target exist in either resolution mode -------------------
changed += ensure_geographiclib_target()

# ---- 6. declare ROS packages the sources include but CMakeLists never asked for --------------
changed += declare_missing_ros_deps()

# ---- report --------------------------------------------------------------------------------
if changed:
    print("hydrobatic_localization: piml_msgs is now optional (nothing removed):")
    for c in changed:
        print("  -", c)
else:
    print("hydrobatic_localization already patched — no change.")

# Anything still demanding piml_msgs unconditionally?
cml_txt, hdr_txt = cml.read_text(), hdr.read_text()
if re.search(r'find_package\(piml_msgs REQUIRED\)', cml_txt):
    sys.exit("ERROR: find_package(piml_msgs REQUIRED) still present")
if re.search(r'^[ \t]*#include\s*<piml_msgs/', hdr_txt, flags=re.M):
    sys.exit("ERROR: an unguarded piml_msgs include remains in state_estimator.h")
print("verified: piml_msgs is referenced only where it is optional.")
