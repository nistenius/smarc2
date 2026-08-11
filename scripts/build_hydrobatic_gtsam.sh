#!/usr/bin/env bash
# build_hydrobatic_gtsam.sh
# One-shot: build the GTSAM that hydrobatic_localization (branch asko26) actually needs,
# then rebuild the node against it.
#
# WHY: hydrobatic uses the post-boost GTSAM API (gtsam::OptionalMatrixType / OptionalNone,
# NavState::rotation()) and gtsam/nonlinear/BatchFixedLagSmoother.h. The FIRST GTSAM tag
# that ships these is 4.3a0. Jazzy's packaged ros-jazzy-gtsam is 4.2-era, so colcon fails
# with "OptionalMatrixType undeclared / BatchFixedLagSmoother.h not found / NavState has no
# rotation()". We build 4.3a0 to /usr/local and point hydrobatic at it explicitly.
#
# NON-DESTRUCTIVE: the packaged ros-jazzy-gtsam (4.2) is left in place; hydrobatic is the
# only GTSAM consumer here and is pinned to /usr/local via -DGTSAM_DIR, and the two libs have
# different SONAMEs so they don't collide at runtime.
# NO `set -u` here (2026-08-11): step [5/5] sources /opt/ros/jazzy/setup.bash, which reads unbound
# variables, so `set -u` kills this script at "AMENT_TRACE_SETUP_FILES: unbound variable" — AFTER
# GTSAM has been built and installed, i.e. it looks like the GTSAM step failed when in fact only
# the hydrobatic colcon build never ran. Same trap as live-sim-runbook.md's symptom table.
set -o pipefail

GTSAM_TAG="${GTSAM_TAG:-4.3a0}"
GTSAM_SRC="${GTSAM_SRC:-$HOME/gtsam}"
WS="${WS:-$HOME/colcon_ws}"
JOBS="${JOBS:-$(nproc)}"
# Extra cmake args for the GTSAM build, if you ever need to experiment without editing this.
GTSAM_EXTRA_CMAKE_ARGS="${GTSAM_EXTRA_CMAKE_ARGS:-}"
# Set to 1 to rebuild GTSAM even when 4.3a0 is already installed in /usr/local.
GTSAM_FORCE_REBUILD="${GTSAM_FORCE_REBUILD:-0}"

step(){ echo; echo "==== $* ===="; }
die(){ echo; echo "!!!! FAILED at: $* !!!!"; exit 1; }

step "[1/5] apt deps (sudo will prompt for your password)"
# A broken borglab gtsam-4.1 PPA (404 on noble) can make apt-get update fail; disable it if present.
sudo sh -c 'grep -rlZ "borglab/gtsam-release-4.1" /etc/apt/sources.list.d 2>/dev/null | xargs -0 -r -I{} mv {} {}.disabled' || true
# Tolerate residual apt-source warnings: the deps below come from the standard Ubuntu repos.
sudo apt-get update || echo "WARN: apt-get update reported errors (continuing; deps come from standard repos)"
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git \
  libboost-all-dev libtbb-dev libeigen3-dev \
  libgeographiclib-dev libyaml-cpp-dev pybind11-dev || die "apt-get install deps"

# Idempotence (2026-08-11): rebuilding GTSAM from scratch is 20-40 min on ARM64. If the right
# version is already installed, go straight to the hydrobatic build — which is what you want when
# the installer is re-run after a failure further down.
# The version-stamped SONAME is the reliable marker: `make install` writes
# /usr/local/lib/libgtsam.so.<TAG> verbatim (seen in the 2026-08-11 install log).
GTSAM_INSTALLED=0
if [ "$GTSAM_FORCE_REBUILD" != "1" ] \
   && [ -f "/usr/local/lib/libgtsam.so.$GTSAM_TAG" ] \
   && [ -f /usr/local/lib/cmake/GTSAM/GTSAMConfig.cmake ]; then
  GTSAM_INSTALLED=1
fi

if [ "$GTSAM_INSTALLED" = "1" ]; then
  step "[2-3/5] GTSAM $GTSAM_TAG already installed in /usr/local — skipping rebuild"
  echo "  (force one with GTSAM_FORCE_REBUILD=1)"
else
step "[2/5] fetch GTSAM $GTSAM_TAG"
if [ ! -d "$GTSAM_SRC/.git" ]; then
  git clone https://github.com/borglab/gtsam.git "$GTSAM_SRC" || die "git clone gtsam"
fi
cd "$GTSAM_SRC" || die "cd $GTSAM_SRC"
git fetch --tags --quiet
git checkout "$GTSAM_TAG" || die "git checkout $GTSAM_TAG"

step "[3/5] build + install GTSAM $GTSAM_TAG (+unstable, system Eigen) -> /usr/local"
rm -rf build && mkdir build && cd build || die "mkdir build"
# GTSAM_COMPILE_OPTIONS_PRIVATE_COMMON: GTSAM's own default is
#     -Werror -Wall -Wpedantic -Wextra ... -Wsuggest-override
# i.e. a BLANKET -Werror on its own sources. On GCC 13.3 (Ubuntu 24.04 noble-updates, 2026-08-11)
# -Woverloaded-virtual fires inside GTSAM's own discrete/DecisionTreeFactor.h, so every file that
# includes it fails: "'...operator*' was hidden [-Werror=overloaded-virtual=]" followed by
# "cc1plus: all warnings being treated as errors". Nothing to do with our code; the same pinned
# tag built fine on the older VM's GCC. GTSAM marks this variable "(User editable)" and sets it
# with a plain `set(... CACHE STRING ...)` (no FORCE), so a -D on the command line wins outright
# -- no source patching, no sed. Warnings stay ON, they just stop being fatal.
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=ON \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_COMPILE_OPTIONS_PRIVATE_COMMON=-Wall \
  $GTSAM_EXTRA_CMAKE_ARGS || die "cmake configure gtsam"
make -j"$JOBS" || die "make gtsam"
sudo make install || die "sudo make install gtsam"
sudo ldconfig
fi

step "[4/5] remove stale hydrobatic build artifacts (were built vs 4.2)"
cd "$WS" || die "cd $WS"
rm -rf build/hydrobatic_localization install/hydrobatic_localization

step "[5/5] colcon build hydrobatic_localization against /usr/local GTSAM"
# The source must actually BE here. 2026-08-11: it wasn't — the submodule was never initialised,
# so colcon printed "ignoring unknown package ... in --packages-select", finished 0 packages, and
# this script cheerfully declared BUILD OK while the estimator did not exist. A build that builds
# nothing is a failure, and it must say so where the estimator is concerned.
HL_PKG_XML="$(find "$WS/src" -name package.xml -path '*hydrobatic_localization*' -print -quit 2>/dev/null)"
if [ -z "$HL_PKG_XML" ]; then
  echo
  echo "!!!! hydrobatic_localization SOURCE IS MISSING from $WS/src !!!!"
  echo "     It is a git submodule (navigation/dead_reckoning/hydrobatic_localization) and the"
  echo "     directory is empty. Without it there is NO state_estimator: the vehicle will accept"
  echo "     missions and sit still. Fetch it (public repo, no SSH key needed):"
  echo
  echo "       cd $WS/src/smarc2 && git -c url.\"https://github.com/\".insteadOf=\"git@github.com:\" \\"
  echo "         submodule update --init --recursive navigation/dead_reckoning/hydrobatic_localization/"
  echo
  exit 1
fi
# shellcheck disable=SC1091
set +u
source /opt/ros/jazzy/setup.bash
# The estimator find_package()s messages that live in THIS workspace (dead_reckoning_msgs,
# sam_msgs, smarc_msgs). Sourcing only /opt/ros leaves them off CMAKE_PREFIX_PATH and CMake
# fails with "Could not find a package configuration file provided by dead_reckoning_msgs"
# (2026-08-11) — they were built minutes earlier, three directories away.
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
set +u

# Name every missing message package up front; a CMake find_package error names only the first.
_missing_deps=""
for _p in dead_reckoning_msgs sam_msgs smarc_msgs; do
  ros2 pkg prefix "$_p" >/dev/null 2>&1 || _missing_deps="$_missing_deps $_p"
done
if [ -n "$_missing_deps" ]; then
  die "these workspace messages are missing, build the workspace first:$_missing_deps"
fi

# GeographicLib: Ubuntu's libgeographiclib-dev ships ONLY the legacy module
# /usr/share/cmake/geographiclib/FindGeographicLib.cmake, which sets variables and defines NO
# imported target. hydrobatic does `find_package(GeographicLib REQUIRED)` (config mode, so the
# module is never even seen -> "Could not find GeographicLibConfig.cmake", 2026-08-11) and then
# links `GeographicLib::GeographicLib`, which only upstream's config package provides.
# Rather than patch upstream's CMakeLists, supply the missing config package over the apt library:
# one file, no second copy of the library, and it disappears the day Ubuntu ships a real one.
GL_CFG_DIR=/usr/local/lib/cmake/GeographicLib
if [ ! -f "$GL_CFG_DIR/GeographicLibConfig.cmake" ] && \
   [ ! -f /usr/lib/cmake/GeographicLib/GeographicLibConfig.cmake ]; then
  step "installing a GeographicLib CMake config shim over the apt library ($GL_CFG_DIR)"
  sudo mkdir -p "$GL_CFG_DIR"
  sudo tee "$GL_CFG_DIR/GeographicLibConfig.cmake" >/dev/null <<'CFGEOF'
# Written by smarc2/scripts/build_hydrobatic_gtsam.sh (data-cube), 2026-08-11.
# Ubuntu's libgeographiclib-dev ships only FindGeographicLib.cmake (variables, no target), but
# consumers such as hydrobatic_localization link the imported target GeographicLib::GeographicLib
# that upstream's own install provides. This exposes the apt-installed library under that name.
# Delete this file if you ever install GeographicLib from source into /usr/local.
if(NOT TARGET GeographicLib::GeographicLib)
  find_path(GeographicLib_INCLUDE_DIR NAMES GeographicLib/Config.h)
  find_library(GeographicLib_LIBRARY NAMES GeographicLib)
  if(GeographicLib_INCLUDE_DIR AND GeographicLib_LIBRARY)
    add_library(GeographicLib::GeographicLib UNKNOWN IMPORTED)
    set_target_properties(GeographicLib::GeographicLib PROPERTIES
      IMPORTED_LOCATION "${GeographicLib_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GeographicLib_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "GeographicLib shim: apt library/headers not found — apt install libgeographiclib-dev")
  endif()
endif()
set(GeographicLib_LIBRARIES GeographicLib::GeographicLib)
set(GeographicLib_INCLUDE_DIRS "${GeographicLib_INCLUDE_DIR}")
set(GeographicLib_FOUND TRUE)
CFGEOF
  echo "  wrote $GL_CFG_DIR/GeographicLibConfig.cmake"
fi

# piml_msgs: the pinned hydrobatic commit find_package()s it REQUIRED, but it exists in no repo we
# can reach and the ESTIMATOR never uses it (one dead include; the only consumer is the
# motion_model executable, and we launch with use_motion_model:=false). Make it OPTIONAL rather
# than remove anything: the motion_model node then builds itself the day piml_msgs turns up.
# Always run — the patch is idempotent and a no-op once applied.
_patch="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_hydrobatic_piml_optional.py"
if [ -f "$_patch" ]; then
  python3 "$_patch" "$(dirname "$HL_PKG_XML")" "$WS" || die "patch_hydrobatic_piml_optional.py failed"
else
  echo "WARN: $_patch not found — CMake will fail on find_package(piml_msgs REQUIRED)."
fi

# Include-resolution preflight (2026-08-11, after tf2_geometry_msgs.h cost a full round trip):
# resolve every <pkg/header> include in the package against the include roots the compiler will
# see, and report EVERY unresolvable one at once — a compiler stops at the first.
# ADVISORY, never a gate: its job is to list every missing include at once (a compiler names only
# the first), not to decide whether the build may proceed. It got that wrong once already — it
# didn't know about the package's own include/ dir and blocked a healthy build — so the compiler
# stays the authority and a preflight complaint only prints.
python3 - "$(dirname "$HL_PKG_XML")" "$WS" <<'PYEOF' || echo "  (advisory only — continuing; the compiler decides)"
import glob, os, re, sys
pkg, ws = sys.argv[1], sys.argv[2]
# The package's OWN include dir comes first: CMakeLists does
# include_directories(${PROJECT_SOURCE_DIR}/include/), so <hydrobatic_localization/DvlFactor.h>
# resolves there. Omitting it made the preflight flag six of the package's own headers as
# missing and block a build that would have been fine (2026-08-11).
roots = [os.path.join(pkg, "include"), pkg,
         "/usr/include", "/usr/include/eigen3", "/usr/local/include"]
for pat in ("/opt/ros/*/include", "/opt/ros/*/include/*", os.path.join(ws, "install/*/include"),
            os.path.join(ws, "install/*/include/*")):
    roots += [d for d in glob.glob(pat) if os.path.isdir(d)]
inc_re = re.compile(r'^\s*#\s*include\s*<([^>]+/[^>]+)>', re.M)
skip_prefixes = ("piml_msgs/",)          # optional by design (guarded with __has_include)
skip_files = {"motion_model.cpp"}        # only built when piml_msgs exists
missing = {}
for f in sorted(glob.glob(os.path.join(pkg, "**", "*.*"), recursive=True)):
    if not f.endswith((".h", ".hpp", ".cpp")) or os.path.basename(f) in skip_files:
        continue
    for inc in inc_re.findall(open(f, errors="replace").read()):
        if inc.startswith(skip_prefixes):
            continue
        if not any(os.path.exists(os.path.join(r, inc)) for r in roots):
            missing.setdefault(inc, []).append(os.path.relpath(f, pkg))
# An include can exist on disk and STILL fail: if CMakeLists never find_package()s the package,
# the target is handed no -I for it (ament_index_cpp, 2026-08-11). The patch script declares
# these automatically; flag any that slipped through.
cml_txt = open(os.path.join(pkg, "CMakeLists.txt"), errors="replace").read()
ros_pkgs = {os.path.basename(d) for d in glob.glob("/opt/ros/*/include/*") +
            glob.glob(os.path.join(ws, "install/*/include/*")) if os.path.isdir(d)}
undeclared = {}
for f in sorted(glob.glob(os.path.join(pkg, "**", "*.*"), recursive=True)):
    if not f.endswith((".h", ".hpp", ".cpp")) or os.path.basename(f) in skip_files:
        continue
    for inc in inc_re.findall(open(f, errors="replace").read()):
        first = inc.split("/")[0]
        if first in ros_pkgs and first != "piml_msgs" and \
                not re.search(r'\b%s\b' % re.escape(first), cml_txt):
            undeclared.setdefault(first, []).append(os.path.relpath(f, pkg))
if undeclared:
    print("PREFLIGHT: these ROS packages are INCLUDED but never find_package()d, so the target "
          "gets no -I for them (the header exists; the compile still fails):")
    for p, files in sorted(undeclared.items()):
        print(f"  {p}   included by: {', '.join(sorted(set(files)))}")
if missing:
    print("PREFLIGHT: these includes resolve NOWHERE on this machine (all of them, not just the "
          "first the compiler would hit):")
    for inc, files in sorted(missing.items()):
        print(f"  <{inc}>   used by: {', '.join(sorted(set(files)))}")
    sys.exit(1)
print("include preflight OK — every package-style include in the estimator resolves.")
PYEOF

# -DBUILD_TESTING=OFF matches the recipe that worked on the old VM
# (data-cube/scripts/vm/sync_vm_gps_fix.sh step 4): the test targets pull CppUnitLite and are
# not what we fly.
#
# Do NOT add -DCMAKE_MODULE_PATH=/usr/share/cmake/geographiclib (2026-08-11): that old script
# used it, but with it find_package(GeographicLib) resolves in MODULE mode against Ubuntu's
# FindGeographicLib.cmake, which sets variables and defines NO imported target — so it silently
# bypasses our config shim and the build then fails at
# `target_link_libraries ... GeographicLib::GeographicLib but the target was not found`.
# ...and a -D from an EARLIER run is still in that build dir's CMakeCache: dropping it from the
# command line does not unset a cache entry. Wipe the build dir if the stale entry is there, or
# the very run that removes the flag would still resolve in module mode.
_hl_cache="$WS/build/hydrobatic_localization/CMakeCache.txt"
if [ -f "$_hl_cache" ] && grep -q '^CMAKE_MODULE_PATH:.*geographiclib' "$_hl_cache"; then
  echo "  clearing build/hydrobatic_localization (stale CMAKE_MODULE_PATH=geographiclib in cache)"
  rm -rf "$WS/build/hydrobatic_localization"
fi

colcon build --packages-select hydrobatic_localization \
  --cmake-args -DGTSAM_DIR=/usr/local/lib/cmake/GTSAM -DCMAKE_BUILD_TYPE=Release \
               -DBUILD_TESTING=OFF \
  || die "colcon build hydrobatic_localization"

[ -d "$WS/install/hydrobatic_localization" ] \
  || die "colcon reported success but $WS/install/hydrobatic_localization does not exist"

echo
echo "##################################################################"
echo "####  BUILD OK - hydrobatic_localization built vs GTSAM $GTSAM_TAG"
echo "##################################################################"
echo "Next:  source $WS/install/setup.bash   then re-run the bringup."
