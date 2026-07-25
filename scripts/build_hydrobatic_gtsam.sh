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
set -uo pipefail

GTSAM_TAG="${GTSAM_TAG:-4.3a0}"
GTSAM_SRC="${GTSAM_SRC:-$HOME/gtsam}"
WS="${WS:-$HOME/colcon_ws}"
JOBS="${JOBS:-$(nproc)}"

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

step "[2/5] fetch GTSAM $GTSAM_TAG"
if [ ! -d "$GTSAM_SRC/.git" ]; then
  git clone https://github.com/borglab/gtsam.git "$GTSAM_SRC" || die "git clone gtsam"
fi
cd "$GTSAM_SRC" || die "cd $GTSAM_SRC"
git fetch --tags --quiet
git checkout "$GTSAM_TAG" || die "git checkout $GTSAM_TAG"

step "[3/5] build + install GTSAM $GTSAM_TAG (+unstable, system Eigen) -> /usr/local"
rm -rf build && mkdir build && cd build || die "mkdir build"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=ON \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_PYTHON=OFF || die "cmake configure gtsam"
make -j"$JOBS" || die "make gtsam"
sudo make install || die "sudo make install gtsam"
sudo ldconfig

step "[4/5] remove stale hydrobatic build artifacts (were built vs 4.2)"
cd "$WS" || die "cd $WS"
rm -rf build/hydrobatic_localization install/hydrobatic_localization

step "[5/5] colcon build hydrobatic_localization against /usr/local GTSAM"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
colcon build --packages-select hydrobatic_localization \
  --cmake-args -DGTSAM_DIR=/usr/local/lib/cmake/GTSAM -DCMAKE_BUILD_TYPE=Release \
  || die "colcon build hydrobatic_localization"

echo
echo "##################################################################"
echo "####  BUILD OK - hydrobatic_localization built vs GTSAM $GTSAM_TAG"
echo "##################################################################"
echo "Next:  source $WS/install/setup.bash   then re-run the bringup."
