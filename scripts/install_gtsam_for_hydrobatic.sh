#!/usr/bin/env bash
# Build + install the GTSAM that `hydrobatic_localization` requires, which the packaged
# ros-<distro>-gtsam does NOT satisfy.
#
# Evidence: hydrobatic_localization uses the post-boost GTSAM API (gtsam::OptionalMatrixType /
# OptionalNone, NavState::rotation()) plus gtsam_unstable (BatchFixedLagSmoother). The FIRST GTSAM
# release tag that contains OptionalMatrixType is 4.3a0; ros-jazzy-gtsam is 4.2-era and fails to
# compile the node (OptionalMatrixType undeclared, BatchFixedLagSmoother.h missing, NavState has
# no rotation()). Pin: 4.3a0 (override GTSAM_TAG only after confirming a newer tag still matches
# hydrobatic's API with @ignaciotb).
set -euo pipefail

GTSAM_TAG="${GTSAM_TAG:-4.3a0}"
GTSAM_SRC="${GTSAM_SRC:-$HOME/gtsam}"
JOBS="${JOBS:-$(nproc)}"

echo "== deps =="
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git \
  libboost-all-dev libtbb-dev libeigen3-dev \
  libgeographiclib-dev libyaml-cpp-dev pybind11-dev

# hydrobatic_localization pulls GTSAM via CMake find_package(GTSAM), NOT via an ament <depend>, and
# is the only GTSAM consumer here. Cleanest is to remove the incompatible packaged GTSAM so
# find_package resolves to the 4.3a0 we install to /usr/local below. (If apt refuses due to other
# dependents, skip this and instead build hydrobatic with:
#   colcon build --packages-select hydrobatic_localization \
#     --cmake-args -DGTSAM_DIR=/usr/local/lib/cmake/GTSAM )
PKG="$(dpkg -l 2>/dev/null | awk '/ros-.*-gtsam/{print $2}' | head -1 || true)"
if [ -n "${PKG:-}" ]; then
  echo "== removing packaged ${PKG} (incompatible 4.2) =="
  sudo apt-get remove -y "$PKG" || echo "  (apt refused; use the -DGTSAM_DIR override noted above)"
fi

echo "== building GTSAM ${GTSAM_TAG} (+ unstable, system Eigen) =="
[ -d "$GTSAM_SRC/.git" ] || git clone https://github.com/borglab/gtsam.git "$GTSAM_SRC"
cd "$GTSAM_SRC"
git fetch --tags --quiet
git checkout "$GTSAM_TAG"
rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=ON \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_PYTHON=OFF
make -j"$JOBS"
sudo make install
sudo ldconfig

cat <<'NEXT'

== GTSAM installed. Finish with: ==
  # the localizer needs its nested smarc_modelling submodule:
  git -C <ws>/src/smarc2/navigation/dead_reckoning/hydrobatic_localization submodule update --init --recursive
  cd <ws>
  colcon build --packages-select hydrobatic_localization
NEXT
