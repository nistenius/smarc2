# Building `hydrobatic_localization` — GTSAM requirement

`hydrobatic_localization` (the SAM dead-reckoning / state estimator) does **not** build against the
packaged `ros-<distro>-gtsam`. It uses the post-boost GTSAM API (`gtsam::OptionalMatrixType`,
`OptionalNone`, `NavState::rotation()`) and the **unstable** module (`BatchFixedLagSmoother`).

- First GTSAM release tag containing `OptionalMatrixType`: **`4.3a0`**.
- `ros-jazzy-gtsam` is 4.2-era → build fails (`OptionalMatrixType` undeclared,
  `gtsam/nonlinear/BatchFixedLagSmoother.h: No such file`, `NavState` has no `rotation()`).

## Fix
```bash
smarc2/scripts/install_gtsam_for_hydrobatic.sh   # builds GTSAM 4.3a0 + unstable, system Eigen
```
then rebuild the package (and init its nested `smarc_modelling` submodule) — see the script's
closing note.

## TODO for the SMaRC team
- Pin the exact GTSAM commit/tag in `hydrobatic_localization`'s `package.xml` / README (currently
  unpinned; this file infers `4.3a0` from the API used). Confirm with @ignaciotb.
- Consider wiring this into `docker/Dockerfile` so `colcon build` works out of the box in CI/image.
- `hydrobatic_localization/package.xml` does not declare a `gtsam` dependency — add one.
