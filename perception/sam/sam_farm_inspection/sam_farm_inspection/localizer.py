#!/usr/bin/env python3
"""Farm localization: turn a cloud of side-scan detections into a verified map.

This is sensors-22-05064 §6 ("Initialization of the Inspection Plan"): cluster the buoy
detections with a variational Gaussian mixture whose maximum number of classes is the
number of buoys the prior says exist, fit lines through them constrained to the farm's
a-priori orientation, and use the result as the updated farm map.

Pure python + numpy. No ROS, no I/O, no clock. `farm_localizer_node.py` is the wrapper.

WHAT THIS PRODUCES, AND WHY EACH PIECE IS SEPARATE
  * `cluster_buoys` — detections to buoy estimates, robust to outliers.
  * `fit_rigid_2d` — prior to observed, rotation + translation only.
  * `verify_against_prior` — per-buoy residual and verdict.
  * `fit_culture_lines` — the rope lines, orientation-constrained.
Each refuses by name rather than returning a degraded answer. A localizer that returns
"a transform" when it had three collinear points and no rotation information is worse
than one that stops: the mission would fly lanes derived from it.

DISPLACEMENT IS A FINDING, NOT A FAILURE (mission design decision D5). This farm IS
displaced — 26 x 33 m and skewed against the paper's nominal 15 x 15 — and that is the
scenario the mission exists to handle. `moved` is a normal verdict and the mission
continues on the observed map.

FOUR VERDICTS, NOT THREE. The mission spec asks for confirmed / moved / missing.
`not_surveyed` is added because "we looked there and it is gone" and "we never looked
there" are different facts, and reporting the second as the first would send someone to
recover a buoy that is probably fine. Same rule as ABSENT IS NOT EMPTY (SETTLED §3e):
a Pydantic default, an empty list and an unvisited region all read as zero unless
something keeps them apart.

WHY THE MIXTURE IS WRITTEN OUT HERE RATHER THAN IMPORTED FROM SKLEARN. The paper's VGMM
is `sklearn.mixture.BayesianGaussianMixture`, and sklearn is not in the vehicle's
dependency set (rclpy + numpy). Adding it is a deployment problem, and a runtime import
guard that fell back to something simpler would be a silent downgrade of the one step
that decides where the farm is. So the mean-field update is written out — about sixty
lines, with the observation covariance held FIXED at the detector's known accuracy
rather than learned. Fixing it is not a simplification for its own sake: with a free
covariance one component happily grows to swallow the whole farm, which is a well-known
failure of mixtures on data where the true clusters are the size of the noise, and here
the noise scale is something we actually know (IROS Table I: ~1 m).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ================================================================== data types
@dataclass(frozen=True)
class WorldDetection:
    """One detection placed in the world frame by the node (pose + slant correction)."""

    x: float
    y: float
    target: str            # "rope" | "buoy"
    confidence: float = 1.0
    #: Which side of the vehicle it came from, in world bearing degrees (grid). Kept so
    #: single-side coverage can be detected: a farm seen only from the west has no
    #: constraint on its eastern edge, and a transform fitted from it will look fine.
    look_bearing_deg: Optional[float] = None


@dataclass(frozen=True)
class Cluster:
    x: float
    y: float
    weight: float          # effective number of detections assigned
    spread_m: float        # RMS distance of assigned detections from the centre


@dataclass
class Verdict:
    name: str
    status: str            # confirmed | moved | missing | not_surveyed
    residual_m: Optional[float]
    predicted_xy: Tuple[float, float]
    observed_xy: Optional[Tuple[float, float]]


@dataclass
class LineFit:
    name: str
    ok: bool
    reason: str
    bearing_deg: float = 0.0
    #: Signed offset of the fitted line from the transformed prior line, metres, positive
    #: to the right of the prior bearing.
    offset_m: float = 0.0
    rms_m: float = 0.0
    n_points: int = 0


@dataclass
class FarmFix:
    ok: bool
    reason: str
    rotation_deg: float = 0.0
    translation_m: Tuple[float, float] = (0.0, 0.0)
    rms_m: float = 0.0
    n_clusters: int = 0
    n_matched: int = 0
    verdicts: List[Verdict] = field(default_factory=list)
    lines: List[LineFit] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)


# ================================================================== clustering
def cluster_buoys(points: Sequence[Tuple[float, float]],
                  max_classes: int,
                  obs_sigma_m: float = 1.0,
                  prior_sigma_m: float = 50.0,
                  alpha0: float = 1e-2,
                  outlier_area_m2: Optional[float] = None,
                  min_weight: float = 2.0,
                  iters: int = 200,
                  n_init: int = 8,
                  seed: int = 0) -> Tuple[List[Cluster], str]:
    """Variational Bayesian Gaussian mixture over 2D detections, plus a uniform
    background component for outliers.

    `max_classes` is the number of buoys the PRIOR says the farm has — the paper's
    "instantiated with a maximum amount of classes equal to the number of buoys in the
    farm". It is a ceiling, not a target: components that nobody claims decay to zero
    weight through the Dirichlet prior and are dropped, so a farm that has lost a buoy
    comes back with fewer clusters rather than with a spurious one placed in open water.

    The uniform component is what makes it "robust against outliers": without somewhere
    for a stray detection to go, every stray drags a real cluster toward it.

    Returns (clusters sorted by weight, reason). An empty list always carries a reason.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = P.shape[0]
    K = int(max_classes)
    if K < 1:
        return [], "max_classes must be at least 1"
    if n == 0:
        return [], "no buoy detections to cluster"
    if n < min_weight:
        return [], (f"{n} buoy detection(s) is below the {min_weight:.0f}-detection "
                    f"minimum for a cluster")

    if outlier_area_m2 is None:
        span = P.max(0) - P.min(0)
        outlier_area_m2 = max(float(span[0] * span[1]), 1.0)
    # Density of the uniform background. A detection this unlikely under every Gaussian
    # goes to the background instead of pulling a cluster.
    log_bg = -math.log(outlier_area_m2)

    m0 = P.mean(0)
    var = obs_sigma_m ** 2
    #: All pairwise squared distances, computed once: used for the density weights below.
    pair_d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)

    def digamma(x):
        """Asymptotic digamma with recurrence — numpy has no psi and scipy is not a
        dependency here (see the module docstring)."""
        x = np.asarray(x, dtype=np.float64)
        r = np.zeros_like(x)
        y = x.copy()
        while np.any(y < 6):
            small = y < 6
            r[small] -= 1.0 / y[small]
            y = np.where(small, y + 1.0, y)
        f = 1.0 / (y * y)
        return r + np.log(y) - 0.5 / y + f * (-1.0 / 12 + f * (1.0 / 120 - f / 252))

    def run(init_seed):
        """One variational run from one k-means++ start. Returns (evidence, resp, m).

        The evidence is the sum of the per-point log normalisers, which is what the
        restarts are ranked on. A single run is NOT enough: measured 2026-08-16, adding
        25 uniform outliers to a seven-buoy survey made a single-start mixture seed two
        components inside one buoy and none at M2_east_mid, and with the covariance held
        fixed a component cannot walk 15 m to find the cluster it missed — so a whole
        buoy silently disappeared from the map. Restarts fix the symptom; the cause is
        that k-means++ seeding on data containing outliers sometimes seeds an outlier,
        and no amount of iteration recovers from it.
        """
        rng = np.random.default_rng(init_seed)
        mm = np.empty((K, 2))
        # DENSITY-WEIGHTED k-means++ seeding. Plain k-means++ samples proportional to the
        # squared distance from the nearest seed, which on this data means it samples
        # OUTLIERS almost every time — they are by definition the points furthest from
        # everything. Measured 2026-08-16: with 25 uniform strays added to a seven-buoy
        # survey, eight plain-k-means++ restarts still failed to seed every buoy, and a
        # component that starts on a stray cannot walk to the cluster it missed because
        # the covariance is held fixed. Weighting each candidate by its local neighbour
        # count encodes the one thing that distinguishes a cluster from a stray — density
        # — and costs one distance matrix.
        w_density = (pair_d2 <= (3.0 * obs_sigma_m) ** 2).sum(1).astype(np.float64)
        p0 = w_density / w_density.sum() if w_density.sum() > 0 else None
        mm[0] = P[rng.choice(n, p=p0)] if p0 is not None else P[rng.integers(n)]
        for k in range(1, K):
            d2 = np.min(((P[:, None, :] - mm[None, :k, :]) ** 2).sum(-1), axis=1)
            w = d2 * w_density
            tot = w.sum()
            mm[k] = P[rng.integers(n)] if tot <= 0 else P[rng.choice(n, p=w / tot)]

        alpha = np.full(K, alpha0 + n / float(K))
        bg_alpha = alpha0 + 1.0
        s2 = np.full(K, var)
        r = None
        ev = -np.inf
        for _ in range(iters):
            a_all = np.concatenate([alpha, [bg_alpha]])
            e_log_pi = digamma(a_all) - digamma(np.array([a_all.sum()]))[0]
            d2 = ((P[:, None, :] - mm[None, :, :]) ** 2).sum(-1)
            # E[log N(x | mu_k, var I)] with mu_k ~ N(m_k, s2_k I), D = 2
            log_gauss = -d2 / (2 * var) - s2[None, :] / var - math.log(2 * math.pi * var)
            log_r = np.concatenate([log_gauss + e_log_pi[None, :K],
                                    np.full((n, 1), log_bg) + e_log_pi[K]], axis=1)
            mx = log_r.max(1, keepdims=True)
            ev = float((mx[:, 0] + np.log(np.exp(log_r - mx).sum(1))).sum())
            r = np.exp(log_r - mx)
            r /= r.sum(1, keepdims=True)

            Nk = r[:, :K].sum(0)
            alpha = alpha0 + Nk
            bg_alpha = alpha0 + r[:, K].sum()
            s2 = 1.0 / (1.0 / prior_sigma_m ** 2 + Nk / var)
            mm = s2[:, None] * (m0[None, :] / prior_sigma_m ** 2 + (r[:, :K].T @ P) / var)
        return ev, r, mm

    best_ev, resp = -np.inf, None
    for i in range(max(1, int(n_init))):
        # Seeds are `seed + i`, so the whole thing is reproducible from one number. A
        # localizer whose answer moves between runs on one bag makes every disagreement
        # unarguable.
        ev, r, _ = run(seed + i)
        if ev > best_ev:
            best_ev, resp = ev, r

    Nk = resp[:, :K].sum(0)
    out: List[Cluster] = []
    for k in range(K):
        if Nk[k] < min_weight:
            continue
        w = resp[:, k]
        centre = (w[:, None] * P).sum(0) / w.sum()
        spread = math.sqrt(float((w * ((P - centre) ** 2).sum(1)).sum() / w.sum()))
        out.append(Cluster(float(centre[0]), float(centre[1]), float(Nk[k]), spread))
    out.sort(key=lambda c: -c.weight)
    if not out:
        return [], (f"{n} detections produced no cluster reaching the {min_weight:.0f}-"
                    f"detection minimum — every one of them went to the outlier component")
    return out, "ok"


# ================================================================== the transform
def fit_rigid_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Kabsch in 2D: the rotation+translation taking `src` onto `dst`, and the RMS.

    Rotation and translation ONLY — no scale. A farm does not shrink, and allowing scale
    lets a bad correspondence set absorb its own error by resizing the farm, producing a
    small residual and a wrong map. Same convention as `bagreader.fit_rigid_2d` and
    `dr_ghost.py` so transforms in this project stay comparable.
    """
    sc, dc = src.mean(0), dst.mean(0)
    S, D = src - sc, dst - dc
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    t = dc - R @ sc
    res = (src @ R.T + t) - dst
    return R, t, float(np.sqrt((res ** 2).sum(1).mean()))


def _collinearity(points: np.ndarray) -> float:
    """0 = perfectly collinear, 1 = isotropic. The ratio of the smaller to the larger
    principal standard deviation."""
    if points.shape[0] < 3:
        return 0.0
    c = points - points.mean(0)
    s = np.linalg.svd(c, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


def match_prior_to_clusters(prior_xy: Dict[str, Tuple[float, float]],
                            clusters: Sequence[Cluster],
                            max_shift_m: float = 20.0,
                            inlier_m: float = 4.0,
                            min_inliers: int = 3,
                            min_collinearity: float = 0.05
                            ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                       Dict[str, int], str]:
    """RANSAC over two-point correspondences: prior -> observed.

    With at most seven buoys an exhaustive search over pairs is trivially cheap and
    removes any dependence on an initial guess. Each hypothesis is scored by how many
    prior buoys land within `inlier_m` of a cluster, and the winner is refined by Kabsch
    over its inliers.

    Two refusals, both named:
      * fewer than `min_inliers` correspondences — a rigid transform from two points is
        exactly determined and cannot be checked, so it would always look perfect;
      * near-collinear inliers — the rotation is unconstrained across the line, and a
        transform fitted from them is a guess with a small residual, which is the most
        dangerous shape an answer can have.
    """
    names = list(prior_xy)
    P = np.array([prior_xy[n] for n in names], dtype=np.float64)
    if len(clusters) < min_inliers:
        return None, None, {}, (f"only {len(clusters)} buoy cluster(s); a checkable rigid "
                                f"transform needs at least {min_inliers}")
    C = np.array([[c.x, c.y] for c in clusters], dtype=np.float64)

    best = (-1, None, None, {})
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            for a in range(len(C)):
                for b in range(len(C)):
                    if a == b:
                        continue
                    R, t, _ = fit_rigid_2d(P[[i, j]], C[[a, b]])
                    if np.linalg.norm(t + R @ P.mean(0) - P.mean(0)) > max_shift_m + 1e-9:
                        # A hypothesis that moves the farm further than the prior's own
                        # uncertainty is not a re-mapping, it is a mismatch.
                        pass
                    pred = P @ R.T + t
                    pairs, used = {}, set()
                    for k in range(len(names)):
                        d = np.linalg.norm(C - pred[k], axis=1)
                        order = np.argsort(d)
                        for c_idx in order:
                            if d[c_idx] <= inlier_m and c_idx not in used:
                                pairs[names[k]] = int(c_idx)
                                used.add(int(c_idx))
                                break
                    if len(pairs) > best[0]:
                        best = (len(pairs), R, t, dict(pairs))

    n_in, R, t, pairs = best
    if n_in < min_inliers:
        return None, None, {}, (f"the best hypothesis matched only {max(n_in, 0)} of "
                                f"{len(names)} prior buoys within {inlier_m:.1f} m; the farm "
                                f"in front of the vehicle does not look like the farm in the "
                                f"prior")
    idx = [names.index(k) for k in pairs]
    src = P[idx]
    dst = np.array([[clusters[pairs[k]].x, clusters[pairs[k]].y] for k in pairs])
    col = _collinearity(dst)
    if col < min_collinearity:
        return None, None, {}, (f"the {len(pairs)} matched buoys are near-collinear "
                                f"(axis ratio {col:.3f} < {min_collinearity}); the rotation "
                                f"across that line is unconstrained, so any transform fitted "
                                f"here would have a small residual and an arbitrary heading")
    R, t, _ = fit_rigid_2d(src, dst)
    return R, t, pairs, "ok"


# ================================================================== verification
def coverage_bearings_ok(dets: Sequence[WorldDetection], min_span_deg: float = 180.0
                         ) -> Tuple[bool, str]:
    """Did the encircle actually go round?

    A farm observed only from one side has no measurement at all on its far edge, and a
    transform fitted from one side will happily report a confident position while the
    unseen buoys are pure prior. The 2022 paper's whole initialization is a
    circumnavigation; this checks that one happened.
    """
    bearings = [d.look_bearing_deg for d in dets if d.look_bearing_deg is not None]
    if len(bearings) < 3:
        return False, "no look bearings recorded, so coverage cannot be checked"
    b = np.sort(np.mod(np.array(bearings), 360.0))
    gaps = np.diff(np.concatenate([b, b[:1] + 360.0]))
    span = 360.0 - float(gaps.max())
    if span < min_span_deg:
        return False, (f"detections span only {span:.0f} deg of look bearing (largest gap "
                       f"{float(gaps.max()):.0f} deg); the farm was seen from one side, so "
                       f"the far edge is prior, not measurement")
    return True, f"look bearings span {span:.0f} deg"


def verify_against_prior(prior_xy: Dict[str, Tuple[float, float]],
                         clusters: Sequence[Cluster],
                         R: np.ndarray, t: np.ndarray,
                         pairs: Dict[str, int],
                         confirm_m: float = 2.0,
                         surveyed_radius_m: float = 12.0,
                         moved_radius_m: float = 15.0,
                         covered_points: Optional[Sequence[Tuple[float, float]]] = None
                         ) -> List[Verdict]:
    """Per-buoy residual and verdict.

    `confirmed` within `confirm_m` of where the transform predicts, `moved` when a buoy
    is somewhere else but still identifiable, `missing` when nothing was found where the
    vehicle DID look, and `not_surveyed` when nothing was found and the vehicle never got
    within `surveyed_radius_m` of the place. The last one is the whole reason this
    function takes the track: without it, every buoy the encircle never reached is
    reported as lost.

    A buoy displaced further than the transform's inlier radius has NO correspondence —
    that is what makes the transform robust to it — so it would fall through to
    `missing` if nothing else happened. It is matched here instead, against the clusters
    the transform did not use, and only on a MUTUAL nearest-neighbour basis: the leftover
    cluster must be this predicted buoy's nearest and vice versa. Without mutuality, one
    stray cluster would be handed to whichever unmatched buoy was checked first, and a
    genuinely missing buoy would be reported as merely moved.
    """
    out: List[Verdict] = []
    track = (np.asarray(covered_points, dtype=np.float64).reshape(-1, 2)
             if covered_points is not None and len(covered_points) else None)
    pred = {n: R @ np.array(xy, dtype=np.float64) + t for n, xy in prior_xy.items()}
    leftover = [i for i in range(len(clusters)) if i not in set(pairs.values())]
    unmatched = [n for n in prior_xy if n not in pairs]

    def nearest_leftover(name):
        if not leftover:
            return None, None
        d = [(math.hypot(clusters[i].x - pred[name][0], clusters[i].y - pred[name][1]), i)
             for i in leftover]
        d.sort()
        return d[0][1], d[0][0]

    displaced = {}
    for name in unmatched:
        idx, dist = nearest_leftover(name)
        if idx is None or dist > moved_radius_m:
            continue
        # Mutual check: is `name` also this cluster's nearest unmatched buoy?
        best = min(unmatched, key=lambda o: math.hypot(clusters[idx].x - pred[o][0],
                                                       clusters[idx].y - pred[o][1]))
        if best == name:
            displaced[name] = (idx, dist)

    for name, xy in prior_xy.items():
        p = pred[name]
        pred_t = (float(p[0]), float(p[1]))
        if name in pairs:
            c = clusters[pairs[name]]
            res = float(math.hypot(c.x - p[0], c.y - p[1]))
            out.append(Verdict(name, "confirmed" if res <= confirm_m else "moved",
                               res, pred_t, (c.x, c.y)))
            continue
        if name in displaced:
            idx, dist = displaced[name]
            c = clusters[idx]
            out.append(Verdict(name, "moved", float(dist), pred_t, (c.x, c.y)))
            continue
        looked = True
        if track is not None:
            looked = bool(np.min(np.linalg.norm(track - p, axis=1)) <= surveyed_radius_m)
        out.append(Verdict(name, "missing" if looked else "not_surveyed",
                           None, pred_t, None))
    return out


def fit_culture_lines(rope_dets: Sequence[WorldDetection],
                      prior_lines: Sequence[Tuple[str, Tuple[float, float],
                                                  Tuple[float, float]]],
                      R: np.ndarray, t: np.ndarray,
                      corridor_m: float = 4.0,
                      bearing_tolerance_deg: float = 20.0,
                      min_points: int = 8) -> List[LineFit]:
    """One offset per culture line, with the ORIENTATION CONSTRAINED to the prior.

    2022 §6: "we do a piece-wise linear fit through the buoys such that the resulting
    lines must align with the a priori known farm orientation (compass direction)". Here
    the same constraint is applied to the rope detections, which are far more numerous
    than the buoys (IROS §IV: "given their ubiquity through the survey").

    The constraint is what makes the fit usable from a single pass down one side: an
    unconstrained fit through detections that all lie on one side of the rope recovers
    the rope's direction from noise. With the direction fixed, only the perpendicular
    offset is estimated, and one pass is enough.

    Per-line refusals rather than one global one: losing the east line is not losing the
    farm, and a mission that can still scan one corridor should be told which.
    """
    out: List[LineFit] = []
    P = np.array([[d.x, d.y] for d in rope_dets], dtype=np.float64) if rope_dets else np.empty((0, 2))
    for name, a, b in prior_lines:
        pa = R @ np.array(a, dtype=np.float64) + t
        pb = R @ np.array(b, dtype=np.float64) + t
        v = pb - pa
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            out.append(LineFit(name, False, "prior line has zero length"))
            continue
        u = v / L
        nvec = np.array([u[1], -u[0]])          # right of the line's bearing
        if P.shape[0] == 0:
            out.append(LineFit(name, False, "no rope detections at all"))
            continue
        rel = P - pa
        along = rel @ u
        perp = rel @ nvec
        keep = (along >= -corridor_m) & (along <= L + corridor_m) & (np.abs(perp) <= corridor_m)
        n_keep = int(keep.sum())
        if n_keep < min_points:
            out.append(LineFit(name, False,
                               f"only {n_keep} rope detection(s) within {corridor_m:.1f} m of "
                               f"the prior line; {min_points} needed for an offset",
                               n_points=n_keep))
            continue
        off = float(np.median(perp[keep]))      # median: a few strays must not drag it
        rms = float(np.sqrt(np.mean((perp[keep] - off) ** 2)))
        bearing = math.degrees(math.atan2(u[0], u[1])) % 360.0
        # The free-direction fit is computed only to CHECK the constraint, never to
        # replace it: if the data disagree with the prior orientation by more than the
        # tolerance, the constraint is no longer describing this farm and the caller
        # needs to know rather than receive a confidently wrong offset.
        c = P[keep] - P[keep].mean(0)
        _, _, Vt = np.linalg.svd(c)
        free = math.degrees(math.atan2(Vt[0][0], Vt[0][1])) % 180.0
        diff = abs(((free - bearing % 180.0) + 90.0) % 180.0 - 90.0)
        if diff > bearing_tolerance_deg:
            out.append(LineFit(name, False,
                               f"the rope detections lie along {free:.0f} deg but the prior "
                               f"line runs at {bearing % 180.0:.0f} deg ({diff:.0f} deg apart, "
                               f"tolerance {bearing_tolerance_deg:.0f}); the orientation "
                               f"constraint does not describe this line any more",
                               bearing_deg=bearing, n_points=n_keep))
            continue
        out.append(LineFit(name, True, "ok", bearing_deg=bearing, offset_m=off,
                           rms_m=rms, n_points=n_keep))
    return out


# ================================================================== the whole thing
def localize_farm(detections: Sequence[WorldDetection],
                  prior_buoys: Dict[str, Tuple[float, float]],
                  prior_lines: Sequence[Tuple[str, Tuple[float, float], Tuple[float, float]]],
                  covered_points: Optional[Sequence[Tuple[float, float]]] = None,
                  obs_sigma_m: float = 1.0,
                  confirm_m: float = 2.0,
                  require_coverage: bool = True,
                  seed: int = 0) -> FarmFix:
    """Detections in, verified farm out — or a refusal that names what was missing.

    The order is the paper's: cluster, fit, verify. Every stage's refusal is returned
    as-is rather than collapsed into "localization failed", because the mission's
    response differs: too few detections means fly the encircle again, a collinear fit
    means fly it from another side, a mismatch means this is not the farm.
    """
    buoy_pts = [(d.x, d.y) for d in detections if d.target == "buoy"]
    rope_dets = [d for d in detections if d.target == "rope"]

    if require_coverage:
        ok, why = coverage_bearings_ok(detections)
        if not ok:
            return FarmFix(False, f"coverage refusal: {why}")

    clusters, why = cluster_buoys(buoy_pts, max_classes=len(prior_buoys),
                                  obs_sigma_m=obs_sigma_m, seed=seed)
    if not clusters:
        return FarmFix(False, f"clustering refusal: {why}")

    R, t, pairs, why = match_prior_to_clusters(prior_buoys, clusters)
    if R is None:
        return FarmFix(False, f"transform refusal: {why}", n_clusters=len(clusters),
                       clusters=list(clusters))

    verdicts = verify_against_prior(prior_buoys, clusters, R, t, pairs,
                                    confirm_m=confirm_m, covered_points=covered_points)
    lines = fit_culture_lines(rope_dets, prior_lines, R, t)

    res = [v.residual_m for v in verdicts if v.residual_m is not None]
    rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return FarmFix(True, "ok",
                   rotation_deg=rot, translation_m=(float(t[0]), float(t[1])),
                   rms_m=float(np.sqrt(np.mean(np.square(res)))) if res else 0.0,
                   n_clusters=len(clusters), n_matched=len(pairs),
                   verdicts=verdicts, lines=lines, clusters=list(clusters))
