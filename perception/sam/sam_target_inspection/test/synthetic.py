"""Synthetic pings and clouds for the off-vehicle tests.

EVERY NUMBER IN HERE IS A MEASURED ONE, and the fixture says which measurement:
the Ideal-fan car (SETTLED §3f0s: +9.2 dB highlight, 1.56 m across, −11.3 dB shadow), the real
2026-12-10 record's speckle (CV 0.28, SETTLED §3f0h), the DeepVision record geometry
(18.55 Hz, 40 m range, 4 cm bins, SETTLED §3f0e) and the Sonar3D15 prefab's fan (150 × 17 rays
over 90° × 40°, tilt −20°).

These are FIXTURES, not data. They exercise the detectors' arithmetic against a signature that
was measured; they are not evidence that the detectors work on real pings, and the only things
in this suite that are evidence of that are the negative-set measurements in
`test_negative_sets_measured.py`, which read real bags.
"""
import math

import numpy as np

RANGE_RES_M = 0.04          # SETTLED §3f0e: 4 cm bins
PING_HZ = 18.55             # SETTLED §3f0e
SPECKLE_CV = 0.28           # SETTLED §3f0h, the real 2024-12-10 record
HIGHLIGHT_DB = 9.2          # SETTLED §3f0s, Ideal fan
SHADOW_DB = -11.3           # SETTLED §3f0s, Ideal fan
ACROSS_M = 1.56             # SETTLED §3f0s
MINI_L, MINI_W, MINI_H = 3.078, 1.416, 1.278     # SETTLED §3f0d


def gain_curve(n_bins, altitude_m, res_m=RANGE_RES_M):
    """A plausible TVG-corrected envelope: it rises off the nadir and falls slowly with range."""
    slant = (np.arange(n_bins) + 0.5) * res_m
    env = 35.0 * np.clip((slant - altitude_m) / 2.0, 0.0, 1.0) * np.exp(-(slant - altitude_m) / 200.0)
    return np.maximum(env, 0.5)


def ping(*, n_bins=1000, altitude_m=6.0, res_m=RANGE_RES_M, target_ground_m=25.0,
         highlight_db=HIGHLIGHT_DB, shadow_db=SHADOW_DB, across_m=ACROSS_M,
         height_m=MINI_H, cv=SPECKLE_CV, seed=0, shadow_len_scale=1.0):
    """One side-scan channel with (optionally) the measured car signature on it."""
    rng = np.random.default_rng(seed)
    env = gain_curve(n_bins, altitude_m, res_m)
    y = np.maximum(env * (1.0 + cv * rng.standard_normal(n_bins)), 0.0)
    if target_ground_m is None:
        return y
    slant = math.hypot(target_ground_m, altitude_m)
    b0 = int(slant / res_m)
    w = max(1, int(across_m / res_m))
    y[b0:b0 + w] = env[b0:b0 + w] * 10 ** (highlight_db / 20.0)
    n_sh = max(0, int(shadow_len_scale * (height_m * target_ground_m / altitude_m) / res_m))
    y[b0 + w:b0 + w + n_sh] = env[b0 + w:b0 + w + n_sh] * 10 ** (shadow_db / 20.0)
    return y


def cloud(objects=(), *, altitude_m=4.0, n_az=150, n_el=17, fan_az_deg=90.0,
          fan_el_deg=40.0, tilt_deg=-20.0, max_range_m=15.0, noise_m=0.02, seed=0):
    """Ray-cast the prefab's own fan onto a flat seabed with axis-aligned boxes standing on it.

    `objects` are `(x, y, length, width, height)` in the body frame (x forward, y left, z up),
    each sitting ON the seabed. The fan is the Sonar3D15 prefab's: 150 beams × 17 rays over
    90° × 40°, centred `tilt_deg` below horizontal.
    """
    rng = np.random.default_rng(seed)
    az = np.radians(np.linspace(-fan_az_deg / 2, fan_az_deg / 2, n_az))
    el = np.radians(np.linspace(tilt_deg - fan_el_deg / 2, tilt_deg + fan_el_deg / 2, n_el))
    pts = []
    for a in az:
        for e in el:
            d = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            best = None
            if d[2] < 0:
                t = -altitude_m / d[2]
                if t <= max_range_m:
                    best = t
            for (cx, cy, L, W, H) in objects:
                lo = np.array([cx - L / 2, cy - W / 2, -altitude_m])
                hi = np.array([cx + L / 2, cy + W / 2, -altitude_m + H])
                tmin, tmax, ok = 0.0, max_range_m, True
                for k in range(3):
                    if abs(d[k]) < 1e-9:
                        if not (lo[k] <= 0 <= hi[k]):
                            ok = False
                            break
                    else:
                        t1, t2 = lo[k] / d[k], hi[k] / d[k]
                        if t1 > t2:
                            t1, t2 = t2, t1
                        tmin, tmax = max(tmin, t1), min(tmax, t2)
                if ok and tmin <= tmax and tmin > 0 and (best is None or tmin < best):
                    best = tmin
            if best is not None:
                pts.append(d * best + noise_m * rng.standard_normal(3))
    return np.asarray(pts)
