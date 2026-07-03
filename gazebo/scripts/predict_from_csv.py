#!/usr/bin/env python3
"""Close the co-simulation loop: take a Gazebo-produced trajectory (traj.csv +
landing.csv from record_landing.py), run the SAME perception+prediction pipeline
used on the analytical track, and report landing error per method.

    python predict_from_csv.py traj.csv landing.csv --fps 120 --sigma0 8 --bad_frac 0.2

This is the drop-in proof that switching the physics backend RK4 -> Gazebo does
not touch the prediction/evaluation code: identical estimators, identical metric.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ttsim.noise import add_noise                       # noqa: E402
from ttsim.physics import predict_landing, R_BALL       # noqa: E402
from ttsim.estimators import fit_trajectory             # noqa: E402

OMEGA_BOUND = 1100.0  # per-component |omega_i| bound rad/s (NOT a norm bound)


def _land_xy(theta):
    # land at z = ball radius to match Gazebo's finite-radius first contact
    lp, _ = predict_landing(theta, table_z=R_BALL)
    return None if lp is None else lp[:2]


def load(path, cols):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("landing")
    ap.add_argument("--fps", type=float, default=120)
    ap.add_argument("--sigma0", type=float, default=8.0, help="mm")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bad_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr = load(a.traj, ["t", "x", "y", "z"])
    ld = load(a.landing, ["x", "y", "t"])
    true_xy = np.array([ld["x"][0], ld["y"][0]])

    # resample dense Gazebo log onto camera frames
    t0, t1 = tr["t"][0], tr["t"][-1]
    fr_t = np.arange(0.0, t1 - t0, 1.0 / a.fps)
    if len(fr_t) < 8:
        sys.exit(f"trajectory too short to fit: {len(fr_t)} frames at {a.fps} fps "
                 f"over {t1 - t0:.4f}s — check traj.csv timestamps")
    P = np.stack([np.interp(fr_t + t0, tr["t"], tr[c]) for c in "xyz"], axis=1)
    sp = np.gradient(P, fr_t, axis=0)
    speeds = np.linalg.norm(sp, axis=1)

    rng = np.random.default_rng(a.seed)
    noisy, sig, keep, conf = add_noise(P, speeds, a.sigma0 / 1000.0, a.alpha,
                                       0.0, rng, bad_frac=a.bad_frac)
    ot, op, cf, sg = fr_t[keep], noisy[keep], conf[keep], sig[keep]

    # all fits omega-bounded (per-component) so a few bad frames can't explode spin
    preds = {
        "M0": _land_xy(fit_trajectory(ot, op, fit_omega=False)),
        "M1": _land_xy(fit_trajectory(ot, op, fit_omega=True, omega_bound=OMEGA_BOUND)),
        "M3_conf": _land_xy(fit_trajectory(ot, op, weights=cf ** 2, fit_omega=True,
                                           omega_bound=OMEGA_BOUND)),
        "M3_oracle": _land_xy(fit_trajectory(ot, op, weights=1.0 / np.maximum(sg, 1e-6) ** 2,
                                             fit_omega=True, omega_bound=OMEGA_BOUND)),
    }
    print(f"Gazebo truth landing: x={true_xy[0]:.3f} y={true_xy[1]:.3f} m "
          f"({len(ot)} frames, bad_frac={a.bad_frac}, landing plane z={R_BALL} m)")
    for m, xy in preds.items():
        err = np.nan if xy is None else 100 * np.linalg.norm(xy - true_xy)
        print(f"   {m:10s} landing error = {err:6.2f} cm")


if __name__ == "__main__":
    main()
