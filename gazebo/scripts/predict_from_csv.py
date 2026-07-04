#!/usr/bin/env python3
"""Close the co-simulation loop: take a Gazebo-produced trajectory (traj.csv +
landing.csv from record_landing.py), run the SAME perception+prediction pipeline
used on the analytical track, and report landing error per method.

    python predict_from_csv.py traj.csv landing.csv --fps 120 --sigma0 8 --bad_frac 0.2

This is the drop-in proof that switching the physics backend RK4 -> Gazebo does
not touch the prediction/evaluation code: identical estimators, identical metric.
`evaluate()` is importable (used by sweep.py for the batch matrix).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ttsim.noise import add_noise                       # noqa: E402
from ttsim.physics import predict_landing, R_BALL       # noqa: E402
from ttsim.estimators import fit_trajectory, fit_trajectory_gated  # noqa: E402

OMEGA_BOUND = 1100.0  # per-component |omega_i| bound rad/s (NOT a norm bound)


def _land_xy(theta):
    # land at z = ball radius to match Gazebo's finite-radius first contact
    lp, _ = predict_landing(theta, table_z=R_BALL)
    return None if lp is None else lp[:2]


def load(path, cols):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def make_observations(tr, fps, sigma0_mm, alpha, bad_frac, seed):
    """Resample a dense Gazebo log onto camera frames and inject the SAME
    heteroscedastic perception noise used by the analytical track.
    Returns (obs_t, obs_p, conf, sigma_true)."""
    t0, t1 = tr["t"][0], tr["t"][-1]
    fr_t = np.arange(0.0, t1 - t0, 1.0 / fps)
    if len(fr_t) < 8:
        raise ValueError(f"trajectory too short to fit: {len(fr_t)} frames at "
                         f"{fps} fps over {t1 - t0:.4f}s — check traj.csv timestamps")
    P = np.stack([np.interp(fr_t + t0, tr["t"], tr[c]) for c in "xyz"], axis=1)
    sp = np.gradient(P, fr_t, axis=0)
    speeds = np.linalg.norm(sp, axis=1)

    rng = np.random.default_rng(seed)
    noisy, sig, keep, conf = add_noise(P, speeds, sigma0_mm / 1000.0, alpha,
                                       0.0, rng, bad_frac=bad_frac)
    return fr_t[keep], noisy[keep], conf[keep], sig[keep]


def evaluate(tr, ld, fps=120.0, sigma0_mm=8.0, alpha=1.0, bad_frac=0.2, seed=0):
    """One noise realization on one Gazebo trajectory -> per-method landing
    error in cm. Returns (errors_dict, n_frames)."""
    true_xy = np.array([ld["x"][0], ld["y"][0]])
    ot, op, cf, sg = make_observations(tr, fps, sigma0_mm, alpha, bad_frac, seed)

    # all fits omega-bounded (per-component) so a few bad frames can't explode spin.
    # M_huber/M_gate: confidence-free robust baselines; landing plane z=R here,
    # hence the local _land_xy projection for every method.
    preds = {
        "M0": _land_xy(fit_trajectory(ot, op, fit_omega=False)),
        "M1": _land_xy(fit_trajectory(ot, op, fit_omega=True, omega_bound=OMEGA_BOUND)),
        "M_huber": _land_xy(fit_trajectory(ot, op, fit_omega=True,
                                           omega_bound=OMEGA_BOUND,
                                           loss="huber", f_scale=0.015)),
        "M_gate": _land_xy(fit_trajectory_gated(ot, op)),
        "M3_conf": _land_xy(fit_trajectory(ot, op, weights=cf ** 2, fit_omega=True,
                                           omega_bound=OMEGA_BOUND)),
        "M3_oracle": _land_xy(fit_trajectory(ot, op, weights=1.0 / np.maximum(sg, 1e-6) ** 2,
                                             fit_omega=True, omega_bound=OMEGA_BOUND)),
    }
    errs = {m: (np.nan if xy is None else 100 * float(np.linalg.norm(xy - true_xy)))
            for m, xy in preds.items()}
    return errs, len(ot)


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
    try:
        errs, n = evaluate(tr, ld, a.fps, a.sigma0, a.alpha, a.bad_frac, a.seed)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Gazebo truth landing: x={ld['x'][0]:.3f} y={ld['y'][0]:.3f} m "
          f"({n} frames, bad_frac={a.bad_frac}, landing plane z={R_BALL} m)")
    for m, err in errs.items():
        print(f"   {m:10s} landing error = {err:6.2f} cm")


if __name__ == "__main__":
    main()
