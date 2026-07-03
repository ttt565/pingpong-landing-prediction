#!/usr/bin/env python3
"""M2: predict the SECOND touchdown from the pre-bounce arc alone.

Pipeline per noise seed:
  1. observe the pre-first-contact arc (same resample+noise as first-landing),
  2. fit theta=(p0,v0,omega) (M1 uniform / M3_conf confidence-weighted),
  3. integrate the fit to the first z=R crossing -> impact state (v, omega),
  4. apply the analytic impulse bounce model (ttsim/bounce.py, params = SDF),
  5. integrate the outgoing state to the next z=R crossing -> predicted 2nd
     touchdown; compare with Gazebo's recorded bounces.csv row 2.

Reference rows:
  * "M2_spinknown": same noisy fit but the TRUE launch spin (from the sweep
    manifest.csv) is handed in — quantifies how much of the M2 error is
    spin-estimation error amplified through the bounce.
  * "M2_clean": noise-free fit — isolates the bounce-model mismatch vs DART's
    contact solver, the floor any M2 estimator inherits from the bounce model.

    python3 predict_second_bounce.py ../sweep_out/v60_top400 --seeds 12
    python3 predict_second_bounce.py ../sweep_out/*          # all conditions
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ttsim.physics import simulate, R_BALL               # noqa: E402
from ttsim.bounce import bounce_state                    # noqa: E402
from ttsim.estimators import fit_trajectory              # noqa: E402
from predict_from_csv import load, make_observations     # noqa: E402

OMEGA_BOUND = 1100.0


def second_touchdown(theta):
    """Integrate flight -> bounce -> flight. Returns (xy2, t2) or None."""
    ts, P, V, lp, lt = simulate(theta[:3], theta[3:6], theta[6:9],
                                dt=1e-3, table_z=R_BALL)
    if lp is None:
        return None
    i = int(np.searchsorted(ts, lt))
    i = min(max(i, 1), len(ts) - 1)
    f = (lt - ts[i - 1]) / (ts[i] - ts[i - 1])
    v_land = V[i - 1] + f * (V[i] - V[i - 1])

    v_out, w_out = bounce_state(v_land, theta[6:9])
    p0 = lp.copy()
    p0[2] = R_BALL
    _, _, _, lp2, lt2 = simulate(p0, v_out, w_out, dt=1e-3, table_z=R_BALL)
    if lp2 is None:
        return None
    return lp2[:2], lt + lt2


def true_spin(rundir):
    """Launch spin from the sweep manifest (spin is constant in flight: no air
    torque in either backend). None when the manifest is absent."""
    mpath = os.path.join(os.path.dirname(rundir.rstrip("/")), "manifest.csv")
    if not os.path.exists(mpath):
        return None
    name = os.path.basename(rundir.rstrip("/"))
    with open(mpath) as f:
        for row in csv.DictReader(f):
            if row["condition"] == name:
                return np.array([float(row["wx"]), float(row["wy"]),
                                 float(row["wz"])])
    return None


def eval_dir(rundir, seeds, fps, sigma0, alpha, bad_frac):
    b = load(os.path.join(rundir, "bounces.csv"), ["n", "x", "y", "t"])
    if len(b["n"]) < 2:
        return None   # first touchdown was off-table: no second bounce exists
    truth2 = np.array([b["x"][1], b["y"][1]])
    tr = load(os.path.join(rundir, "traj.csv"), ["t", "x", "y", "z"])
    w_true = true_spin(rundir)

    errs = {"M2_M1": [], "M2_conf": [], "M2_spinknown": []}
    for seed in range(seeds):
        ot, op, cf, _ = make_observations(tr, fps, sigma0, alpha, bad_frac, seed)
        fits = {
            "M2_M1": fit_trajectory(ot, op, fit_omega=True, omega_bound=OMEGA_BOUND),
            "M2_conf": fit_trajectory(ot, op, weights=cf ** 2, fit_omega=True,
                                      omega_bound=OMEGA_BOUND),
        }
        if w_true is not None:
            fits["M2_spinknown"] = fit_trajectory(ot, op, fit_omega=False,
                                                  fixed_omega=w_true)
        for m, th in fits.items():
            pred = second_touchdown(th)
            errs[m].append(np.nan if pred is None
                           else 100 * float(np.linalg.norm(pred[0] - truth2)))

    # noise-free ceiling: bounce-model error alone
    ot, op, _, _ = make_observations(tr, fps, 0.0, 0.0, 0.0, 0)
    th = fit_trajectory(ot, op, fit_omega=True, omega_bound=OMEGA_BOUND)
    pred = second_touchdown(th)
    clean = (np.nan if pred is None
             else 100 * float(np.linalg.norm(pred[0] - truth2)))
    return truth2, errs, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+",
                    help="dir(s) with traj.csv + bounces.csv (from --bounces 2)")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--fps", type=float, default=120)
    ap.add_argument("--sigma0", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bad_frac", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="write a markdown table here")
    a = ap.parse_args()

    lines = ["# M2: second-touchdown prediction vs Gazebo truth\n",
             f"{a.seeds} noise seeds; error in cm at the 2nd z=R crossing. "
             "M2_spinknown = noisy fit with the TRUE spin handed in (isolates "
             "the spin-estimation penalty). M2_clean = noise-free fit -> pure "
             "bounce-model mismatch vs DART.\n",
             "| condition | truth2 x,y (m) | M2_M1 | M2_conf | M2_spinknown | M2_clean |",
             "|---|---|---|---|---|---|"]
    rundirs = [r.rstrip("/") for r in a.rundirs
               if os.path.exists(os.path.join(r, "bounces.csv"))]
    with ProcessPoolExecutor(max_workers=min(len(rundirs), os.cpu_count())) as ex:
        all_res = list(ex.map(eval_dir, rundirs,
                              [a.seeds] * len(rundirs), [a.fps] * len(rundirs),
                              [a.sigma0] * len(rundirs), [a.alpha] * len(rundirs),
                              [a.bad_frac] * len(rundirs)))
    for rundir, res in zip(rundirs, all_res):
        name = os.path.basename(rundir)
        if res is None:
            lines.append(f"| {name} | (1st touchdown off table) | — | — | — | — |")
            continue
        truth2, errs, clean = res
        cells = []
        for m in ("M2_M1", "M2_conf", "M2_spinknown"):
            if errs.get(m):
                e = np.array(errs[m])
                cells.append(f"{np.nanmean(e):.2f} ± {np.nanstd(e):.2f}")
            else:
                cells.append("—")
        lines.append(f"| {name} | {truth2[0]:.4f}, {truth2[1]:.4f} | "
                     f"{cells[0]} | {cells[1]} | {cells[2]} | {clean:.2f} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
