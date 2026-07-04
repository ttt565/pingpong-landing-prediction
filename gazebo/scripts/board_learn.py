#!/usr/bin/env python3
"""Contact-board self-supervision, stage 1: DATA GENERATION.

A sensing board sits just past the table end (worlds/table_tennis_board.sdf).
Every serve yields one free ground-truth label: where the ball strikes the
board. This script:

  1. samples N serves from a structured "serve-machine repertoire"
     (topspin / flat / backspin clusters with speed + direction jitter),
  2. simulates each in Gazebo (parallel, isolated via GZ_PARTITION),
  3. per serve: observes the pre-bounce arc with the standard perception
     noise (ONE realization — a real system sees each serve once), fits
     theta with the deployable estimator (M3_conf), and pushes the physics
     pipeline (flight -> calibrated bounce -> flight) to the board plane,
  4. writes board_dataset.csv: fitted state + physics prediction + true
     contact + references (true-spin oracle prediction), one row per serve.

Stage 2 (learn_board_residual.py) consumes only board_dataset.csv, so the
learning experiment is reproducible without re-simulating.

    python3 board_learn.py --n 120 --jobs 3
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GZ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(GZ))
from sweep import patch_model                             # noqa: E402
from predict_from_csv import load, make_observations      # noqa: E402
from ttsim.physics import simulate, R_BALL                # noqa: E402
from ttsim.bounce import bounce_state                     # noqa: E402
from ttsim.estimators import fit_trajectory               # noqa: E402

X_BOARD = 2.80                 # board front face (worlds/table_tennis_board.sdf)
X_PLANE = X_BOARD - R_BALL     # ball-center plane at contact
OMEGA_BOUND = 1100.0

# serve-machine repertoire: cluster -> (vx range, vz range, wy mean/std)
CLUSTERS = [
    ("topspin",  (5.2, 6.3), (0.75, 1.00), (330.0, 40.0)),
    ("flat",     (4.8, 6.0), (0.75, 1.00), (20.0, 30.0)),
    ("backspin", (4.6, 5.4), (0.75, 0.95), (-230.0, 35.0)),
]


def sample_serves(n, rng):
    serves = []
    for i in range(n):
        name, vxr, vzr, (wy_m, wy_s) = CLUSTERS[i % len(CLUSTERS)]
        v = (rng.uniform(*vxr), rng.uniform(-0.25, 0.25), rng.uniform(*vzr))
        w = (rng.normal(0, 15), rng.normal(wy_m, wy_s), rng.normal(0, 25))
        serves.append((name, v, w))
    return serves


def run_serve(idx, v, w, outdir):
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        patch_model(tmp, v, w)
        env = os.environ.copy()
        env["GZ_SIM_RESOURCE_PATH"] = tmp
        env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.path.join(
            GZ, "plugins", "aero_launch", "build")
        env["GZ_PARTITION"] = f"board_serve_{idx}"   # isolate parallel sims
        rec = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "record_landing.py"),
             "--outdir", outdir, "--bounces", "2", "--timeout", "30"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        subprocess.run(
            ["gz", "sim", "-s", "-r", "--iterations", "2500",
             os.path.join(GZ, "worlds", "table_tennis_board.sdf")],
            env=env, capture_output=True, timeout=180)
        rec.wait(timeout=40)


def true_board_contact(tf, t_after):
    """First forward crossing of X_PLANE after t_after in the full record."""
    t, x = tf["t"], tf["x"]
    for i in range(1, len(t)):
        if t[i - 1] >= t_after and x[i - 1] < X_PLANE <= x[i]:
            f = (X_PLANE - x[i - 1]) / (x[i] - x[i - 1])
            return (tf["y"][i - 1] + f * (tf["y"][i] - tf["y"][i - 1]),
                    tf["z"][i - 1] + f * (tf["z"][i] - tf["z"][i - 1]),
                    t[i - 1] + f * (t[i] - t[i - 1]))
    return None


def predict_board_contact(theta, max_bounces=3, bounce_params=None):
    """Physics pipeline to the board plane: flight -> calibrated bounce ->
    flight, up to max_bounces table contacts. Returns (y, z, t_rel) or None.
    bounce_params: optional (e, mu, alpha) override — used by the
    differentiable bounce-parameter learning in board_ood.py."""
    p = np.array(theta[:3], float)
    v = np.array(theta[3:6], float)
    w = np.array(theta[6:9], float)
    t_acc = 0.0
    for _ in range(max_bounces + 1):
        ts, P, V, lp, lt = simulate(p, v, w, dt=1e-3, t_max=3.0, table_z=R_BALL)
        hit = np.where((P[1:, 0] >= X_PLANE) & (P[:-1, 0] < X_PLANE))[0]
        if len(hit):
            i = hit[0] + 1
            f = (X_PLANE - P[i - 1, 0]) / (P[i, 0] - P[i - 1, 0])
            y = P[i - 1, 1] + f * (P[i, 1] - P[i - 1, 1])
            z = P[i - 1, 2] + f * (P[i, 2] - P[i - 1, 2])
            return y, z, t_acc + ts[i - 1] + f * (ts[i] - ts[i - 1])
        if lp is None:
            return None
        i = int(np.clip(np.searchsorted(ts, lt), 1, len(ts) - 1))
        f = (lt - ts[i - 1]) / (ts[i] - ts[i - 1])
        v_land = V[i - 1] + f * (V[i] - V[i - 1])
        if bounce_params is None:
            v, w = bounce_state(v_land, w)
        else:
            v, w = bounce_state(v_land, w, e=bounce_params[0],
                                mu=bounce_params[1], alpha=bounce_params[2])
        p = lp.copy()
        p[2] = R_BALL
        t_acc += lt
    return None


def extract_row(args):
    idx, cluster, v0, w0, outdir, fps, sigma0, alpha, bad_frac = args
    try:
        tr = load(os.path.join(outdir, "traj.csv"), ["t", "x", "y", "z"])
        tf = load(os.path.join(outdir, "traj_full.csv"), ["t", "x", "y", "z"])
        b = load(os.path.join(outdir, "bounces.csv"), ["n", "x", "y", "t"])
    except Exception:
        return None
    if not (0.0 <= b["x"][0] <= 2.74 and abs(b["y"][0]) <= 0.7625):
        return None                       # first bounce off table
    truth = true_board_contact(tf, b["t"][0])
    if truth is None:
        return None                       # never reached the board

    ot, op, cf, _ = make_observations(tr, fps, sigma0, alpha, bad_frac, seed=idx)
    th = fit_trajectory(ot, op, weights=cf ** 2, fit_omega=True,
                        omega_bound=OMEGA_BOUND)
    pred = predict_board_contact(th)
    if pred is None:
        return None
    # true-spin oracle: same fitted p0/v0, true spin (upper reference)
    th_o = np.concatenate([th[:6], w0])
    pred_o = predict_board_contact(th_o)
    t0 = tr["t"][0]
    row = [idx, cluster, *np.round(v0, 4), *np.round(w0, 2),
           *np.round(th[3:9], 4),
           round(pred[0], 4), round(pred[1], 4), round(t0 + pred[2], 4),
           round(truth[0], 4), round(truth[1], 4), round(truth[2], 4),
           len(ot)]
    if pred_o is not None:
        row += [round(pred_o[0], 4), round(pred_o[1], 4)]
    else:
        row += ["", ""]
    row += list(np.round(th[:3], 4))    # fitted p0: needed to re-run the
    return row                          # pipeline under other bounce params


HEADER = ["serve", "cluster", "vx0", "vy0", "vz0", "wx0", "wy0", "wz0",
          "vhx", "vhy", "vhz", "whx", "why", "whz",
          "pred_y", "pred_z", "pred_t", "true_y", "true_z", "true_t",
          "n_frames", "pred_oracle_y", "pred_oracle_z",
          "phx", "phy", "phz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=3, help="parallel Gazebo sims")
    ap.add_argument("--fps", type=float, default=120)
    ap.add_argument("--sigma0", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bad_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()

    root = os.path.join(GZ, "board_out")
    os.makedirs(root, exist_ok=True)
    serves = sample_serves(a.n, np.random.default_rng(a.seed))

    if not a.skip_sim:
        print(f"[board] simulating {a.n} serves ({a.jobs} parallel) ...")
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(run_serve, i, v, w,
                              os.path.join(root, f"serve_{i:03d}"))
                    for i, (_, v, w) in enumerate(serves)]
            for k, f in enumerate(futs):
                f.result()
                if (k + 1) % 10 == 0:
                    print(f"[board] {k + 1}/{a.n} done")

    print("[board] extracting dataset ...")
    jobs = [(i, c, np.array(v), np.array(w),
             os.path.join(root, f"serve_{i:03d}"),
             a.fps, a.sigma0, a.alpha, a.bad_frac)
            for i, (c, v, w) in enumerate(serves)]
    with ProcessPoolExecutor() as ex:
        rows = [r for r in ex.map(extract_row, jobs) if r is not None]

    out = os.path.join(root, "board_dataset.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"[board] wrote {out}: {len(rows)}/{a.n} valid serves")


if __name__ == "__main__":
    main()
