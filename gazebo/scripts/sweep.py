#!/usr/bin/env python3
"""Sweep launch conditions in Gazebo and reproduce the method-comparison matrix
on Gazebo physics (the analytical track's matrix, but with DART as the truth
backend). For each condition:

  1. patch <init_linear>/<init_angular> into a temp copy of the ball model,
  2. run one headless serve, recording through the SECOND touchdown
     (traj.csv / landing.csv / bounces.csv / traj_full.csv per condition
     under sweep_out/<name>/ — bounces.csv doubles as M2 ground truth),
  3. evaluate M0/M1/M3_conf/M3_oracle over --seeds noise realizations,
  4. aggregate into sweep_out/sweep_summary.csv + ../results_sweep.md.

Run inside the Gazebo environment (Linux / WSL2):
    python3 scripts/sweep.py            # full sweep
    python3 scripts/sweep.py --skip-sim # re-evaluate existing recordings
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))   # gazebo/scripts
GZ = os.path.dirname(HERE)                          # gazebo/
sys.path.insert(0, HERE)
from predict_from_csv import evaluate, load          # noqa: E402

# name, init_linear (m/s), init_angular (rad/s; topspin = +y)
CONDITIONS = [
    ("v45_flat",    (4.5, 0.0, 0.9), (0.0,    0.0, 0.0)),
    ("v45_top200",  (4.5, 0.0, 0.9), (0.0,  200.0, 0.0)),
    ("v45_top400",  (4.5, 0.0, 0.9), (0.0,  400.0, 0.0)),
    ("v60_flat",    (6.0, 0.0, 0.9), (0.0,    0.0, 0.0)),
    ("v60_top200",  (6.0, 0.0, 0.9), (0.0,  200.0, 0.0)),
    ("v60_top400",  (6.0, 0.0, 0.9), (0.0,  400.0, 0.0)),   # baseline serve
    ("v60_back200", (6.0, 0.0, 0.9), (0.0, -200.0, 0.0)),   # backspin floats
    ("v70_top400",  (7.0, 0.0, 0.9), (0.0,  400.0, 0.0)),
    ("v60_mixed",   (6.0, 0.3, 0.9), (50.0, 350.0, 80.0)),  # side+top, lateral
]
METHODS = ["M0", "M1", "M_huber", "M_gate", "M3_conf", "M3_oracle"]
ITERATIONS = 2000   # 2 s sim time: covers both touchdowns for all conditions


def patch_model(tmpdir, v, w):
    dst = os.path.join(tmpdir, "pingpong_ball")
    shutil.copytree(os.path.join(GZ, "models", "pingpong_ball"), dst)
    sdf_path = os.path.join(dst, "model.sdf")
    with open(sdf_path) as f:
        sdf = f.read()
    sdf = re.sub(r"<init_linear>.*?</init_linear>",
                 f"<init_linear>{v[0]} {v[1]} {v[2]}</init_linear>", sdf)
    sdf = re.sub(r"<init_angular>.*?</init_angular>",
                 f"<init_angular>{w[0]} {w[1]} {w[2]}</init_angular>", sdf)
    with open(sdf_path, "w") as f:
        f.write(sdf)


def run_condition(name, v, w, outdir):
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        patch_model(tmp, v, w)
        env = os.environ.copy()
        env["GZ_SIM_RESOURCE_PATH"] = tmp
        env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.path.join(
            GZ, "plugins", "aero_launch", "build")
        rec = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "record_landing.py"),
             "--outdir", outdir, "--bounces", "2", "--timeout", "60"],
            env=env)
        time.sleep(1.0)   # let the subscriber come up
        sim = subprocess.run(
            ["gz", "sim", "-s", "-r", "--iterations", str(ITERATIONS),
             os.path.join(GZ, "worlds", "table_tennis.sdf")],
            env=env, capture_output=True, text=True, timeout=180)
        if sim.returncode != 0:
            print(f"[sweep] gz sim exited {sim.returncode}:\n{sim.stderr[-2000:]}",
                  file=sys.stderr)
        rc = rec.wait(timeout=70)
    return rc == 0 and sim.returncode == 0


def _eval_one(args):
    name, outdir, seed, fps, sigma0, alpha, bad_frac = args
    tr = load(os.path.join(outdir, "traj.csv"), ["t", "x", "y", "z"])
    ld = load(os.path.join(outdir, "landing.csv"), ["x", "y", "t"])
    errs, n = evaluate(tr, ld, fps, sigma0, alpha, bad_frac, seed)
    return name, seed, errs, n


def fmt_vec(v):
    return "(" + ", ".join(f"{x:g}" for x in v) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--fps", type=float, default=120)
    ap.add_argument("--sigma0", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bad_frac", type=float, default=0.2)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--skip-sim", action="store_true",
                    help="reuse existing sweep_out recordings")
    a = ap.parse_args()

    sweep_root = os.path.join(GZ, "sweep_out")
    os.makedirs(sweep_root, exist_ok=True)

    # manifest: single source of truth for launch params per condition
    # (consumed by predict_second_bounce.py / calibrate_bounce.py / closure_check.py)
    with open(os.path.join(sweep_root, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "vx", "vy", "vz", "wx", "wy", "wz"])
        for name, v, wvec in CONDITIONS:
            w.writerow([name, *v, *wvec])

    # ---- phase 1: simulate + record -------------------------------------
    if not a.skip_sim:
        for name, v, w in CONDITIONS:
            outdir = os.path.join(sweep_root, name)
            print(f"[sweep] {name}: v0={fmt_vec(v)} w0={fmt_vec(w)}")
            ok = run_condition(name, v, w, outdir)
            if not ok:
                print(f"[sweep] {name}: recorder reported a problem "
                      f"(see files in {outdir})", file=sys.stderr)

    # ---- phase 2: evaluate ----------------------------------------------
    jobs = []
    for name, v, w in CONDITIONS:
        outdir = os.path.join(sweep_root, name)
        if not os.path.exists(os.path.join(outdir, "landing.csv")):
            print(f"[sweep] {name}: no landing.csv, skipping eval", file=sys.stderr)
            continue
        for seed in range(a.seeds):
            jobs.append((name, outdir, seed, a.fps, a.sigma0, a.alpha, a.bad_frac))
    results = {}   # name -> {method: [errs...]}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for name, seed, errs, n in ex.map(_eval_one, jobs):
            for m, e in errs.items():
                results.setdefault(name, {}).setdefault(m, []).append(e)

    # ---- phase 3: aggregate ----------------------------------------------
    summary_csv = os.path.join(sweep_root, "sweep_summary.csv")
    md = []
    md.append("# Gazebo working-condition sweep\n")
    md.append(f"{len(CONDITIONS)} conditions x {a.seeds} noise seeds, "
              f"fps={a.fps:g}, sigma0={a.sigma0:g} mm, alpha={a.alpha:g}, "
              f"bad_frac={a.bad_frac:g}. Truth = DART (Gazebo Harmonic), "
              f"recording at 1000 Hz. Landing plane z = ball radius.\n")

    md.append("## Ground truth per condition (from Gazebo)\n")
    md.append("| condition | v0 m/s | omega rad/s | 1st touchdown x,y (m) | t1 (s) "
              "| 2nd touchdown x,y (m) | t2 (s) |")
    md.append("|---|---|---|---|---|---|---|")
    for name, v, w in CONDITIONS:
        outdir = os.path.join(sweep_root, name)
        b_path = os.path.join(outdir, "bounces.csv")
        if not os.path.exists(b_path):
            md.append(f"| {name} | {fmt_vec(v)} | {fmt_vec(w)} | — | — | — | — |")
            continue
        b = load(b_path, ["n", "x", "y", "t"])
        c1 = f"{b['x'][0]:.4f}, {b['y'][0]:.4f}"
        t1 = f"{b['t'][0]:.4f}"
        if len(b["n"]) > 1:
            c2 = f"{b['x'][1]:.4f}, {b['y'][1]:.4f}"
            off = "" if 0.0 <= b["x"][1] <= 2.74 and abs(b["y"][1]) <= 0.7625 \
                else " (off table)"
            c2 += off
            t2 = f"{b['t'][1]:.4f}"
        else:
            c2 = t2 = "—"
        md.append(f"| {name} | {fmt_vec(v)} | {fmt_vec(w)} | {c1} | {t1} | {c2} | {t2} |")

    md.append("\n## First-landing prediction error, mean ± std cm over seeds\n")
    md.append("Gain columns are PAIRED per-seed differences (same noise "
              "realization) with 95% t-intervals. M1−M3_conf = the headline "
              "gain over uniform weighting; M_huber−M3_conf = the marginal "
              "value of the confidence signal over a tuned confidence-free "
              "robust loss (the honest bar).\n")
    md.append("| condition | " + " | ".join(METHODS) +
              " | M1−M3_conf [CI] | M_huber−M3_conf [CI] |")
    md.append("|---" * (len(METHODS) + 3) + "|")
    from scipy.stats import t as t_dist

    def paired_gain(a, b):
        g = np.array(a) - np.array(b)
        g = g[~np.isnan(g)]
        half = t_dist.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
        return g, half

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "method", "mean_cm", "std_cm", "seeds"])
        for name, _, _ in CONDITIONS:
            if name not in results:
                continue
            cells = []
            for m in METHODS:
                e = np.array(results[name][m])
                w.writerow([name, m, f"{np.nanmean(e):.3f}",
                            f"{np.nanstd(e):.3f}", len(e)])
                cells.append(f"{np.nanmean(e):.2f} ± {np.nanstd(e):.2f}")
            for ref, tag in (("M1", "M1_minus_M3conf"),
                             ("M_huber", "Mhuber_minus_M3conf")):
                g, half = paired_gain(results[name][ref], results[name]["M3_conf"])
                w.writerow([name, tag, f"{g.mean():.3f}",
                            f"{g.std(ddof=1):.3f}", len(g)])
                cells.append(f"**{g.mean():.2f}** [{g.mean()-half:.2f}, {g.mean()+half:.2f}]")
            md.append(f"| {name} | " + " | ".join(cells) + " |")

    md_path = os.path.join(GZ, "results_sweep.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {summary_csv} and {md_path}")


if __name__ == "__main__":
    main()
