#!/usr/bin/env python3
"""Dynamics closure across the sweep: for every recorded condition, integrate
the analytical engine (ttsim RK4) from the ball state right after the Gazebo
launch step and compare the first landing with Gazebo's landing.csv.

The launch velocity command holds v0 exactly through the first physics step
and the aero force starts one step later, so the state at the first recorded
frame is (p_frame0, v0, w0) with v0/w0 from the sweep manifest.

Residuals quantify the DART (semi-implicit Euler) vs RK4 integrator gap —
they are the backend-equivalence budget for every Gazebo-vs-analytic claim.

    python3 closure_check.py ../sweep_out/*/ --out ../results_closure.md
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ttsim.physics import simulate, R_BALL               # noqa: E402
from predict_from_csv import load                        # noqa: E402


def manifest(rundir):
    mpath = os.path.join(os.path.dirname(rundir.rstrip("/")), "manifest.csv")
    with open(mpath) as f:
        return {row["condition"]: row for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mf = manifest(a.rundirs[0])
    lines = ["# Dynamics closure: ttsim RK4 vs Gazebo (DART), first landing\n",
             "Same initial state, both integrated at dt=1 ms to the z=R crossing. "
             "The residual is the integrator gap (semi-implicit Euler vs RK4), "
             "i.e. the backend-equivalence budget.\n",
             "| condition | Gazebo x (m) | RK4 x (m) | dx (mm) | dt (ms) |",
             "|---|---|---|---|---|"]
    dxs = []
    for rd in a.rundirs:
        rd = rd.rstrip("/")
        name = os.path.basename(rd)
        if name not in mf:
            continue
        tr = load(os.path.join(rd, "traj.csv"), ["t", "x", "y", "z"])
        ld = load(os.path.join(rd, "landing.csv"), ["x", "y", "t"])
        row = mf[name]
        p0 = [tr["x"][0], tr["y"][0], tr["z"][0]]
        v0 = [float(row["vx"]), float(row["vy"]), float(row["vz"])]
        w0 = [float(row["wx"]), float(row["wy"]), float(row["wz"])]
        _, _, _, lp, lt = simulate(p0, v0, w0, dt=1e-3, table_z=R_BALL)
        if lp is None:
            lines.append(f"| {name} | {ld['x'][0]:.4f} | (no landing) | — | — |")
            continue
        dx = np.hypot(lp[0] - ld["x"][0], lp[1] - ld["y"][0]) * 1000
        dt = (lt + tr["t"][0] - ld["t"][0]) * 1000
        dxs.append(dx)
        lines.append(f"| {name} | {ld['x'][0]:.4f} | {lp[0]:.4f} | "
                     f"{dx:.1f} | {dt:+.1f} |")
    if dxs:
        lines.append(f"\nmean |dx| = {np.mean(dxs):.1f} mm, "
                     f"max = {np.max(dxs):.1f} mm over {len(dxs)} conditions.")

    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
