#!/usr/bin/env python3
"""Measure the EFFECTIVE bounce response of the Gazebo (DART) table contact
from recorded trajectories, and compare with ttsim.bounce.bounce_state.

For each run dir (needs traj_full.csv + bounces.csv from --bounces 2):
  * fit linear velocity over a short window before/after the first touchdown,
  * report measured restitution e_eff = -vz_out/vz_in and tangential delta_v,
  * report what bounce_state() would predict for the same incoming state
    (incoming spin = launch spin: no air torque is modeled, so spin is
    constant during flight in both backends).

Use the output to set/justify E_TABLE / MU_TABLE / ALPHA_I in ttsim/bounce.py.

    python3 calibrate_bounce.py ../sweep_out/*/
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ttsim.bounce import bounce_state                    # noqa: E402
from predict_from_csv import load                        # noqa: E402

# launch spin per sweep condition (must mirror sweep.py CONDITIONS)
SPINS = {
    "v45_flat": (0, 0, 0), "v45_top200": (0, 200, 0), "v45_top400": (0, 400, 0),
    "v60_flat": (0, 0, 0), "v60_top200": (0, 200, 0), "v60_top400": (0, 400, 0),
    "v60_back200": (0, -200, 0), "v70_top400": (0, 400, 0),
    "v60_mixed": (50, 350, 80),
}
WIN = (0.004, 0.040)   # fit window: t1 +/- [4, 40] ms, contact frames excluded


def vel_fit(tr, t_lo, t_hi):
    m = (tr["t"] >= t_lo) & (tr["t"] <= t_hi)
    if m.sum() < 4:
        return None
    t = tr["t"][m]
    return np.array([np.polyfit(t, tr[c][m], 1)[0] for c in "xyz"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+")
    a = ap.parse_args()

    rows = []
    for rd in a.rundirs:
        rd = rd.rstrip("/")
        name = os.path.basename(rd)
        f_traj = os.path.join(rd, "traj_full.csv")
        f_b = os.path.join(rd, "bounces.csv")
        if not (os.path.exists(f_traj) and os.path.exists(f_b) and name in SPINS):
            continue
        tr = load(f_traj, ["t", "x", "y", "z"])
        t1 = load(f_b, ["n", "x", "y", "t"])["t"][0]
        v_in = vel_fit(tr, t1 - WIN[1], t1 - WIN[0])
        v_out = vel_fit(tr, t1 + WIN[0], t1 + WIN[1])
        if v_in is None or v_out is None:
            continue
        w_in = np.array(SPINS[name], float)
        v_pred, w_pred = bounce_state(v_in, w_in)
        rows.append((name, v_in, v_out, v_pred, w_in))

    print(f"{'condition':12s} {'e_eff':>6s} {'vx_in':>7s} {'vx_out':>7s} "
          f"{'vx_pred':>8s} {'vy_out':>7s} {'vy_pred':>8s}")
    e_all = []
    for name, v_in, v_out, v_pred, w_in in rows:
        e_eff = -v_out[2] / v_in[2]
        e_all.append(e_eff)
        print(f"{name:12s} {e_eff:6.3f} {v_in[0]:7.3f} {v_out[0]:7.3f} "
              f"{v_pred[0]:8.3f} {v_out[1]:7.3f} {v_pred[1]:8.3f}")
    if e_all:
        print(f"\nmean e_eff = {np.mean(e_all):.4f}  (bounce.py E_TABLE)")
        print(json.dumps({"e_eff": [round(float(e), 4) for e in e_all]}))


if __name__ == "__main__":
    main()
