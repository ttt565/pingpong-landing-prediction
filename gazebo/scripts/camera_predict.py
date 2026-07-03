#!/usr/bin/env python3
"""Route B closure: triangulate the two per-camera pixel tracks into a 3D
trajectory and run the SAME estimators on it — perception noise now comes from
actual rendering + pixel quantization instead of the synthetic Gaussian model.

Inputs (a run dir produced by run_camera.sh):
    detections_side.csv / detections_back.csv   from camera_track.py
    traj.csv / landing.csv                      truth from record_landing.py

Camera geometry below mirrors worlds/table_tennis_cam.sdf — keep in sync.

Outputs: traj_cam.csv (t,x,y,z,conf triangulated), printed + optional
markdown report: stereo 3D residual vs truth, landing error per method.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ttsim.physics import predict_landing, R_BALL       # noqa: E402
from ttsim.estimators import fit_trajectory             # noqa: E402
from predict_from_csv import load                       # noqa: E402

OMEGA_BOUND = 1100.0
W, H, HFOV = 640, 480, 1.2
FX = (W / 2) / np.tan(HFOV / 2)
CX, CY = W / 2.0, H / 2.0


def yaw_R(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


CAMS = {
    # name: (position, R world<-body); body: +x fwd, +y left, +z up
    "side": (np.array([1.37, -2.5, 0.6]), yaw_R(np.pi / 2)),
    "back": (np.array([-2.0, 0.0, 0.8]), yaw_R(0.0)),
}


def pixel_ray(cam, u, v):
    """World-frame (origin, unit direction) for pixel (u, v)."""
    pos, R = CAMS[cam]
    d = np.array([1.0, -(u - CX) / FX, -(v - CY) / FX])
    d = R @ d
    return pos, d / np.linalg.norm(d)


def triangulate(o1, d1, o2, d2):
    """Midpoint of the common perpendicular of two rays."""
    r = o2 - o1
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return None
    s = (c * (d1 @ r) - b * (d2 @ r)) / denom
    t = (b * (d1 @ r) - a * (d2 @ r)) / denom
    return 0.5 * (o1 + s * d1 + o2 + t * d2)


def _land_xy(theta):
    lp, _ = predict_landing(theta, table_z=R_BALL)
    return None if lp is None else lp[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--match_tol", type=float, default=2e-3,
                    help="stamp matching tolerance between cameras (s)")
    ap.add_argument("--out", default=None, help="markdown report path")
    a = ap.parse_args()

    side = load(os.path.join(a.rundir, "detections_side.csv"), ["t", "u", "v", "npix"])
    back = load(os.path.join(a.rundir, "detections_back.csv"), ["t", "u", "v", "npix"])
    tr = load(os.path.join(a.rundir, "traj.csv"), ["t", "x", "y", "z"])
    ld = load(os.path.join(a.rundir, "landing.csv"), ["x", "y", "t"])
    t_land = ld["t"][0]
    true_xy = np.array([ld["x"][0], ld["y"][0]])

    # match frames across cameras by sim timestamp, keep pre-landing arc
    pts, times, conf = [], [], []
    for i, t in enumerate(side["t"]):
        if t > t_land:
            continue
        j = int(np.argmin(np.abs(back["t"] - t)))
        if abs(back["t"][j] - t) > a.match_tol:
            continue
        o1, d1 = pixel_ray("side", side["u"][i], side["v"][i])
        o2, d2 = pixel_ray("back", back["u"][j], back["v"][j])
        p = triangulate(o1, d1, o2, d2)
        if p is None:
            continue
        times.append(t)
        pts.append(p)
        conf.append(np.sqrt(side["npix"][i] * back["npix"][j]))
    if len(pts) < 8:
        sys.exit(f"only {len(pts)} stereo matches — not enough to fit")
    times = np.array(times)
    P = np.stack(pts)
    conf = np.array(conf)

    # stereo perception quality vs 1 kHz ground truth
    truth = np.stack([np.interp(times, tr["t"], tr[c]) for c in "xyz"], axis=1)
    res = np.linalg.norm(P - truth, axis=1)
    rms_mm = 1000 * float(np.sqrt(np.mean(res ** 2)))

    with open(os.path.join(a.rundir, "traj_cam.csv"), "w") as f:
        f.write("t,x,y,z,conf\n")
        for t, p, c in zip(times, P, conf):
            f.write(f"{t},{p[0]},{p[1]},{p[2]},{c}\n")

    # same estimators, REAL rendered-pixel noise (no synthetic injection)
    w_conf = (conf / conf.mean()) ** 2
    preds = {
        "M0": _land_xy(fit_trajectory(times, P, fit_omega=False)),
        "M1": _land_xy(fit_trajectory(times, P, fit_omega=True,
                                      omega_bound=OMEGA_BOUND)),
        "M3_conf": _land_xy(fit_trajectory(times, P, weights=w_conf, fit_omega=True,
                                           omega_bound=OMEGA_BOUND)),
    }

    lines = ["# Route B: stereo-camera perception -> prediction\n",
             f"{len(times)} stereo frames @120 fps (640x480, hfov 1.2), "
             f"stereo 3D RMS vs truth = {rms_mm:.1f} mm, "
             f"truth landing x={true_xy[0]:.4f} y={true_xy[1]:.4f} m\n",
             "| method | landing error (cm) |", "|---|---|"]
    print(f"stereo frames: {len(times)}, 3D RMS = {rms_mm:.1f} mm")
    for m, xy in preds.items():
        err = np.nan if xy is None else 100 * float(np.linalg.norm(xy - true_xy))
        print(f"   {m:8s} landing error = {err:6.2f} cm")
        lines.append(f"| {m} | {err:.2f} |")
    lines.append("\nNoise here is real rendering/quantization noise — roughly "
                 "homoscedastic per arc, so M3 ≈ M1 is the EXPECTED outcome "
                 "(that is the H≈0 null of the killer experiment, measured on "
                 "rendered pixels instead of assumed).")

    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
