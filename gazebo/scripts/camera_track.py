#!/usr/bin/env python3
"""Route B capture: subscribe to the two rendered camera streams and detect the
(orange) ball per frame by color threshold — the Gazebo stand-in for a
TrackNet-style detector. Writes per-camera pixel tracks:

    detections_side.csv / detections_back.csv : t,u,v,npix

`npix` (blob size) doubles as the confidence proxy. Triangulation + prediction
happen offline in camera_predict.py.

Exit: after --duration wall seconds, or early once frames stop arriving
(sim ended) with at least a few detections in hand.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

try:                                  # Harmonic
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
except Exception:                     # Garden / older
    from gz.transport12 import Node
    from gz.msgs9.image_pb2 import Image


def detect_ball(img, W, H):
    """Orange-blob centroid. Returns (u, v, npix) or None."""
    a = np.frombuffer(img, dtype=np.uint8)
    if a.size != W * H * 3:
        return None
    a = a.reshape(H, W, 3).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 110) & (r > g * 5 // 4) & (r > b * 7 // 5)
    npix = int(mask.sum())
    if npix < 3:
        return None
    vs, us = np.nonzero(mask)
    return float(us.mean()), float(vs.mean()), npix


class CamRecorder:
    def __init__(self, name):
        self.name = name
        self.rows = []
        self.last_rx = None

    def cb(self, msg: Image):
        self.last_rx = time.time()
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        det = detect_ball(msg.data, msg.width, msg.height)
        if det is not None:
            self.rows.append((t, *det))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="max wall seconds to capture")
    ap.add_argument("--idle", type=float, default=4.0,
                    help="exit after this many idle wall-seconds once data has arrived")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    node = Node()
    cams = {"side": CamRecorder("side"), "back": CamRecorder("back")}
    for name, rec in cams.items():
        topic = f"/camera/{name}"
        if not node.subscribe(Image, topic, rec.cb):
            print(f"failed to subscribe to {topic}", file=sys.stderr)
            sys.exit(1)
    print("capturing /camera/side + /camera/back ...")

    t0 = time.time()
    while time.time() - t0 < a.duration:
        time.sleep(0.2)
        rx = [r.last_rx for r in cams.values() if r.last_rx]
        if rx and time.time() - max(rx) > a.idle \
                and all(len(r.rows) >= 5 for r in cams.values()):
            break

    for name, rec in cams.items():
        path = os.path.join(a.outdir, f"detections_{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "u", "v", "npix"])
            w.writerows(rec.rows)
        print(f"wrote {path} ({len(rec.rows)} detections)")
    if any(len(r.rows) < 5 for r in cams.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
