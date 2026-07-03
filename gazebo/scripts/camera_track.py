#!/usr/bin/env python3
"""Route B capture: subscribe to the two rendered camera streams and detect the
ball per frame — the Gazebo stand-in for a TrackNet-style detector.

Exposure emulation: with --blend N, N consecutive renders are averaged into one
output frame stamped at their mean time (e.g. 500 Hz renders + --blend 4 =
125 fps full-shutter frames with REAL speed-dependent motion blur).

Detection: the scene is static except the ball, so the detector is background
subtraction — per-pixel L1 distance to the temporal-mean background:
    npix   : pixels with diff > DIFF_LO   (the whole, possibly smeared, blob)
    nsharp : pixels with diff > DIFF_HI   (crisp core; collapses under motion
             blur or partial occlusion -> the confidence proxy for M3)

Writes per-camera pixel tracks: detections_side.csv / detections_back.csv
with columns t,u,v,npix,nsharp. Triangulation + prediction happen offline in
camera_predict.py.
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

DIFF_LO = 60.0    # summed |RGB| distance to background: blob membership
DIFF_HI = 250.0   # near-pure ball over background: sharp-core membership


class CamRecorder:
    def __init__(self, name, blend):
        self.name = name
        self.blend = blend
        self.acc = None
        self.acc_t = []
        self.frames = []   # (t, uint8 HxWx3) exposure-blended frames
        self.last_rx = None

    def cb(self, msg: Image):
        self.last_rx = time.time()
        if msg.pixel_format_type not in (0, 3):   # unset / RGB_INT8
            return
        a = np.frombuffer(msg.data, dtype=np.uint8)
        if a.size != msg.width * msg.height * 3:
            return
        a = a.reshape(msg.height, msg.width, 3)
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        if self.blend <= 1:
            self.frames.append((t, a.copy()))
            return
        if self.acc is None:
            self.acc = np.zeros(a.shape, np.float32)
        self.acc += a
        self.acc_t.append(t)
        if len(self.acc_t) >= self.blend:
            frame = (self.acc / len(self.acc_t)).astype(np.uint8)
            self.frames.append((float(np.mean(self.acc_t)), frame))
            self.acc = None
            self.acc_t = []

    def detect_all(self):
        """Two-pass: temporal-MEDIAN background, then per-frame diff blobs.

        Median, not mean: where the ball's image moves slowly (e.g. receding
        from the back camera) it covers the same pixels for many consecutive
        frames — a mean background absorbs an orange bias comparable to
        DIFF_LO and grows a ghost trail that drags the centroid. The ball
        never covers a pixel for >50% of frames, so the median is immune.
        Computed on <=31 evenly-spaced frames, in row blocks, to bound memory.
        """
        if len(self.frames) < 5:
            return []
        stack = [f for _, f in self.frames]
        sub = np.stack(stack[::max(1, len(stack) // 31)])
        H = sub.shape[1]
        bg = np.empty(sub.shape[1:], np.float32)
        for r in range(0, H, 60):
            bg[r:r + 60] = np.median(sub[:, r:r + 60], axis=0)
        rows = []
        for t, f in self.frames:
            diff = np.abs(f.astype(np.float32) - bg).sum(axis=2)
            mask = diff > DIFF_LO
            npix = int(mask.sum())
            if npix < 3:
                continue
            vs, us = np.nonzero(mask)
            nsharp = int((diff > DIFF_HI).sum())
            rows.append((t, float(us.mean()), float(vs.mean()), npix, nsharp))
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--blend", type=int, default=1,
                    help="renders averaged per output frame (exposure emulation)")
    ap.add_argument("--duration", type=float, default=180.0,
                    help="max wall seconds to capture")
    ap.add_argument("--idle", type=float, default=4.0,
                    help="exit after this many idle wall-seconds once data arrived")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    node = Node()
    cams = {"side": CamRecorder("side", a.blend),
            "back": CamRecorder("back", a.blend)}
    for name, rec in cams.items():
        topic = f"/camera/{name}"
        if not node.subscribe(Image, topic, rec.cb):
            print(f"failed to subscribe to {topic}", file=sys.stderr)
            sys.exit(1)
    print(f"capturing /camera/side + /camera/back (blend={a.blend}) ...")

    t0 = time.time()
    while time.time() - t0 < a.duration:
        time.sleep(0.2)
        rx = [r.last_rx for r in cams.values() if r.last_rx]
        if rx and time.time() - max(rx) > a.idle \
                and all(len(r.frames) >= 5 for r in cams.values()):
            break

    ok = True
    for name, rec in cams.items():
        rows = rec.detect_all()
        path = os.path.join(a.outdir, f"detections_{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "u", "v", "npix", "nsharp"])
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} detections from {len(rec.frames)} frames)")
        ok = ok and len(rows) >= 5
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
