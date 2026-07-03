#!/usr/bin/env python3
"""Subscribe to the ball pose stream from Gazebo and record its trajectory,
detecting downward crossings of the table plane (z = ball radius): touchdowns.

Listens to /model/pingpong_ball/pose (gz::msgs::Pose_V), published by the
PosePublisher plugin attached to the ball MODEL with publish_model_pose=true —
a top-level model's pose is its world pose (= ball center). Link poses are
relative to the model (identity here), which is why the model pose is used.
Timestamps live on each individual pose.header.stamp (the Pose_V top-level
header is left empty by PosePublisher).

Outputs (all to --outdir, default cwd) in the unified schema of the
analytical track:
    traj.csv      : t,x,y,z   pre-FIRST-contact arc (input to the predictors)
    landing.csv   : x,y,t     first touchdown (one row)
    bounces.csv   : n,x,y,t   every touchdown        (written when --bounces > 1)
    traj_full.csv : t,x,y,z   full record incl. bounces (when --bounces > 1)

A "touchdown" is the interpolated downward crossing of z = RADIUS. With
--bounces 2 the recorder keeps going through the table bounce and stops at the
second crossing — the M2 (second-bounce) ground truth. If a touchdown lands
outside the table footprint there is nothing to bounce off, so recording stops
there regardless (the crossing itself is still a valid plane-crossing metric).

Run on a Linux host with Gazebo Harmonic. The gz.transport / gz.msgs python
module names are version-suffixed (Harmonic = transport13 / msgs10). CLI
fallback if the bindings are missing:
    gz topic -e -t /model/pingpong_ball/pose --json-output > poses.json
"""
import argparse
import csv
import os
import sys
import time

try:                                  # Harmonic
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
except Exception:                     # Garden / older
    from gz.transport12 import Node
    from gz.msgs9.pose_v_pb2 import Pose_V

MODEL_NAME = "pingpong_ball"
RADIUS = 0.02
TOPIC = f"/model/{MODEL_NAME}/pose"
TABLE_X = (0.0, 2.74)          # table top footprint, top surface at z=0
TABLE_Y = (-0.7625, 0.7625)
REARM_Z = RADIUS + 0.005       # ball must rise above this to arm the next touchdown


class Recorder:
    def __init__(self, n_bounces):
        self.n_bounces = n_bounces
        self.traj = []        # pre-first-contact arc
        self.traj_full = []   # everything up to the stop condition
        self.bounces = []     # (x, y, t) interpolated touchdown points
        self.done = False
        self.armed = True
        self.prev = None

    def cb(self, msg: Pose_V):
        if self.done:
            return
        for p in msg.pose:
            if p.name != MODEL_NAME:
                continue
            st = p.header.stamp
            if st.sec == 0 and st.nsec == 0:
                st = msg.header.stamp
            t = st.sec + st.nsec * 1e-9
            x, y, z = p.position.x, p.position.y, p.position.z
            prev, self.prev = self.prev, (t, x, y, z)

            if prev is not None:
                tp, xp, yp, zp = prev
                if self.armed and zp > RADIUS and z <= RADIUS:
                    f = (zp - RADIUS) / (zp - z)
                    bx = xp + f * (x - xp)
                    by = yp + f * (y - yp)
                    bt = tp + f * (t - tp)
                    self.bounces.append((bx, by, bt))
                    self.armed = False
                    on_table = (TABLE_X[0] <= bx <= TABLE_X[1]
                                and TABLE_Y[0] <= by <= TABLE_Y[1])
                    n = len(self.bounces)
                    print(f"TOUCHDOWN {n}  x={bx:.4f} y={by:.4f} m  t={bt:.4f} s  "
                          f"(n_frames={len(self.traj_full)})"
                          + ("" if on_table else "  [off table]"))
                    if n >= self.n_bounces or not on_table:
                        self.done = True
                        return
                elif not self.armed and z > REARM_Z:
                    self.armed = True

            if not self.bounces:
                self.traj.append((t, x, y, z))
            self.traj_full.append((t, x, y, z))


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="directory for the CSVs")
    ap.add_argument("--bounces", type=int, default=1,
                    help="stop after this many touchdowns (2 = M2 ground truth)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="wall-clock seconds to wait")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rec = Recorder(a.bounces)
    node = Node()
    if not node.subscribe(Pose_V, TOPIC, rec.cb):
        print(f"failed to subscribe to {TOPIC}", file=sys.stderr)
        sys.exit(1)
    print(f"recording {TOPIC} (target: {a.bounces} touchdown(s)) ...")
    t_start = time.time()
    while not rec.done:
        if time.time() - t_start > a.timeout:
            print(f"timeout after {a.timeout:.0f}s: {len(rec.traj_full)} frames, "
                  f"{len(rec.bounces)} touchdown(s)", file=sys.stderr)
            break
        time.sleep(0.01)

    if not rec.bounces:
        if rec.traj:
            _write(os.path.join(a.outdir, "traj.csv"), ["t", "x", "y", "z"], rec.traj)
            print("wrote traj.csv (partial, no landing.csv)", file=sys.stderr)
        sys.exit(1)

    _write(os.path.join(a.outdir, "traj.csv"), ["t", "x", "y", "z"], rec.traj)
    bx, by, bt = rec.bounces[0]
    _write(os.path.join(a.outdir, "landing.csv"), ["x", "y", "t"], [(bx, by, bt)])
    out = ["traj.csv", "landing.csv"]
    if a.bounces > 1:
        _write(os.path.join(a.outdir, "bounces.csv"), ["n", "x", "y", "t"],
               [(i + 1, *b) for i, b in enumerate(rec.bounces)])
        _write(os.path.join(a.outdir, "traj_full.csv"), ["t", "x", "y", "z"],
               rec.traj_full)
        out += ["bounces.csv", "traj_full.csv"]
    print("wrote " + " + ".join(out))
    # success = full bounce count, or a deliberate early stop (off-table touchdown)
    ok = len(rec.bounces) >= a.bounces or rec.done
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
