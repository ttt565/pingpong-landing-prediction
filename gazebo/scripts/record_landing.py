#!/usr/bin/env python3
"""Subscribe to the ball pose stream from Gazebo and record its trajectory until
the first downward crossing of the table top (z = ball radius), i.e. the LANDING.

Listens to /model/pingpong_ball/pose (gz::msgs::Pose_V), published by the
PosePublisher plugin attached to the ball MODEL with publish_model_pose=true —
a top-level model's pose is its world pose (= ball center). Link poses are
relative to the model (identity here), which is why the model pose is used.

Emits the unified schema also produced by the analytical track:
    traj.csv     : t,x,y,z          (full pre-landing arc, ball center)
    landing.csv  : x,y,t            (one row: first table contact)

so the SAME perception+prediction pipeline (predict_from_csv.py / ttsim) can
consume Gazebo truth — that is the sim-to-real / co-simulation closure.

Run on a Linux host with Gazebo Harmonic. The gz.transport / gz.msgs python
module names are version-suffixed; adjust the import below to your install
(Harmonic = transport13 / msgs10). CLI fallback if the bindings are missing:
    gz topic -e -t /model/pingpong_ball/pose --json-output > poses.json
"""
import argparse
import csv
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

traj = []
landed = {"done": False}


def _cb(msg: Pose_V):
    if landed["done"]:
        return
    for p in msg.pose:
        if p.name == MODEL_NAME:
            # PosePublisher stamps each individual pose's header; the Pose_V
            # top-level header is left empty. Fall back to it just in case.
            st = p.header.stamp
            if st.sec == 0 and st.nsec == 0:
                st = msg.header.stamp
            t = st.sec + st.nsec * 1e-9
            x, y, z = p.position.x, p.position.y, p.position.z
            if traj and traj[-1][3] > RADIUS and z <= RADIUS:
                # interpolate the crossing; do NOT append the post-contact
                # sample (its velocity may already include the bounce impulse)
                tp, xp, yp, zp = traj[-1]
                f = (zp - RADIUS) / (zp - z)
                lx = xp + f * (x - xp)
                ly = yp + f * (y - yp)
                lt = tp + f * (t - tp)
                landed.update(done=True, x=lx, y=ly, t=lt)
                print(f"LANDING  x={lx:.4f} y={ly:.4f} m  t={lt:.4f} s  "
                      f"(n_frames={len(traj)})")
                return
            traj.append((t, x, y, z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="wall-clock seconds to wait for the landing")
    a = ap.parse_args()

    node = Node()
    if not node.subscribe(Pose_V, TOPIC, _cb):
        print(f"failed to subscribe to {TOPIC}", file=sys.stderr)
        sys.exit(1)
    print(f"recording {TOPIC} ... (Ctrl-C to stop)")
    t_start = time.time()
    while not landed["done"]:
        if time.time() - t_start > a.timeout:
            print(f"timeout after {a.timeout:.0f}s: {len(traj)} frames recorded, "
                  f"no landing detected", file=sys.stderr)
            if traj:
                with open("traj.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "x", "y", "z"])
                    w.writerows(traj)
                print("wrote traj.csv (partial, no landing.csv)", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.01)

    with open("traj.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "x", "y", "z"]); w.writerows(traj)
    with open("landing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x", "y", "t"])
        w.writerow([landed["x"], landed["y"], landed["t"]])
    print("wrote traj.csv + landing.csv")


if __name__ == "__main__":
    main()
