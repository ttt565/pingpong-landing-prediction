#!/usr/bin/env python3
"""Subscribe to the ball pose stream from Gazebo and record its trajectory until
the first downward crossing of the table top (z = ball radius), i.e. the LANDING.

Emits the unified schema also produced by the analytical track:
    traj.csv     : t,x,y,z          (full pre-landing arc)
    landing.csv  : x,y,t            (one row: first table contact)

so the SAME perception+prediction pipeline (predict_from_csv.py / ttsim) can
consume Gazebo truth — that is the sim-to-real / co-simulation closure.

Run on a Linux host with Gazebo Harmonic. The gz.transport / gz.msgs python
module names are version-suffixed; adjust the import below to your install
(Harmonic = transport13 / msgs10). CLI fallback if the bindings are missing:
    gz topic -e -t /world/table_tennis/pose/info --json > poses.json
"""
import csv
import sys
import time

try:                                  # Harmonic
    from gz.transport13 import Node
    from gz.msgs10.pose_v_pb2 import Pose_V
except Exception:                     # Garden / older
    from gz.transport12 import Node
    from gz.msgs9.pose_v_pb2 import Pose_V

BALL_LINK = "ball_link"
RADIUS = 0.02
TOPIC = "/world/table_tennis/pose/info"

traj = []
landed = {"done": False}


def _cb(msg: Pose_V):
    if landed["done"]:
        return
    t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
    for p in msg.pose:
        if p.name.endswith(BALL_LINK):
            x, y, z = p.position.x, p.position.y, p.position.z
            if traj and traj[-1][3] > RADIUS and z <= RADIUS:
                f = (traj[-1][3] - RADIUS) / (traj[-1][3] - z)
                lx = traj[-1][1] + f * (x - traj[-1][1])
                ly = traj[-1][2] + f * (y - traj[-1][2])
                landed.update(done=True, x=lx, y=ly, t=t)
                print(f"LANDING  x={lx:.4f} y={ly:.4f} m  t={t:.4f} s  "
                      f"(n_frames={len(traj)})")
            traj.append((t, x, y, z))


def main():
    node = Node()
    if not node.subscribe(Pose_V, TOPIC, _cb):
        print(f"failed to subscribe to {TOPIC}", file=sys.stderr)
        sys.exit(1)
    print(f"recording {TOPIC} ... (Ctrl-C to stop)")
    while not landed["done"]:
        time.sleep(0.01)

    with open("traj.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "x", "y", "z"]); w.writerows(traj)
    with open("landing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x", "y", "t"])
        w.writerow([landed["x"], landed["y"], landed["t"]])
    print("wrote traj.csv + landing.csv")


if __name__ == "__main__":
    main()
