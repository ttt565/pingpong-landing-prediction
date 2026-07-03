# Gazebo co-simulation track

Runs the **same** drag + Magnus flight model as the analytical `ttsim` engine,
but inside Gazebo's rigid-body/contact solver — so the bounce, multi-body and
(later) simulated-camera physics come for free. The prediction + evaluation code
is untouched: Gazebo emits a trajectory CSV, the analytical pipeline consumes it.

> **Status: executed and validated** on WSL2 (Ubuntu 24.04, Gazebo Harmonic /
> gz-sim 8.14.0, DART physics). One default serve (`v0=(6,0,0.9) m/s`,
> topspin `ω=(0,400,0) rad/s`): Gazebo lands at **x=1.7531 m, t=0.2849 s**;
> ttsim RK4 from the same post-launch state lands 8.8 mm / 1.7 ms away
> (integrator difference: DART semi-implicit Euler vs RK4, both dt=1 ms) —
> the two backends implement the same dynamics. The prediction pipeline runs
> unchanged on the Gazebo trajectory (see Run below). The numbers in
> `../results/` still come from the analytical engine; Gazebo's added value is
> contact/bounce (M2, second-bounce) and the rendered-camera path (Route B).

## Why Gazebo needs a plugin here

Gazebo's physics engines model gravity + contact but **not** aerodynamic drag or
Magnus lift on a ball. `plugins/aero_launch` is a `gz::sim::System` plugin that:
1. launches the ball with a prescribed initial velocity + spin (one shot), and
2. every step adds `F = m·(−drag_coeff·|v|·v + magnus_coeff·(ω×v))`,
with the **same constants as `ttsim/physics.py`** (`drag_coeff=0.1117`,
`magnus_coeff=0.0016`). So Gazebo and the analytical sim are the same dynamics.

## Layout

```
worlds/table_tennis.sdf          world: gravity, table (top at z=0)
models/pingpong_ball/            40mm/2.7g ball + PosePublisher + AeroLaunch (edit launch here)
plugins/aero_launch/             C++ aero+launch System plugin + CMake
scripts/record_landing.py        subscribe to /model/pingpong_ball/pose -> traj.csv + landing.csv
scripts/predict_from_csv.py      run M0/M1/M3 on the Gazebo trajectory (same metric)
scripts/run.sh                   build plugin -> simulate one serve -> record -> predict
```

The ball's world pose comes from a `PosePublisher` attached to the ball **model**
(`publish_model_pose=true`, 1000 Hz). PosePublisher is a model plugin — at world
scope it does nothing — and link poses are published relative to the model
(identity here), which is why the model pose is the one recorded. Timestamps
live on each individual `pose.header.stamp`, not the `Pose_V` top-level header.

One gz-sim quirk the plugin handles: `Link::SetLinearVelocity/SetAngularVelocity`
create `*VelocityCmd` components that the physics system re-applies every step
and zeroes after each step (they are never auto-removed) — left in place they
pin the ball's velocity to zero. `AeroLaunch` removes both components on the
step after launch so the flight is ballistic from then on.

## Run (Linux + Gazebo Harmonic)

```bash
sudo apt install gz-harmonic libgz-sim8-dev \
     python3-gz-transport13 python3-gz-msgs10 \
     python3-numpy python3-scipy              # repo: packages.osrfoundation.org
cd gazebo && bash scripts/run.sh
```

### Windows host (WSL2) — the validated path

```powershell
wsl --install -d Ubuntu-24.04
```

then inside Ubuntu: add the osrfoundation apt repo, install the packages above,
and run `bash scripts/run.sh` from `gazebo/` under `/mnt/c/...`. Expected output:

```
LANDING  x=1.7531 y=0.0000 m  t=0.2849 s  (n_frames=284)
Gazebo truth landing: x=1.753 y=0.000 m (34 frames, bad_frac=0.2, landing plane z=0.02 m)
   M0         landing error =   2.92 cm
   M1         landing error =   5.28 cm
   M3_conf    landing error =   0.31 cm
   M3_oracle  landing error =   0.60 cm
```

(single noise seed — method ranking fluctuates run to run; the analytical
track's Monte-Carlo matrix is where the statistical claims live)

Sweep working conditions by editing `<init_linear>` / `<init_angular>` in
`models/pingpong_ball/model.sdf` (topspin = +y), or script many serves and pipe
each `traj.csv` through `predict_from_csv.py` to reproduce the analytical matrix
on Gazebo physics.

## Plugin build only

```bash
cmake -S plugins/aero_launch -B plugins/aero_launch/build
cmake --build plugins/aero_launch/build
export GZ_SIM_SYSTEM_PLUGIN_PATH=$PWD/plugins/aero_launch/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
```

For Garden (gz-sim7) change `gz-sim8`→`gz-sim7` in `CMakeLists.txt` and the
transport/msgs imports in `record_landing.py` (transport12 / msgs9).
