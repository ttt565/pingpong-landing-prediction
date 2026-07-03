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
worlds/table_tennis_cam.sdf      + lighting + two 120fps cameras (Route B)
models/pingpong_ball/            40mm/2.7g ball + PosePublisher + AeroLaunch (edit launch here)
plugins/aero_launch/             C++ aero+launch System plugin + CMake
scripts/record_landing.py        pose stream -> traj/landing CSVs; --bounces 2 = M2 truth
scripts/predict_from_csv.py      run M0/M1/M3 on a Gazebo trajectory (same metric)
scripts/run.sh                   build plugin -> simulate one serve -> record -> predict
scripts/sweep.py                 9-condition launch sweep -> results_sweep.md
scripts/predict_second_bounce.py M2: fit arc -> bounce model -> 2nd touchdown vs truth
scripts/calibrate_bounce.py      measure DART's effective bounce response from recordings
scripts/camera_track.py          Route B: color-detect the ball in both camera streams
scripts/camera_predict.py        Route B: stereo triangulation -> same estimators
scripts/run_camera.sh            Route B end-to-end (rendered serve -> prediction)
sweep_out/, cam_out/             committed ground-truth recordings + summaries
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

## Working-condition sweep (`scripts/sweep.py`)

Reproduces the method matrix with **DART as the truth backend**: 9 launch
conditions (speed 4.5–7 m/s; topspin/backspin/sidespin 0–400 rad/s), each
simulated once and recorded through the **second** touchdown, then evaluated
over 12 noise seeds. Results: [results_sweep.md](results_sweep.md), raw
recordings under `sweep_out/`. The analytical track's mechanism carries over
unchanged: M0 degrades with spin (1.8→6.3 cm), M3_conf ≈ M3_oracle ≈ 1.2–1.8 cm
in the bad-frame regime.

```bash
python3 scripts/sweep.py             # ~3 min: 9 sims + 108 fits
python3 scripts/sweep.py --skip-sim  # re-evaluate existing recordings
```

## M2 — second touchdown (`scripts/predict_second_bounce.py`)

Pipeline: fit the pre-bounce arc → integrate to first contact → analytic
impulse bounce (`ttsim/bounce.py`) → integrate to the second z=R crossing.
Results: [results_m2.md](results_m2.md).

Two findings worth the trip:
1. **Calibrate against the backend, not its config**: DART's *effective*
   restitution is **0.777** (measured across all 9 conditions by
   `calibrate_bounce.py`), not the 0.9 in the SDF. Switching E_TABLE to the
   measured value cut the noise-free M2 mismatch from 9–20 cm to **1.8–7.5 cm**.
2. With perception noise, M2 error is 20–60 cm — dominated by **spin-estimation
   error amplified through the bounce** (spin sets the tangential impulse AND
   the second flight's Magnus). Same lesson as Experiment B: the M2 lever is a
   spin prior, not more precision weighting.

## Route B — rendered cameras (`scripts/run_camera.sh`)

`worlds/table_tennis_cam.sdf` adds two 120 fps 640×480 cameras (side + behind).
`camera_track.py` color-thresholds the ball per frame (TrackNet stand-in,
blob size = confidence), `camera_predict.py` triangulates the stereo pair and
runs the same estimators — **perception noise is now real rendering/pixel
quantization, not the synthetic Gaussian**. Results: [results_camera.md](results_camera.md).

Measured on the baseline serve: stereo 3D RMS **5.3 mm**; landing errors
M0 = 6.24 cm (spin bias), M1 = 0.53 cm, M3_conf = 0.55 cm. M3 ≈ M1 is the
*expected* outcome — rendered noise is near-homoscedastic per arc (H≈0), which
is exactly the killer experiment's null, now observed on pixels instead of
assumed. Heteroscedasticity (and M3's edge) needs motion blur / occlusion /
detector confusion, which pure ogre2 rendering does not produce.

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
