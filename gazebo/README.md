# Gazebo co-simulation track

Runs the **same** drag + Magnus flight model as the analytical `ttsim` engine,
but inside Gazebo's rigid-body/contact solver — so the bounce, multi-body and
(later) simulated-camera physics come for free. The prediction + evaluation code
is untouched: Gazebo emits a trajectory CSV, the analytical pipeline consumes it.

> **Status:** this package targets **Gazebo Harmonic (gz-sim8) on Linux**. It was
> authored on macOS without a Gazebo runtime, so it has **not been executed** —
> treat it as ready-to-run-on-your-Linux-rig, not as already-validated. The
> numbers in `../results/` come from the validated analytical engine, which is
> the correct ground truth for pure ballistic+Magnus flight. Gazebo's value is
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
worlds/table_tennis.sdf          world: gravity, table (top at z=0), pose publisher
models/pingpong_ball/            40mm/2.7g ball + AeroLaunch plugin (edit launch here)
plugins/aero_launch/             C++ aero+launch System plugin + CMake
scripts/record_landing.py        subscribe to pose stream -> traj.csv + landing.csv
scripts/predict_from_csv.py      run M0/M1/M3 on the Gazebo trajectory (same metric)
scripts/run.sh                   build plugin -> simulate one serve -> record -> predict
```

## Run (Linux + Gazebo Harmonic)

```bash
sudo apt install gz-harmonic libgz-sim8-dev   # or per https://gazebosim.org
cd gazebo && bash scripts/run.sh
```

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
