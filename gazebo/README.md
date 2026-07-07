# Gazebo co-simulation track

Runs the **same** drag + Magnus flight model as the analytical `ttsim` engine,
but inside Gazebo's rigid-body/contact solver — so the bounce, multi-body and
(later) simulated-camera physics come for free. The prediction + evaluation code
is untouched: Gazebo emits a trajectory CSV, the analytical pipeline consumes it.

> **Status: executed and validated** on WSL2 (Ubuntu 24.04, Gazebo Harmonic /
> gz-sim 8.14.0, DART physics). Dynamics closure holds across all 9 sweep
> conditions: ttsim RK4 and Gazebo land **9.2 mm apart on average (max
> 11.9 mm)** from identical initial states — the integrator gap (DART
> semi-implicit Euler vs RK4, both dt=1 ms), see
> [results_closure.md](results_closure.md). The prediction pipeline runs
> unchanged on Gazebo trajectories. The numbers in `../results/` still come
> from the analytical engine; Gazebo's added value is contact/bounce (M2,
> second-bounce) and the rendered-camera path (Route B).

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
worlds/table_tennis_cam.sdf      + lighting + two 120fps cameras (Route B sharp)
worlds/table_tennis_cam_blur.sdf + net + 500Hz cameras + sensor noise (Route B blur)
models/pingpong_ball/            40mm/2.7g ball + PosePublisher + AeroLaunch (edit launch here)
plugins/aero_launch/             C++ aero+launch System plugin + CMake
scripts/record_landing.py        pose stream -> traj/landing CSVs; --bounces 2 = M2 truth
scripts/predict_from_csv.py      run M0/M1/M3 on a Gazebo trajectory (same metric)
scripts/run.sh                   build plugin -> simulate one serve -> record -> predict
scripts/sweep.py                 9-condition launch sweep -> results_sweep.md
scripts/predict_second_bounce.py M2: fit arc -> bounce model -> 2nd touchdown vs truth
scripts/calibrate_bounce.py      measure DART's effective bounce response from recordings
scripts/closure_check.py         RK4-vs-Gazebo landing gap across all recorded conditions
scripts/camera_track.py          Route B: median-background ball detector (+exposure blend)
scripts/camera_predict.py        Route B: stereo triangulation -> same estimators
scripts/run_camera.sh            Route B end-to-end; "blur" arg = realism pack
scripts/board_learn.py           contact-board dataset: N random serves -> labels
scripts/learn_board_residual.py  ridge residual learning + sample-efficiency curve
worlds/table_tennis_board.sdf    + sensing board just past the table end
sweep_out/, cam_out/, cam_blur_out/  committed ground-truth recordings + summaries
board_out/board_dataset.csv      committed per-serve dataset (raw dirs gitignored)
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
over 30 noise seeds. Results: [results_sweep.md](results_sweep.md), raw
recordings + `manifest.csv` (single source of truth for launch params) under
`sweep_out/`. The analytical track's mechanism carries over unchanged: M0
degrades with spin (1.9→6.7 cm), M3_conf ≈ M3_oracle, and the **paired**
per-seed gain M1−M3_conf is positive with a 95% CI excluding zero in *all
nine conditions* (point estimates 1.1–2.5 cm). The robust-baseline columns
tell the same story as the analytical track (see ../LIMITATIONS.md): the
confidence-attributable remainder **M_huber−M3_conf is 0.1–0.5 cm with CIs
including zero in 8/9 conditions** — robust fitting without any confidence
signal already collects nearly all of the gain on DART physics too.

```bash
python3 scripts/sweep.py             # ~4 min: 9 sims + 1080 fits
python3 scripts/sweep.py --skip-sim  # re-evaluate existing recordings
```

## M2 — second touchdown (`scripts/predict_second_bounce.py`)

Pipeline: fit the pre-bounce arc → integrate to first contact → analytic
impulse bounce (`ttsim/bounce.py`) → integrate to the second z=R crossing.
Results: [results_m2.md](results_m2.md).

Four findings worth the trip:
1. **Calibrate against the backend, not its config**: DART's *effective*
   restitution is **0.777** (measured across all 9 conditions by
   `calibrate_bounce.py`), not the 0.9 in the SDF. Switching E_TABLE to the
   measured value cut the noise-free M2 mismatch from 9–20 cm to **1.8–7.5 cm**.
2. With perception noise, M2 error is 24–66 cm — dominated by **spin-estimation
   error amplified through the bounce** (spin sets the tangential impulse AND
   the second flight's Magnus).
3. That claim is now *tested*, not asserted: **M2_spinknown** (same noisy fit,
   true spin handed in) collapses the error to **3.7–7.8 cm — essentially the
   bounce-model floor** (M2_clean 1.8–7.5 cm). Same lesson as Experiment B:
   the M2 lever is a spin prior, not more precision weighting.
4. And the lever is now *pulled*: **M2_prior** — the same fit with a Gaussian
   MAP session prior (σ = 60 rad/s, `fit_trajectory(omega_prior=...)`) —
   lands at **6.6–14.3 cm**, most of the way from M2_M1 (24–66 cm) to the
   spin-known reference. A session prior is exactly what warm-up fits or the
   contact board provide.

## Route B — rendered cameras (`scripts/run_camera.sh`)

`worlds/table_tennis_cam.sdf` adds two 120 fps 640×480 cameras (side + behind).
`camera_track.py` detects the ball per frame by background subtraction against
a **temporal-median** background (a mean background absorbs an orange bias
where the ball's image moves slowly and grows a ghost trail — measured 2.7×
worse 3D RMS before the fix). Blob size `npix` and sharp-core count `nsharp`
(the TrackNet-peak-score stand-in) come out per detection;
`camera_predict.py` triangulates the stereo pair and runs the same
estimators — **perception noise is real rendering, not the synthetic
Gaussian**.

Two regimes, one serve each:

| mode | world | noise sources | 3D RMS | M0 | M1 | M3_conf |
|---|---|---|---|---|---|---|
| sharp ([results_camera.md](results_camera.md)) | `table_tennis_cam.sdf` | quantization | 5.1 mm | 6.13 | 0.51 | 0.49 cm |
| blur ([results_camera_blur.md](results_camera_blur.md)) | `table_tennis_cam_blur.sdf` | + full-shutter motion blur (500 Hz renders, `--blend 4` → 125 fps), sensor noise, **net occluding the back camera near landing** | 5.8 mm | 8.68 | 1.47 | 1.45 cm |

```bash
bash scripts/run_camera.sh          # sharp
bash scripts/run_camera.sh blur     # realism pack
```

The punchline is the same in both: **M3 ≈ M1 even under blur + occlusion**,
because rendered failures manifest as *dropouts* (the occluded/over-blurred
frames simply vanish — 5 of 62 back-camera frames in blur mode) rather than
as flaggable degraded detections; the within-arc noise that remains is
near-homoscedastic (H≈0). This is the killer experiment's null measured on
pixels, and it sharpens the Phase-1 exit test: M3's regime requires
*detector-level* confusion (real TrackNet on cluttered scenes) — measure H
there before investing in it.

## Contact-board self-supervision (`scripts/board_learn.py` → `learn_board_residual.py`)

A sensing board just past the table end (`worlds/table_tennis_board.sdf`,
face at x = 2.80 m; a real plate would be piezo/mic-array) turns **every serve
into a free ground-truth label** — where the ball strikes it. That label
supervises exactly what the M2 experiment showed is missing: the spin prior
and the residual bounce mismatch.

Experiment: 120 serves sampled from a 3-cluster repertoire (topspin / flat /
backspin, with speed/direction jitter), ONE perception-noise realization per
serve; the physics pipeline (M3_conf fit → calibrated bounce → flight)
predicts the board contact; a multi-output ridge on 9 prediction-time
features learns the residual. Results
([results_board.md](results_board.md), [fig_board_learning.png](fig_board_learning.png)):

| model | board-contact error (held-out serves) |
|---|---|
| physics only | 20.8 cm |
| physics + ridge residual (90 labels) | **3.9 cm** |
| true-spin oracle | 6.8 cm |

Sample efficiency: **10 labels → 7.3 cm (already at the oracle), 20 → 5.1,
80 → 3.9 cm.** Two conclusions: (1) a dozen self-labeled serves effectively
*learn the spin prior* — the board buys the exact lever M2 needs; (2) beyond
~40 labels the learner beats the true-spin oracle, because it also absorbs
the analytic bounce model's systematic bias vs DART. The physics model
remains the out-of-distribution fallback — the learner only adds a
correction on top.

`scripts/board_robustness.py`
([results_board_robustness.md](results_board_robustness.md)) then closes the
two deployment questions:

- **Plate accuracy is not the binding constraint.** With Gaussian label noise
  up to σ=2 cm (a realistic piezo/mic plate) the result is unchanged
  (3.9 → 4.0 cm at 80 labels, 5.1 → 5.6 at 20); even a poor σ=5 cm plate
  still reaches 4.5 cm vs 20.8 physics-only. The residual being corrected is
  ~20 cm, so the label SNR is ~10:1 and ridge averages the rest out
  ([fig_board_label_noise.png](fig_board_label_noise.png)).
- **Learning-while-playing converges in ~20 serves.** Online RLS scored
  prequentially (predict serve k, then update on its noisy label) reaches the
  true-spin-oracle level by ~serve 20 and batch-level ~4 cm by serve 50,
  with or without 2 cm label noise
  ([fig_board_online.png](fig_board_online.png)).

`scripts/board_ood.py` ([results_board_ood.md](results_board_ood.md)) stress-tests
deployment under distribution shift (leave-one-cluster-out):

- **Ridge extrapolates across spin clusters better than feared**: on a
  never-seen cluster it still cuts the error 2–3× vs physics (e.g. 26.4 →
  11.6 cm with topspin held out) — the residual is a smooth function of the
  fitted state, and the clusters overlap in feature space.
- **The Mahalanobis gate works as a detector** (68 % fallback rate on the most
  distinct held-out cluster, 6–8 % elsewhere) **but is priced as insurance**:
  falling back to physics forfeits the extrapolation gain (21.4 vs 11.6 cm).
  Use it where a wrong correction is costly, not to improve averages.
- **Caveat on "learning physics through the pipeline"**: fitting (e, μ, α)
  through flight→bounce→flight transfers OK (OOD 4.2–12.9 cm) but the
  recovered parameters are NOT the true contact parameters — μ and α slam
  into their bounds because the board label constrains the *whole pipeline
  including spin-estimation bias*. Clean bounce calibration still belongs to
  the direct pre/post-velocity measurement (`calibrate_bounce.py`).

```bash
python3 scripts/board_learn.py --n 120 --jobs 3   # ~7 min: simulate + extract
python3 scripts/learn_board_residual.py           # instant, from the CSV
python3 scripts/board_robustness.py               # label-noise + online RLS
python3 scripts/board_ood.py                      # leave-one-cluster-out
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
