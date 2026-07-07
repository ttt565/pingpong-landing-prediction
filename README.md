# Phase 1 — Table-tennis first-landing prediction (sim) & the M3 "killer experiment"

> ⚠️ **Read [LIMITATIONS.md](LIMITATIONS.md) before citing any number.** This is a
> mechanism sandbox under synthetic assumptions (inverse crime, baked-in confidence
> calibration). The headline figures are **not** real-system claims.
>
> **Robust-baseline update (v2):** against the honest reference — a tuned,
> confidence-FREE Huber loss or residual gating — the marginal value of the
> confidence signal at the operating point is **+0.16 cm [0.02, 0.31]**
> (gating: +0.03 [−0.16, +0.27]), an order of magnitude below the ~3 cm floor.
> The old M1-relative gap (+2.17 cm [1.83, 2.52] vs oracle, reproduced
> bit-exactly) measured "weighting vs plain OLS", most of which a robust loss
> collects for free. **Deployment guidance: use Huber/gating; wire in
> confidence only if it is unusually well calibrated (log-noise ≲ 0.6) and
> bad frames are extreme** — see `results/miscalibration.txt` + fig5.

Goal of this phase: **before building any hardware**, decide whether precision-weighted
residual correction (M3) can beat the plain residual/physics fit (M1) by *more than the
real-world measurement floor* (~2–4 cm). If even an idealized M3 can't, the thesis needs
rethinking — and we learn that in CPU-minutes instead of lab-months.

## What is modeled

- **Physics** (`ttsim/physics.py`): gravity + quadratic drag (Cd≈0.4, 40 mm / 2.7 g ball)
  + Magnus `a = K_MAG·(ω×v)`, RK4 integration to the first downward `z=0` crossing.
  Topspin (`ω` along +y) makes the ball dive — landing ~34 cm shorter than no-spin, so a
  no-spin model (M0) is visibly biased long.
  *Note:* the **bounce model is deliberately absent** — the first landing point is set
  entirely by the flight, so the bounce only matters for M2 / future work.
- **Perception noise** (`ttsim/noise.py`): per-frame Gaussian position noise with two
  heteroscedastic sources:
  - **speed** (`alpha`): `σ = σ0·(1+α·|v|/v_ref)` — smooth, but barely varies over a single
    short arc, so it produces almost no *within-arc* spread (this is a key finding).
  - **bad frames** (`bad_frac`, `bad_mult`): a random subset of detections with σ inflated
    (motion-blur spike / occlusion / background confusion). This is the realistic source of
    within-arc heteroscedasticity, and it is what a TrackNet confidence score can flag.
  - `confidence`: a noisy `~1/σ` proxy standing in for a TrackNet heatmap-peak score.

## Methods — one estimator, weights are the only difference

All methods fit `θ=(p0,v0,ω)` to the noisy observed arc by **weighted nonlinear least
squares** against the physics model, then integrate to the landing point
(`ttsim/estimators.py`). M1 vs M3 differ **only** in the per-frame weights — a clean ablation.

| method      | weights / loss                        | role |
|-------------|---------------------------------------|------|
| `M0`        | uniform, ω fixed = 0                  | lower bound (no spin) |
| `M1`        | uniform                               | existing residual/full fit |
| `M_huber`   | uniform, Huber loss (tuned f_scale)   | **robust baseline, no confidence** |
| `M_gate`    | uniform, MAD residual gating + refit  | **robust baseline, no confidence** |
| `M3_conf`   | `confidence²` (realizable)            | confidence-weighted candidate |
| `M3_oracle` | `1/σ_true²`                           | ceiling of *any* precision scheme |
| `M4`        | uniform, noise-free obs               | upper bound |

`M3` must beat `M_huber`/`M_gate` — not `M1` — for the confidence signal to
matter; at the operating point it does not (gap ≈ 0.0–0.2 cm).

`H = std(σ)/mean(σ)` over the observed frames is the **heteroscedasticity index** — *one
diagnostic among several* for whether precision weighting can help (H=0 ⇒ optimal weights
are uniform ⇒ M3≡M1). It is **not** a standalone go/no-go metric: the actual gain also
depends on the bad-frame rate, *where* in the arc bad frames fall, their temporal
correlation, and how well confidence is calibrated to true σ. See [LIMITATIONS.md](LIMITATIONS.md).

## Run

```bash
python run_killer.py --trials 150      # full run (parallel across trials)
python run_killer.py --quick           # 30 trials, fast smoke test
python run_miscalibration.py           # confidence-quality threshold sweep
```

Outputs to `results/`:
- `summary.txt` — all numbers + a GO/NO-GO verdict
- `fig1_methods.png` — method comparison at a realistic operating point
- `fig2_hetero_sweep.png` — **the main figure**: M3 gain vs bad-frame frequency / H,
  now with the Huber baseline and the residual confidence-value curve
- `fig3_nobs_sweep.png` — M3 gain vs number of observed frames
- `fig5_miscalibration.png` + `miscalibration.txt` — how good the confidence
  signal must be (log-noise, mis-scaling γ) before M3 beats a robust loss

## How to read it

The mechanism is Gauss–Markov: weighting by `1/σ²` beats uniform weighting **only** in
proportion to how much σ varies across the observations you actually have. So:

1. If your real TrackNet noise is roughly homoscedastic per arc (H≈0), M3 ≈ M1 — do not
   pursue M3, the gain is below the measurement floor.
2. M3 earns its keep specifically when **some frames are much worse than others and you can
   identify them** (confidence) — i.e. it is really *robustness to bad detections*.
3. **But robustness to bad detections does not require a confidence signal.** A tuned
   Huber loss / MAD gating recovers nearly all of the gain with no side information
   (op. point: M_huber 1.84, M_gate 1.71, M3_conf 1.68 cm); the confidence-attributable
   remainder stays ≤ 1.3 cm even at 50 % bad frames, and turns *negative* once
   confidence log-noise exceeds ~0.6 (fig5) — miscalibrated confidence actively hurts.
   **Rider (fig7):** in the *early-prediction* regime (half the arc observed) the
   Huber→M3 gap opens to +9…+14 cm mean even with zero model mismatch — with few
   frames, confidence weighting mainly suppresses catastrophic fits. Confidence
   earns its keep early in the arc; for full-arc prediction use a robust loss.

So the Phase-1 exit test becomes concrete: **measure H *and* the confidence-vs-precision
calibration from actual TrackNet detections** on a few recorded arcs. Unless the
confidence is unusually informative, the deployable answer is a robust loss, not M3.

## The production ladder (`run_prior.py` + `run_residual_flight.py`)

What Phase 1 converged on, each step a strict add-on, evaluated in the hardest
cell (STRONG model mismatch, half-arc observations, 20 % bad frames):

| step | mean landing error | what it fixes |
|---|---|---|
| `M_huber` (robust loss) | 42.3 cm | bad frames, no side info needed |
| + MAP spin prior | **6.2 cm** | spin identifiability (the Fisher floor) |
| + ridge residual on landing self-labels | **4.0 cm** | systematic model mismatch + fit bias |

The spin prior lives in the estimator (`fit_trajectory(omega_prior=...)`);
it can be **learned from ~20 warm-up full-arc fits with no extra sensor**
(`run_prior.py`: learned ≈ ideal prior, 8-frame error 130 → 25 cm ≈
spin-known — the empirical realization of the fig6 CRLB curves; the weak
direction of the learned prior is exactly the information-free `ω∥v`, which
is harmless). The residual step needs only the landing-board labels the
measurement protocol already assumes (1 cm label noise included; 10 labels
already give 5.4 cm). See `results/fig8_prior.png` / `fig9_residual_flight.png`.

## Experiment B — convergence + spin observability (`run_convergence.py`)

Predicts the landing from only the first *k* frames (all fits ω-bounded). Compares `M1`
(must infer spin) against `M1_spinknown` (true spin handed in). Result
(`results/fig4_convergence.png`): at 8 frames the spin-inferring error is **~43 cm** while
spin-known is **~7 cm** — early first-landing error is dominated by the **estimation
degrees-of-freedom tied to unknown spin, not by perception noise**, and precision weighting
cannot fix that.

This is now backed by a Fisher/CRLB analysis (`run_observability.py`,
`results/fig6_observability.png`): the empirical bounded fit tracks the no-prior
CRLB within ×0.9–1.7 (information floor, not optimizer artifact), **spin
observability is rank-2** — the `ω∥v` component is information-free under
Magnus physics (conditional std ~1 400 rad/s even at full arc) — and a
100 rad/s spin prior moves the 8-frame CRLB from 185 cm to 14 cm. The lever
for early prediction is more arc / a spin prior (learnable from the contact
board); precision weighting is for the bad-frame regime instead.

## Repository layout

```
ttsim/                 analytical engine: physics(+rich truth), noise, bounce, estimators
run_killer.py          M3 go/no-go: operating point + H sweep + frame-rate sweep
run_miscalibration.py  confidence-quality thresholds vs the robust baseline (fig5)
run_observability.py   Fisher/CRLB: rank-2 spin observability, prior value (fig6)
run_mismatch.py        inverse crime broken: rich truth vs simple predictor (fig7)
run_prior.py           MAP spin prior in the estimator: learned = ideal (fig8)
run_residual_flight.py physics + learned residual, honest truth/labels (fig9)
run_convergence.py     experiment B: convergence + spin observability
sanity.py              physics realism + timing self-check
results/               figures + summaries (committed)
gazebo/                Gazebo Harmonic co-simulation package (see gazebo/README.md)
real/                  exit test on real 120fps footage (see real/README.md)
```

## Real-video exit test (`real/`)

The go/no-go numbers, measured on OpenTTGames (120 fps, per-frame ball
labels) with a lightweight background-subtraction detector (TrackNet
stand-in; the harness scores any detector's CSV): calibration **γ̂ = 1.06**
and confidence log-noise **0.17** — both PASS the fig5 thresholds — with σ
spanning 3.6→1.2 px across confidence deciles, within-arc **H = 0.31**, 9.4 %
dropouts. Reading: the confidence signal is good enough to use, but the
measured heterogeneity is well below the synthetic operating point (H≈1.15),
so the expected M3-vs-robust-loss gain is small; the robust loss stays the
default. See [real/results_exit_test.md](real/results_exit_test.md).

## Gazebo co-simulation track (`gazebo/`)

Same drag+Magnus model inside Gazebo's contact solver via a custom `gz::sim` aero
plugin, so the prediction/evaluation code runs unchanged on Gazebo trajectories.
**Executed and validated on WSL2 (Ubuntu 24.04 + Gazebo Harmonic 8.14)**. What runs:

- single serve→record→predict cycle; dynamics closure across all 9 recorded
  conditions: RK4 and Gazebo land 9.2 mm apart on average, max 11.9 mm
  ([gazebo/results_closure.md](gazebo/results_closure.md));
- 9-condition launch sweep reproducing the method matrix on DART physics; the
  paired M1−M3_conf gain is positive with 95% CI excluding zero in all nine
  conditions ([gazebo/results_sweep.md](gazebo/results_sweep.md));
- **M2 second-touchdown prediction** via an impulse bounce model calibrated to
  DART's *measured* effective restitution 0.777 (not the SDF's 0.9) —
  noise-free mismatch 1.8–7.5 cm; with noise M2 is 24–66 cm, and
  **M2_spinknown collapses it to 3.7–7.8 cm**: spin estimation is the
  bottleneck, tested not asserted ([gazebo/results_m2.md](gazebo/results_m2.md));
- **Route B**: rendered stereo cameras → median-background detection →
  triangulation (3D RMS 5.1 mm sharp / 5.8 mm with motion blur + net
  occlusion) → same estimators; in BOTH regimes rendered failures are
  dropouts, residual noise is near-homoscedastic, M3 ≈ M1 — the H≈0 null
  observed on pixels ([gazebo/results_camera.md](gazebo/results_camera.md),
  [gazebo/results_camera_blur.md](gazebo/results_camera_blur.md));
- **contact-board self-supervision**: a sensing board past the table end
  labels every serve for free; a ridge residual learner on top of the physics
  pipeline drops the board-contact error 20.8 → 3.9 cm, reaching the
  true-spin-oracle level with ~10–20 labeled serves and beating it beyond
  ~40 (it also absorbs bounce-model bias) —
  the spin prior, learned ([gazebo/results_board.md](gazebo/results_board.md)).

The committed numbers in `results/` still come from the analytical engine. See
[gazebo/README.md](gazebo/README.md).
