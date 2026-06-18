# Phase 1 — Table-tennis first-landing prediction (sim) & the M3 "killer experiment"

> ⚠️ **Read [LIMITATIONS.md](LIMITATIONS.md) before citing any number.** This is a
> mechanism sandbox under synthetic assumptions (inverse crime, baked-in confidence
> calibration, no robust baselines). The headline figures are **not** real-system claims.
> The fair, bounded operating-point M3 advantage is **+2.17 cm [95% CI 1.82, 2.52]**, not
> the unbounded +2.8 cm — and still rests on the assumptions listed there.

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

| method      | weights                    | role |
|-------------|----------------------------|------|
| `M0`        | uniform, ω fixed = 0       | lower bound (no spin) |
| `M1`        | uniform                    | existing residual/full fit |
| `M3_conf`   | `confidence²` (realizable) | **the deployable contribution** |
| `M3_oracle` | `1/σ_true²`                | ceiling of *any* precision scheme |
| `M4`        | uniform, noise-free obs    | upper bound |

`H = std(σ)/mean(σ)` over the observed frames is the **heteroscedasticity index** — the
single number that predicts whether precision weighting can help (H=0 ⇒ optimal weights are
uniform ⇒ M3≡M1).

## Run

```bash
python run_killer.py --trials 150      # full run (parallel across trials)
python run_killer.py --quick           # 30 trials, fast smoke test
```

Outputs to `results/`:
- `summary.txt` — all numbers + a GO/NO-GO verdict
- `fig1_methods.png` — method comparison at a realistic operating point
- `fig2_hetero_sweep.png` — **the main figure**: M3 gain vs bad-frame frequency / H, with
  the ~3 cm measurement floor drawn in
- `fig3_nobs_sweep.png` — M3 gain vs number of observed frames

## How to read it

The mechanism is Gauss–Markov: weighting by `1/σ²` beats uniform weighting **only** in
proportion to how much σ varies across the observations you actually have. So:

1. If your real TrackNet noise is roughly homoscedastic per arc (H≈0), M3 ≈ M1 — do not
   pursue M3, the gain is below the measurement floor.
2. M3 earns its keep specifically when **some frames are much worse than others and you can
   identify them** (confidence) — i.e. it is really *robustness to bad detections*.

So the Phase-1 exit test becomes concrete: **measure H from the actual TrackNet detections**
on a few recorded arcs. That number tells you whether M3 is worth the hardware.

## Experiment B — convergence + spin observability (`run_convergence.py`)

Predicts the landing from only the first *k* frames. Compares `M1` (must infer spin)
against `M1_spinknown` (true spin handed in). Result (`results/fig4_convergence.png`):
at 8 frames the spin-inferring error is **~72 cm** while spin-known is **~7 cm** — early
first-landing error is dominated by **spin unobservability, not perception noise**, and
precision weighting cannot fix that. The lever for early prediction is more arc / a spin
prior; precision weighting is for the high-H bad-frame regime instead.

## Repository layout

```
ttsim/                 analytical engine: physics, noise, estimators, experiment
run_killer.py          M3 go/no-go: operating point + H sweep + frame-rate sweep
run_convergence.py     experiment B: convergence + spin observability
sanity.py              physics realism + timing self-check
results/               figures + summaries (committed)
gazebo/                Gazebo Harmonic co-simulation package (see gazebo/README.md)
```

## Gazebo co-simulation track (`gazebo/`)

Same drag+Magnus model inside Gazebo's contact solver via a custom `gz::sim` aero
plugin, so the prediction/evaluation code runs unchanged on Gazebo trajectories.
**Targets Gazebo Harmonic on Linux; authored on macOS without a Gazebo runtime, so it
is ready-to-run but not yet executed** — the committed numbers come from the validated
analytical engine. See [gazebo/README.md](gazebo/README.md).
