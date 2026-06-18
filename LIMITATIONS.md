# LIMITATIONS — read before citing any number

This repo is a **Phase-1 mechanism sandbox**, not evidence that precision-weighted
residual correction (M3) works in a real table-tennis system. The headline numbers
(`~2.x cm` M3 advantage, `~90%` of the oracle, an `H` go/no-go threshold) hold **only
under the current synthetic assumptions** and must not be reported as real-system results.
Two independent reviews (summarized below) drove these caveats.

## What is NOT yet established

| # | Limitation | Why it matters | Status |
|---|---|---|---|
| 1 | **M1 is weighted physics fitting, not residual correction.** M0/M1/M3 all fit `p0,v0,ω` by weighted NLS; only the weights differ. | This tests "weighting vs no weighting," not the proposal's "physics + learned residual." | open (v2) |
| 2 | **Inverse crime.** Truth and predictor use the *same* drag/Magnus equations + parameters, so `M4 = 0 cm`. | Only estimation error is measured; **zero model error** ⇒ says nothing about sim-to-real. | open (v2, top priority) |
| 3 | **`M3_conf` calibration is baked in:** `confidence = 1/σ · lognormal`. | "90% of oracle" only shows *if* the TrackNet score ≈ true inverse-σ, weighting helps. Circular. | open (v2) |
| 4 | **Early-prediction error was optimizer blow-up, not clean unobservability.** | The `72 cm` figure came from unbounded `ω` exploding (see below). | corrected here |
| 5 | **`H` is not a standalone go/no-go metric.** Same `H≈1.15` gave 2.0 vs 3.5 cm gains; past 35% bad frames `H` falls while gain rises. Also: real per-frame `σ` is unknown without position-labeled data. | The decision depends on bad-frame rate, location, temporal correlation, and confidence calibration jointly. | reframed (v2) |
| 6 | **The "3 cm floor" is not a hard wall.** | A mean difference below single-measurement noise is still detectable with enough paired samples. | reframed (below) |
| 7 | **No robust baselines.** Bad frames are i.i.d. zero-mean high-variance Gaussian; M1 is plain OLS. | M3 may only beat *deliberately naive* OLS. The real competitors are Huber/Cauchy, RANSAC/residual-gating, robust/adaptive Kalman, confidence-threshold rejection. | open (v2, top priority) |
| 8 | **Gazebo coordinate bug (fixed):** recorder used contact at `z = R_ball`, predictor landed at `z = 0`. | ~2 cm systematic bias. | fixed (predictor now lands at `z = R_ball` in the bridge) |

## Corrected finding (point 4) — bounded, with CI

Re-run at the operating point (σ₀=8 mm, α=1, p_miss=0.10, fps=120, bad_frac=0.20),
160 trials, with a **per-component** spin bound `|ω_i| ≤ 1100 rad/s` (NOTE: this is a box
constraint, **not** a norm bound — `‖ω‖` can still reach `1100·√3 ≈ 1905`):

```
8-frame fit, +6 mm noise:  ‖ω‖ unbounded ≈ 61 900 rad/s  (true ~400!) → bounded ≈ 1735
operating point:           median ‖ω‖ of M1_unbounded ≈ 3009 rad/s, M1_bounded ≈ 1188
gap (M1_unbounded − M3) = +2.82 cm [95% CI 2.32, 3.38]   (INFLATED by ω blow-up)
gap (M1_bounded   − M3) = +2.17 cm [95% CI 1.82, 2.52]   (FAIR baseline)
```

**Correct wording:** *under the current synthetic noise and confidence-generation
assumptions, after adding a physical spin bound, the precision-weighting advantage
persists (+2.17 cm [1.82, 2.52]).* It shows the advantage is **not purely a parameter-
explosion artifact** — it does **not** show the mechanism holds for real TrackNet, because
the +2.17 cm still rests on (a) an artificial bad-frame model, (b) `confidence ≈ 1/σ`,
(c) predictor ≡ truth (inverse crime), and (d) no comparison against robust baselines.

## Spin "observability" (point 4, restrained wording)

`bounded M1 ≈ 41 cm` vs `spin-known ≈ 7 cm` at 8 frames shows **the estimation
degrees-of-freedom tied to unknown spin are an important bottleneck** for early
prediction — but this is **not yet** a rigorous "spin is unobservable" claim. The gap
also mixes optimizer local minima, an over-parameterized 3-component `ω`, no spin
prior, and parameter coupling. Establishing unobservability requires a Jacobian
singular-value / profile-likelihood / Fisher-information analysis (v2).

## The "3 cm" threshold, reframed (point 6)

Do not treat measurement noise as an uncrossable line. Separate three things:
- **statistical significance** — set by variance, sample size, and the paired design;
- **engineering significance** — set by paddle size, acceptable landing tolerance, and
  interception success rate;
- **measurement capability** — set by the landing-board (TDOA) error.

Pre-register a **minimum practically-important difference** (e.g. `Δ = 2 cm`) and run
equivalence / superiority tests, rather than drawing a fixed noise line. The current
evaluation also still compares to the noise-free truth; it does **not** yet inject the
landing-board measurement noise the protocol calls for.

## Gazebo does NOT break the inverse crime

The Gazebo plugin uses the **same** quadratic drag + constant `Cd` + constant Magnus
coefficient + constant spin as the analytical predictor, and contact does not participate
in the *first* landing. Swapping integrators adds only numerical differences, so `M4`
would still be ≈ 0. To create genuine model mismatch the **truth** must contain physics the
predictor lacks — and this does **not** require Gazebo; it can be done in the analytical
truth first:
- `Cd(Re)` (Reynolds-dependent drag), spin decay, Magnus coefficient vs spin ratio,
  wind/turbulence, per-ball parameter randomization, and camera projection + motion blur +
  TrackNet detection error.

## v2 roadmap (priority order)

1. **Physical constraints** — norm/axis spin prior (not just box bounds), regularization,
   Jacobian-conditioning / Fisher analysis for observability.
2. **Robust baselines** — Huber/Cauchy, RANSAC/residual-gating, robust/adaptive Kalman,
   confidence-threshold rejection. M3 must beat these to matter.
3. **Artificial model-mismatch matrix** — richer analytical truth (above) vs simplified
   predictor; break the inverse crime; re-introduce a real residual-correction M1.
4. **Confidence-miscalibration sweep** — degrade `confidence` away from `1/σ` and measure
   sensitivity; flag that real calibration needs position-labeled arcs.
5. **Gazebo / TrackNet route** — only after 1–4; add `Cd(Re)`/spin-decay to the plugin and
   the rendered-camera path so Gazebo contributes real mismatch + perception error.
