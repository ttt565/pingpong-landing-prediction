"""Experiment B: truncated-observation convergence curve + spin observability.

Predict the landing point from only the first k frames of the arc (early
prediction). Three estimators isolate WHERE the error comes from:

    M0          no spin                       -> model bias (spin ignored)
    M1          fit p0,v0,omega               -> must infer spin from a short arc
    M1_spinknown fit p0,v0 with TRUE omega     -> spin handed to it for free
    M3_conf     M1 + confidence weighting

The gap (M1 - M1_spinknown) at small k = the *spin-observability penalty*: how
much of the early-prediction error is spin being unidentifiable from a short
arc, rather than perception noise. If M1 >> M1_spinknown while M1_spinknown is
already low, the bottleneck for early first-landing prediction is SPIN, not
per-frame noise -- which is exactly the regime precision weighting does NOT fix.

Outputs: results/convergence.txt, results/fig4_convergence.png
"""
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttsim.physics import simulate
from ttsim.noise import add_noise
from ttsim.experiment import sample_launch, make_observations
from ttsim import estimators as E

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FPS = 120
SIGMA0, ALPHA, P_MISS = 0.006, 1.0, 0.0   # clean-ish noise to isolate the spin effect
KS = [8, 10, 12, 15, 20, 25, 30]
TRIALS = 150


def _err(xy, land):
    return np.nan if xy is None else 100.0 * np.linalg.norm(xy - land[:2])


def _worker(seed):
    rng = np.random.default_rng(seed)
    p0, v0, om = sample_launch(rng)
    times, pos, vel, land, t_land = simulate(p0, v0, om)
    if land is None:
        return None
    fr_t, P, sp = make_observations(times, pos, vel, t_land, FPS)
    noisy, sig, keep, conf = add_noise(P, sp, SIGMA0, ALPHA, P_MISS, rng)
    out = {}
    for k in KS:
        if k > len(fr_t):
            continue
        ot, op, cf = fr_t[:k], noisy[:k], conf[:k]
        out[k] = dict(
            M0=_err(E.predict_M0(ot, op), land),
            M1=_err(E.predict_M1(ot, op), land),
            M1sk=_err(E.predict_M1_spinknown(ot, op, om), land),
            M3c=_err(E.predict_M3_conf(ot, op, cf), land),
        )
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    workers = max(1, os.cpu_count() - 1)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_worker, range(1, TRIALS + 1), chunksize=4):
            if r is not None:
                results.append(r)

    lines = [f"Experiment B: convergence + spin observability  (trials={len(results)}, "
             f"fps={FPS}, sigma0={SIGMA0*1000:.0f}mm, no bad frames)", "=" * 68,
             f"  median landing error (cm) by #frames observed from launch", ""]
    lines.append(f"  {'k':>3s} {'t_obs':>6s} {'M0':>7s} {'M1':>7s} {'M1_spinknown':>13s} "
                 f"{'M3conf':>7s} {'spin_penalty':>12s}")
    curves = {m: [] for m in ["M0", "M1", "M1sk", "M3c"]}
    pen = []
    for k in KS:
        vals = {m: np.array([r[k][m] for r in results if k in r], float) for m in ["M0", "M1", "M1sk", "M3c"]}
        med = {m: np.nanmedian(vals[m]) for m in vals}
        for m in curves:
            curves[m].append(med[m])
        sp_pen = med["M1"] - med["M1sk"]
        pen.append(sp_pen)
        lines.append(f"  {k:3d} {1000*k/FPS:5.0f}ms {med['M0']:7.2f} {med['M1']:7.2f} "
                     f"{med['M1sk']:13.2f} {med['M3c']:7.2f} {sp_pen:+12.2f}")

    # verdict
    i_small = 0  # k=8
    dominates = pen[i_small] > curves["M1sk"][i_small]
    lines += ["", "READING (all fits omega-bounded per-component |w_i|<=1100):",
              f"  - At k={KS[i_small]} frames ({1000*KS[i_small]/FPS:.0f} ms of arc): "
              f"M1={curves['M1'][i_small]:.1f}cm vs M1_spinknown={curves['M1sk'][i_small]:.1f}cm.",
              f"    unknown-spin estimation penalty = {pen[i_small]:.1f}cm -> "
              f"{'unknown-spin DoF dominate early-prediction error' if dominates else 'noise, not spin, dominates'}.",
              f"  - With true spin known, error is already ~{curves['M1sk'][i_small]:.1f}cm at "
              f"{KS[i_small]} frames: geometry/noise is fine; inferring omega is the hard part.",
              "  - NOTE: this is the *estimation DoF tied to unknown spin*, NOT yet a rigorous",
              "    'spin unobservable' claim -- that needs Jacobian-singular-value / Fisher",
              "    analysis (v2). The penalty also mixes local minima and omega over-parameterization.",
              "  - Implication: the lever for early prediction is more arc / a spin prior,",
              "    not per-frame precision weighting (which helps the bad-frame regime instead)."]

    print("\n".join(lines))
    with open(os.path.join(RESULTS, "convergence.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t_ms = [1000 * k / FPS for k in KS]
    ax.plot(t_ms, curves["M0"], "o-", color="#888", label="M0 (no spin)")
    ax.plot(t_ms, curves["M1"], "o-", color="#1f77b4", label="M1 (infer spin)")
    ax.plot(t_ms, curves["M3c"], "s-", color="#2ca02c", label="M3_conf")
    ax.plot(t_ms, curves["M1sk"], "^-", color="#d62728", label="M1 (TRUE spin known)")
    ax.fill_between(t_ms, curves["M1sk"], curves["M1"], color="#1f77b4", alpha=.10,
                    label="unknown-spin estimation penalty")
    ax.set_xlabel("observation window from launch (ms)  /  more frames →")
    ax.set_ylabel("median landing error (cm)")
    ax.set_title("Early first-landing prediction: unknown-spin DoF dominate (bounded ω)")
    ax.set_ylim(0, min(40, np.nanmax(curves["M1"]) * 1.1))
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig4_convergence.png"), dpi=130)
    print(f"\nsaved -> {RESULTS}/convergence.txt + fig4_convergence.png")


if __name__ == "__main__":
    main()
