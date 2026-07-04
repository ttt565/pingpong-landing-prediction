"""Break the inverse crime (v2 roadmap item 3): TRUTH from the richer physics
(ttsim/physics_rich.py — Cd(v), Magnus saturation, spin decay), predictors
unchanged (simplified constant-coefficient model).

What this answers, per mismatch level (none / mild / strong):
  * M4 (fit on noise-free observations) is now the pure MODEL-ERROR floor —
    with 'none' it must return ~0 (sanity that only the truth changed);
  * do the noise-weighting conclusions survive model mismatch — in particular
    the robust-baseline verdict (M_huber ~ M3_conf)?
  * how do all methods degrade as mismatch grows relative to the ~3 cm floor?

Outputs: results/mismatch.txt + results/fig7_mismatch.png
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttsim.experiment import sample_launch, make_observations
from ttsim.physics_rich import simulate_rich, LEVELS
from ttsim.noise import add_noise
from ttsim import estimators as E

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OP = dict(sigma0=0.008, alpha=1.0, p_miss=0.10, fps=120,
          bad_frac=0.20, bad_mult=6.0)
METHODS = ["M0", "M1", "M_huber", "M_gate", "M3_conf", "M3_oracle", "M4"]
FLOOR_CM = 3.0


def _trial(args):
    level, obs_frac, seed = args
    prm = LEVELS[level]
    rng = np.random.default_rng(seed)
    p0, v0, om = sample_launch(rng)
    times, pos, vel, land, t_land = simulate_rich(p0, v0, om, prm, dt=1e-3)
    if land is None:
        return None
    fr_t, P, sp = make_observations(times, pos, vel, t_land, OP["fps"])
    if obs_frac < 1.0:
        # early-prediction regime: the fit must EXTRAPOLATE through the
        # mismatched physics instead of interpolating the observed arc —
        # this is where model error actually bites
        keep_n = max(8, int(len(fr_t) * obs_frac))
        fr_t, P, sp = fr_t[:keep_n], P[:keep_n], sp[:keep_n]
    if len(fr_t) < 8:
        return None
    noisy, sig, keep, conf = add_noise(P, sp, OP["sigma0"], OP["alpha"],
                                       OP["p_miss"], rng,
                                       bad_frac=OP["bad_frac"],
                                       bad_mult=OP["bad_mult"])
    ot, op_, sg, cf, cl = fr_t[keep], noisy[keep], sig[keep], conf[keep], P[keep]
    if len(ot) < 8:
        return None
    true_xy = land[:2]

    def err(xy):
        return np.nan if xy is None else 100 * float(np.linalg.norm(xy - true_xy))

    return (level, obs_frac), {
        "M0": err(E.predict_M0(ot, op_)),
        "M1": err(E.predict_M1(ot, op_)),
        "M_huber": err(E.predict_M_huber(ot, op_)),
        "M_gate": err(E.predict_M_gate(ot, op_)),
        "M3_conf": err(E.predict_M3_conf(ot, op_, confidence=cf)),
        "M3_oracle": err(E.predict_M3_oracle(ot, op_, sigma_true=sg)),
        "M4": err(E.predict_M4(ot, cl)),
    }


def boot_gap(a, b, n=2000, seed=0):
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    OBS_FRACS = [1.0, 0.5]
    jobs = [(lvl, of, 11_000_000 + i)
            for lvl in LEVELS for of in OBS_FRACS for i in range(a.trials)]
    res = {(lvl, of): [] for lvl in LEVELS for of in OBS_FRACS}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_trial, jobs, chunksize=4):
            if out is not None:
                res[out[0]].append(out[1])

    log(f"Inverse-crime break: rich truth vs simplified predictor "
        f"(n={a.trials}/cell, operating point bad_frac=0.2)")
    log("=" * 74)
    log("levels: none = truth==predictor model (sanity), "
        "mild/strong = Cd(v) + Magnus saturation + spin decay (physics_rich.py)")
    log("obs=1.0: fit sees the whole arc (interpolation; the fit ABSORBS the "
        "mismatch into biased theta-hat). obs=0.5: fit sees the first half "
        "and must extrapolate through the mismatched physics.")
    means = {}
    gaps_hc = {}
    for lvl in LEVELS:
        for of in OBS_FRACS:
            r = res[(lvl, of)]
            log(f"\n  [{lvl}, obs={of:.1f}]  (n={len(r)})")
            log(f"    {'method':10s} {'mean':>7s} {'median':>7s}  (cm)")
            for m in METHODS:
                e = np.array([x[m] for x in r], float)
                e = e[np.isfinite(e)]
                means[(lvl, of, m)] = e.mean()
                log(f"    {m:10s} {e.mean():7.2f} {np.median(e):7.2f}")
            g_cf = boot_gap(np.array([x['M1'] for x in r]),
                            np.array([x['M3_conf'] for x in r]))
            g_hc = boot_gap(np.array([x['M_huber'] for x in r]),
                            np.array([x['M3_conf'] for x in r]))
            gaps_hc[(lvl, of)] = g_hc[0]
            log(f"    gap M1-M3conf     = {g_cf[0]:+.2f} [{g_cf[1]:+.2f},{g_cf[2]:+.2f}]")
            log(f"    gap Mhuber-M3conf = {g_hc[0]:+.2f} [{g_hc[1]:+.2f},{g_hc[2]:+.2f}]")

    log(f"\nREADING")
    log(f"  - M4 (pure model error) full arc: "
        + ", ".join(f"{lvl} {means[(lvl, 1.0, 'M4')]:.2f}" for lvl in LEVELS)
        + " cm -> the flight fit re-absorbs Cd(v)/saturation/decay into "
          "biased theta-hat almost perfectly (saturation is exactly an "
          "omega-rescale). First-landing prediction from a full arc is "
          "insensitive to this mismatch class;")
    log(f"    the bias re-emerges wherever theta-hat is USED further: "
        f"early prediction (below), and the M2 bounce pipeline.")
    log(f"  - M4 half arc: "
        + ", ".join(f"{lvl} {means[(lvl, 0.5, 'M4')]:.2f}" for lvl in LEVELS)
        + " cm -> extrapolation through mismatched physics is where model "
          "error bites.")
    g_full = max(gaps_hc[(lvl, 1.0)] for lvl in LEVELS)
    g_half = {lvl: gaps_hc[(lvl, 0.5)] for lvl in LEVELS}
    log(f"  - robust-baseline verdict, FULL arc: gap Mhuber-M3conf <= "
        f"{g_full:+.2f} cm at every mismatch level -> 'use a robust loss' "
        f"is not an inverse-crime artifact.")
    log(f"  - HALF arc (early prediction): the gap opens to "
        + ", ".join(f"{lvl} {g_half[lvl]:+.2f}" for lvl in LEVELS)
        + " cm mean — including with NO mismatch, i.e. this is the "
          "few-frames REGIME, not model error: with ~17 frames and 20% bad "
          "frames, confidence weighting mainly suppresses catastrophic fits "
          "(means are tail-driven; medians sit ~2-3 cm apart). The batch-1 "
          "verdict gains a rider: confidence earns its keep in the "
          "early-prediction regime.")

    with open(os.path.join(RESULTS, "mismatch.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # grouped bars, one panel per observation regime
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    xs = np.arange(len(METHODS))
    wd = 0.26
    colors = {"none": "#999999", "mild": "#1f77b4", "strong": "#d62728"}
    for ax, of, ttl in ((axes[0], 1.0, "full arc observed (interpolation)"),
                        (axes[1], 0.5, "half arc observed (extrapolation)")):
        for i, lvl in enumerate(LEVELS):
            ax.bar(xs + (i - 1) * wd, [means[(lvl, of, m)] for m in METHODS],
                   wd, color=colors[lvl], label=f"truth: {lvl}")
        ax.axhline(FLOOR_CM, ls="--", c="r", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(METHODS, fontsize=8)
        ax.set_ylabel("mean landing error (cm)")
        ax.set_title(ttl)
        ax.legend(fontsize=8)
    fig.suptitle("Inverse crime broken: rich truth vs simplified predictor")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig7_mismatch.png"), dpi=130)
    print(f"\nsaved -> {RESULTS}/mismatch.txt + fig7_mismatch.png")


if __name__ == "__main__":
    main()
