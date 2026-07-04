"""Confidence-miscalibration sweep: how good must a TrackNet-style confidence
signal be before M3_conf beats the confidence-FREE robust baseline (M_huber)?

Confidence model (generalizing ttsim.noise.add_noise's fixed conf):

    conf = (1/sigma_true)^gamma * exp(N(0, conf_noise))

  * gamma = 1, conf_noise = 0    -> perfectly calibrated (M3_conf ~ M3_oracle)
  * gamma = 1, conf_noise = 0.3  -> the default used everywhere else
  * gamma = 0                    -> confidence carries NO information
  * gamma != 1                   -> systematic mis-scaling (over/under-trust)

Paired design: each trial generates ONE arc + ONE noise realization; every
confidence-quality setting is evaluated on that same realization, and the
reference methods (M1, M_huber, M_gate, M3_oracle) are fit once per trial.

Outputs: results/miscalibration.txt + results/fig5_miscalibration.png
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttsim.experiment import sample_launch, make_observations
from ttsim.physics import simulate
from ttsim.noise import add_noise
from ttsim import estimators as E

RESULTS = os.path.join(os.path.dirname(__file__), "results")

CONF_NOISES = [0.0, 0.15, 0.3, 0.6, 1.0, 1.5]     # gamma fixed at 1
GAMMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]          # conf_noise fixed at 0.3
OP = dict(sigma0=0.008, alpha=1.0, p_miss=0.10, fps=120,
          bad_frac=0.20, bad_mult=6.0)


def _one_trial(seed):
    rng = np.random.default_rng(seed)
    p0, v0, om = sample_launch(rng)
    times, pos, vel, land, t_land = simulate(p0, v0, om, dt=1e-3)
    if land is None:
        return None
    fr_t, P, sp = make_observations(times, pos, vel, t_land, OP["fps"])
    noisy, sig, keep, _ = add_noise(P, sp, OP["sigma0"], OP["alpha"],
                                    OP["p_miss"], rng,
                                    bad_frac=OP["bad_frac"],
                                    bad_mult=OP["bad_mult"])
    ot, op_, sg = fr_t[keep], noisy[keep], sig[keep]
    if len(ot) < 8:
        return None
    true_xy = land[:2]

    def err(xy):
        return np.nan if xy is None else 100 * float(np.linalg.norm(xy - true_xy))

    out = {"M1": err(E.predict_M1(ot, op_)),
           "M_huber": err(E.predict_M_huber(ot, op_)),
           "M_gate": err(E.predict_M_gate(ot, op_)),
           "M3_oracle": err(E.predict_M3_oracle(ot, op_, sigma_true=sg))}

    # one shared lognormal draw, scaled per conf_noise -> smooth paired sweep
    z = rng.standard_normal(len(sg))
    for cn in CONF_NOISES:
        conf = (1.0 / np.maximum(sg, 1e-9)) * np.exp(z * cn)
        out[f"cn{cn}"] = err(E.predict_M3_conf(ot, op_, confidence=conf))
    for gm in GAMMAS:
        conf = (1.0 / np.maximum(sg, 1e-9)) ** gm * np.exp(z * 0.3)
        out[f"gm{gm}"] = err(E.predict_M3_conf(ot, op_, confidence=conf))
    return out


def agg(results, key):
    x = np.array([r[key] for r in results], float)
    x = x[np.isfinite(x)]
    return x.mean(), x.std() / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = [r for r in ex.map(_one_trial,
                                     [7_000_000 + i for i in range(a.trials)],
                                     chunksize=2) if r is not None]

    lines = [f"Confidence miscalibration sweep  (n={len(results)} paired trials, "
             f"operating point: sigma0=8mm bad_frac=0.2 bad_mult=6 p_miss=0.1 fps=120)",
             "=" * 74]
    refs = {m: agg(results, m)[0] for m in ("M1", "M_huber", "M_gate", "M3_oracle")}
    lines.append("  references: " + "  ".join(f"{m}={v:.2f}cm" for m, v in refs.items()))

    lines.append(f"\n  (a) conf_noise sweep (gamma=1):")
    lines.append(f"      {'conf_noise':>10s} {'M3_conf':>8s} {'vs Huber':>9s}")
    cn_mean, cn_sem = [], []
    for cn in CONF_NOISES:
        m, s = agg(results, f"cn{cn}")
        cn_mean.append(m); cn_sem.append(s)
        lines.append(f"      {cn:10.2f} {m:8.2f} {m - refs['M_huber']:+9.2f}")

    lines.append(f"\n  (b) gamma sweep (conf_noise=0.3):")
    lines.append(f"      {'gamma':>10s} {'M3_conf':>8s} {'vs Huber':>9s}")
    gm_mean, gm_sem = [], []
    for gm in GAMMAS:
        m, s = agg(results, f"gm{gm}")
        gm_mean.append(m); gm_sem.append(s)
        lines.append(f"      {gm:10.2f} {m:8.2f} {m - refs['M_huber']:+9.2f}")

    # where does confidence stop paying?
    beat = [cn for cn, m in zip(CONF_NOISES, cn_mean) if m < refs["M_huber"]]
    lines.append("\n  VERDICT: M3_conf beats the tuned confidence-free Huber baseline "
                 f"only for conf_noise <= {max(beat) if beat else 'NONE'} "
                 f"(gamma=1). A confidence signal weaker than that is not worth "
                 "wiring in — use a robust loss instead.")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "miscalibration.txt"), "w") as f:
        f.write(text)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, xs, ys, es, xlab, ttl in (
            (a1, CONF_NOISES, cn_mean, cn_sem, "confidence log-noise (gamma=1)",
             "(a) noisy confidence"),
            (a2, GAMMAS, gm_mean, gm_sem, "gamma  (conf ~ (1/sigma)^gamma, noise=0.3)",
             "(b) mis-scaled confidence")):
        ax.errorbar(xs, ys, yerr=es, fmt="s-", color="#2ca02c", capsize=3,
                    label="M3_conf")
        ax.axhline(refs["M1"], color="#1f77b4", ls="--", lw=1,
                   label=f"M1 ({refs['M1']:.2f})")
        ax.axhline(refs["M_huber"], color="#d62728", ls="--", lw=1,
                   label=f"M_huber ({refs['M_huber']:.2f})")
        ax.axhline(refs["M3_oracle"], color="#9467bd", ls=":", lw=1,
                   label=f"M3_oracle ({refs['M3_oracle']:.2f})")
        ax.set_xlabel(xlab); ax.grid(alpha=.3); ax.set_title(ttl)
    a2.axvline(1.0, color="gray", lw=0.8, ls=":")
    a1.set_ylabel("mean landing error (cm)")
    a1.legend(fontsize=8)
    fig.suptitle("How good must TrackNet confidence be to beat a robust loss?")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig5_miscalibration.png"), dpi=130)
    print(f"saved -> {RESULTS}/miscalibration.txt + fig5_miscalibration.png")


if __name__ == "__main__":
    main()
