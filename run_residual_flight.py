"""Flight-segment residual learning — the original "physics + learned
residual" M1 (LIMITATIONS item 1's second half), evaluated where it matters.

Regime: STRONG rich truth (Cd(v) + Magnus saturation + spin decay), noisy
operating point (20% bad frames), HALF-arc observations — the cell where
spin error, model mismatch and noise all bite (fig7: M_huber ~ 44 cm there).

The production ladder, each step a strict add-on:

    M_huber                robust loss                       (batch-1 verdict)
    + spin prior           MAP with the serve-machine prior  (run_prior.py)
    + ridge residual       landing-residual regressor trained on self-labels
                           from the landing-board sensor (label noise 1 cm),
                           features = fitted state + own prediction

Labels are exactly what the project's measurement protocol already assumes
(TDOA landing board); the contact-board experiments showed the same learning
recipe survives label noise and runs online (RLS).

Outputs: results/residual_flight.txt + results/fig9_residual_flight.png
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttsim.experiment import NOMINAL, sample_launch, make_observations
from ttsim.physics_rich import simulate_rich, STRONG
from ttsim.noise import add_noise
from ttsim import estimators as E

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OP = dict(sigma0=0.008, alpha=1.0, p_miss=0.10, fps=120,
          bad_frac=0.20, bad_mult=6.0)
OBS_FRAC = 0.5
IDEAL_MU = NOMINAL["omega"]
IDEAL_SD = np.array([25.0, 35.0, 25.0])
LABEL_SD = 0.01          # landing-board label noise (m)
N_TEST = 40
TRAIN_SIZES = [10, 20, 40, 80]
N_RESAMPLE = 30
LAM = 1.0


def _trial(seed):
    rng = np.random.default_rng(seed)
    p0, v0, om = sample_launch(rng)
    times, pos, vel, land, t_land = simulate_rich(p0, v0, om, STRONG, dt=1e-3)
    if land is None:
        return None
    fr_t, P, sp = make_observations(times, pos, vel, t_land, OP["fps"])
    keep_n = max(8, int(len(fr_t) * OBS_FRAC))
    fr_t, P, sp = fr_t[:keep_n], P[:keep_n], sp[:keep_n]
    noisy, sig, keep, conf = add_noise(P, sp, OP["sigma0"], OP["alpha"],
                                       OP["p_miss"], rng,
                                       bad_frac=OP["bad_frac"],
                                       bad_mult=OP["bad_mult"])
    ot, op_ = fr_t[keep], noisy[keep]
    if len(ot) < 8:
        return None
    true_xy = land[:2]

    th_h = E.fit_trajectory(ot, op_, fit_omega=True, omega_bound=E.OMEGA_BOUND,
                            loss="huber", f_scale=0.015)
    th_p = E.fit_trajectory(ot, op_, fit_omega=True, omega_bound=E.OMEGA_BOUND,
                            loss="huber", f_scale=0.015,
                            omega_prior=IDEAL_MU, omega_prior_std=IDEAL_SD)
    xy_h = E._landing_xy(th_h)
    xy_p = E._landing_xy(th_p)
    if xy_h is None or xy_p is None:
        return None
    label = true_xy + rng.normal(0.0, LABEL_SD, 2)
    return dict(true=true_xy, label=label, xy_h=np.asarray(xy_h),
                xy_p=np.asarray(xy_p), th=th_p, n=len(ot))


def ridge_fit(X, Y, lam):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ Y)


def ridge_predict(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serves", type=int, default=150)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        trials = [t for t in ex.map(_trial,
                                    [15_000_000 + i for i in range(a.serves)],
                                    chunksize=2) if t is not None]
    n = len(trials)
    true = np.stack([t["true"] for t in trials])
    label = np.stack([t["label"] for t in trials])
    xy_h = np.stack([t["xy_h"] for t in trials])
    xy_p = np.stack([t["xy_p"] for t in trials])
    feats = np.stack([np.concatenate([t["th"][3:9], t["xy_p"], [t["n"]]])
                      for t in trials])
    # residual target uses the NOISY label (self-supervision); evaluation
    # is against the true landing
    resid_lab = label - xy_p
    e_h = 100 * np.linalg.norm(xy_h - true, axis=1)
    e_p = 100 * np.linalg.norm(xy_p - true, axis=1)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    test, pool = perm[:N_TEST], perm[N_TEST:]
    mu_f, sd_f = feats[pool].mean(0), feats[pool].std(0) + 1e-9
    X = (feats - mu_f) / sd_f

    W = ridge_fit(X[pool], resid_lab[pool], LAM)
    corr = xy_p + ridge_predict(W, X)
    e_r = 100 * np.linalg.norm(corr - true, axis=1)

    log(f"Flight-segment residual learning "
        f"(STRONG rich truth, half-arc, operating-point noise; n={n} serves, "
        f"{N_TEST} held out, label noise {100*LABEL_SD:.0f} cm)")
    log("=" * 74)
    log(f"\n  production ladder (mean cm on held-out serves):")
    log(f"    M_huber                : {e_h[test].mean():7.2f}")
    log(f"    + spin prior (MAP)     : {e_p[test].mean():7.2f}")
    log(f"    + ridge residual (n={len(pool)}): {e_r[test].mean():7.2f}")

    curve = []
    for m in TRAIN_SIZES:
        errs = []
        for _ in range(N_RESAMPLE):
            sub = rng.choice(pool, size=m, replace=False)
            Wm = ridge_fit(X[sub], resid_lab[sub], LAM)
            cm_ = xy_p[test] + ridge_predict(Wm, X[test])
            errs.append(float(np.mean(100 * np.linalg.norm(cm_ - true[test],
                                                           axis=1))))
        curve.append((m, float(np.mean(errs)), float(np.std(errs))))
    log(f"\n  residual sample efficiency (labels -> test error cm):")
    for m, mean, std in curve:
        log(f"    {m:4d}  {mean:6.2f} ± {std:4.2f}")

    log("\nREADING")
    log(f"  - the ladder is compositional: robust loss handles bad frames, "
        f"the MAP prior restores spin identifiability "
        f"({e_h[test].mean():.1f} -> {e_p[test].mean():.1f} cm), and the "
        f"residual regressor absorbs what remains systematic — model "
        f"mismatch + fit bias — from self-labels alone "
        f"({e_p[test].mean():.1f} -> {e_r[test].mean():.1f} cm).")
    log(f"  - this is the original 'physics + learned residual' M1, now with "
        f"an honest truth (inverse crime broken) and honest labels "
        f"(1 cm board noise). LIMITATIONS item 1 closes here.")

    with open(os.path.join(RESULTS, "residual_flight.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    bars = [("M_huber", e_h[test].mean(), "#d62728"),
            ("+ spin prior", e_p[test].mean(), "#ff7f0e"),
            ("+ ridge residual", e_r[test].mean(), "#2ca02c")]
    a1.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars])
    a1.set_ylabel("mean landing error (cm)")
    a1.set_title("Production ladder (half arc, rich truth)")
    ms = [c[0] for c in curve]
    a2.errorbar(ms, [c[1] for c in curve], yerr=[c[2] for c in curve],
                marker="o", color="#2ca02c", capsize=3, label="+ ridge residual")
    a2.axhline(e_p[test].mean(), color="#ff7f0e", ls="--", lw=1,
               label=f"physics+prior ({e_p[test].mean():.1f})")
    a2.set_xlabel("landing labels used")
    a2.set_ylabel("test error (cm)")
    a2.set_title("Residual sample efficiency")
    a2.legend(fontsize=8)
    a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig9_residual_flight.png"), dpi=130)
    print(f"\nsaved -> {RESULTS}/residual_flight.txt + fig9_residual_flight.png")


if __name__ == "__main__":
    main()
