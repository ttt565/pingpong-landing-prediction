#!/usr/bin/env python3
"""Contact-board learning, stage 3: HOW DOES IT DEGRADE TOWARD REALITY?

Two robustness questions on top of learn_board_residual.py, both answered
from the committed board_dataset.csv (no re-simulation):

1. LABEL NOISE — a real plate (piezo / mic-array triangulation) localizes the
   strike to ~1-3 cm, not exactly. Train labels get Gaussian noise of
   sigma in {0..5} cm; evaluation stays against the TRUE contact. The
   resulting floor-vs-sigma table is the sensor spec: it says how much plate
   accuracy the learning gain actually requires.

2. ONLINE (RLS) — replace batch ridge with recursive least squares updated
   one serve at a time, scored PREQUENTIALLY (predict serve k with weights
   from serves 1..k-1, then update). This is the "learn while playing"
   deployment mode; the curve shows how fast the corrector converges live.

Outputs: ../results_board_robustness.md,
         ../fig_board_label_noise.png, ../fig_board_online.png
"""
import argparse
import os

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GZ = os.path.dirname(HERE)

from learn_board_residual import (load_dataset, build_features,        # noqa: E402
                                  build_residuals, ridge_fit,
                                  ridge_predict, err_cm)

SIGMAS_CM = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
TRAIN_SIZES = [10, 20, 40, 80]
N_RESAMPLE = 30
N_TEST = 24          # same held-out protocol (and, with --seed 0, the same
LAM = 1.0            # test serves) as learn_board_residual.py
RLS_SIGMAS_CM = [0.0, 2.0]
N_SHUFFLE = 30
ROLL = 10            # rolling-mean window for the online curve


def rls_prequential(X, resid, sigma_m, rng, n_shuffle):
    """Prequential error (cm) per serve position, averaged over orderings.
    Weights start at the ridge prior (W=0, P=I/LAM); labels used for the
    UPDATE carry plate noise, the score is against the true residual."""
    n, p = X.shape
    curves = np.empty((n_shuffle, n))
    for s in range(n_shuffle):
        order = rng.permutation(n)
        P = np.eye(p + 1) / LAM
        W = np.zeros((p + 1, 2))
        for k, idx in enumerate(order):
            x = np.append(X[idx], 1.0)
            corr = W.T @ x
            curves[s, k] = 100 * np.hypot(*(resid[idx] - corr))
            r_noisy = resid[idx] + rng.normal(0.0, sigma_m, 2)
            Px = P @ x
            K = Px / (1.0 + x @ Px)
            W += np.outer(K, r_noisy - corr)
            P -= np.outer(K, Px)
    return curves.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=os.path.join(GZ, "board_out", "board_dataset.csv"))
    ap.add_argument("--out",
                    default=os.path.join(GZ, "results_board_robustness.md"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d, n = load_dataset(a.dataset)
    feats = build_features(d)
    resid = build_residuals(d)
    rng = np.random.default_rng(a.seed)

    perm = rng.permutation(n)
    test, pool = perm[:N_TEST], perm[N_TEST:]
    mu, sd = feats[pool].mean(0), feats[pool].std(0) + 1e-9
    X = (feats - mu) / sd
    e_phys = err_cm(resid[test, 0], resid[test, 1]).mean()
    e_oracle = float(np.nanmean(err_cm(d["true_y"] - d["pred_oracle_y"],
                                       d["true_z"] - d["pred_oracle_z"])[test]))

    # ---- 1. label-noise sweep (batch ridge) ------------------------------
    grid = {}
    for sig in SIGMAS_CM:
        for m in TRAIN_SIZES:
            errs = []
            for _ in range(N_RESAMPLE):
                sub = rng.choice(pool, size=m, replace=False)
                noisy = resid[sub] + rng.normal(0.0, sig / 100.0, (m, 2))
                W = ridge_fit(X[sub], noisy, LAM)
                corr = resid[test] - ridge_predict(W, X[test])
                errs.append(err_cm(corr[:, 0], corr[:, 1]).mean())
            grid[(sig, m)] = (float(np.mean(errs)), float(np.std(errs)))

    # ---- 2. online RLS, prequential --------------------------------------
    online = {sig: rls_prequential(X, resid, sig / 100.0, rng, N_SHUFFLE)
              for sig in RLS_SIGMAS_CM}

    # ---- report -----------------------------------------------------------
    lines = ["# Contact-board learning: label noise + online RLS\n",
             f"Same dataset/protocol as results_board.md ({n} serves, "
             f"{N_TEST} held-out; physics only = {e_phys:.1f} cm, true-spin "
             f"oracle = {e_oracle:.1f} cm). Train labels get Gaussian plate "
             "noise; evaluation is always against the true contact.\n",
             "## 1. Test error (cm) vs training-label noise\n",
             "| train serves | " + " | ".join(f"σ={s:g} cm" for s in SIGMAS_CM) + " |",
             "|---" * (len(SIGMAS_CM) + 1) + "|"]
    for m in TRAIN_SIZES:
        cells = [f"{grid[(s, m)][0]:.1f} ± {grid[(s, m)][1]:.1f}" for s in SIGMAS_CM]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["\n## 2. Online RLS, prequential error (cm) at serve #k\n",
              "| update-label noise | k=10 | k=20 | k=50 | k=100 | last-20 mean |",
              "|---|---|---|---|---|---|"]
    for sig, curve in online.items():
        pts = [f"{curve[k - 1]:.1f}" for k in (10, 20, 50, 100) if k <= len(curve)]
        lines.append(f"| σ={sig:g} cm | " + " | ".join(pts) +
                     f" | {curve[-20:].mean():.1f} |")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(a.out, "w") as f:
        f.write(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4))
        for sig in SIGMAS_CM:
            ys = [grid[(sig, m)][0] for m in TRAIN_SIZES]
            ax.plot(TRAIN_SIZES, ys, marker="o", label=f"σ = {sig:g} cm")
        ax.axhline(e_phys, color="k", ls="--", lw=1,
                   label=f"physics only ({e_phys:.1f})")
        ax.axhline(e_oracle, color="gray", ls=":", lw=1,
                   label=f"true-spin oracle ({e_oracle:.1f})")
        ax.set_xlabel("labeled serves")
        ax.set_ylabel("test error (cm)")
        ax.set_title("Learning gain vs plate label noise")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(GZ, "fig_board_label_noise.png"), dpi=140)

        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        kernel = np.ones(ROLL) / ROLL
        for sig, curve in online.items():
            roll = np.convolve(curve, kernel, mode="valid")
            ax2.plot(np.arange(ROLL, ROLL + len(roll)), roll,
                     label=f"RLS, label σ = {sig:g} cm")
        ax2.axhline(e_phys, color="k", ls="--", lw=1,
                    label=f"physics only ({e_phys:.1f})")
        ax2.axhline(e_oracle, color="gray", ls=":", lw=1,
                    label=f"true-spin oracle ({e_oracle:.1f})")
        ax2.set_xlabel("serve # (predict, then update)")
        ax2.set_ylabel(f"prequential error, rolling {ROLL} (cm)")
        ax2.set_title("Learning while playing: online RLS convergence")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(os.path.join(GZ, "fig_board_online.png"), dpi=140)
        print("wrote fig_board_label_noise.png + fig_board_online.png")
    except Exception as e:
        print(f"(no figures: {e})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
