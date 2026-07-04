#!/usr/bin/env python3
"""Contact-board learning, stage 4: OUT-OF-DISTRIBUTION protection and the
generalizing alternative — one experiment, leave-one-cluster-out.

Protocol: hold out one spin cluster entirely (the robot has never seen this
serve type). Train on the other two clusters. Evaluate on:
    ID  — held-out serves from the two training clusters,
    OOD — every serve of the excluded cluster.

Models compared (board-contact error, cm):
    physics        calibrated pipeline, no learning        (the fallback)
    ridge          residual regressor from board labels    (expected to FAIL OOD)
    ridge+gate     ridge, but fall back to physics when the feature vector is
                   far from the training distribution (Mahalanobis distance
                   over standardized features > chi2_{0.999, 9})
    bounce-fit     differentiable re-calibration of the PHYSICAL bounce params
                   (e, mu, alpha) through the flight->bounce->flight map on the
                   training labels (expected to TRANSFER: parameters are
                   physics, not repertoire)

Outputs: ../results_board_ood.md
    python3 board_ood.py            # uses ../board_out/board_dataset.csv
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

HERE = os.path.abspath(os.path.dirname(__file__))
GZ = os.path.dirname(HERE)

from learn_board_residual import (load_dataset, build_features,        # noqa: E402
                                  build_residuals, ridge_fit,
                                  ridge_predict, err_cm)
from board_learn import predict_board_contact                          # noqa: E402
from ttsim.bounce import E_TABLE, MU_TABLE, ALPHA_I                    # noqa: E402

CLUSTERS = ["topspin", "flat", "backspin"]
LAM = 1.0
GATE_Q = 0.999
ID_HOLDOUT = 0.25


def _contact_for(args):
    theta, params = args
    pred = predict_board_contact(theta, bounce_params=params)
    return (np.nan, np.nan) if pred is None else (pred[0], pred[1])


def contacts(thetas, params, pool):
    out = list(pool.map(_contact_for, [(t, params) for t in thetas]))
    return np.array(out)


def fit_bounce_params(thetas, truths, pool):
    """Least-squares over (e, mu, alpha) through the full pipeline."""
    def resid(x):
        pr = contacts(thetas, x, pool)
        r = (pr - truths).ravel()
        return np.where(np.isfinite(r), r, 0.5)   # penalize lost balls
    sol = least_squares(resid, x0=[E_TABLE, MU_TABLE, ALPHA_I],
                        bounds=([0.4, 0.05, 0.2], [0.99, 0.8, 0.9]),
                        diff_step=0.03, max_nfev=15)
    return sol.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=os.path.join(GZ, "board_out", "board_dataset.csv"))
    ap.add_argument("--out", default=os.path.join(GZ, "results_board_ood.md"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d, n = load_dataset(a.dataset)
    feats = build_features(d)
    resid = build_residuals(d)
    thetas = np.stack([d["phx"], d["phy"], d["phz"],
                       d["vhx"], d["vhy"], d["vhz"],
                       d["whx"], d["why"], d["whz"]], axis=1)
    truths = np.stack([d["true_y"], d["true_z"]], axis=1)
    e_phys_all = err_cm(resid[:, 0], resid[:, 1])
    rng = np.random.default_rng(a.seed)
    thresh = chi2.ppf(GATE_Q, df=feats.shape[1])

    lines = ["# Board learning under distribution shift (leave-one-cluster-out)\n",
             f"Hold out one spin cluster entirely; train on the other two "
             f"({n} serves total). Gate: Mahalanobis distance over the "
             f"standardized features vs chi2(q={GATE_Q}, df={feats.shape[1]}). "
             "Bounce-fit: least-squares over (e, mu, alpha) through the "
             "flight->bounce->flight map on the training labels.\n",
             "| held-out cluster | split | physics | ridge | ridge+gate "
             "| bounce-fit | gate fallback rate |",
             "|---|---|---|---|---|---|---|"]

    pool = ProcessPoolExecutor()
    fitted_params = {}
    for hold in CLUSTERS:
        ood = np.where(d["cluster"] == hold)[0]
        id_all = np.where(d["cluster"] != hold)[0]
        id_all = rng.permutation(id_all)
        n_hold = int(len(id_all) * ID_HOLDOUT)
        id_test, train = id_all[:n_hold], id_all[n_hold:]

        mu_f = feats[train].mean(0)
        sd_f = feats[train].std(0) + 1e-9
        X = (feats - mu_f) / sd_f
        W = ridge_fit(X[train], resid[train], LAM)

        # Mahalanobis over standardized training features
        cov = np.cov(X[train].T) + 1e-6 * np.eye(X.shape[1])
        cov_inv = np.linalg.inv(cov)
        mu_x = X[train].mean(0)

        params = fit_bounce_params(thetas[train], truths[train], pool)
        fitted_params[hold] = params
        for split, idx in (("ID", id_test), ("OOD", ood)):
            corr = resid[idx] - ridge_predict(W, X[idx])
            e_ridge = err_cm(corr[:, 0], corr[:, 1])
            dx = X[idx] - mu_x
            D2 = np.einsum("ij,jk,ik->i", dx, cov_inv, dx)
            use_phys = D2 > thresh
            e_gate = np.where(use_phys, e_phys_all[idx], e_ridge)
            pr = contacts(thetas[idx], params, pool)
            e_fit = err_cm(pr[:, 0] - truths[idx, 0], pr[:, 1] - truths[idx, 1])
            lines.append(
                f"| {hold} | {split} ({len(idx)}) | {e_phys_all[idx].mean():.1f} "
                f"| {np.nanmean(e_ridge):.1f} | {np.nanmean(e_gate):.1f} "
                f"| {np.nanmean(e_fit):.1f} | {use_phys.mean() * 100:.0f}% |")

    pool.shutdown()
    lines.append("\nfitted bounce params per fold (default e=0.7765 mu=0.25 "
                 "alpha=0.4):")
    for hold, p in fitted_params.items():
        lines.append(f"- held-out {hold}: e={p[0]:.3f}  mu={p[1]:.3f}  "
                     f"alpha={p[2]:.3f}")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(a.out, "w") as f:
        f.write(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
