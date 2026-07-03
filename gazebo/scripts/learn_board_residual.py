#!/usr/bin/env python3
"""Contact-board self-supervision, stage 2: LEARNING.

Consumes board_out/board_dataset.csv (one row per serve: fitted state,
physics-pipeline prediction of the board contact, true contact, true-spin
oracle prediction) and asks the money question:

    starting from the calibrated physics pipeline, how many self-labeled
    serves does a small ridge regressor on the residual need to approach the
    true-spin ceiling?

Model: multi-output ridge on standardized features
    phi = [v_hat, w_hat, pred_y, pred_z, pred_t_rel]   (9 dims + bias)
    target = (true_y - pred_y, true_z - pred_z)
No physics is re-run here — the learner only corrects the physics output,
so the physics model stays as the fallback for out-of-distribution serves.

Outputs: ../results_board.md + ../fig_board_learning.png

    python3 learn_board_residual.py            # uses ../board_out/board_dataset.csv
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
GZ = os.path.dirname(HERE)

TRAIN_SIZES = [10, 20, 40, 80]
N_RESAMPLE = 30     # random train subsets per size (closed-form ridge = cheap)
N_TEST = 24
LAM = 1.0


def load_dataset(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    d = {}
    for k in rows[0]:
        if k == "cluster":
            d[k] = np.array([r[k] for r in rows])
        else:
            d[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
    return d, len(rows)


def ridge_fit(X, Y, lam):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam            # do not penalize the bias
    return np.linalg.solve(A, Xb.T @ Y)


def ridge_predict(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def err_cm(dy, dz):
    return 100 * np.hypot(dy, dz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=os.path.join(GZ, "board_out", "board_dataset.csv"))
    ap.add_argument("--out", default=os.path.join(GZ, "results_board.md"))
    ap.add_argument("--fig", default=os.path.join(GZ, "fig_board_learning.png"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d, n = load_dataset(a.dataset)
    rng = np.random.default_rng(a.seed)

    # all features are available at prediction time (fitted state + the
    # physics pipeline's own output); pred_t centered on its dataset mean
    feats = np.stack([d["vhx"], d["vhy"], d["vhz"],
                      d["whx"], d["why"], d["whz"],
                      d["pred_y"], d["pred_z"],
                      d["pred_t"] - d["pred_t"].mean()], axis=1)
    resid = np.stack([d["true_y"] - d["pred_y"],
                      d["true_z"] - d["pred_z"]], axis=1)

    e_phys = err_cm(resid[:, 0], resid[:, 1])
    e_oracle = err_cm(d["true_y"] - d["pred_oracle_y"],
                      d["true_z"] - d["pred_oracle_z"])   # NaN if no oracle row

    # fixed held-out test set
    perm = rng.permutation(n)
    test, pool = perm[:N_TEST], perm[N_TEST:]
    mu, sd = feats[pool].mean(0), feats[pool].std(0) + 1e-9
    Xall = (feats - mu) / sd

    curve = []
    for m in TRAIN_SIZES:
        errs = []
        for _ in range(N_RESAMPLE):
            sub = rng.choice(pool, size=m, replace=False)
            W = ridge_fit(Xall[sub], resid[sub], LAM)
            corr = resid[test] - ridge_predict(W, Xall[test])
            errs.append(err_cm(corr[:, 0], corr[:, 1]).mean())
        curve.append((m, float(np.mean(errs)), float(np.std(errs))))

    # full-pool model for the per-cluster breakdown
    W = ridge_fit(Xall[pool], resid[pool], LAM)
    corr_test = resid[test] - ridge_predict(W, Xall[test])
    e_learn_test = err_cm(corr_test[:, 0], corr_test[:, 1])

    lines = ["# Contact-board self-supervision: residual learning\n",
             f"{n} valid serves from a 3-cluster serve repertoire "
             f"(topspin/flat/backspin), one perception-noise realization per "
             f"serve, physics pipeline = M3_conf fit -> calibrated bounce -> "
             f"flight to the board plane (x = {2.78} m). Test set: "
             f"{N_TEST} held-out serves; ridge on 9 features, lambda={LAM:g}.\n",
             "## Board-contact error (cm, mean over test serves)\n",
             "| model | error |", "|---|---|",
             f"| physics only | {e_phys[test].mean():.1f} |",
             f"| physics + ridge residual (full pool, n={len(pool)}) "
             f"| {e_learn_test.mean():.1f} |",
             f"| true-spin oracle (ceiling) | "
             f"{np.nanmean(e_oracle[test]):.1f} |",
             "\n## Sample efficiency (test error vs labeled serves)\n",
             "| train serves | error (cm) mean ± std over subsets |", "|---|---|"]
    for m, mean, std in curve:
        lines.append(f"| {m} | {mean:.1f} ± {std:.1f} |")

    lines.append("\n## Per-cluster (full-pool model, test set)\n")
    lines.append("| cluster | physics | + learned | oracle |")
    lines.append("|---|---|---|---|")
    for c in ("topspin", "flat", "backspin"):
        mte = [i for i in test if d["cluster"][i] == c]
        if not mte:
            continue
        i_loc = [list(test).index(i) for i in mte]
        lines.append(f"| {c} | {e_phys[mte].mean():.1f} | "
                     f"{e_learn_test[i_loc].mean():.1f} | "
                     f"{np.nanmean(e_oracle[mte]):.1f} |")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(a.out, "w") as f:
        f.write(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ms = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        es = [c[2] for c in curve]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(e_phys[test].mean(), color="tab:red", ls="--",
                   label=f"physics only ({e_phys[test].mean():.1f} cm)")
        ax.axhline(np.nanmean(e_oracle[test]), color="tab:green", ls="--",
                   label=f"true-spin oracle ({np.nanmean(e_oracle[test]):.1f} cm)")
        ax.errorbar(ms, ys, yerr=es, marker="o", color="tab:blue",
                    label="physics + ridge residual")
        ax.set_xlabel("labeled serves (board contacts)")
        ax.set_ylabel("board-contact error on held-out serves (cm)")
        ax.set_title("Self-supervised residual learning from a contact board")
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.fig, dpi=140)
        print(f"wrote {a.fig}")
    except Exception as e:                       # matplotlib optional
        print(f"(no figure: {e})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
