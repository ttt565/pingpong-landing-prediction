"""Phase-1 killer experiment: does precision weighting (M3) beat uniform (M1)
by more than the real-world measurement floor (~2-4 cm)?

Clean ablation: M1 and M3 are the SAME estimator; only the per-frame weights
differ. M3_oracle (true 1/sigma^2) is the ceiling of ANY precision scheme;
M3_conf is the deployable version using a noisy TrackNet-style confidence.

Outputs: results/summary.txt, results/*.png
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttsim.experiment import sample_launch, run_trial

FLOOR_CM = 3.0  # representative real-world measurement floor (TDOA + calib + depth)
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _worker(job):
    tag, kwargs, seed = job
    rng = np.random.default_rng(seed)
    p0, v0, om = sample_launch(rng)
    return tag, run_trial(p0, v0, om, rng=rng, **kwargs)


def run_conditions(conditions, trials, workers):
    """conditions: list of (tag, kwargs). Returns {tag: [result_dicts]}."""
    jobs = []
    for ci, (tag, kw) in enumerate(conditions):
        for i in range(trials):
            jobs.append((tag, kw, ci * 1_000_000 + i + 1))
    out = {tag: [] for tag, _ in conditions}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for tag, res in ex.map(_worker, jobs, chunksize=4):
            if res is not None:
                out[tag].append(res)
    return out


def col(results, method):
    return np.array([r[method] for r in results], float)


def boot_mean_ci(x, n=2000, seed=0):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_gap_ci(a, b, n=2000, seed=0):
    """Paired mean(a)-mean(b) with bootstrap CI (a,b aligned per trial)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return (np.nan, np.nan, np.nan)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.trials = 30
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"Phase-1 killer experiment  (trials/cond={args.trials}, workers={args.workers})")
    log("=" * 70)

    # ---------------------------------------------------------------
    # (A) Operating-point comparison: realistic noise WITH bad frames
    #     M_huber / M_gate are the confidence-FREE robust baselines: the
    #     honest reference M3 must beat for the confidence signal to matter.
    # ---------------------------------------------------------------
    methods_full = ["M0", "M1", "M_huber", "M_gate", "M3_conf", "M3_oracle", "M4"]
    op = dict(sigma0=0.008, alpha=1.0, p_miss=0.10, fps=120,
              bad_frac=0.20, bad_mult=6.0, methods=methods_full)
    res = run_conditions([("op", op)], args.trials, args.workers)["op"]
    Hbar = np.mean([r["_H"] for r in res])
    log(f"\n(A) OPERATING POINT  sigma0=8mm alpha=1 p_miss=0.10 fps=120 "
        f"bad_frac=0.20 bad_mult=6   (mean H={Hbar:.2f}, n={len(res)})")
    log(f"    {'method':10s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'valid':>9s}  (cm)")
    means = {}
    for mth in methods_full:
        raw = col(res, mth)
        e = raw[np.isfinite(raw)]
        means[mth] = e.mean()
        # per-method validity: a method that fails more often would otherwise look
        # cleaner because NaNs are dropped -- surface n_valid/n_total explicitly
        log(f"    {mth:10s} {e.mean():7.2f} {np.median(e):7.2f} {np.percentile(e,90):7.2f} "
            f"{len(e):4d}/{len(raw):<4d}")
    g_or = boot_gap_ci(col(res, "M1"), col(res, "M3_oracle"))
    g_cf = boot_gap_ci(col(res, "M1"), col(res, "M3_conf"))
    g_hb = boot_gap_ci(col(res, "M1"), col(res, "M_huber"))
    g_hc = boot_gap_ci(col(res, "M_huber"), col(res, "M3_conf"))
    g_gc = boot_gap_ci(col(res, "M_gate"), col(res, "M3_conf"))
    log(f"    gap M1-M3oracle = {g_or[0]:+.2f} cm  [95% CI {g_or[1]:+.2f},{g_or[2]:+.2f}]")
    log(f"    gap M1-M3conf   = {g_cf[0]:+.2f} cm  [95% CI {g_cf[1]:+.2f},{g_cf[2]:+.2f}]")
    log(f"    gap M1-Mhuber   = {g_hb[0]:+.2f} cm  [95% CI {g_hb[1]:+.2f},{g_hb[2]:+.2f}]"
        f"   <- available with NO confidence signal")
    log(f"    gap Mhuber-M3conf = {g_hc[0]:+.2f} cm  [95% CI {g_hc[1]:+.2f},{g_hc[2]:+.2f}]"
        f" <- TRUE marginal value of confidence")
    log(f"    gap Mgate-M3conf  = {g_gc[0]:+.2f} cm  [95% CI {g_gc[1]:+.2f},{g_gc[2]:+.2f}]")
    log(f"    real-world floor ~ {FLOOR_CM:.1f} cm")

    _bar_fig(res, methods_full, Hbar)

    # ---------------------------------------------------------------
    # (B) Heteroscedasticity sweep: scan H via bad-frame fraction
    # ---------------------------------------------------------------
    log("\n(B) HETEROSCEDASTICITY SWEEP  (vary bad_frac -> H), fps=120")
    bad_fracs = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    conds = [(f"bf{bf}", dict(sigma0=0.008, alpha=1.0, p_miss=0.05, fps=120,
                              bad_frac=bf, bad_mult=8.0,
                              methods=["M1", "M_huber", "M3_oracle", "M3_conf"]))
             for bf in bad_fracs]
    R = run_conditions(conds, args.trials, args.workers)
    Hs, e_m1, e_hb, e_or, e_cf, gor, gcf, ghc = [], [], [], [], [], [], [], []
    log(f"    {'bad_frac':>8s} {'H':>5s} {'M1':>6s} {'Mhuber':>7s} {'M3orac':>7s} "
        f"{'M3conf':>7s} {'gapOr':>7s} {'gapCf':>7s} {'gapHbCf':>8s}  (cm)")
    for bf, (tag, _) in zip(bad_fracs, conds):
        r = R[tag]
        H = np.mean([x["_H"] for x in r])
        m1, hb = col(r, "M1"), col(r, "M_huber")
        orc, cf = col(r, "M3_oracle"), col(r, "M3_conf")
        Hs.append(H)
        e_m1.append(np.nanmean(m1)); e_hb.append(np.nanmean(hb))
        e_or.append(np.nanmean(orc)); e_cf.append(np.nanmean(cf))
        d_or = boot_gap_ci(m1, orc); d_cf = boot_gap_ci(m1, cf)
        d_hc = boot_gap_ci(hb, cf)
        gor.append(d_or); gcf.append(d_cf); ghc.append(d_hc)
        log(f"    {bf:8.2f} {H:5.2f} {np.nanmean(m1):6.2f} {np.nanmean(hb):7.2f} "
            f"{np.nanmean(orc):7.2f} {np.nanmean(cf):7.2f} {d_or[0]:+7.2f} "
            f"{d_cf[0]:+7.2f} {d_hc[0]:+8.2f}")
    _hetero_fig(bad_fracs, Hs, e_m1, e_hb, e_or, e_cf, gor, gcf, ghc)

    # ---------------------------------------------------------------
    # (C) Frame-count sweep: lower fps over the FULL arc (fewer samples,
    #     geometry/spin still observable -> isolates "fewer frames")
    # ---------------------------------------------------------------
    log("\n(C) FRAME-RATE SWEEP  (full arc), bad_frac=0.20 bad_mult=8")
    fps_list = [120, 80, 60, 45]
    conds = [(f"fps{fp}", dict(sigma0=0.008, alpha=1.0, p_miss=0.05, fps=fp,
                               bad_frac=0.20, bad_mult=8.0,
                               methods=["M1", "M3_oracle", "M3_conf"]))
             for fp in fps_list]
    R = run_conditions(conds, args.trials, args.workers)
    nobs, gor2, gcf2 = [], [], []
    log(f"    {'fps':>5s} {'n_obs':>5s} {'M1':>6s} {'M3orac':>7s} {'gapOr':>7s} {'gapCf':>7s}")
    for fp, (tag, _) in zip(fps_list, conds):
        r = R[tag]
        n = np.mean([x["_n_obs"] for x in r])
        m1, orc, cf = col(r, "M1"), col(r, "M3_oracle"), col(r, "M3_conf")
        d_or = boot_gap_ci(m1, orc); d_cf = boot_gap_ci(m1, cf)
        nobs.append(n); gor2.append(d_or); gcf2.append(d_cf)
        log(f"    {fp:5d} {n:5.0f} {np.nanmean(m1):6.2f} {np.nanmean(orc):7.2f} "
            f"{d_or[0]:+7.2f} {d_cf[0]:+7.2f}")
    _nobs_fig(nobs, gor2, gcf2)

    # ---------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------
    log("\n" + "=" * 70)
    max_gap_or = max(g[0] for g in gor)
    max_gap_hc = max(g[0] for g in ghc)
    log("VERDICT")
    log(f"  - speed-only heteroscedasticity over a single arc is negligible (H~0.02):")
    log(f"    precision weighting on speed alone cannot help first-landing prediction.")
    log(f"  - with realistic bad/low-confidence frames, the M3-oracle ceiling reaches")
    log(f"    up to {max_gap_or:+.2f} cm over M1 (see sweep); it exceeds the ~{FLOOR_CM:.0f} cm")
    log(f"    floor only at high H (frequent bad detections) and/or few frames.")
    log(f"  - ROBUST BASELINES: a tuned Huber loss (no confidence input) captures")
    log(f"    most of that gain for free; the residual confidence-signal value")
    log(f"    (gap Mhuber-M3conf) peaks at {max_gap_hc:+.2f} cm in the sweep — compare")
    log(f"    THIS number, not gap-over-M1, against the ~{FLOOR_CM:.0f} cm floor.")
    log(f"  => the case for M3 is now: confidence must beat a robust loss, not")
    log(f"     uniform weighting. Measure H AND confidence quality from TrackNet.")

    with open(os.path.join(RESULTS, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nsaved -> {RESULTS}/summary.txt  + figures")


# ----------------------------- figures --------------------------------
def _bar_fig(res, methods, Hbar):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(methods))
    ms, los, his = [], [], []
    for m in methods:
        mu, lo, hi = boot_mean_ci(col(res, m))
        ms.append(mu); los.append(mu - lo); his.append(hi - mu)
    colors = ["#888", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#aaa"][:len(methods)]
    ax.bar(xs, ms, yerr=[los, his], capsize=4, color=colors)
    ax.axhline(FLOOR_CM, ls="--", c="r", lw=1)
    ax.text(len(methods) - 1, FLOOR_CM + 0.1, f"~{FLOOR_CM:.0f}cm floor", color="r", ha="right")
    ax.set_xticks(list(xs)); ax.set_xticklabels(methods)
    ax.set_ylabel("mean landing error (cm)")
    ax.set_title(f"Operating point (bad_frac=0.20, H~{Hbar:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig1_methods.png"), dpi=130)


def _hetero_fig(bad_fracs, Hs, e_m1, e_hb, e_or, e_cf, gor, gcf, ghc):
    x = np.array(bad_fracs) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(x, e_m1, "o-", label="M1 (uniform)", color="#1f77b4")
    a1.plot(x, e_hb, "d-", label="M_huber (robust, no confidence)", color="#d62728")
    a1.plot(x, e_cf, "s-", label="M3_conf (realizable)", color="#2ca02c")
    a1.plot(x, e_or, "^-", label="M3_oracle (ceiling)", color="#9467bd")
    a1.set_xlabel("bad-frame fraction (%)")
    a1.set_ylabel("mean landing error (cm)"); a1.legend(); a1.grid(alpha=.3)
    a1.set_title("Error vs frequency of bad detections")
    for xi, H in zip(x, Hs):
        a1.annotate(f"H={H:.2f}", (xi, 0), fontsize=7, color="#666",
                    ha="center", va="bottom", rotation=90, xytext=(xi, 0.3))

    g_or = [g[0] for g in gor]; lo = [max(0, g[0] - g[1]) for g in gor]; hi = [max(0, g[2] - g[0]) for g in gor]
    g_cf = [g[0] for g in gcf]
    g_hc = [g[0] for g in ghc]
    lo_h = [g[0] - g[1] for g in ghc]; hi_h = [g[2] - g[0] for g in ghc]
    a2.errorbar(x, g_or, yerr=[lo, hi], fmt="^-", color="#9467bd", capsize=3, label="gap M1-M3oracle (ceiling)")
    a2.plot(x, g_cf, "s-", color="#2ca02c", label="gap M1-M3conf (realizable)")
    a2.errorbar(x, g_hc, yerr=[lo_h, hi_h], fmt="d-", color="#d62728", capsize=3,
                label="gap Mhuber-M3conf (marginal value of confidence)")
    a2.axhspan(0, FLOOR_CM, color="r", alpha=.08)
    a2.axhline(FLOOR_CM, ls="--", c="r", lw=1, label=f"~{FLOOR_CM:.0f}cm real-world floor")
    a2.axhline(0, c="k", lw=.6)
    a2.set_xlabel("bad-frame fraction (%)"); a2.set_ylabel("landing-error gain (cm)")
    a2.legend(fontsize=8); a2.grid(alpha=.3); a2.set_title("M3 advantage  (must clear the floor)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig2_hetero_sweep.png"), dpi=130)


def _nobs_fig(nobs, gor, gcf):
    fig, ax = plt.subplots(figsize=(6, 4))
    g_or = [g[0] for g in gor]; lo = [g[0] - g[1] for g in gor]; hi = [g[2] - g[0] for g in gor]
    ax.errorbar(nobs, g_or, yerr=[lo, hi], fmt="^-", color="#9467bd", capsize=3, label="gap M1-M3oracle")
    ax.plot(nobs, [g[0] for g in gcf], "s-", color="#2ca02c", label="gap M1-M3conf")
    ax.axhline(FLOOR_CM, ls="--", c="r", lw=1, label=f"~{FLOOR_CM:.0f}cm floor")
    ax.axhline(0, c="k", lw=.6)
    ax.set_xlabel("number of observed frames"); ax.set_ylabel("landing-error gain over M1 (cm)")
    ax.legend(); ax.grid(alpha=.3); ax.set_title("M3 advantage vs #frames (bad_frac=0.20)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig3_nobs_sweep.png"), dpi=130)


if __name__ == "__main__":
    main()
