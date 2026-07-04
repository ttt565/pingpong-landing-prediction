"""Jacobian / Fisher-information observability analysis (v2 roadmap item 1).

Questions answered, at the nominal serve with the good-frame noise model
(no bad frames — CRLB assumes known Gaussian sigma):

  (1) What landing accuracy does the INFORMATION in the first k frames allow
      at all (CRLB), and does the empirical bounded-M1 fit approach it?
      -> if yes, early-prediction error is an information floor, not an
         optimizer artifact.
  (2) WHICH spin direction is unobservable? Magnus acceleration K*(omega x v)
      is exactly blind to the omega-component parallel to v; gravity bends v
      by only ~25 deg over the arc, so omega_parallel stays nearly invisible.
      -> eigen-analysis of the spin marginal covariance.
  (3) How much does a spin PRIOR buy (sigma_prior in {inf, 400, 100} rad/s)?
      -> posterior CRLB; connects to the contact-board result, which showed a
         learned prior closing exactly this gap.

Outputs: results/observability.txt + results/fig6_observability.png
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttsim.experiment import NOMINAL, make_observations
from ttsim.physics import simulate, predict_positions_batch, predict_landing
from ttsim.noise import sigma_profile
from ttsim import estimators as E

RESULTS = os.path.join(os.path.dirname(__file__), "results")
STEPS = E._STEPS
SIGMA0, ALPHA, FPS = 0.008, 1.0, 120.0
K_LIST = [8, 12, 16, 20, 26, 34]
PRIORS = [None, 400.0, 100.0]          # rad/s Gaussian prior on each omega_i
N_EMP = 60                             # empirical trials per k


def position_jacobian(theta, obs_t):
    """J: (3k, 9) via the same batched finite-diff scheme as the estimator."""
    thetas = np.tile(theta, (10, 1))
    for j in range(9):
        thetas[j + 1, j] += STEPS[j]
    preds = predict_positions_batch(thetas, obs_t, 2e-3)   # (10, k, 3)
    return np.stack([(preds[j + 1] - preds[0]).ravel() / STEPS[j]
                     for j in range(9)], axis=1)


def landing_jacobian(theta):
    """G: (2, 9) d(landing_xy)/d(theta)."""
    base, _ = predict_landing(theta)
    G = np.empty((2, 9))
    for j in range(9):
        th = theta.copy()
        th[j] += STEPS[j]
        lp, _ = predict_landing(th)
        G[:, j] = (lp[:2] - base[:2]) / STEPS[j]
    return G


def _emp_trial(args):
    seed, k, theta = args
    rng = np.random.default_rng(seed)
    p0, v0, om = theta[:3], theta[3:6], theta[6:9]
    times, pos, vel, land, t_land = simulate(p0, v0, om, dt=1e-3)
    fr_t, P, sp = make_observations(times, pos, vel, t_land, FPS)
    fr_t, P, sp = fr_t[:k], P[:k], sp[:k]
    sig = sigma_profile(sp, SIGMA0, ALPHA)
    noisy = P + rng.standard_normal(P.shape) * sig[:, None]
    e1 = es = np.nan
    xy = E.predict_M1(fr_t, noisy)
    if xy is not None:
        e1 = 100 * np.linalg.norm(xy - land[:2])
    xy = E.predict_M1_spinknown(fr_t, noisy, true_omega=om)
    if xy is not None:
        es = 100 * np.linalg.norm(xy - land[:2])
    return k, e1, es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    theta = np.concatenate([NOMINAL["p0"], NOMINAL["v0"], NOMINAL["omega"]])
    times, pos, vel, land, t_land = simulate(theta[:3], theta[3:6], theta[6:9],
                                             dt=1e-3)
    fr_t, P, sp = make_observations(times, pos, vel, t_land, FPS)
    sig = sigma_profile(sp, SIGMA0, ALPHA)
    G = landing_jacobian(theta)
    v0hat = theta[3:6] / np.linalg.norm(theta[3:6])

    log("Fisher/CRLB observability at the nominal serve "
        f"(sigma0={SIGMA0*1000:.0f}mm alpha={ALPHA:g} fps={FPS:.0f}, "
        f"{len(fr_t)} frames to landing)")
    log("=" * 74)

    crlb = {p: [] for p in PRIORS}
    spin_std = []          # per k: sqrt eigvals of spin marginal cov (no prior)
    spin_worst_angle = []  # angle(worst eigvec, v0)
    for k in K_LIST:
        J = position_jacobian(theta, fr_t[:k])
        w = np.repeat(1.0 / sig[:k], 3)
        Jw = J * w[:, None]
        I = Jw.T @ Jw
        for p in PRIORS:
            Ip = I.copy()
            if p is not None:
                Ip[6:, 6:] += np.eye(3) / p ** 2
            C = np.linalg.pinv(Ip, rcond=1e-12)
            cov_xy = G @ C @ G.T
            crlb[p].append(100 * np.sqrt(np.trace(cov_xy)))
        # spin information GIVEN the other params: Schur complement. Using the
        # covariance from pinv would silently zero out the truly singular
        # direction (pinv truncation) and mislabel it as "perfectly observed".
        I_pp, I_pw, I_ww = I[:6, :6], I[:6, 6:], I[6:, 6:]
        S = I_ww - I_pw.T @ np.linalg.solve(I_pp, I_pw)
        ev, evec = np.linalg.eigh(S)          # ascending information
        with np.errstate(divide="ignore"):
            spin_std.append(1.0 / np.sqrt(np.maximum(ev, 0))[::-1])  # worst last
        u = evec[:, 0]                        # LEAST-informed spin direction
        spin_worst_angle.append(np.degrees(np.arccos(min(1, abs(u @ v0hat)))))

    log(f"\n(1) landing CRLB (cm) vs frames, priors on omega:")
    log(f"    {'k':>4s} " + " ".join(f"{('no prior' if p is None else f'sp={p:.0f}'):>10s}"
                                     for p in PRIORS))
    for i, k in enumerate(K_LIST):
        log(f"    {k:4d} " + " ".join(f"{crlb[p][i]:10.2f}" for p in PRIORS))

    log(f"\n(2) spin conditional std (rad/s, no prior; Schur complement — "
        f"inf = information-free) + angle(LEAST-informed dir, v0):")
    for k, ss, ang in zip(K_LIST, spin_std, spin_worst_angle):
        cells = ", ".join("inf" if not np.isfinite(s) else f"{s:8.1f}" for s in ss)
        log(f"    k={k:3d}  std=({cells})  least-informed dir angle to v0 = "
            f"{ang:5.1f} deg")
    log("    -> the information-free spin direction is omega parallel to v: "
        "Magnus K*(omega x v) is exactly blind to it; only gravity's bending "
        "of v over the arc makes it faintly visible by the last frames.")

    # ---- empirical overlay -------------------------------------------------
    jobs = [(9_000_000 + 1000 * k + i, k, theta)
            for k in K_LIST for i in range(N_EMP)]
    emp1 = {k: [] for k in K_LIST}
    emps = {k: [] for k in K_LIST}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, e1, es in ex.map(_emp_trial, jobs, chunksize=4):
            emp1[k].append(e1)
            emps[k].append(es)
    rms1 = [float(np.sqrt(np.nanmean(np.array(emp1[k]) ** 2))) for k in K_LIST]
    rmss = [float(np.sqrt(np.nanmean(np.array(emps[k]) ** 2))) for k in K_LIST]

    log(f"\n(3) empirical RMS landing error (n={N_EMP}/k, same noise, "
        "fixed nominal launch):")
    log(f"    {'k':>4s} {'CRLB':>8s} {'M1 (bounded)':>13s} {'ratio':>6s} "
        f"{'spin-known':>11s} {'CRLB sp=100':>12s}")
    for i, k in enumerate(K_LIST):
        log(f"    {k:4d} {crlb[None][i]:8.2f} {rms1[i]:13.2f} "
            f"{rms1[i]/crlb[None][i]:6.2f} {rmss[i]:11.2f} {crlb[100.0][i]:12.2f}")
    log("\n    reading: bounded-M1 tracks the no-prior CRLB within a small "
        "factor -> early error is an INFORMATION floor, not an optimizer "
        "artifact; the spin-known fit tracks the strong-prior CRLB -> a spin "
        "prior (e.g. learned from the contact board) is what moves the floor.")

    with open(os.path.join(RESULTS, "observability.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure ------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(K_LIST, crlb[None], "k-", lw=2, label="CRLB, no spin prior")
    a1.plot(K_LIST, crlb[400.0], "-", color="#9467bd",
            label="CRLB, prior 400 rad/s")
    a1.plot(K_LIST, crlb[100.0], "-", color="#2ca02c",
            label="CRLB, prior 100 rad/s")
    a1.plot(K_LIST, rms1, "o--", color="#1f77b4", label="empirical M1 (bounded)")
    a1.plot(K_LIST, rmss, "s--", color="#d62728", label="empirical spin-known")
    a1.set_yscale("log")
    a1.set_xlabel("frames observed k"); a1.set_ylabel("landing error RMS (cm)")
    a1.grid(alpha=.3, which="both"); a1.legend(fontsize=8)
    a1.set_title("Information floor vs achieved error")

    ss = np.array(spin_std)
    ss_plot = np.where(np.isfinite(ss), ss, np.nan)
    for j, lab in enumerate(["best", "mid", "worst (≈ ∥ v, inf early)"]):
        a2.plot(K_LIST, ss_plot[:, j], "o-", label=f"spin std, {lab} direction")
    a2.set_yscale("log")
    a2.set_xlabel("frames observed k"); a2.set_ylabel("spin marginal std (rad/s)")
    a2.grid(alpha=.3, which="both"); a2.legend(fontsize=8)
    a2.set_title("Anisotropic spin observability (no prior)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig6_observability.png"), dpi=130)
    print(f"\nsaved -> {RESULTS}/observability.txt + fig6_observability.png")


if __name__ == "__main__":
    main()
