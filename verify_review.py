"""Verify Codex review point #4 (optimizer blowup) and whether it confounds the
killer result. Two checks:

  (1) short-arc fitted-spin magnitude, unbounded vs bounded
  (2) operating-point killer gap with M1 UNBOUNDED vs M1 BOUNDED (|omega|<=1100):
      if M3's advantage over M1 survives bounding M1, the killer conclusion is not
      just "M3 dodges an optimizer artifact"; if it collapses, the reviewer is right.
"""
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from ttsim.physics import simulate, predict_landing
from ttsim.noise import add_noise
from ttsim.experiment import sample_launch, make_observations
from ttsim import estimators as E
from ttsim.estimators import fit_trajectory

OMB = 1100.0  # physical spin bound rad/s (~10500 rpm)


def _land_err(theta, true_xy):
    lp, _ = predict_landing(theta)
    return np.nan if lp is None else 100 * np.linalg.norm(lp[:2] - true_xy)


def short_arc_worker(seed):
    rng = np.random.default_rng(seed)
    p0, v0, om = sample_launch(rng)
    t, pos, vel, land, tl = simulate(p0, v0, om)
    if land is None:
        return None
    fr_t, P, sp = make_observations(t, pos, vel, tl, 120)
    if len(fr_t) < 8:
        return None
    # clean and 6mm-noise, first 8 frames
    out = {}
    for tag, sig in [("clean", 0.0), ("noisy6mm", 0.006)]:
        noisy, _, _, _ = add_noise(P, sp, sig, 1.0, 0.0, rng)
        ot, op = fr_t[:8], noisy[:8]
        thu = fit_trajectory(ot, op, fit_omega=True)
        thb = fit_trajectory(ot, op, fit_omega=True, omega_bound=OMB)
        out[tag] = (np.linalg.norm(thu[6:9]), np.linalg.norm(thb[6:9]),
                    _land_err(thu, land[:2]), _land_err(thb, land[:2]))
    out["_true_om"] = np.linalg.norm(om)
    return out


def killer_worker(seed):
    rng = np.random.default_rng(seed)
    p0, v0, om = sample_launch(rng)
    t, pos, vel, land, tl = simulate(p0, v0, om)
    if land is None:
        return None
    fr_t, P, sp = make_observations(t, pos, vel, tl, 120)
    noisy, sig, keep, conf = add_noise(P, sp, 0.008, 1.0, 0.10, rng,
                                       bad_frac=0.20, bad_mult=6.0)
    ot, op, sg = fr_t[keep], noisy[keep], sig[keep]
    if len(ot) < 8:
        return None
    txy = land[:2]
    w = 1.0 / np.maximum(sg, 1e-6) ** 2
    th_M1u = fit_trajectory(ot, op, fit_omega=True)
    th_M1b = fit_trajectory(ot, op, fit_omega=True, omega_bound=OMB)
    th_M3b = fit_trajectory(ot, op, weights=w, fit_omega=True, omega_bound=OMB)
    return dict(
        M1_unb=_land_err(th_M1u, txy),
        M1_bnd=_land_err(th_M1b, txy),
        M3_bnd=_land_err(th_M3b, txy),
        om_M1_unb=np.linalg.norm(th_M1u[6:9]),
        om_M1_bnd=np.linalg.norm(th_M1b[6:9]),
    )


def main():
    W = max(1, os.cpu_count() - 1)

    print("=== (1) 8-frame fitted |omega| (true ~400 rad/s) ===")
    rows = [r for r in ProcessPoolExecutor(W).map(short_arc_worker, range(1, 61)) if r]
    for tag in ["clean", "noisy6mm"]:
        omu = np.median([r[tag][0] for r in rows])
        omb = np.median([r[tag][1] for r in rows])
        eu = np.nanmedian([r[tag][2] for r in rows])
        eb = np.nanmedian([r[tag][3] for r in rows])
        print(f"  {tag:9s}: |om|_unbounded={omu:9.0f}  |om|_bounded={omb:6.0f}  "
              f"land_err unb={eu:6.1f}cm  bnd={eb:6.1f}cm")

    print("\n=== (2) operating-point killer: does M3's gap survive bounding M1? ===")
    rk = [r for r in ProcessPoolExecutor(W).map(killer_worker, range(1, 161)) if r]

    def m(k):
        return np.nanmean([r[k] for r in rk])

    def paired_ci(akey, bkey, n=5000, seed=0):
        a = np.array([r[akey] for r in rk]); b = np.array([r[bkey] for r in rk])
        msk = np.isfinite(a) & np.isfinite(b)
        d = a[msk] - b[msk]
        rng = np.random.default_rng(seed)
        bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"  trials={len(rk)}")
    print(f"  median ||om|| of M1_unbounded = {np.median([r['om_M1_unb'] for r in rk]):.0f} rad/s"
          f"   (true ~400)")
    print(f"  median ||om|| of M1_bounded   = {np.median([r['om_M1_bnd'] for r in rk]):.0f} rad/s"
          f"   (per-component bound {OMB:.0f} => ||om|| can reach {OMB*3**0.5:.0f})")
    print(f"  M1_unbounded  mean err = {m('M1_unb'):6.2f} cm")
    print(f"  M1_BOUNDED    mean err = {m('M1_bnd'):6.2f} cm   <- fair M1")
    print(f"  M3_oracle bnd mean err = {m('M3_bnd'):6.2f} cm")
    gu = paired_ci("M1_unb", "M3_bnd"); gb = paired_ci("M1_bnd", "M3_bnd")
    print(f"  gap (M1_unb - M3_bnd) = {gu[0]:+.2f} cm  [95% CI {gu[1]:+.2f}, {gu[2]:+.2f}]  (inflated)")
    print(f"  gap (M1_BND - M3_bnd) = {gb[0]:+.2f} cm  [95% CI {gb[1]:+.2f}, {gb[2]:+.2f}]  (fair)")


if __name__ == "__main__":
    main()
