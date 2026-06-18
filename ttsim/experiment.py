"""Trajectory generation, observation sampling, and one-trial evaluation."""
import numpy as np

from .physics import simulate
from .noise import add_noise, hetero_index
from . import estimators as E

# Nominal strong-topspin serve (ball machine at one table end, shooting +x).
# topspin = omega along +y -> Magnus pushes the ball down (dives), shortening flight.
NOMINAL = dict(
    p0=np.array([0.20, 0.00, 0.30]),
    v0=np.array([6.0, 0.0, 0.9]),
    omega=np.array([0.0, 400.0, 0.0]),   # ~64 rev/s topspin
)


def sample_launch(rng):
    return (
        NOMINAL["p0"] + rng.normal(0, [0.02, 0.02, 0.02]),
        NOMINAL["v0"] + rng.normal(0, [0.25, 0.15, 0.10]),
        NOMINAL["omega"] + rng.normal(0, [25, 35, 25]),
    )


def make_observations(times, pos, vel, t_land, fps):
    """Sample frames from launch up to (not including) landing."""
    fr_t = np.arange(0.0, t_land, 1.0 / fps)
    P = np.stack([np.interp(fr_t, times, pos[:, ax]) for ax in range(3)], axis=1)
    V = np.stack([np.interp(fr_t, times, vel[:, ax]) for ax in range(3)], axis=1)
    return fr_t, P, np.linalg.norm(V, axis=1)


def run_trial(p0, v0, omega, sigma0, alpha, p_miss, fps, rng, methods,
              bad_frac=0.0, bad_mult=6.0, obs_frac=1.0):
    """Returns dict method -> horizontal landing error in cm (np.nan on failure).

    obs_frac<1 truncates observation to the first fraction of the pre-landing arc
    (early-prediction / fewer-frames regime).
    """
    times, pos, vel, land, t_land = simulate(p0, v0, omega, dt=1e-3)
    if land is None:
        return None
    fr_t, P, sp = make_observations(times, pos, vel, t_land, fps)
    if obs_frac < 1.0:
        keep_n = max(8, int(len(fr_t) * obs_frac))
        fr_t, P, sp = fr_t[:keep_n], P[:keep_n], sp[:keep_n]
    if len(fr_t) < 8:
        return None
    noisy, sig_true, keep, conf = add_noise(P, sp, sigma0, alpha, p_miss, rng,
                                            bad_frac=bad_frac, bad_mult=bad_mult)
    obs_t, obs_p = fr_t[keep], noisy[keep]
    sig, clean_p, cf = sig_true[keep], P[keep], conf[keep]
    if len(obs_t) < 8:
        return None

    true_xy = land[:2]
    out = {}
    for name in methods:
        try:
            if name == "M0":
                xy = E.predict_M0(obs_t, obs_p)
            elif name == "M1":
                xy = E.predict_M1(obs_t, obs_p)
            elif name == "M3_oracle":
                xy = E.predict_M3_oracle(obs_t, obs_p, sigma_true=sig)
            elif name == "M3_conf":
                xy = E.predict_M3_conf(obs_t, obs_p, confidence=cf)
            elif name == "M3_rule":
                xy = E.predict_M3_rule(obs_t, obs_p, sigma0=sigma0, alpha=alpha)
            elif name == "M4":
                xy = E.predict_M4(obs_t, clean_p)
            else:
                raise ValueError(name)
            out[name] = np.nan if xy is None else 100.0 * np.linalg.norm(xy - true_xy)
        except Exception:
            out[name] = np.nan
    out["_true_xy"] = true_xy
    out["_n_obs"] = len(obs_t)
    out["_H"] = hetero_index(sig)
    return out
