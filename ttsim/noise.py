"""Heteroscedastic perception-noise model (Route A: structured noise on truth).

The whole M3 (precision-weighting) thesis lives or dies on the noise being
*heteroscedastic* and identifiable. The dominant heteroscedastic source here is
speed: faster ball -> more motion blur -> larger detection error. alpha controls
how strongly sigma grows with speed; alpha=0 is homoscedastic (the null where
precision weighting must give nothing).
"""
import numpy as np

V_REF = 10.0  # m/s, speed scale for the noise model


def sigma_profile(speeds, sigma0, alpha, v_ref=V_REF):
    """Per-frame std of position noise: sigma0 * (1 + alpha * |v| / v_ref)."""
    return sigma0 * (1.0 + alpha * np.asarray(speeds) / v_ref)


def add_noise(positions, speeds, sigma0, alpha, p_miss, rng,
              bad_frac=0.0, bad_mult=6.0, conf_noise=0.3, v_ref=V_REF):
    """Inject heteroscedastic noise. Two sources of per-frame sigma variation:

      * speed (alpha): smooth, but barely varies over a short arc -> small spread
      * bad frames (bad_frac/bad_mult): a random subset of detections with sigma
        inflated x bad_mult (motion blur spike / occlusion / background confusion).
        THIS is what creates real within-arc heteroscedasticity.

    Returns (noisy_positions, true_sigma_per_frame, keep_mask, confidence).
    `confidence` is a realizable precision proxy (~1/sigma with multiplicative
    log-noise) standing in for a TrackNet heatmap-peak score.
    """
    n = len(positions)
    sig = sigma_profile(speeds, sigma0, alpha, v_ref)
    is_bad = rng.random(n) < bad_frac
    sig = sig * np.where(is_bad, bad_mult, 1.0)
    noisy = positions + rng.standard_normal(positions.shape) * sig[:, None]
    keep = rng.random(n) >= p_miss
    conf = (1.0 / np.maximum(sig, 1e-9)) * np.exp(rng.normal(0.0, conf_noise, n))  # noisy 1/sigma
    return noisy, sig, keep, conf


def hetero_index(sigma):
    """Heteroscedasticity index H = std/mean of per-frame sigma (0 = homoscedastic)."""
    return float(np.std(sigma) / np.mean(sigma))
