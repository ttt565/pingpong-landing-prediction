"""Prediction methods, all in ONE estimator framework so M1 vs M3 differ ONLY
in the per-frame weights (clean ablation: precision weighting, nothing else).

Fit theta=(p0,v0,omega) to noisy observed positions by weighted nonlinear least
squares against the physics forward model, then integrate to the landing point.

    M0  no-spin physics  (omega fixed 0, uniform weights)      -> lower bound
    M1  residual/full fit (omega free,  uniform weights)        -> existing method
    M3  precision-weighted (omega free, w = estimated 1/sigma^2)-> contribution
    M3-oracle  same but w = TRUE 1/sigma^2                      -> ceiling of any
                                                                  precision scheme
    M4  M1 on noise-free observations                           -> upper bound
"""
import numpy as np
from scipy.optimize import least_squares

from .physics import predict_positions, predict_positions_batch, predict_landing
from .noise import sigma_profile

# per-parameter finite-diff steps and scales (omega lives on a ~100x larger scale)
_STEPS = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1])
_XSCALE = np.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0])


def _initial_guess(obs_t, obs_p, fit_omega):
    p0 = obs_p[0].copy()
    k = min(6, len(obs_t))
    v0 = np.array([np.polyfit(obs_t[:k], obs_p[:k, ax], 1)[0] for ax in range(3)])
    return np.concatenate([p0, v0, [0.0, 0.0, 0.0]]) if fit_omega else np.concatenate([p0, v0])


def fit_trajectory(obs_t, obs_p, weights=None, fit_omega=True, fixed_omega=None,
                   omega_bound=None, dt_fit=4e-3):
    """Weighted NLS fit with an analytic (batched finite-diff) Jacobian.

    weights are per-frame (length N); None -> uniform. M1 vs M3 differ ONLY here.
    fixed_omega: spin value to hold when fit_omega=False (default 0 = no spin).
    omega_bound: if set, constrain |omega_i| <= omega_bound rad/s (physical prior).
    """
    if weights is None:
        weights = np.ones(len(obs_t))
    if fixed_omega is None:
        fixed_omega = np.zeros(3)
    sw = np.sqrt(weights)[:, None]
    nfree = 9 if fit_omega else 6
    x0 = _initial_guess(obs_t, obs_p, fit_omega)[:nfree]
    steps, xscale = _STEPS[:nfree], _XSCALE[:nfree]

    def theta_of(x):
        return x if fit_omega else np.concatenate([x, fixed_omega])

    def resid(x):
        return (sw * (predict_positions(theta_of(x), obs_t, dt_fit) - obs_p)).ravel()

    def jac(x):
        base = theta_of(x)
        thetas = np.tile(base, (nfree + 1, 1))
        for j in range(nfree):
            thetas[j + 1, j] += steps[j]
        preds = predict_positions_batch(thetas, obs_t, dt_fit)  # (nfree+1, N, 3)
        r0 = (sw * (preds[0] - obs_p)).ravel()
        J = np.empty((r0.size, nfree))
        for j in range(nfree):
            J[:, j] = ((sw * (preds[j + 1] - obs_p)).ravel() - r0) / steps[j]
        return J

    if fit_omega and omega_bound is not None:
        lb = np.r_[np.full(6, -np.inf), np.full(3, -omega_bound)]
        ub = np.r_[np.full(6, np.inf), np.full(3, omega_bound)]
        bounds = (lb, ub)
    else:
        bounds = (-np.inf, np.inf)
    sol = least_squares(resid, x0, jac=jac, method="trf", x_scale=xscale,
                        bounds=bounds, max_nfev=80)
    return theta_of(sol.x)


def _landing_xy(theta):
    lp, _ = predict_landing(theta)
    return None if lp is None else lp[:2]


def predict_M0(obs_t, obs_p, **kw):
    return _landing_xy(fit_trajectory(obs_t, obs_p, fit_omega=False))


def predict_M1(obs_t, obs_p, **kw):
    return _landing_xy(fit_trajectory(obs_t, obs_p, fit_omega=True))


def predict_M1_spinknown(obs_t, obs_p, true_omega, **kw):
    """Fit only (p0,v0) with the TRUE spin given. Isolates the spin-estimation
    penalty: gap to M1 = cost of having to infer omega from the arc."""
    return _landing_xy(fit_trajectory(obs_t, obs_p, fit_omega=False, fixed_omega=true_omega))


def predict_M3_oracle(obs_t, obs_p, sigma_true, **kw):
    w = 1.0 / np.maximum(sigma_true, 1e-6) ** 2
    return _landing_xy(fit_trajectory(obs_t, obs_p, weights=w, fit_omega=True))


def predict_M3_conf(obs_t, obs_p, confidence, **kw):
    """Realizable precision weighting from a TrackNet-style confidence proxy
    (Pi ~ confidence^2). This is the deployable M3."""
    w = np.asarray(confidence) ** 2
    w = w / np.mean(w)
    return _landing_xy(fit_trajectory(obs_t, obs_p, weights=w, fit_omega=True))


def predict_M3_rule(obs_t, obs_p, sigma0, alpha, **kw):
    """Realizable precision: 2-stage. First an unweighted (M1) fit to get a smooth
    speed estimate per frame, then weight by the (known-form) sigma(speed)."""
    theta1 = fit_trajectory(obs_t, obs_p, fit_omega=True)
    # speeds from the fitted trajectory via finite difference of predicted positions
    dt = 1e-3
    pp = predict_positions(theta1, np.clip(obs_t, dt, None), 2e-3)
    pm = predict_positions(theta1, np.clip(obs_t - dt, 0, None), 2e-3)
    speeds = np.linalg.norm((pp - pm) / dt, axis=1)
    sig_hat = sigma_profile(speeds, sigma0, alpha)
    w = 1.0 / np.maximum(sig_hat, 1e-6) ** 2
    return _landing_xy(fit_trajectory(obs_t, obs_p, weights=w, fit_omega=True))


def predict_M4(obs_t, clean_p, **kw):
    return _landing_xy(fit_trajectory(obs_t, clean_p, fit_omega=True))
