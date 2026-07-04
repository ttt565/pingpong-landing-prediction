"""RICHER truth physics — used ONLY as ground truth to break the inverse
crime (the predictors keep the simplified constant-coefficient model in
physics.py). Three mechanisms the simple model lacks:

  * speed-dependent drag       Cd(v)   = cd0 + cd_amp * exp(-|v| / cd_vref)
  * Magnus saturation          K_mag(S)= K_MAG / (1 + sat * S),  S = |w| r / |v|
  * spin decay in flight       w(t)    = w0 * exp(-t / tau)

With NONE the model reduces exactly to physics.py (sanity: M4 = 0 again).
Parameter levels are chosen so the induced landing shift is comparable to the
effects under study (mild ~ small vs the 3 cm floor, strong ~ above it) —
the point is a controlled mismatch matrix, not aerodynamic truth.
"""
import numpy as np

from .physics import (G, M_BALL, R_BALL, RHO_AIR, A_CROSS, K_MAG, TABLE_Z)

NONE = dict(cd0=0.40, cd_amp=0.00, cd_vref=8.0, sat=0.00, tau=np.inf)
MILD = dict(cd0=0.38, cd_amp=0.06, cd_vref=8.0, sat=0.08, tau=8.0)
STRONG = dict(cd0=0.35, cd_amp=0.14, cd_vref=6.0, sat=0.20, tau=3.0)
LEVELS = {"none": NONE, "mild": MILD, "strong": STRONG}


def accel_rich(vel, omega, prm):
    speed = np.linalg.norm(vel)
    cd = prm["cd0"] + prm["cd_amp"] * np.exp(-speed / prm["cd_vref"])
    k_drag = 0.5 * RHO_AIR * cd * A_CROSS / M_BALL
    S = np.linalg.norm(omega) * R_BALL / max(speed, 1e-9)
    k_mag = K_MAG / (1.0 + prm["sat"] * S)
    return (np.array([0.0, 0.0, -G])
            - k_drag * speed * vel
            + k_mag * np.cross(omega, vel))


def simulate_rich(p0, v0, omega0, prm, dt=1e-3, t_max=2.0, table_z=TABLE_Z):
    """RK4 with time-decaying spin; same interface/return as physics.simulate."""
    omega0 = np.asarray(omega0, float)

    def om(t):
        return omega0 * np.exp(-t / prm["tau"]) if np.isfinite(prm["tau"]) \
            else omega0

    def deriv(state, t):
        return np.concatenate([state[3:], accel_rich(state[3:], om(t), prm)])

    state = np.concatenate([np.asarray(p0, float), np.asarray(v0, float)])
    times = [0.0]
    states = [state.copy()]
    t = 0.0
    landing_pos = landing_t = None
    for _ in range(int(t_max / dt)):
        prev = state
        k1 = deriv(state, t)
        k2 = deriv(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = deriv(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = deriv(state + dt * k3, t + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        times.append(t)
        states.append(state.copy())
        if prev[2] > table_z and state[2] <= table_z and state[5] < 0:
            frac = (prev[2] - table_z) / (prev[2] - state[2])
            landing_pos = prev[:3] + frac * (state[:3] - prev[:3])
            landing_t = (t - dt) + frac * dt
            break
    S = np.array(states)
    return np.array(times), S[:, :3], S[:, 3:], landing_pos, landing_t
