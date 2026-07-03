"""Instantaneous impulse model of the table bounce (for M2, second-touchdown
prediction).

Normal direction: coefficient of restitution `e` (SDF: 0.9).
Tangential direction: Coulomb friction impulse `mu*Jn` opposing the contact
point's slip, capped at the impulse that *stops* the slip (stick), with the
ball inertia I = alpha * m * r^2. `alpha=0.4` matches the SDF's solid-sphere
inertia (a real celluloid ball is hollow, alpha=2/3 — but the Gazebo model is
the truth being predicted here, so we match the SDF).

Sanity check (baseline serve): topspin in with slip arrested -> outgoing
contact-point velocity ~0, forward speed-up ~0.6 m/s, spin partially consumed.
This mirrors what DART's contact solver produces for the same surface params.
"""
import numpy as np

from .physics import M_BALL, R_BALL

# EFFECTIVE restitution, MEASURED from the Gazebo recordings themselves
# (scripts/calibrate_bounce.py over the 9-condition sweep: 0.770..0.782,
# mean 0.7765). The SDF says 0.9 on both surfaces, but DART's contact
# resolution at 1 kHz stepping yields ~0.78 effective — calibrate against
# the backend you predict, not against its config file.
E_TABLE = 0.7765
MU_TABLE = 0.25  # friction coefficient (matches SDF; tangential response fits)
ALPHA_I = 0.4    # I/(m r^2): solid sphere, matches the model.sdf inertia


def bounce_state(v_in, w_in, e=E_TABLE, mu=MU_TABLE, alpha=ALPHA_I):
    """Map incoming (velocity, spin) at table contact to outgoing (v, w).

    v_in: (3,) ball-center velocity at contact (v_in[2] < 0)
    w_in: (3,) spin; topspin for +x travel is +w_y
    Returns (v_out, w_out).
    """
    v_in = np.asarray(v_in, float)
    w_in = np.asarray(w_in, float)
    v_out = v_in.copy()
    w_out = w_in.copy()

    # normal: restitution
    v_out[2] = -e * v_in[2]
    Jn = M_BALL * (1.0 + e) * abs(v_in[2])

    # tangential: contact-point slip velocity (bottom of the ball)
    #   v_c = v + w x (-r zhat) -> (vx - r*wy, vy + r*wx)
    s = np.array([v_in[0] - R_BALL * w_in[1],
                  v_in[1] + R_BALL * w_in[0]])
    smag = np.linalg.norm(s)
    if smag < 1e-12:
        return v_out, w_out

    shat = s / smag
    # impulse that would arrest the slip: d(slip)/dJt = (1/m)(1 + 1/alpha)
    J_stick = smag * M_BALL / (1.0 + 1.0 / alpha)
    Jt = min(mu * Jn, J_stick)

    v_out[:2] += -(Jt / M_BALL) * shat
    # angular impulse: dw = (Jt r / I) (zhat x shat), I = alpha m r^2
    inertia = alpha * M_BALL * R_BALL ** 2
    zxs = np.array([-shat[1], shat[0], 0.0])
    w_out += (Jt * R_BALL / inertia) * zxs
    return v_out, w_out
