"""Table-tennis flight physics: gravity + quadratic drag + Magnus.

First-landing prediction only needs the *flight* dynamics (the bounce model is
off the critical path for the first table contact), so this module integrates a
ball state until its first downward crossing of the table plane z=0.

Constants are physical (40 mm / 2.7 g ball) and the lumped drag/Magnus
coefficients are calibrated to give realistic table-tennis arcs. K_MAG is the
one knob to dial spin influence up or down.
"""
import numpy as np

# --- Ball & air constants ---
G = 9.81                       # m/s^2
M_BALL = 2.7e-3                # kg
R_BALL = 0.02                  # m  (40 mm diameter)
RHO_AIR = 1.20                 # kg/m^3
CD = 0.40                      # drag coefficient
A_CROSS = np.pi * R_BALL**2    # frontal area

# drag acceleration = -K_DRAG * |v| * v   (~0.112 /m  => ~1.1 g of drag at 10 m/s)
K_DRAG = 0.5 * RHO_AIR * CD * A_CROSS / M_BALL
# Magnus acceleration = K_MAG * (omega x v); lumped coeff (Cl-slope ~0.3) -> ~1.6e-3
K_MAG = 1.6e-3

TABLE_Z = 0.0                  # landing plane (height above table surface)


def accel(vel, omega):
    speed = np.linalg.norm(vel)
    a_grav = np.array([0.0, 0.0, -G])
    a_drag = -K_DRAG * speed * vel
    a_mag = K_MAG * np.cross(omega, vel)
    return a_grav + a_drag + a_mag


def _deriv(state, omega):
    return np.concatenate([state[3:], accel(state[3:], omega)])


def rk4_step(state, omega, dt):
    k1 = _deriv(state, omega)
    k2 = _deriv(state + 0.5 * dt * k1, omega)
    k3 = _deriv(state + 0.5 * dt * k2, omega)
    k4 = _deriv(state + dt * k3, omega)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(p0, v0, omega, dt=1e-3, t_max=2.0, table_z=TABLE_Z):
    """Integrate from (p0,v0) with constant spin until first downward crossing of
    `table_z`. Use table_z=0 for a point-ball landing plane, or table_z=R_BALL to
    match a finite-radius ball's first *contact* (e.g. the Gazebo bridge).

    Returns (times, positions, velocities, landing_pos, landing_t). landing_* are
    None if the ball never lands within t_max.
    """
    state = np.concatenate([np.asarray(p0, float), np.asarray(v0, float)])
    times = [0.0]
    states = [state.copy()]
    t = 0.0
    landing_pos = landing_t = None
    for _ in range(int(t_max / dt)):
        prev = state
        state = rk4_step(state, omega, dt)
        t += dt
        times.append(t)
        states.append(state.copy())
        if prev[2] > table_z and state[2] <= table_z and state[5] < 0:
            frac = (prev[2] - table_z) / (prev[2] - state[2])
            landing_pos = prev[:3] + frac * (state[:3] - prev[:3])
            landing_t = (t - dt) + frac * dt
            break
    return np.array(times), np.array(states)[:, :3], np.array(states)[:, 3:], landing_pos, landing_t


def _accel_batch(vel, omega):
    """Batched acceleration. vel,omega: (B,3) -> (B,3)."""
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    a = np.zeros_like(vel)
    a[:, 2] = -G
    a += -K_DRAG * speed * vel
    a += K_MAG * np.cross(omega, vel)
    return a


def _rk4_batch(state, omega, dt):
    """state: (B,6), omega: (B,3)."""
    def d(s):
        return np.concatenate([s[:, 3:], _accel_batch(s[:, 3:], omega)], axis=1)
    k1 = d(state)
    k2 = d(state + 0.5 * dt * k1)
    k3 = d(state + 0.5 * dt * k2)
    k4 = d(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_batch(states0, omegas, t_end, dt):
    """Integrate a batch of trajectories on a shared grid. Returns (times, traj_xyz)
    with traj_xyz shape (nsteps+1, B, 3). One Python loop handles all B."""
    state = states0.copy()
    nsteps = int(np.ceil(t_end / dt))
    traj = np.empty((nsteps + 1, states0.shape[0], 3))
    traj[0] = state[:, :3]
    for i in range(nsteps):
        state = _rk4_batch(state, omegas, dt)
        traj[i + 1] = state[:, :3]
    return np.arange(nsteps + 1) * dt, traj


def predict_positions_batch(thetas, obs_times, dt=4e-3):
    """thetas: (B,9). Returns (B, N, 3) positions at obs_times for each theta."""
    states0 = thetas[:, :6]
    omegas = thetas[:, 6:9]
    ts, traj = integrate_batch(states0, omegas, obs_times.max() + dt, dt)
    B = thetas.shape[0]
    out = np.empty((B, len(obs_times), 3))
    for b in range(B):
        for ax in range(3):
            out[b, :, ax] = np.interp(obs_times, ts, traj[:, b, ax])
    return out


def predict_positions(theta, obs_times, dt=4e-3):
    """Positions at obs_times for trajectory params theta=(p0,v0,omega)."""
    return predict_positions_batch(theta[None, :], obs_times, dt)[0]


def predict_landing(theta, dt=1e-3, t_max=2.0, table_z=TABLE_Z):
    p0, v0, omega = theta[:3], theta[3:6], theta[6:9]
    _, _, _, lp, lt = simulate(p0, v0, omega, dt, t_max, table_z=table_z)
    return lp, lt
