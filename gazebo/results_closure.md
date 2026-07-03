# Dynamics closure: ttsim RK4 vs Gazebo (DART), first landing

Same initial state, both integrated at dt=1 ms to the z=R crossing. The residual is the integrator gap (semi-implicit Euler vs RK4), i.e. the backend-equivalence budget.

| condition | Gazebo x (m) | RK4 x (m) | dx (mm) | dt (ms) |
|---|---|---|---|---|
| v45_flat | 1.6480 | 1.6562 | 8.2 | +2.1 |
| v45_top200 | 1.5308 | 1.5384 | 7.5 | +1.9 |
| v45_top400 | 1.4341 | 1.4411 | 7.0 | +1.7 |
| v60_back200 | 2.3330 | 2.3449 | 11.9 | +2.4 |
| v60_flat | 2.0868 | 2.0974 | 10.6 | +2.1 |
| v60_mixed | 1.7872 | 1.7962 | 9.1 | +1.7 |
| v60_top200 | 1.9000 | 1.9096 | 9.6 | +1.8 |
| v60_top400 | 1.7531 | 1.7620 | 8.8 | +1.7 |
| v70_top400 | 1.9482 | 1.9582 | 10.0 | +1.6 |

mean |dx| = 9.2 mm, max = 11.9 mm over 9 conditions.
