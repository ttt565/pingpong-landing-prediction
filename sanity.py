import time, numpy as np
from ttsim.experiment import NOMINAL, sample_launch, run_trial
from ttsim.physics import simulate

# --- physics realism: truth landing, flight time, spin effect ---
p0,v0,om = NOMINAL["p0"], NOMINAL["v0"], NOMINAL["omega"]
t,pos,vel,land,tl = simulate(p0,v0,om)
_,_,_,land0,tl0 = simulate(p0,v0,np.zeros(3))  # no-spin counterfactual
print(f"topspin : land_xy=({land[0]:.3f},{land[1]:.3f}) m  t_land={tl:.3f}s  v0={np.linalg.norm(v0):.1f}")
print(f"no-spin : land_xy=({land0[0]:.3f},{land0[1]:.3f}) m  t_land={tl0:.3f}s")
print(f"spin shortens landing by {100*(land0[0]-land[0]):.1f} cm in x  (=> M0 should be biased long)")
print(f"frames @120fps before landing: {int(tl*120)}")

# --- timing one full trial (all methods) ---
rng = np.random.default_rng(0)
methods=["M0","M1","M3_oracle","M3_rule","M4"]
t0=time.time()
r = run_trial(*sample_launch(rng), sigma0=0.008, alpha=4.0, p_miss=0.0, fps=120, rng=rng, methods=methods)
dt=time.time()-t0
print(f"\none trial ({len(methods)} methods): {dt*1000:.0f} ms   n_obs={r['_n_obs']}")
for m in methods: print(f"   {m:10s} err={r[m]:.2f} cm")
