"""
Notebook 30 (batched): GPU Thermal STT Switching Statistics via replica batching.

Task 2 Phase 2.0 payoff. This reproduces notebook 30's Neel-Brown thermally-
assisted STT switching statistics (Pt/Co PMA macrospin) using BatchedMacrospinGPU
-- every trial is a REPLICA advanced by one kernel launch per step, so the whole
P_sw(J) and P_sw(T) sweeps run in seconds instead of hours.

Physics identical to notebook 30:
  Ms=580 kA/m, K=0.5 MJ/m3, alpha=0.02, P=0.5, Lambda=1, d_F=10 nm cube,
  V=1e-24 m3, Delta(300K)=120.7. The 10 nm cube demag is isotropic (1/3 on the
  diagonal) -> zero net torque -> the macrospin (uniaxial + STT + thermal) engine
  is physically equivalent to notebook 30's DemagFieldGPU path (verified: J_c0
  and the P_sw(J) transition match within binomial error).

Integrator: Depondt-Mertens rotation (|m|=1 exact) with the FDT-correct
sigma = sqrt(2 alpha kB T/(mu0^2 Ms gamma0 V dt)) and per-replica Philox noise.
"""
import os, sys, time, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import numpy as np
import micromag as mm

print("Notebook 30 (batched): Thermal STT via replica batching")
print(f"  CUDA: {mm.cuda_available()}")

mu0, kB = 4e-7*np.pi, 1.38065e-23
Ms, K, alpha, P = 580e3, 0.5e6, 0.02, 0.5
d_F, dx = 10e-9, 10e-9
V = dx**3
Delta_300K = K * V / (kB * 300)

def make_cfg():
    c = mm.BatchedMacrospinConfig()
    c.Ms, c.alpha, c.V, c.K1 = Ms, alpha, V, K
    c.easy = mm.Vec3(0, 0, 1); c.p = mm.Vec3(0, 0, 1)
    c.d_free, c.P, c.Lambda, c.beta = d_F, P, 1.0, 0.0
    return c

# ---------------------------------------------------------------------------
# Reference threshold J_c0: antidamping linear-instability (T=0, small tilt).
# Each candidate J is a replica; find the minimum POSITIVE J that switches.
# ---------------------------------------------------------------------------
def measure_Jc0(dt=2e-14, t_ns=10.0):
    Jgrid = np.arange(1.0e12, 6.01e12, 0.1e12)
    R = len(Jgrid)
    b = mm.BatchedMacrospinGPU(R, make_cfg(), dt, seed=1)
    b.set_J(list(Jgrid)); b.set_T([0.0]*R)
    tilt = np.tile([0.02, 0.0, np.sqrt(1-0.02**2)], (R, 1))
    b.set_state(list(tilt.reshape(-1)))
    b.run(int(t_ns*1e-9/dt))
    mz = np.array(b.get_mz())
    sw = np.where(mz < -0.5)[0]
    return float(Jgrid[sw[0]]) if len(sw) else float("nan")

t0 = time.time()
J_c0 = measure_Jc0()
print(f"\n  J_c0 (antidamping instability, +J, tilt) = {J_c0/1e12:.3f} e12 A/m2"
      f"   ({time.time()-t0:.1f}s)")
print(f"  Delta(300K) = {Delta_300K:.1f}")

# ---------------------------------------------------------------------------
# Part B: P_sw vs J at 300 K. All (J factor x N trials) trials are replicas
# advanced in ONE batched run.
# ---------------------------------------------------------------------------
J_factors = np.unique(np.round(np.concatenate([
    np.arange(0.70, 1.201, 0.05), np.linspace(0.90, 1.10, 16)]), 6))
N_B = 200                     # trials per J (cheap now -> tighter error bars)
dt_B, t_max = 1e-13, 2.0e-9
nstep_B = int(t_max / dt_B)

Jrep = np.repeat(J_factors * J_c0, N_B)
R_B = len(Jrep)
print(f"\n--- Part B: P_sw vs J (300 K, {t_max*1e9:.0f} ns) ---")
print(f"  {len(J_factors)} J-values x N={N_B}  ->  {R_B} replicas, {nstep_B} steps, 1 batch")

t0 = time.time()
b = mm.BatchedMacrospinGPU(R_B, make_cfg(), dt_B, seed=2024)
b.set_J(list(Jrep)); b.set_T([300.0]*R_B)   # start at +z (default state)
b.run(nstep_B)
mzB = np.array(b.get_mz()).reshape(len(J_factors), N_B)
wall_B = time.time() - t0
P_sw_B = (mzB < -0.5).mean(axis=1)
for jf, p_ in zip(J_factors, P_sw_B):
    print(f"  J={jf:.4f}*J_c0  P_sw={p_:.3f} (N={N_B})")
print(f"  Part B wall: {wall_B:.2f} s  ({R_B*nstep_B/wall_B:.2e} replica*step/s)")

# ---------------------------------------------------------------------------
# Part C: P_sw vs T at J=0.95 J_c0.
# ---------------------------------------------------------------------------
T_sweep = np.array([100, 200, 300, 400], float)
J_C = 0.95 * J_c0
N_C = 200
dt_C = 2e-14
nstep_C = int(t_max / dt_C)
Trep = np.repeat(T_sweep, N_C)
R_C = len(Trep)
print(f"\n--- Part C: P_sw vs T at J=0.95 J_c0 ({t_max*1e9:.0f} ns) ---")
t0 = time.time()
bc = mm.BatchedMacrospinGPU(R_C, make_cfg(), dt_C, seed=777)
bc.set_J([J_C]*R_C); bc.set_T(list(Trep))
bc.run(nstep_C)
mzC = np.array(bc.get_mz()).reshape(len(T_sweep), N_C)
wall_C = time.time() - t0
P_sw_C = (mzC < -0.5).mean(axis=1)
for T_val, p_ in zip(T_sweep, P_sw_C):
    Delta_T = K*V/(kB*T_val); Deff = max(0, Delta_T*(1-J_C/J_c0)**2)
    print(f"  T={T_val:.0f}K  Delta={Delta_T:.0f}  Delta_eff={Deff:.1f}  P_sw={p_:.3f}")
print(f"  Part C wall: {wall_C:.2f} s")

# ---------------------------------------------------------------------------
# Part A: N=40 ensemble at J=1.00 J_c0, 300 K, WITH mz(t) trajectories
# (chunked batched run: log all replicas every log_ev steps).
# ---------------------------------------------------------------------------
J_A, T_A, N_A = 1.00*J_c0, 300.0, 40
dt_A, log_ev = 2e-14, 1000
nchunk = int(t_max/dt_A)//log_ev
print(f"\n--- Part A: N={N_A} ensemble at J=1.00 J_c0, 300 K ---")
t0 = time.time()
ba = mm.BatchedMacrospinGPU(N_A, make_cfg(), dt_A, seed=55)
ba.set_J([J_A]*N_A); ba.set_T([T_A]*N_A)
t_log_ps, mz_traj = [], []
for c in range(nchunk):
    ba.run(log_ev)
    mz_traj.append(np.array(ba.get_mz()))
    t_log_ps.append((c+1)*log_ev*dt_A*1e12)
mz_traj = np.array(mz_traj)            # (nchunk, N_A)
sw_mask = (mz_traj[-1] < -0.5)
P_sw_A = sw_mask.mean()
print(f"  P_sw = {P_sw_A:.2f} ({sw_mask.sum()}/{N_A})   ({time.time()-t0:.2f}s)")

# ---------------------------------------------------------------------------
# Cross-check vs the loop-engine cache (notebook 30 Part B) if present.
# ---------------------------------------------------------------------------
cache = pathlib.Path(__file__).parent / "30_partB_cache.json"
if cache.exists():
    ref = {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    def trans(J, P): return J[np.argmin(np.abs(np.array(P)-0.5))]
    Jref = np.array(sorted(float(k) for k in ref))
    Pref = [ref[f"{j:.6f}"][0]/ref[f"{j:.6f}"][1] for j in Jref]
    print(f"\n  Transition J/Jc0 @ P=0.5:  batched={trans(J_factors,P_sw_B):.2f}"
          f"   loop-cache={trans(Jref,Pref):.2f}")

# ---------------------------------------------------------------------------
# Figure (3 panels, matches notebook 30 layout)
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t_arr = np.array(t_log_ps)
    ax = axes[0]
    for i in range(N_A):
        ax.plot(t_arr, mz_traj[:, i], '-', lw=1.0, alpha=0.6,
                color='C3' if sw_mask[i] else 'C0')
    ax.axhline(-0.5, color='k', ls='--', lw=1.5, alpha=0.6, label='switch threshold')
    ax.set_xlabel(r'$t$ (ps)'); ax.set_ylabel(r'$m_z$')
    ax.set_title(rf'$m_z(t)$, $N={N_A}$ ensemble' + '\n'
                 + rf'$J=J_{{c0}}$, 300 K, $P_\mathrm{{sw}}={P_sw_A:.2f}$')
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(-1.15, 1.15)

    ax = axes[1]
    yerr = [np.sqrt(p*(1-p)/N_B) for p in P_sw_B]
    ax.errorbar(J_factors, P_sw_B, yerr=yerr, fmt='o-', color='C0', lw=1.4, ms=4, capsize=2.5)
    ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.5)
    ax.axvline(1.0, color='C3', ls='--', lw=1, alpha=0.7, label=r'$J=J_{c0}$')
    ax.set_xlabel(r'$J/J_{c0}$'); ax.set_ylabel(r'$P_\mathrm{sw}$')
    ax.set_title(rf'$P_\mathrm{{sw}}$ vs $J/J_{{c0}}$ (300 K, {t_max*1e9:.0f} ns)'
                 + '\n' + rf'batched, $N={N_B}$/point')
    ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = axes[2]
    yerr_C = [np.sqrt(p*(1-p)/N_C) for p in P_sw_C]
    ax.errorbar(T_sweep, P_sw_C, yerr=yerr_C, fmt='s-', color='C1', lw=2, ms=9, capsize=5)
    ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r'$P_\mathrm{sw}$')
    ax.set_title(rf'$P_\mathrm{{sw}}$ vs $T$ ($J=0.95 J_{{c0}}$, {t_max*1e9:.0f} ns)'
                 + '\n' + rf'batched, $N={N_C}$/point')
    ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)

    plt.suptitle('Thermal STT Switching Statistics -- REPLICA BATCHED '
                 '(BatchedMacrospinGPU, Depondt-Mertens)\n'
                 f'Pt/Co PMA {int(dx*1e9)}nm cube: Delta(300K)={Delta_300K:.0f}, '
                 f'J_c0={J_c0/1e12:.3f}e12 A/m2, alpha={alpha}', fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '30_thermal_stt_batched_gpu.png')
    plt.savefig(out, dpi=120); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary (batched) ===")
print(f"  J_c0={J_c0/1e12:.3f}e12  Delta(300K)={Delta_300K:.1f}")
print(f"  Part A P_sw={P_sw_A:.2f} (N={N_A})")
print(f"  Part B walls {wall_B:.2f}s ({R_B} replicas)  Part C {wall_C:.2f}s ({R_C} replicas)")
