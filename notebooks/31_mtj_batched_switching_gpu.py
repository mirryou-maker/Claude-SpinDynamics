"""
Notebook 31: Spatially-resolved MTJ free-layer switching statistics (batched).

Task 2 Phase 2.5 showcase of the physically-complete batched engine
(BatchedLLGGPU + enable_demag): exchange + uniaxial PMA + demag + Slonczewski
STT + FDT thermal, with every trial a REPLICA advanced by one launch per step.

Unlike the macrospin (notebook 30), the free layer is resolved on a grid, so
exchange stiffness and the true (non-isotropic thin-film) demag enter — the
switching is nucleation-mediated, not coherent. We compare the multi-cell
P_sw(J) transition against the single-cell macrospin approximation.

Free layer: CoFeB-like PMA square, 80x80x1.5 nm, 5 nm cells (16x16x1):
  Ms=1.0 MA/m, A=1.5e-11 J/m, K_u=1.0 MJ/m3 (net PMA vs mu0 Ms^2/2=0.63 MJ/m3),
  alpha=0.02, P=0.5, p=+z. T=300 K, 2 ns window.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import numpy as np
import micromag as mm

print("Notebook 31: MTJ free-layer batched switching")
print(f"  CUDA: {mm.cuda_available()}")

mu0, kB = 4e-7*np.pi, 1.380649e-23
Ms, A, Ku, alpha, P = 1.0e6, 1.5e-11, 1.0e6, 0.02, 0.5
nx, ny, nz = 16, 16, 1
dx = dy = 5e-9; dz = 1.5e-9
d_free = dz
grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
N = nx*ny*nz
V_cell = dx*dy*dz
V_tot  = V_cell * N
Delta_300 = Ku * V_tot / (kB*300)   # macrospin thermal stability of the whole layer
print(f"  Free layer 80x80x1.5 nm, {nx}x{ny} cells; Delta(300K,coherent)={Delta_300:.0f}")

def cfg():
    c = mm.BatchedLLGConfig()
    c.Ms, c.alpha, c.A, c.K1 = Ms, alpha, A, Ku
    c.easy = mm.Vec3(0,0,1); c.p = mm.Vec3(0,0,1)
    c.d_free, c.P, c.Lambda, c.beta = d_free, P, 1.0, 0.0
    return c

def macro_cfg():
    # WHOLE-LAYER coherent macrospin: thermal volume = the full free layer
    # (V_tot), not one cell -> the correct coherent baseline to contrast with
    # the exchange/demag-textured multi-cell switching.
    c = mm.BatchedMacrospinConfig()
    c.Ms, c.alpha, c.V, c.K1 = Ms, alpha, V_tot, Ku
    c.easy = mm.Vec3(0,0,1); c.p = mm.Vec3(0,0,1)
    c.d_free, c.P, c.Lambda = d_free, P, 1.0
    return c

# ---------------------------------------------------------------------------
# J_c0: min +J that switches the spatially-resolved layer at T=0 (2% tilt).
# ---------------------------------------------------------------------------
def measure_Jc0_multicell(dt=1e-13, t_ns=10.0):
    Jgrid = np.arange(0.2e12, 4.01e12, 0.2e12)
    R = len(Jgrid)
    b = mm.BatchedLLGGPU(R, grid, cfg(), dt, seed=1); b.enable_demag()
    b.set_J(list(Jgrid)); b.set_T([0.0]*R)
    b.set_uniform(0.02, 0.0, np.sqrt(1-0.02**2))
    b.run(int(t_ns*1e-9/dt))
    mz = np.array(b.get_avg_m()).reshape(R,3)[:,2]
    sw = np.where(mz < -0.5)[0]
    return float(Jgrid[sw[0]]) if len(sw) else float("nan")

def measure_Jc0_macro(dt=1e-13, t_ns=10.0):
    Jgrid = np.arange(0.2e12, 4.01e12, 0.2e12)
    R = len(Jgrid)
    b = mm.BatchedMacrospinGPU(R, macro_cfg(), dt, seed=1)
    b.set_J(list(Jgrid)); b.set_T([0.0]*R)
    tilt = np.tile([0.02, 0.0, np.sqrt(1-0.02**2)], (R,1))
    b.set_state(list(tilt.reshape(-1)))
    b.run(int(t_ns*1e-9/dt))
    mz = np.array(b.get_mz())
    sw = np.where(mz < -0.5)[0]
    return float(Jgrid[sw[0]]) if len(sw) else float("nan")

t0 = time.time()
Jc0 = measure_Jc0_multicell()          # normalises the multi-cell curve
Jc0_ma = measure_Jc0_macro()           # normalises the macrospin curve (own threshold)
print(f"  J_c0 multi-cell = {Jc0/1e12:.3f} e12,  macrospin = {Jc0_ma/1e12:.3f} e12  "
      f"({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# P_sw(J) at 300 K — multi-cell vs macrospin. Each (J factor x N trials) = replica.
# ---------------------------------------------------------------------------
J_factors = np.round(np.arange(0.70, 1.31, 0.05), 4)
N_TR = 96
dt, t_max = 1e-13, 2.0e-9
nstep = int(t_max/dt)
Jrep = np.repeat(J_factors*Jc0, N_TR)
R = len(Jrep)
print(f"\n  P_sw(J): {len(J_factors)} J-values x N={N_TR} = {R} replicas, {nstep} steps")

t0 = time.time()
bm = mm.BatchedLLGGPU(R, grid, cfg(), dt, seed=2024); bm.enable_demag()
bm.set_J(list(Jrep)); bm.set_T([300.0]*R); bm.set_uniform(0,0,1)
bm.run(nstep)
mz_mc = np.array(bm.get_avg_m()).reshape(len(J_factors), N_TR, 3)[:,:,2]
P_mc = (mz_mc < -0.5).mean(axis=1)
wall_mc = time.time()-t0
print(f"  multi-cell: {wall_mc:.1f} s ({R*nstep*N/wall_mc:.2e} cell*step/s)")

t0 = time.time()
Jrep_ma = np.repeat(J_factors*Jc0_ma, N_TR)    # macrospin normalised by its OWN Jc0
ba = mm.BatchedMacrospinGPU(R, macro_cfg(), dt, seed=2024)
ba.set_J(list(Jrep_ma)); ba.set_T([300.0]*R)
ba.run(nstep)
mz_ma = np.array(ba.get_mz()).reshape(len(J_factors), N_TR)
P_ma = (mz_ma < -0.5).mean(axis=1)
wall_ma = time.time()-t0
print(f"  macrospin:  {wall_ma:.1f} s")

def transition(P): return J_factors[np.argmin(np.abs(P-0.5))]
print(f"\n  Transition J/Jc0 @ P=0.5:  multi-cell={transition(P_mc):.2f}  macrospin={transition(P_ma):.2f}")
for jf, pmc, pma in zip(J_factors, P_mc, P_ma):
    print(f"   J={jf:.2f}*Jc0   multi-cell P={pmc:.2f}   macrospin P={pma:.2f}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.errorbar(J_factors, P_mc, yerr=np.sqrt(P_mc*(1-P_mc)/N_TR), fmt='o-',
                color='C0', ms=5, capsize=3, label=f'multi-cell {nx}x{ny} (exchange+demag)')
    ax.errorbar(J_factors, P_ma, yerr=np.sqrt(P_ma*(1-P_ma)/N_TR), fmt='s--',
                color='C1', ms=5, capsize=3, label='macrospin (coherent, own $J_{c0}$)')
    ax.axhline(0.5, color='k', ls=':', lw=1, alpha=0.6)
    ax.axvline(1.0, color='C3', ls='--', lw=1, alpha=0.7, label=r'own $J_{c0}$ (T=0)')
    ax.set_xlabel(r'$J/J_{c0}^\mathrm{own}$'); ax.set_ylabel(r'$P_\mathrm{sw}$ (2 ns, 300 K)')
    ax.set_title('MTJ free-layer STT switching: spatially resolved vs macrospin\n'
                 f'CoFeB PMA 80x80x1.5 nm, $J_{{c0}}$={Jc0/1e12:.2f}e12 A/m$^2$, '
                 f'{R} batched replicas')
    ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '31_mtj_batched_switching_gpu.png')
    fig.savefig(out, dpi=130); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  J_c0={Jc0/1e12:.3f}e12 A/m2  (multi-cell, T=0)")
print(f"  transition multi-cell={transition(P_mc):.2f} Jc0  macrospin={transition(P_ma):.2f} Jc0")
print(f"  {R} replicas x {nstep} steps: multi-cell {wall_mc:.1f}s, macrospin {wall_ma:.1f}s")
