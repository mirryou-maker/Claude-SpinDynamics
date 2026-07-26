"""
Notebook 32: MTJ free-layer switching — mesh-convergence of P_sw(J).

Same physical problem as notebook 31 (CoFeB PMA free layer, 80x80x1.5 nm) but
the SAME structure is discretised on progressively finer meshes:
    8x8   -> 10.00 nm cells
    16x16 ->  5.00 nm cells   (= notebook 31)
    32x32 ->  2.50 nm cells
    64x64 ->  1.25 nm cells
Coarse cells (>= l_ex ~ 4.9 nm) force near-coherent reversal; fine cells resolve
nucleation, so J_c0 and the P_sw(J) transition converge as the mesh refines.

For each mesh we increase the ABSOLUTE current density J until P_sw = 1
(fully deterministic switching within the 2 ns / 300 K window).

Batched engine: BatchedLLGGPU + enable_demag() (exchange + uniaxial PMA + demag
+ Slonczewski STT + FDT thermal). Every trial is a replica; each J-point is one
batched run over N_trials replicas.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import numpy as np
import micromag as mm

print("Notebook 32: MTJ mesh-convergence of P_sw(J)")
print(f"  CUDA: {mm.cuda_available()}")

mu0, kB = 4e-7*np.pi, 1.380649e-23
Ms, A, Ku, alpha, P = 1.0e6, 1.5e-11, 1.0e6, 0.02, 0.5
L, tz = 80e-9, 1.5e-9          # fixed physical structure: 80x80x1.5 nm
d_free = tz
l_ex = np.sqrt(2*A/(mu0*Ms**2))
print(f"  Structure 80x80x1.5 nm; l_ex={l_ex*1e9:.2f} nm; Ku={Ku/1e6:.1f} MJ/m3, "
      f"mu0Ms^2/2={mu0*Ms**2/2/1e6:.2f} MJ/m3")

MESHES = [8, 16, 32, 64]
t_max = 2.0e-9
N_TRIALS = {8: 96, 16: 96, 32: 64, 64: 24}

# The exchange field ~ 2A/(mu0 Ms dx^2) stiffens as the mesh refines, so the
# time step must scale with dx^2 for the dynamics to stay accurate (Depondt is
# norm-exact and never blows up, but too-large dt suppresses switching entirely
# — verified: 1.25 nm cells fail to switch at dt=5e-14/2.5e-14 but switch
# correctly at dt<=1.25e-14). Anchor dt=5e-14 at the 2.5 nm cell, capped at 1e-13.
def dt_for(dxy):
    return min(1e-13, 5e-14 * (dxy / 2.5e-9)**2)

def cfg():
    c = mm.BatchedLLGConfig()
    c.Ms, c.alpha, c.A, c.K1 = Ms, alpha, A, Ku
    c.easy = mm.Vec3(0,0,1); c.p = mm.Vec3(0,0,1)
    c.d_free, c.P, c.Lambda, c.beta = d_free, P, 1.0, 0.0
    return c

def measure_Jc0(grid, dt_, t_ns=3.0):
    """min +J switching a 2%-tilted uniform state at T=0."""
    Jgrid = np.arange(0.15e12, 3.01e12, 0.15e12)
    R = len(Jgrid)
    b = mm.BatchedLLGGPU(R, grid, cfg(), dt_, seed=1); b.enable_demag()
    b.set_J(list(Jgrid)); b.set_T([0.0]*R)
    b.set_uniform(0.02, 0.0, np.sqrt(1-0.02**2))
    b.run(int(t_ns*1e-9/dt_))
    mz = np.array(b.get_avg_m()).reshape(R,3)[:,2]
    sw = np.where(mz < -0.5)[0]
    return float(Jgrid[sw[0]]) if len(sw) else float("nan")

results = {}
for n in MESHES:
    dxy = L/n
    dt = dt_for(dxy)
    nstep = int(t_max/dt)
    grid = mm.StructuredGrid(n, n, 1, dxy, dxy, tz)
    Ntr = N_TRIALS[n]
    t0 = time.time()
    Jc0 = measure_Jc0(grid, dt)
    print(f"\n[{n}x{n}] cell={dxy*1e9:.2f} nm  N={n*n}  dt={dt:.2e}s  {nstep} steps  "
          f"J_c0={Jc0/1e12:.3f}e12  (cells {'>' if dxy>l_ex else '<='} l_ex)")
    # absolute-J sweep anchored to this mesh's Jc0; extend until P_sw=1.
    Jfac = np.round(np.arange(0.6, 3.001, 0.15), 4)
    Jv, Pv = [], []
    for f in Jfac:
        J = f*Jc0
        b = mm.BatchedLLGGPU(Ntr, grid, cfg(), dt, seed=2024); b.enable_demag()
        b.set_J([J]*Ntr); b.set_T([300.0]*Ntr); b.set_uniform(0,0,1)
        b.run(nstep)
        mz = np.array(b.get_avg_m()).reshape(Ntr,3)[:,2]
        Psw = float((mz < -0.5).mean())
        Jv.append(J); Pv.append(Psw)
        print(f"    J={J/1e12:5.3f}e12 ({f:.2f}Jc0)  P_sw={Psw:.3f}")
        if Psw >= 0.999:
            break
    Jv, Pv = np.array(Jv), np.array(Pv)
    Jsat = Jv[np.argmax(Pv >= 0.999)] if np.any(Pv >= 0.999) else np.nan
    results[n] = dict(dxy=dxy, Jc0=Jc0, J=Jv, P=Pv, Jsat=Jsat, N=Ntr, dt=dt)
    print(f"    -> J(P_sw=1) = {Jsat/1e12:.3f}e12 A/m2   ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Figure: P_sw vs absolute J for the 4 meshes + convergence of Jc0 / Jsat.
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {8:'C0', 16:'C1', 32:'C2', 64:'C3'}
    for n in MESHES:
        r = results[n]
        ax.errorbar(r['J']/1e12, r['P'], yerr=np.sqrt(r['P']*(1-r['P'])/r['N']),
                    fmt='o-', color=colors[n], ms=4, capsize=2,
                    label=f'{n}x{n} ({r["dxy"]*1e9:.2f} nm)')
    ax.axhline(1.0, color='k', ls=':', lw=1, alpha=0.5)
    ax.axhline(0.5, color='0.6', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$J$ (10$^{12}$ A/m$^2$)'); ax.set_ylabel(r'$P_\mathrm{sw}$ (2 ns, 300 K)')
    ax.set_title('Mesh convergence of MTJ switching\n80x80x1.5 nm CoFeB PMA, batched')
    ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=8.5, title='mesh (cell)'); ax.grid(alpha=0.3)

    cells = np.array([results[n]['dxy']*1e9 for n in MESHES])
    Jc0s  = np.array([results[n]['Jc0']/1e12 for n in MESHES])
    Jsats = np.array([results[n]['Jsat']/1e12 for n in MESHES])
    ax2.plot(cells, Jc0s, 'o-', color='C4', label=r'$J_{c0}$ (T=0 threshold)')
    ax2.plot(cells, Jsats, 's-', color='C5', label=r'$J(P_\mathrm{sw}=1)$')
    ax2.axvline(l_ex*1e9, color='k', ls='--', lw=1, alpha=0.6, label=r'$l_\mathrm{ex}$')
    ax2.set_xlabel('cell size (nm)'); ax2.set_ylabel(r'$J$ (10$^{12}$ A/m$^2$)')
    ax2.set_title('Threshold vs discretisation'); ax2.invert_xaxis()
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '32_mtj_mesh_convergence_gpu.png')
    fig.savefig(out, dpi=130); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary (80x80x1.5 nm, 300 K, 2 ns) ===")
for n in MESHES:
    r = results[n]
    print(f"  {n:2d}x{n:<2d} ({r['dxy']*1e9:5.2f} nm, dt={r['dt']:.1e}): "
          f"Jc0={r['Jc0']/1e12:.3f}e12  J(P=1)={r['Jsat']/1e12:.3f}e12  N={r['N']}")
