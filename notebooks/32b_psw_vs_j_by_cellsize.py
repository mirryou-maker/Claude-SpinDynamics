"""
Notebook 32b: P_sw vs J for each cell size (dense curves).

Companion to notebook 32 — same 80x80x1.5 nm CoFeB PMA free layer on 8/16/32/64
meshes, but a DENSE absolute-J sweep (no early stop) so each cell size gets a
smooth P_sw(J) S-curve. Reuses the T=0 thresholds J_c0 measured by notebook 32
(hardcoded below) to skip re-measuring. Per-mesh dt ~ dx^2 (see nb32).

Caches results to 32b_psw_cache.json so the figure can be redrawn instantly.
"""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import numpy as np
import micromag as mm

print("Notebook 32b: dense P_sw(J) by cell size;  CUDA:", mm.cuda_available())

mu0, kB = 4e-7*np.pi, 1.380649e-23
Ms, A, Ku, alpha, P = 1.0e6, 1.5e-11, 1.0e6, 0.02, 0.5
L, tz = 80e-9, 1.5e-9
l_ex = np.sqrt(2*A/(mu0*Ms**2))
MESHES = [8, 16, 32, 64]
JC0 = {8: 0.75e12, 16: 0.60e12, 32: 0.60e12, 64: 0.60e12}   # from nb32 (T=0)
N_TRIALS = {8: 96, 16: 96, 32: 64, 64: 24}
t_max = 2.0e-9
JFAC = np.round(np.arange(0.50, 1.301, 0.05), 4)            # dense, no early stop

def dt_for(dxy):
    return min(1e-13, 5e-14*(dxy/2.5e-9)**2)

def cfg():
    c = mm.BatchedLLGConfig()
    c.Ms, c.alpha, c.A, c.K1 = Ms, alpha, A, Ku
    c.easy = mm.Vec3(0,0,1); c.p = mm.Vec3(0,0,1)
    c.d_free, c.P, c.Lambda, c.beta = tz, P, 1.0, 0.0
    return c

CACHE = Path(__file__).parent / "32b_psw_cache.json"
data = json.loads(CACHE.read_text()) if CACHE.exists() else {}

for n in MESHES:
    key = str(n)
    if key in data and len(data[key]["P"]) == len(JFAC):
        print(f"[{n}x{n}] cached"); continue
    dxy = L/n; dt = dt_for(dxy); nstep = int(t_max/dt); Ntr = N_TRIALS[n]
    grid = mm.StructuredGrid(n, n, 1, dxy, dxy, tz)
    Jc0 = JC0[n]
    Jv, Pv = [], []
    t0 = time.time()
    for f in JFAC:
        J = f*Jc0
        b = mm.BatchedLLGGPU(Ntr, grid, cfg(), dt, seed=2024); b.enable_demag()
        b.set_J([J]*Ntr); b.set_T([300.0]*Ntr); b.set_uniform(0,0,1)
        b.run(nstep)
        mz = np.array(b.get_avg_m()).reshape(Ntr,3)[:,2]
        Jv.append(J); Pv.append(float((mz < -0.5).mean()))
    data[key] = dict(dxy=dxy, Jc0=Jc0, dt=dt, N=Ntr, J=Jv, P=Pv)
    CACHE.write_text(json.dumps(data))
    print(f"[{n}x{n}] {dxy*1e9:.2f} nm  {len(JFAC)} pts  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.0, 5.0))
colors = {8:'C0', 16:'C1', 32:'C2', 64:'C3'}
for n in MESHES:
    r = data[str(n)]
    J = np.array(r["J"])/1e12; Psw = np.array(r["P"]); N = r["N"]
    ax.errorbar(J, Psw, yerr=np.sqrt(Psw*(1-Psw)/N), fmt='o-', color=colors[n],
                ms=4, lw=1.4, capsize=2, label=f'{n}x{n}  ({r["dxy"]*1e9:.2f} nm)')
ax.axhline(0.5, color='0.6', ls=':', lw=0.8)
ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.6)
ax.set_xlabel(r'$J$ (10$^{12}$ A/m$^2$)')
ax.set_ylabel(r'$P_\mathrm{sw}$ (2 ns, 300 K)')
ax.set_title('MTJ switching probability vs current, by cell size\n'
             f'80x80x1.5 nm CoFeB PMA (fixed structure); $l_\\mathrm{{ex}}$={l_ex*1e9:.2f} nm')
ax.set_ylim(-0.05, 1.08); ax.legend(title='mesh (cell)', fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
out = Path(__file__).parent / "32b_psw_vs_j_by_cellsize.png"
fig.savefig(out, dpi=140); print("Plot saved:", out.name)

for n in MESHES:
    r = data[str(n)]; J = np.array(r["J"]); Psw = np.array(r["P"])
    j50 = J[np.argmin(np.abs(Psw-0.5))]/1e12
    j1  = J[np.argmax(Psw>=0.999)]/1e12 if np.any(Psw>=0.999) else float('nan')
    print(f"  {n:2d}x{n:<2d} ({r['dxy']*1e9:5.2f} nm): J(P=0.5)={j50:.3f}e12  J(P=1)={j1:.3f}e12")
