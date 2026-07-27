"""
Notebook 32d: fine-J refinement of the MTJ switching transition (run on ubuntu98).

Companion to nb32/32b — same 80x80x1.5 nm CoFeB PMA free layer on 8/16/32/64
meshes, but a FINE absolute-J grid focused on the switching window
(J/Jc0 in [0.70, 1.05], step 0.025) so the transition is densely sampled.
Reuses the T=0 thresholds Jc0 from nb32 (hardcoded) to skip re-measuring.
Per-mesh dt ~ dx^2. Caches to 32d_psw_fine_cache.json; two plots (absolute J
and own-Jc0-normalised, both zoomed to the transition).
"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# explicit linux CUDA build path (ubuntu98) in addition to micromag_locate
for p in [HERE.parent / "build/linux-gcc-cuda/python"]:
    if p.exists():
        sys.path.insert(0, str(p))
try:
    from micromag_locate import add_micromag_to_path; add_micromag_to_path()
except Exception:
    pass
import numpy as np
import micromag as mm

print("Notebook 32d: fine-J switching refinement;  CUDA:", mm.cuda_available())

mu0, kB = 4e-7*np.pi, 1.380649e-23
Ms, A, Ku, alpha, P = 1.0e6, 1.5e-11, 1.0e6, 0.02, 0.5
L, tz = 80e-9, 1.5e-9
l_ex = np.sqrt(2*A/(mu0*Ms**2))
MESHES = [8, 16, 32, 64]
JC0 = {8: 0.75e12, 16: 0.60e12, 32: 0.60e12, 64: 0.60e12}   # from nb32 (T=0)
N_TRIALS = {8: 128, 16: 128, 32: 96, 64: 32}
t_max = 2.0e-9
JFAC = np.round(np.arange(0.70, 1.051, 0.025), 4)           # FINE, switching window

def dt_for(dxy):
    return min(1e-13, 5e-14*(dxy/2.5e-9)**2)

def cfg():
    c = mm.BatchedLLGConfig()
    c.Ms, c.alpha, c.A, c.K1 = Ms, alpha, A, Ku
    c.easy = mm.Vec3(0,0,1); c.p = mm.Vec3(0,0,1)
    c.d_free, c.P, c.Lambda, c.beta = tz, P, 1.0, 0.0
    return c

CACHE = HERE / "32d_psw_fine_cache.json"
data = json.loads(CACHE.read_text()) if CACHE.exists() else {}

for n in MESHES:
    key = str(n)
    if key in data and len(data[key]["P"]) == len(JFAC):
        print(f"[{n}x{n}] cached"); continue
    dxy = L/n; dt = dt_for(dxy); nstep = int(t_max/dt); Ntr = N_TRIALS[n]
    grid = mm.StructuredGrid(n, n, 1, dxy, dxy, tz); Jc0 = JC0[n]
    Jv, Pv, Nswv, mz_all = [], [], [], []; t0 = time.time()
    for f in JFAC:
        J = f*Jc0
        b = mm.BatchedLLGGPU(Ntr, grid, cfg(), dt, seed=2024); b.enable_demag()
        b.set_J([J]*Ntr); b.set_T([300.0]*Ntr); b.set_uniform(0,0,1)
        b.run(nstep)
        mz = np.array(b.get_avg_m()).reshape(Ntr,3)[:,2]
        Jv.append(J); Nswv.append(int((mz < -0.5).sum())); Pv.append(float((mz < -0.5).mean()))
        mz_all.append(mz)                             # per-replica final <mz> (raw)
    # aggregated (curve-reconstruction) data + explicit switch counts
    data[key] = dict(dxy=dxy, Jc0=Jc0, dt=dt, N=Ntr, J=Jv, P=Pv, n_sw=Nswv)
    CACHE.write_text(json.dumps(data))
    # RAW per-replica final mz, one .npz per mesh (nJ x N_trials), for later reuse
    np.savez_compressed(HERE / f"32d_raw_mesh{n}.npz",
                        J=np.array(Jv), mz=np.array(mz_all), Jc0=Jc0, dt=dt,
                        dxy=dxy, N=Ntr, JFAC=JFAC)
    print(f"[{n}x{n}] {dxy*1e9:.2f} nm  {len(JFAC)} fine pts  ({time.time()-t0:.0f}s)  "
          f"-> 32d_raw_mesh{n}.npz")

# ---------------------------------------------------------------------------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
colors = {8:'C0',16:'C1',32:'C2',64:'C3'}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for n in MESHES:
    r = data[str(n)]; J = np.array(r["J"]); Psw = np.array(r["P"]); N = r["N"]
    err = np.sqrt(Psw*(1-Psw)/N); lab = f'{n}$\\times${n} ({r["dxy"]*1e9:.2f} nm)'
    a1.errorbar(J/1e12, Psw, yerr=err, fmt='o-', color=colors[n], ms=3.5, lw=1.2, capsize=1.5, label=lab)
    a2.errorbar(J/r["Jc0"], Psw, yerr=err, fmt='o-', color=colors[n], ms=3.5, lw=1.2, capsize=1.5, label=lab)
for ax in (a1, a2):
    ax.axhline(0.5, color='0.6', ls=':', lw=0.8); ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_ylabel(r'$P_\mathrm{sw}$ (2\,ns, 300\,K)'); ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
a1.set_xlabel(r'$J$ (10$^{12}$ A/m$^2$)'); a1.set_title('(a) absolute current (fine)'); a1.legend(fontsize=8, title='mesh (cell)')
a2.set_xlabel(r'$J/J_{c0}^\mathrm{own}$'); a2.set_title('(b) normalised by own $J_{c0}$'); a2.axvline(1.0, color='C3', ls='--', lw=1, alpha=0.5)
fig.suptitle(f'MTJ switching transition — fine-J sampling ({len(JFAC)} pts/mesh in $J/J_{{c0}}\\in[0.70,1.05]$)\n'
             f'80$\\times$80$\\times$1.5 nm CoFeB PMA; $l_\\mathrm{{ex}}$={l_ex*1e9:.2f} nm', fontsize=10)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(HERE / "32d_psw_switching_fine.png", dpi=150)
print("wrote 32d_psw_switching_fine.png")

# tabular raw export (human-readable, spreadsheet-friendly)
import csv
with open(HERE / "32d_psw_fine.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["mesh_n", "cell_nm", "dt_s", "Jc0_A_per_m2", "J_A_per_m2",
                "J_over_Jc0", "n_switched", "N_trials", "P_sw"])
    for n in MESHES:
        r = data[str(n)]
        nsw = r.get("n_sw", [int(round(p*r["N"])) for p in r["P"]])
        for J, p, ns in zip(r["J"], r["P"], nsw):
            w.writerow([n, round(r["dxy"]*1e9, 3), r["dt"], r["Jc0"], J,
                        round(J/r["Jc0"], 4), ns, r["N"], p])
print("wrote 32d_psw_fine.csv")

for n in MESHES:
    r = data[str(n)]; Jn = np.array(r["J"])/r["Jc0"]; Psw = np.array(r["P"])
    print(f"  {n:2d}x{n:<2d} ({r['dxy']*1e9:5.2f} nm): J/Jc0(P=0.5)={Jn[np.argmin(np.abs(Psw-0.5))]:.3f}")
