"""
Notebook 32c: P_sw vs J/Jc0 (own-threshold-normalised) by cell size.

Redraws the notebook-32b dense sweep with each curve normalised by its OWN
T=0 threshold J_c0, isolating the transition SHAPE (thermal/nucleation width)
from the absolute-threshold shift. Reads 32b_psw_cache.json (no recompute).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
data = json.loads((HERE / "32b_psw_cache.json").read_text())
mu0, Ms, A = 4e-7*np.pi, 1.0e6, 1.5e-11
l_ex = np.sqrt(2*A/(mu0*Ms**2))
MESHES = [8, 16, 32, 64]; colors = {8:'C0',16:'C1',32:'C2',64:'C3'}

fig, ax = plt.subplots(figsize=(7.0, 5.0))
for n in MESHES:
    r = data[str(n)]
    Jn = np.array(r["J"])/r["Jc0"]; Psw = np.array(r["P"]); N = r["N"]
    ax.errorbar(Jn, Psw, yerr=np.sqrt(Psw*(1-Psw)/N), fmt='o-', color=colors[n],
                ms=4, lw=1.4, capsize=2,
                label=f'{n}x{n}  ({r["dxy"]*1e9:.2f} nm, $J_{{c0}}$={r["Jc0"]/1e12:.2f}e12)')
ax.axhline(0.5, color='0.6', ls=':', lw=0.8)
ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.6)
ax.axvline(1.0, color='C3', ls='--', lw=1, alpha=0.5, label=r'own $J_{c0}$ (T=0)')
ax.set_xlabel(r'$J / J_{c0}^\mathrm{own}$')
ax.set_ylabel(r'$P_\mathrm{sw}$ (2 ns, 300 K)')
ax.set_title('MTJ switching vs current, normalised by own $J_{c0}$, by cell size\n'
             f'80x80x1.5 nm CoFeB PMA (fixed structure); $l_\\mathrm{{ex}}$={l_ex*1e9:.2f} nm')
ax.set_ylim(-0.05, 1.08); ax.legend(title='mesh (cell, $J_{c0}$)', fontsize=8.5)
ax.grid(alpha=0.3); fig.tight_layout()
out = HERE / "32c_psw_vs_jnorm_by_cellsize.png"
fig.savefig(out, dpi=140); print("Plot saved:", out.name)

for n in MESHES:
    r = data[str(n)]; Jn = np.array(r["J"])/r["Jc0"]; Psw = np.array(r["P"])
    print(f"  {n:2d}x{n:<2d} ({r['dxy']*1e9:5.2f} nm): "
          f"J/Jc0(P=0.5)={Jn[np.argmin(np.abs(Psw-0.5))]:.2f}  "
          f"J/Jc0(P=1)={(Jn[np.argmax(Psw>=0.999)] if np.any(Psw>=0.999) else float('nan')):.2f}")
