"""
12_energy_landscape.py — Edens_* full energy landscape (all field terms)

Physics
-------
- SP#4 element: 500×125×3 nm Permalloy, 5 nm cells
- S-state initial condition (uniform mx)
- After relaxation: visualise per-cell energy density for each field term
- Demonstrates mumax3 Edens_* analogs

Output
------
- energy_landscape_sp4.png  (4-panel: Edens_exch, Edens_demag, Edens_zee, Edens_total)
- table_energy.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc', 'python'))

import micromag as mm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# SP#4 grid
# ---------------------------------------------------------------------------
Lx, Ly, Lz = 500e-9, 125e-9, 3e-9
dx = 5e-9
nx, ny, nz = int(Lx/dx), int(Ly/dx), 1
g = mm.StructuredGrid(nx, ny, nz, dx, dx, Lz)

mat = mm.Material.permalloy()
mat.alpha = 0.5   # high damping for fast relax

print(f"Grid: {nx}×{ny}×{nz} cells, dx={dx*1e9:.0f} nm")

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
demag  = mm.DemagField(g)
exch   = mm.ExchangeField(mm.BoundaryCondition.Neumann)
zeeman = mm.ZeemanField(mm.Vec3(-24600, 0, 0))   # µMAG field A

heff = mm.EffectiveFieldSum()
heff.add(demag)
heff.add(exch)
heff.add(zeeman)

# ---------------------------------------------------------------------------
# Relax from uniform state (S-state)
# ---------------------------------------------------------------------------
print("Relaxing SP#4 element...")
m = mm.uniform_mag(g, mm.Vec3(1, 0, 0))
opts = mm.RelaxOptions()
opts.tol = 1e-7
mm.relax(m, mat, heff, opts)

mx_avg, my_avg, mz_avg = mm.mean_magnetization(m)
print(f"  <mx,my,mz> = ({mx_avg:.4f}, {my_avg:.4f}, {mz_avg:.4f})")

# ---------------------------------------------------------------------------
# Per-cell energy densities (mumax3 Edens_* analogs)
# ---------------------------------------------------------------------------
ed_exch  = exch.energy_density(m, mat)
ed_demag = demag.energy_density(m, mat)
ed_zee   = zeeman.energy_density(m, mat)
ed_total = heff.energy_density(m, mat)

def sf_to_np(sf, ny, nx):
    arr = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            arr[iy, ix] = sf[g.linear_index(ix, iy, 0)]
    return arr

Ed_exch  = sf_to_np(ed_exch,  ny, nx)
Ed_demag = sf_to_np(ed_demag, ny, nx)
Ed_zee   = sf_to_np(ed_zee,   ny, nx)
Ed_total = sf_to_np(ed_total, ny, nx)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes = axes.ravel()
extent = [0, Lx*1e9, 0, Ly*1e9]

panels = [
    (Ed_exch  * 1e-3, "Edens_exch (kJ/m³)",  'hot'),
    (Ed_demag * 1e-3, "Edens_demag (kJ/m³)", 'viridis'),
    (Ed_zee   * 1e-3, "Edens_zee (kJ/m³)",   'RdBu'),
    (Ed_total * 1e-3, "Edens_total (kJ/m³)", 'plasma'),
]

for ax, (data, title, cmap) in zip(axes, panels):
    im = ax.imshow(data, origin='lower', cmap=cmap, extent=extent, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    plt.colorbar(im, ax=ax)

plt.suptitle("SP#4 Permalloy — per-cell energy density (mumax3 Edens_*)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "energy_landscape_sp4.png"), dpi=150)
print("  Saved energy_landscape_sp4.png")
plt.close()

# ---------------------------------------------------------------------------
# Table: energy breakdown
# ---------------------------------------------------------------------------
E_exch  = exch.energy(m, mat)
E_demag = demag.energy(m, mat)
E_zee   = zeeman.energy(m, mat)
E_total = heff.total_energy(m, mat)
print(f"\nEnergy breakdown:")
print(f"  E_exch  = {E_exch*1e18:.2f} aJ")
print(f"  E_demag = {E_demag*1e18:.2f} aJ")
print(f"  E_zee   = {E_zee*1e18:.2f} aJ")
print(f"  E_total = {E_total*1e18:.2f} aJ")

# Save Table
tbl = mm.Table()
tbl.add_row(0.0, m, mat=mat, heff=heff,
            extra={"E_exch_aJ": E_exch*1e18,
                   "E_demag_aJ": E_demag*1e18,
                   "E_zee_aJ": E_zee*1e18})
tbl.save(os.path.join(out_dir, "table_energy.csv"))
print("  Saved table_energy.csv")
