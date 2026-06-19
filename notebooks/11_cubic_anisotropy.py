"""
11_cubic_anisotropy.py — Fe cubic anisotropy domain patterns

Physics
-------
- 100×100×1 Fe film, 5 nm cells
- CubicAnisotropyField: Kc1=+48e3 J/m³ (Fe easy-axes along <100>)
- Random initial state → relax → domain pattern
- Energy density map showing Edens_* per term

Output
------
- cubic_domains_mx.png   (mx domain map)
- cubic_energy_map.png   (per-cell energy density)
- table_cubic.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc', 'python'))

import micromag as mm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Grid + material (Fe)
# ---------------------------------------------------------------------------
Lx, Ly, Lz = 100e-9, 100e-9, 5e-9
nx, ny      = 20, 20
g = mm.StructuredGrid(nx, ny, 1, Lx/nx, Ly/ny, Lz)

mat = mm.Material()
mat.Ms         = 1.71e6      # Fe saturation magnetization
mat.A_exchange = 2.1e-11     # Fe exchange stiffness
mat.K_uniaxial = 0.0
mat.easy_axis  = mm.Vec3(0, 0, 1)
mat.alpha      = 0.5         # high damping for fast relax

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
demag = mm.DemagField(g)
exch  = mm.ExchangeField(mm.BoundaryCondition.Neumann)
cubic = mm.CubicAnisotropyField(Kc1=48e3, Kc2=0.0,
                                 c1=mm.Vec3(1,0,0), c2=mm.Vec3(0,1,0))

heff = mm.EffectiveFieldSum()
heff.add(demag)
heff.add(exch)
heff.add(cubic)

# ---------------------------------------------------------------------------
# Relax from random state
# ---------------------------------------------------------------------------
print("Relaxing Fe film (cubic anisotropy Kc1=48 kJ/m³)...")
m = mm.random_mag(g, seed=42)
opts = mm.RelaxOptions()
opts.tol = 1e-6
mm.relax(m, mat, heff, opts)

# ---------------------------------------------------------------------------
# Domain pattern
# ---------------------------------------------------------------------------
m_np = mm.to_numpy(m)
mx = m_np[0, :, :, 0]  # shape (ny, nx)
my = m_np[0, :, :, 1]
mz = m_np[0, :, :, 2]

mx_avg, my_avg, mz_avg = mm.mean_magnetization(m)
print(f"  <mx,my,mz> = ({mx_avg:.3f}, {my_avg:.3f}, {mz_avg:.3f})")

# Energy densities
ed_exch  = exch.energy_density(m, mat)
ed_cubic = cubic.energy_density(m, mat)
ed_demag = demag.energy_density(m, mat)

import math
def field_to_np(sf):
    arr = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            arr[iy, ix] = sf[g.linear_index(ix, iy, 0)]
    return arr

ed_exch_np  = field_to_np(ed_exch)
ed_cubic_np = field_to_np(ed_cubic)
ed_demag_np = field_to_np(ed_demag)

# ---------------------------------------------------------------------------
# Plot 1: domain pattern (mx)
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

im0 = axes[0].imshow(mx, origin='lower', cmap='RdBu', vmin=-1, vmax=1,
                     extent=[0, Lx*1e9, 0, Ly*1e9])
axes[0].set_title("mx (after relax)")
axes[0].set_xlabel("x (nm)"); axes[0].set_ylabel("y (nm)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(my, origin='lower', cmap='RdBu', vmin=-1, vmax=1,
                     extent=[0, Lx*1e9, 0, Ly*1e9])
axes[1].set_title("my")
axes[1].set_xlabel("x (nm)")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(mz, origin='lower', cmap='RdBu', vmin=-1, vmax=1,
                     extent=[0, Lx*1e9, 0, Ly*1e9])
axes[2].set_title("mz")
axes[2].set_xlabel("x (nm)")
plt.colorbar(im2, ax=axes[2])

plt.suptitle("Fe film: cubic anisotropy Kc1=48 kJ/m³ domain relaxation")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cubic_domains_mx.png"), dpi=150)
print("  Saved cubic_domains_mx.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: energy density map
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

vmax_ex = max(abs(ed_exch_np.min()), ed_exch_np.max())
im0 = axes[0].imshow(ed_exch_np*1e-3, origin='lower', cmap='hot',
                     extent=[0, Lx*1e9, 0, Ly*1e9])
axes[0].set_title("Edens_exch (kJ/m³)")
axes[0].set_xlabel("x (nm)"); axes[0].set_ylabel("y (nm)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(ed_cubic_np*1e-3, origin='lower', cmap='RdBu',
                     extent=[0, Lx*1e9, 0, Ly*1e9])
axes[1].set_title("Edens_cubic (kJ/m³)")
axes[1].set_xlabel("x (nm)")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(ed_demag_np*1e-3, origin='lower', cmap='viridis',
                     extent=[0, Lx*1e9, 0, Ly*1e9])
axes[2].set_title("Edens_demag (kJ/m³)")
axes[2].set_xlabel("x (nm)")
plt.colorbar(im2, ax=axes[2])

plt.suptitle("Fe film: per-cell energy density maps (Edens_*)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cubic_energy_map.png"), dpi=150)
print("  Saved cubic_energy_map.png")
plt.close()

# Energy breakdown
E_exch  = exch.energy(m, mat)
E_cubic = cubic.energy(m, mat)
E_demag = demag.energy(m, mat)
print(f"  E_exch  = {E_exch*1e18:.2f} aJ")
print(f"  E_cubic = {E_cubic*1e18:.2f} aJ")
print(f"  E_demag = {E_demag*1e18:.2f} aJ")
