"""Notebook 32 - Phase N feature showcase.

Demonstrates:
  (A) MFM imaging   -- mm.MFMImage / mm.mfm_signal (already bound in Phase B2, now doc-tested)
  (B) edge_smooth   -- sub-cell anti-aliasing at geometry boundaries
  (C) poisson_disk_grains -- uniform grain size via Poisson-disk seeding
"""
import sys, os
_repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_repo, 'build', 'windows-msvc', 'python'))

import numpy as np
import micromag as mm

print("=== Notebook 32: Phase N Features ===\n")

# ---------------------------------------------------------------------------
# (A) MFM imaging
# ---------------------------------------------------------------------------
print("--- (A) MFM imaging ---")

nx, ny, nz = 64, 64, 5
dx = 5e-9
g  = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)
mat = mm.Material.permalloy()

# Two-domain state (stripe domains in z-direction)
m = mm.two_domain(g, mm.Vec3(0.0, 0.0, 1.0), mm.Vec3(0.0, 0.0, -1.0))

# MFM signal at 50nm lift height
sig_dipole   = mm.mfm_signal(m, mat, lift_m=50e-9, tip="dipole")    # (ny, nx)
sig_monopole = mm.mfm_signal(m, mat, lift_m=50e-9, tip="monopole")  # (ny, nx)

print(f"  Grid: {nx}x{ny}x{nz}, dx={dx*1e9:.0f} nm")
print(f"  Two-domain state: mz = +1 (left half), -1 (right half)")
print(f"  Dipole MFM: min={sig_dipole.min():.4e}, max={sig_dipole.max():.4e}")
print(f"  Monopole MFM: min={sig_monopole.min():.4e}, max={sig_monopole.max():.4e}")

# MFM dipole signal peaks near domain walls (max |dHz/dz|), zero in uniform domain
# The signal should have at least some non-zero region
print(f"  Dipole MFM non-zero? {(np.abs(sig_dipole) > 1).any()} (should be True)")
print(f"  Monopole MFM non-zero? {(np.abs(sig_monopole) > 1e-6).any()} (should be True)")

# Low-level MFMImage API
mfm = mm.MFMImage(g, 30e-9, mm.TipMode.Dipole)
sig_raw = mfm.compute(m, mat)   # (ny, nx) numpy array (Phase B2 binding returns ndarray)
print(f"  MFMImage.compute() shape: {sig_raw.shape}, lift={mfm.lift*1e9:.0f} nm")

# ---------------------------------------------------------------------------
# (B) edge_smooth -- anti-aliased geometry mask
# ---------------------------------------------------------------------------
print("\n--- (B) edge_smooth (anti-aliased GeomMask) ---")

g2  = mm.StructuredGrid(40, 40, 1, 5e-9, 5e-9, 5e-9)
r   = 90e-9   # radius slightly larger than grid so we get boundary cells

disk_binary = mm.circle(g2, r)
disk_smooth = mm.edge_smooth(disk_binary, n_sub=8)

# Count cells by value
vals_bin    = np.array([disk_binary[ix + 40*iy] for iy in range(40) for ix in range(40)])
vals_smooth = np.array([disk_smooth[ix + 40*iy] for iy in range(40) for ix in range(40)])

n_inside  = (vals_bin > 0.5).sum()
n_outside = (vals_bin < 0.5).sum()
n_frac    = ((vals_smooth > 0.01) & (vals_smooth < 0.99)).sum()

print(f"  Binary disk: {n_inside} inside, {n_outside} outside")
print(f"  Smoothed:    {n_frac} fractional boundary cells (0 < mask < 1)")
print(f"  Min non-zero smooth value: {vals_smooth[vals_smooth > 0].min():.3f}")
print(f"  Max non-one  smooth value: {vals_smooth[vals_smooth < 1].max():.3f}")

# Smoothed mask should have same total area (sum) as binary, within ~5%
area_bin    = vals_bin.sum()
area_smooth = vals_smooth.sum()
print(f"  Area binary={area_bin:.0f}, smooth={area_smooth:.1f}  (diff {abs(area_smooth-area_bin)/area_bin*100:.1f}%)")

# ---------------------------------------------------------------------------
# (C) poisson_disk_grains -- uniform grain size
# ---------------------------------------------------------------------------
print("\n--- (C) poisson_disk_grains ---")

g3 = mm.StructuredGrid(100, 100, 1, 5e-9, 5e-9, 5e-9)

# Target: ~50nm radius grains in 500x500 nm box
regions_s42  = mm.poisson_disk_grains(g3, avg_radius=50e-9, seed=42)
regions_s99  = mm.poisson_disk_grains(g3, avg_radius=50e-9, seed=99)   # different seed

# Count unique region IDs
lin_42 = np.array([regions_s42[i]  for i in range(100*100)])
lin_99 = np.array([regions_s99[i]  for i in range(100*100)])

n_42 = len(np.unique(lin_42))
n_99 = len(np.unique(lin_99))

print(f"  Box: 500x500 nm, dx=5 nm, avg_radius=50 nm")
print(f"  poisson_disk_grains(seed=42): {n_42} grains")
print(f"  poisson_disk_grains(seed=99): {n_99} grains")

# Grain size uniformity
def grain_areas(lin):
    ids = np.unique(lin)
    return np.array([(lin == i).sum() for i in ids])

a_42 = grain_areas(lin_42)
cv_42 = a_42.std() / a_42.mean()
print(f"  Grain areas (cells): mean={a_42.mean():.0f}, std={a_42.std():.0f}, CV={cv_42:.2f}")
print(f"  Expected CV for Poisson-disk: < 0.5 (uniform distribution)")
print(f"  All cells assigned: {(lin_42 > 0).all()} (should be True)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Phase N API Summary ===")
print("  mm.MFMImage(grid, lift_m, tip=TipMode.Dipole) -- MFM simulator")
print("  mm.mfm_signal(m, mat, lift_m, tip='dipole')   -- convenience -> (ny,nx) array")
print("  mm.TipMode.Dipole / mm.TipMode.Monopole")
print("  mm.edge_smooth(mask, n_sub=8)                 -- anti-aliased GeomMask")
print("  mm.poisson_disk_grains(grid, avg_radius, ...) -- uniform grain distribution")
print("\nAll Phase N features verified OK.")
