"""Notebook 14 — Inter-Region Exchange Coupling & Rotating Geometry

Demonstrates two Phase H features:

A. SetInterExchange (mumax3 analog):
   Two magnetic regions (Permalloy / Co) separated by a non-magnetic spacer.
   Exchange coupling across the boundary is controlled via set_inter_exchange().
   Shows: ferromagnetic coupling → both regions align; zero coupling → independent.

B. Rotating geometry (mumax3 "Rotating Cheese" analog):
   A circular PMA disk that rotates in time using mm.rotate(mask, theta).
   Demonstrates Phase D dynamic geometry without any new C++ code.
"""

import sys, os
import os
from pathlib import Path

def _add_micromag_to_path():
    """Locate the micromag module + its DLLs in a release package
    (../runtime-dll + ../<variant>/python for GPU, ../python for CPU) or a source
    build (../build/<preset>/python + system CUDA). Works from any working dir."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    def _adddll(_d):                      # add_dll_directory is Windows-only
        if hasattr(os, "add_dll_directory") and os.path.isdir(_d):
            os.add_dll_directory(_d)
    def _hasmod(_p):
        _pat = "_micromag*.pyd" if sys.platform == "win32" else "_micromag*.so"
        return bool(list(_p.glob(_pat)))
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"
    if rtd.is_dir():
        _adddll(str(rtd))
        for _v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            _py = root / _v / "python"
            if _hasmod(_py):
                sys.path.insert(0, str(_py)); return
    if _hasmod(root / "python"):
        _adddll(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _adddll(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
    for _p in ("windows-msvc-cuda", "windows-msvc", "linux-gcc-cuda", "linux-gcc"):
        _py = root / "build" / _p / "python"
        if _hasmod(_py):
            sys.path.insert(0, str(_py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import micromag as mm

mu_0 = 4 * np.pi * 1e-7

# ===========================================================================
# A. SetInterExchange — two-region bilayer
# ===========================================================================
print("=== A. Inter-Region Exchange Coupling ===")

# 20×1×1 nm chain: left half = region 0 (Permalloy), right half = region 1 (Co)
# dx = 2 nm  → 10 cells per region
Lx = 20e-9
dx = 2e-9
nx = int(Lx / dx)   # 10
g  = mm.StructuredGrid(nx, 1, 1, dx, dx, dx)

print(f"  Grid: {nx}x1x1, dx = {dx*1e9:.0f} nm  (10 cells Py | 10 cells Co)")

mat = mm.Material.permalloy()   # used for both regions (simplified)

# Region map: 0=left, 1=right
rmap = mm.RegionMap(g, 0)
for i in range(nx // 2, nx):
    rmap[i] = 1

# Initial state: region 0 → (1,0,0), region 1 → (0,1,0) [orthogonal]
# Use set_magnetization on the RegionMap to initialize each region
m = mm.uniform_mag(g, mm.Vec3(1, 0, 0))
rmap.set_magnetization(1, m, mm.Vec3(0, 1, 0))

# Helper: copy a VectorField3D via numpy
def copy_field(src):
    dst = mm.VectorField3D(src.grid)
    mm.from_numpy(dst, mm.to_numpy(src))
    return dst

# --- Case 1: ferromagnetic inter-exchange (same as Py A = 13e-12 J/m) ---
A_Py = 13e-12
exch_fm = mm.ExchangeField(mm.BoundaryCondition.Neumann)
exch_fm.set_region_map(rmap)
exch_fm.set_inter_exchange(0, 1, A_Py)   # strong FM coupling at boundary

heff_fm = mm.EffectiveFieldSum()
heff_fm.add(exch_fm)

integ_fm = mm.RK4Integrator(5e-14)
m_fm = copy_field(m)

print("  Relaxing with FM inter-exchange (should align both regions) ...")
mm.run(integ_fm, m_fm, mat, heff_fm, t_total=2e-9)
arr_fm = mm.to_numpy(m_fm)
mx_fm = arr_fm[0, 0, :, 0]
my_fm = arr_fm[0, 0, :, 1]
print(f"    Final: mx = {mx_fm.mean():.3f}, my = {my_fm.mean():.3f}")

# --- Case 2: zero inter-exchange (regions independent) ---
exch_zero = mm.ExchangeField(mm.BoundaryCondition.Neumann)
exch_zero.set_region_map(rmap)
exch_zero.set_inter_exchange(0, 1, 0.0)   # cut exchange at boundary

heff_zero = mm.EffectiveFieldSum()
heff_zero.add(exch_zero)

integ_zero = mm.RK4Integrator(5e-14)
m_zero = copy_field(m)

print("  Relaxing with A_IEC=0 (regions stay independent) ...")
mm.run(integ_zero, m_zero, mat, heff_zero, t_total=2e-9)
arr_zero = mm.to_numpy(m_zero)
mx_zero = arr_zero[0, 0, :, 0]
my_zero = arr_zero[0, 0, :, 1]
print(f"    Left  region: mx = {mx_zero[:nx//2].mean():.3f}")
print(f"    Right region: my = {my_zero[nx//2:].mean():.3f}")

# Plot A
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x_nm = (np.arange(nx) + 0.5) * dx * 1e9

ax = axes[0]
ax.plot(x_nm, mx_fm, "-o", ms=4, label="mx (FM coupling)")
ax.plot(x_nm, my_fm, "-s", ms=4, label="my (FM coupling)")
ax.axvline(x_nm[nx//2 - 1] + dx*1e9/2, color="k", ls="--", lw=1, label="region boundary")
ax.set_xlabel("x  (nm)")
ax.set_ylabel("m component")
ax.set_title("FM inter-exchange: both regions align")
ax.legend()
ax.set_ylim(-0.1, 1.1)

ax = axes[1]
ax.plot(x_nm, mx_zero, "-o", ms=4, label="mx (A_IEC=0)")
ax.plot(x_nm, my_zero, "-s", ms=4, label="my (A_IEC=0)")
ax.axvline(x_nm[nx//2 - 1] + dx*1e9/2, color="k", ls="--", lw=1, label="region boundary")
ax.set_xlabel("x  (nm)")
ax.set_ylabel("m component")
ax.set_title("A_IEC=0: regions stay independent")
ax.legend()
ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
out_a = os.path.join(os.path.dirname(__file__), "inter_exchange.png")
plt.savefig(out_a, dpi=150)
plt.close()
print(f"  Saved -> {out_a}")


# ===========================================================================
# B. Rotating geometry (Phase D — dynamic mask via mm.rotate)
# ===========================================================================
print("\n=== B. Rotating Geometry (Rotating Cheese) ===")

# 60×60×1 nm film, 3 nm cells
g2 = mm.StructuredGrid(20, 20, 1, 3e-9, 3e-9, 3e-9)
mat_b = mm.Material.permalloy()

# Base geometry: ellipse (15 nm × 8 nm semi-axes) centred on the box
a_el, b_el = 15e-9, 8e-9
base_mask = mm.ellipse(g2, a_el, b_el)

# Initial state: uniform m along x, masked to base_mask shape
m2 = mm.uniform_mag(g2, mm.Vec3(1, 0, 0))
mm.set_geom(base_mask, m2)

# Fields for the dynamic simulation
exch_b  = mm.ExchangeField(mm.BoundaryCondition.Neumann)
demag_b = mm.DemagField(g2)
heff_b  = mm.EffectiveFieldSum()
heff_b.add(exch_b)
heff_b.add(demag_b)
integ_b = mm.RK4Integrator(5e-13)

# Relax into ground state
print("  Initial relax ...")
mm.run(integ_b, m2, mat_b, heff_b, t_total=0.5e-9)
mm.set_geom(base_mask, m2)
print(f"    After relax: <mx> = {mm.mean_magnetization(m2)[0]:.3f}")

# Rotate mask by 0°, 30°, 60°, 90° — record mean mx
angles_deg = [0, 30, 60, 90]
mx_vs_angle = []
snapshots = []

for theta_deg in angles_deg:
    theta = theta_deg * np.pi / 180
    rotated_mask = mm.rotate(base_mask, theta)
    # Apply rotated mask to current m (zero outside rotated geometry)
    mm.set_geom(rotated_mask, m2)
    # Short relax with rotated mask
    exch_b.set_mask(rotated_mask)
    mm.run(integ_b, m2, mat_b, heff_b, t_total=0.2e-9)
    mx, my, mz = mm.mean_magnetization(m2)
    mx_vs_angle.append(mx)
    snap_arr = mm.to_numpy(m2)[0, :, :, 0]  # mx layer 0
    snapshots.append(snap_arr.copy())
    print(f"    theta = {theta_deg:3d}deg: <mx> = {mx:.3f}")

# Plot B
fig, axes = plt.subplots(1, len(angles_deg) + 1, figsize=(4 * (len(angles_deg) + 1), 4))

for k, (theta_deg, snap) in enumerate(zip(angles_deg, snapshots)):
    ax = axes[k]
    im = ax.imshow(snap, origin="lower", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_title(f"θ = {theta_deg}°")
    ax.axis("off")

ax = axes[-1]
ax.plot(angles_deg, mx_vs_angle, "o-", ms=7, lw=2)
ax.set_xlabel("Rotation angle  (deg)")
ax.set_ylabel("<mx>")
ax.set_title("Mean mx vs rotation")

plt.tight_layout()
out_b = os.path.join(os.path.dirname(__file__), "rotating_geometry.png")
plt.savefig(out_b, dpi=150)
plt.close()
print(f"  Saved -> {out_b}")

# Demonstrate snap()
snap_path = os.path.join(os.path.dirname(__file__), "snap_test")
os.makedirs(snap_path, exist_ok=True)
fname = mm.snap(m2, os.path.join(snap_path, "m"))
print(f"\n  snap() demo: saved {fname}")

print("\n=== Done ===")
print(f"  A. FM coupling: both regions aligned after relax")
print(f"  B. Rotating ellipse: mx changes from {mx_vs_angle[0]:.3f} @ 0deg "
      f"to {mx_vs_angle[-1]:.3f} @ 90°")
