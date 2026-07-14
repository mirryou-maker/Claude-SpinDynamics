"""Notebook 16 -- Spinning Hard Disk (Granular PMA Media + Moving Write Head)

Demonstrates Phase D+C1 integration:
  - Granular PMA recording medium: voronoi_grains with per-cell Ku variation
  - Moving write head: Gaussian ZeemanFieldSpatial swept along the track
  - Bit writing and readback: mz > 0 -> '1', mz < 0 -> '0'

Physical setup:
  Track: 500 nm x 100 nm x 5 nm (100x20x1 cells, dx = 5 nm)
  PMA medium: Co-like Ku = 4e5 J/m3, Ms = 1.4 MA/m, easy_axis = z-hat
  Write head: Gaussian field sigma = 15 nm, H_write = +/-5e5 A/m swept at v = 100 m/s
"""

import sys, os
import os
from pathlib import Path

def _add_micromag_to_path():
    """Locate the micromag module + its DLLs in a release package
    (../runtime-dll + ../<variant>/python for GPU, ../python for CPU) or a source
    build (../build/<preset>/python + system CUDA). Works from any working dir."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"
    if rtd.is_dir():
        os.add_dll_directory(str(rtd))
        for _v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            _py = root / _v / "python"
            if list(_py.glob("_micromag*.pyd")):
                sys.path.insert(0, str(_py)); return
    if list((root / "python").glob("_micromag*.pyd")):
        os.add_dll_directory(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _cuda = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _p in ("windows-msvc-cuda", "windows-msvc"):
        _py = root / "build" / _p / "python"
        if _py.is_dir():
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
# A. Granular PMA recording medium
# ===========================================================================
print("=== A. Granular PMA Medium ===")

nx, ny, nz = 100, 20, 1
dx = 5e-9                  # 5 nm cells  =>  500 nm x 100 nm x 5 nm
g  = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)

# Base material: Co-like PMA
base_mat = mm.Material()
base_mat.Ms         = 1.4e6       # A/m
base_mat.A_exchange = 30e-12      # J/m
base_mat.K_uniaxial = 4e5         # J/m3  (PMA)
base_mat.easy_axis  = mm.Vec3(0, 0, 1)
base_mat.alpha      = 0.5

# Granular structure: 25 grains, 10% Ku variation
matf = mm.voronoi_grains(g, n_grains=25,
                          base=base_mat,
                          sigma_K=4e4,
                          seed=42)

print(f"  Grid: {nx}x{ny}x{nz}, dx = {dx*1e9:.0f} nm  (500x100x5 nm)")
print(f"  25 Co-like grains, Ku = 400 kJ/m3 +- 10%")

# Easy-axis z-component as a proxy for grain identity (varies grain-to-grain)
arr_ea    = mm.to_numpy(matf.easy_axis_field)    # (nz, ny, nx, 3)
grain_viz = arr_ea[0, :, :, 2]                   # easy_axis.z per cell  (ny, nx)

# ===========================================================================
# B. Initial state: random PMA domains after relax
# ===========================================================================
print("\n=== B. Initial Demagnetised State ===")

m = mm.random_mag(g, seed=123)

aniso = mm.UniaxialAnisotropyField()
aniso.set_material_field(matf)
exch  = mm.ExchangeField(mm.BoundaryCondition.Neumann)
exch.set_material_field(matf)

heff_relax = mm.EffectiveFieldSum()
heff_relax.add(aniso)
heff_relax.add(exch)

n_steps = mm.relax(m, base_mat, heff_relax,
                   mm.RelaxOptions(threshold=5.0, max_steps=200_000),
                   matf)
print(f"  Relaxed in {n_steps} steps")
arr_init = mm.to_numpy(m)
mz_init  = arr_init[0, :, :, 2]   # (ny, nx)
print(f"  <mz> = {mz_init.mean():.3f}")

# ===========================================================================
# C. Write head sweep: write 5 alternating bits (1 0 1 0 1)
# ===========================================================================
print("\n=== C. Write Head Sweep ===")

sigma_head = 15e-9        # Gaussian write-head width [m]
H_write    = 5e5          # A/m (well above Co coercivity)
v_head     = 100.0        # m/s
bit_period = 100e-9       # 100 nm per bit -> 5 bits on 500 nm track

H_head_field = mm.VectorField3D(g)
zeeman_head  = mm.ZeemanFieldSpatial(H_head_field)

heff_write = mm.EffectiveFieldSum()
heff_write.add(aniso)
heff_write.add(exch)
heff_write.add(zeeman_head)

integ = mm.RK4Integrator(5e-14)   # 50 fs

# Pre-compute head-field arrays for positive and negative polarity
arr_H_pos = np.zeros((nz, ny, nx, 3))
arr_H_neg = np.zeros((nz, ny, nx, 3))
xc_cells  = (np.arange(nx) + 0.5) * dx - nx * dx / 2   # box-centred x [m]
yc_cells  = (np.arange(ny) + 0.5) * dx - ny * dx / 2   # box-centred y [m]
XX, YY    = np.meshgrid(xc_cells, yc_cells)              # (ny, nx)

def update_head_field(x_head: float, polarity: float):
    r2 = (XX - x_head)**2 + YY**2
    hz = polarity * H_write * np.exp(-r2 / (2 * sigma_head**2))
    arr_H = np.zeros((nz, ny, nx, 3))
    arr_H[0, :, :, 2] = hz
    mm.from_numpy(H_head_field, arr_H)

t_write    = (nx * dx) / v_head     # time to cross full track
n_bits     = int(nx * dx / bit_period)
dt_cb      = 20e-12

def head_callback(t: float, m_field):
    x_head  = v_head * t - nx * dx / 2
    bit_idx = int((x_head + nx * dx / 2) / bit_period)
    pol     = +1.0 if (bit_idx % 2 == 0) else -1.0
    update_head_field(x_head, pol)

print(f"  Sweep: v={v_head:.0f} m/s, t_write={t_write*1e9:.1f} ns, {n_bits} bits")
print("  Writing ...")
mm.run(integ, m, base_mat, heff_write, t_total=t_write,
       callback=head_callback, callback_dt=dt_cb)

# Zero field after sweep + short relax
mm.from_numpy(H_head_field, np.zeros((nz, ny, nx, 3)))
mm.run(integ, m, base_mat, heff_write, t_total=0.2e-9)

arr_after = mm.to_numpy(m)
mz_after  = arr_after[0, :, :, 2]
print(f"  <mz> after write = {mz_after.mean():.3f}")

# Decode bits
bit_x_m = (np.arange(n_bits) + 0.5) * bit_period - nx * dx / 2
bit_mz  = []
for b in range(n_bits):
    ix0 = int(b * bit_period / dx)
    ix1 = min(int((b + 1) * bit_period / dx), nx)
    bit_mz.append(mz_after[:, ix0:ix1].mean())

bits_read   = ["1" if v > 0 else "0" for v in bit_mz]
bits_expect = ["1" if b % 2 == 0 else "0" for b in range(n_bits)]
n_correct   = sum(a == b for a, b in zip(bits_read, bits_expect))
print(f"\n  Expected:  {' '.join(bits_expect)}")
print(f"  Readback:  {' '.join(bits_read)}   ({n_correct}/{n_bits} correct)")

# ===========================================================================
# D. Thermal stability at 300 K (0.5 ns)
# ===========================================================================
print("\n=== D. Thermal Stability (300 K, 0.5 ns) ===")

heff_T = mm.EffectiveFieldSum()
heff_T.add(aniso)
heff_T.add(exch)

mm.thermalize(m, base_mat, heff_T, T_K=300.0, t_therm=0.5e-9, dt=5e-14, seed=77)
arr_therm = mm.to_numpy(m)
mz_therm  = arr_therm[0, :, :, 2]
print(f"  After 300 K / 0.5 ns:  <mz> = {mz_therm.mean():.3f}")

dw_x = mm.domain_wall_pos(m, component=2, threshold=0.0, axis=0)
print(f"  First domain wall at x = {dw_x*1e9:.1f} nm"
      f"  (bit 0->1 transition expected at x ~ {bit_period*1e9/2:.0f} nm from centre-left)")

# ===========================================================================
# E. Plotting
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

x_nm = xc_cells * 1e9
y_nm = yc_cells * 1e9
ext  = [x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]]

kw = dict(origin="lower", extent=ext, aspect="auto")

ax = axes[0, 0]
im = ax.imshow(grain_viz, cmap="twilight_shifted", vmin=-1, vmax=1, **kw)
ax.set_title("Grain structure (easy_axis.z)")
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(mz_init, cmap="RdBu", vmin=-1, vmax=1, **kw)
ax.set_title("Initial demagnetised state (mz)")
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
plt.colorbar(im, ax=ax)

ax = axes[0, 2]
im = ax.imshow(mz_after, cmap="RdBu", vmin=-1, vmax=1, **kw)
ax.set_title("After write sweep (5 alternating bits)")
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
plt.colorbar(im, ax=ax)

ax = axes[1, 0]
im = ax.imshow(mz_therm, cmap="RdBu", vmin=-1, vmax=1, **kw)
ax.set_title("After 300 K thermalization (0.5 ns)")
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
ax.plot(x_nm, mz_init.mean(axis=0), "gray", lw=1, label="initial", alpha=0.6)
ax.plot(x_nm, mz_after.mean(axis=0), "royalblue", lw=1.5, label="after write")
ax.plot(x_nm, mz_therm.mean(axis=0), "tomato", lw=1, ls="--", label="after 300 K")
for b in range(n_bits):
    xb = (bit_x_m[b] + bit_period / 2) * 1e9
    ax.axvline(xb, color="k", ls=":", lw=0.8, alpha=0.4)
ax.set_xlabel("x (nm)"); ax.set_ylabel("<mz>"); ax.set_ylim(-1.2, 1.2)
ax.set_title("Track profile (y-averaged)")
ax.legend(fontsize=8)

ax = axes[1, 2]
colors = ["steelblue" if v > 0 else "salmon" for v in bit_mz]
ax.bar(range(n_bits), bit_mz, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("bit index"); ax.set_ylabel("<mz>")
ax.set_title(f"Bit readback: {n_correct}/{n_bits} correct")
ax.set_xticks(range(n_bits))
ax.set_xticklabels([f"{b}\n(exp. {e})" for b, e in zip(bits_read, bits_expect)],
                   fontsize=8)

plt.suptitle("Spinning Hard Disk (Phase D+C1): Granular PMA Media + Write Head Sweep",
             fontsize=12)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "hard_disk.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {out}")
print("\n=== Done ===")
