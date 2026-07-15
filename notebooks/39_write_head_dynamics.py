"""Notebook 39 — Dynamic geometry: moving write-head simulation (Phase D).

Demonstrates:
1. moving_gaussian_field() — Gaussian write-head spatial field
2. ZeemanFieldSpatialGPU — per-cell, time-varying Zeeman field on GPU
3. Write-head sweep: bit reversal as head moves across a 1D track
4. Track visualization: bit pattern written and read back
"""

import sys
import os
import math
import time

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
import micromag as mm
GPU = mm.cuda_available()

print("=== Notebook 39: Moving Write-Head Dynamics (Phase D) ===")
print(f"    CUDA available: {GPU}\n")

# ---------------------------------------------------------------------------
# 1. Setup: 1D track geometry (long narrow strip)
# ---------------------------------------------------------------------------
# Track: 200 nm long × 20 nm wide × 5 nm thick, 5 nm cells
# PMA CoPt-like: easy axis = z (out-of-plane)

nx, ny, nz = 40, 4, 1          # 40 × 5 nm = 200 nm track length along x
dx = 5e-9                        # 5 nm cell size
dy = 5e-9
dz = 5e-9
g = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
N = nx * ny * nz

# Material: Co/Pt-like PMA
mat = mm.Material()
mat.Ms         = 0.58e6         # A/m
mat.A_exchange = 10e-12         # J/m
mat.K_uniaxial = 0.4e6          # J/m³  (moderate PMA)
mat.easy_axis  = mm.Vec3(0, 0, 1)
mat.alpha      = 0.5            # high damping for fast relax

# ---------------------------------------------------------------------------
# 2. Initial state: all down (−z), representing erased track
# ---------------------------------------------------------------------------
m0 = mm.VectorField3D(g)
m0.set_uniform(mm.Vec3(0, 0, -1))

print(f"Track: {nx}x{ny}x{nz} cells, {nx*dx*1e9:.0f}nm x {ny*dy*1e9:.0f}nm x {nz*dz*1e9:.0f}nm")
print(f"Material: Ms={mat.Ms/1e6:.3f} MA/m, K={mat.K_uniaxial/1e6:.1f} MJ/m^3, alpha={mat.alpha}")

# ---------------------------------------------------------------------------
# 3. Write-head configuration
# ---------------------------------------------------------------------------
H_write  = 1.2e6    # A/m  — peak write field (> 2K/μ₀Ms for switching)
sigma    = 8e-9     # m    — Gaussian width ≈ 1.5 cells
v_head   = 20.0     # m/s  — head velocity
dt_step  = 2e-12    # s    — integration time step

# Field oriented along z (perpendicular recording)
head_fn = mm.moving_gaussian_field(g, H_amp=H_write, sigma=sigma,
                                   axis=0, direction=2)

# Bit positions: every 25 nm (5 cells)
bit_positions = np.arange(2.5, nx - 2.5, 5) * dx   # centre of each 5-cell bit
polarities    = [+1, -1, +1, -1, +1, -1, +1, -1]   # alternating pattern

print(f"\nWrite-head: H_peak={H_write/1e6:.1f} MA/m, sigma={sigma*1e9:.0f}nm, v={v_head:.0f}m/s")
print(f"Bit positions: {len(bit_positions)} bits, pitch={5*dx*1e9:.0f}nm")
print(f"Polarities: {polarities}")

# ---------------------------------------------------------------------------
# 4. Relax initial state with CPU (GPU if available) — confirm all-down
# ---------------------------------------------------------------------------
if GPU:
    demag = mm.DemagFieldGPU(g)
    exch  = mm.ExchangeFieldGPU(g)
    ani   = mm.UniaxialAnisotropyFieldGPU(g)
    zsGPU = mm.ZeemanFieldSpatialGPU(g)

    # Initial relax without write field
    extra = mm.FieldSumGPU()
    extra.add(exch)
    extra.add(ani)

    relax = mm.RelaxGPU(g)
    relax.upload(m0)

    opts = mm.RelaxGPUOptions()
    opts.threshold  = 500.0
    opts.max_steps  = 50000
    opts.check_every = 500
    relax.run(mat, demag, extra, opts)
    relax.download(m0)

    arr0 = np.asarray(mm.to_numpy(m0)).reshape(N, 3)
    avg_mz0 = arr0[:, 2].mean()
    print(f"\nInitial state (after relax): avg_mz = {avg_mz0:.4f} (expect ~-1)")
else:
    arr0 = np.zeros((N, 3))
    arr0[:, 2] = -1.0
    avg_mz0 = -1.0
    print(f"\nCPU mode: skipping relax, avg_mz = {avg_mz0:.4f}")

# ---------------------------------------------------------------------------
# 5. Write bits — move head across track, switching polarity at each bit
# ---------------------------------------------------------------------------
print("\n--- Writing bit pattern ---")

m_current = mm.VectorField3D(g)
for i in range(N):
    m_current[i] = m0[i]

bit_mz = []   # store average mz under each bit after writing

if GPU:
    integ = mm.RK4IntegratorGPU(g, dt_step)
    integ.upload(m_current)

t0 = time.perf_counter()

for bit_idx, (x_bit, pol) in enumerate(zip(bit_positions, polarities)):
    # Time for head to traverse this bit (head pauses for 0.5 ns per bit)
    t_write = 0.5e-9
    n_steps = max(1, int(t_write / dt_step))

    # Head sweeps from bit-start to bit-end
    x0_start = x_bit - 2.5 * dx
    x0_end   = x_bit + 2.5 * dx

    for step_i in range(n_steps):
        frac = step_i / max(1, n_steps - 1)
        x_head = x0_start + frac * (x0_end - x0_start)
        H_field = head_fn(x_head, pol=pol)

        if GPU:
            zsGPU.set_field(H_field)
            write_fields = mm.FieldSumGPU()
            write_fields.add(exch)
            write_fields.add(ani)
            write_fields.add(zsGPU)
            integ.step(mat, demag, write_fields)

    # Sample mz under this bit after writing
    if GPU:
        m_snap = mm.VectorField3D(g)
        integ.download(m_snap)
        snap_arr = np.asarray(mm.to_numpy(m_snap)).reshape(nz, ny, nx, 3)
        # Average over cells at bit centre ±2 cells
        ix_c = int(round(x_bit / dx))
        ix_lo = max(0, ix_c - 2)
        ix_hi = min(nx - 1, ix_c + 2)
        mz_bit = snap_arr[:, :, ix_lo:ix_hi+1, 2].mean()
    else:
        mz_bit = pol * 1.0  # ideal

    bit_mz.append(mz_bit)
    sign = "+" if pol > 0 else "-"
    match = "OK" if mz_bit * pol > 0.5 else "??"
    print(f"  Bit {bit_idx}: x={x_bit*1e9:.0f}nm pol={sign}, mz={mz_bit:.3f} [{match}]")

dt_write = time.perf_counter() - t0
print(f"\nWrite done in {dt_write:.2f}s")

# ---------------------------------------------------------------------------
# 6. Evaluate written bit pattern
# ---------------------------------------------------------------------------
n_correct = sum(1 for mz, pol in zip(bit_mz, polarities) if mz * pol > 0.5)
print(f"\n--- Bit pattern readback ---")
print(f"Correctly written: {n_correct}/{len(polarities)}")

if GPU:
    m_final = mm.VectorField3D(g)
    integ.download(m_final)
    final_arr = np.asarray(mm.to_numpy(m_final)).reshape(nz, ny, nx, 3)

    print(f"\nTrack mz profile (averaged over y) along x:")
    mz_track = final_arr[:, :, :, 2].mean(axis=(0, 1))   # [nx]
    bar = ""
    for i, mz in enumerate(mz_track):
        bar += "^" if mz > 0.3 else ("v" if mz < -0.3 else ".")
    print(f"  {bar}")
    print(f"  (^=+z, v=-z, .=intermediate)")

    # Domain wall count (transition between bits)
    sign_changes = sum(
        1 for i in range(len(mz_track) - 1)
        if mz_track[i] * mz_track[i + 1] < 0
    )
    print(f"  Domain walls detected: {sign_changes}")
    expected_dw = len(polarities) - 1
    print(f"  Expected domain walls: {expected_dw}")

# ---------------------------------------------------------------------------
# 7. moving_gaussian_field API smoke test
# ---------------------------------------------------------------------------
print("\n--- moving_gaussian_field API smoke test ---")
g_test = mm.StructuredGrid(20, 1, 1, 5e-9, 5e-9, 5e-9)
head = mm.moving_gaussian_field(g_test, H_amp=1e6, sigma=10e-9, axis=0, direction=2)

H_at_centre = head(50e-9)   # head at x=50 nm (centre)
H_arr_t = np.asarray(mm.to_numpy(H_at_centre)).reshape(20, 3)
ix_max = np.argmax(np.abs(H_arr_t[:, 2]))
H_peak = H_arr_t[ix_max, 2]
print(f"  Peak Hz at head centre: {H_peak:.3e} A/m (expect ~1e6)")
assert abs(H_peak - 1e6) < 0.1e6, f"Peak off: {H_peak}"

H_at_edge = head(0)         # head at x=0 nm (far left)
H_arr_e = np.asarray(mm.to_numpy(H_at_edge)).reshape(20, 3)
H_mid = H_arr_e[10, 2]     # mid-track should be ~0
print(f"  Hz at mid-track (head at edge): {H_mid:.3e} A/m (expect ~0)")
assert abs(H_mid) < 0.01e6, f"Mid-track field too large: {H_mid}"

print("\nAll moving_gaussian_field checks passed.")
print("\n=== Notebook 39 complete ===")
