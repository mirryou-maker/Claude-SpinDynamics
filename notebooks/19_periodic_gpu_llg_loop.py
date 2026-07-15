"""Notebook 19 — Fully Periodic GPU LLG Loop

Demonstrates a complete all-GPU micromagnetic simulation with PERIODIC boundary
conditions on all fields:

  DemagFieldPeriodicGPU  + ExchangeFieldGPU(Periodic)  → RK4IntegratorGPU

Physics: spin-wave dispersion in a 1-D chain of permalloy cells.
  - Ground state: uniform saturation along +x
  - Excite with a sinc-pulse field applied to the centre cell at t=0
  - Record m_z(x, t) over 1 ns
  - 2D FFT → ω(k) dispersion spectrum

Comparison:
  - CPU periodic (DemagFieldPeriodic + ExchangeField(Periodic) + RK4)
  - GPU periodic (this demo) — same physics, much faster

Expected dispersion: ω² = γ²μ₀Ms[μ₀Ms·Nxx(k) + 2A/μ₀Ms · k²] (Damon-Eshbach)
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
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import micromag as mm

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if not mm.cuda_available():
    print("CUDA not available -- skipping notebook 19")
    sys.exit(0)

mu_0   = 4 * np.pi * 1e-7
mat    = mm.Material.permalloy()
gamma  = mm.gamma_0 / (1 + mat.alpha**2)

print("Notebook 19: Fully Periodic GPU LLG Loop")
print(f"  Material: Ms={mat.Ms/1e3:.0f} kA/m  A={mat.A_exchange:.2e} J/m  alpha={mat.alpha}")

# ---------------------------------------------------------------------------
# Grid — 1D chain for spin-wave demo, periodic in x
# ---------------------------------------------------------------------------
Nx, Ny, Nz = 128, 1, 1
dx = 5e-9   # 5 nm cells  →  640 nm chain
g  = mm.StructuredGrid(Nx, Ny, Nz, dx, dx, dx)

# ---------------------------------------------------------------------------
# Ground state: uniform along +x, small z-tilt to seed spin wave (sinc pulse)
# We apply a transverse sinc pulse instead of tilting.
# ---------------------------------------------------------------------------
m0 = mm.VectorField3D(g)
arr = mm.to_numpy(m0)
arr[..., 0] = 1.0; arr[..., 1] = 0.0; arr[..., 2] = 0.0
mm.from_numpy(m0, arr)

# ---------------------------------------------------------------------------
# Periodic GPU fields
# ---------------------------------------------------------------------------
demag_gpu = mm.DemagFieldPeriodicGPU(g)
exch_gpu  = mm.ExchangeFieldGPU(g, mm.BoundaryCondition.Periodic)
H_sinc_amp = 1e3 / mu_0   # 1 mT transverse sinc pulse in A/m
H_bias     = mm.Vec3(80e3, 0, 0)   # 80 kA/m DC bias along x
H_combined = mm.Vec3(80e3, 0, H_sinc_amp)  # bias + pulse together
zeeman_gpu       = mm.ZeemanFieldGPU(g, H_bias)
zeeman_gpu_pulse = mm.ZeemanFieldGPU(g, H_combined)

# ---------------------------------------------------------------------------
# Time parameters
# ---------------------------------------------------------------------------
dt         = 5e-13   # 0.5 ps per step
T_total    = 1e-9    # 1 ns total
N_steps    = int(T_total / dt)
N_save     = Nx      # record every step (thin 1D grid)
save_every = max(1, N_steps // 500)   # ~500 time samples

print(f"\nGrid: {Nx}x{Ny}x{Nz}  ({Nx*Ny*Nz} cells)  dx={dx*1e9:.0f} nm")
print(f"dt={dt*1e12:.1f} ps  T={T_total*1e9:.1f} ns  N_steps={N_steps}")
print(f"Save every {save_every} steps -> {N_steps//save_every} time samples")

# ---------------------------------------------------------------------------
# GPU periodic simulation
# ---------------------------------------------------------------------------
print("\n--- GPU periodic LLG ---")

integ_gpu = mm.RK4IntegratorGPU(g, dt)

# Apply sinc pulse: add a short transient z-field at all cells
# We use a uniform ZeemanFieldGPU and change the field for the first few steps
# Reset magnetisation
m_reset = mm.VectorField3D(g)
arr2 = mm.to_numpy(m_reset)
arr2[..., 0] = 1.0; arr2[..., 1] = 0.0; arr2[..., 2] = 1e-3   # tiny seed
arr2 /= np.sqrt((arr2**2).sum(axis=-1, keepdims=True))
mm.from_numpy(m_reset, arr2)
integ_gpu.upload(m_reset)

mz_t_gpu   = []   # shape: (N_t, Nx)
t_samples  = []
t0 = time.perf_counter()

# sinc pulse duration: 10 steps
N_pulse = 10
for step in range(N_steps):
    if step < N_pulse:
        integ_gpu.step(mat, demag_gpu, exch_gpu, zeeman_gpu_pulse)
    else:
        integ_gpu.step(mat, demag_gpu, exch_gpu, zeeman_gpu)

    if step % save_every == 0:
        m_now = mm.VectorField3D(g)
        integ_gpu.download(m_now)
        a = mm.to_numpy(m_now)
        mz_t_gpu.append(a[0, 0, :, 2].copy())   # mz along x; shape (Nz,Ny,Nx,3) → index [iz,iy,ix,c]
        t_samples.append(step * dt)

t_gpu = time.perf_counter() - t0
mz_t_gpu = np.array(mz_t_gpu)   # shape (N_t, Nx)
t_samples = np.array(t_samples)
print(f"  GPU: {t_gpu:.2f} s  ({N_steps} steps)  "
      f"[{t_gpu/N_steps*1e3:.3f} ms/step]")

# ---------------------------------------------------------------------------
# CPU periodic simulation (for validation)
# ---------------------------------------------------------------------------
print("\n--- CPU periodic LLG ---")

demag_cpu   = mm.DemagFieldPeriodic(g)
exch_cpu    = mm.ExchangeField(mm.BoundaryCondition.Periodic)
zeeman_cpu  = mm.ZeemanField(mm.Vec3(80e3, 0, 0))
zeeman_cpu_pulse = mm.ZeemanField(mm.Vec3(80e3, 0, H_sinc_amp))

m_cpu = mm.VectorField3D(g)
mm.from_numpy(m_cpu, arr2)

integ_cpu = mm.RK4Integrator(dt)

heff_pulse = mm.EffectiveFieldSum()
heff_pulse.add(demag_cpu); heff_pulse.add(exch_cpu)
heff_pulse.add(zeeman_cpu_pulse)

heff_base = mm.EffectiveFieldSum()
heff_base.add(demag_cpu); heff_base.add(exch_cpu)
heff_base.add(zeeman_cpu)

mz_t_cpu = []
N_steps_cpu = min(N_steps, 2000)   # only short CPU run for validation
t1 = time.perf_counter()
for step in range(N_steps_cpu):
    if step < N_pulse:
        integ_cpu.step(m_cpu, mat, heff_pulse)
    else:
        integ_cpu.step(m_cpu, mat, heff_base)
    if step % save_every == 0:
        a = mm.to_numpy(m_cpu)
        mz_t_cpu.append(a[0, 0, :, 2].copy())
t2 = time.perf_counter() - t1
mz_t_cpu = np.array(mz_t_cpu)
t_samples_cpu = t_samples[:len(mz_t_cpu)]
print(f"  CPU: {t2:.2f} s  ({N_steps_cpu} steps)  "
      f"[{t2/N_steps_cpu*1e3:.3f} ms/step]")
print(f"  GPU speedup: {t2/N_steps_cpu / (t_gpu/N_steps):.1f}x  "
      f"(GPU ran {N_steps//N_steps_cpu}x longer)")

# Validate: compare first N_steps_cpu steps
n_cmp = min(len(mz_t_cpu), len(mz_t_gpu))
rms_err = np.sqrt(np.mean((mz_t_gpu[:n_cmp] - mz_t_cpu[:n_cmp])**2))
print(f"  CPU vs GPU mz RMS error: {rms_err:.2e}  (first {n_cmp} snapshots)")

# ---------------------------------------------------------------------------
# Dispersion: 2D FFT of mz(x, t) → S(k, omega)
# ---------------------------------------------------------------------------
print("\n--- Dispersion spectrum (GPU data, 1 ns) ---")

mz = mz_t_gpu - mz_t_gpu.mean(axis=0, keepdims=True)   # remove DC

# 2D FFT: time (rows) → freq, space (cols) → k
S  = np.abs(np.fft.fftshift(np.fft.fft2(mz)))**2

N_t, N_x = mz.shape
dt_eff = t_samples[1] - t_samples[0] if len(t_samples) > 1 else dt * save_every
freqs  = np.fft.fftshift(np.fft.fftfreq(N_t, d=dt_eff)) * 1e-9   # GHz
kvals  = np.fft.fftshift(np.fft.fftfreq(N_x, d=dx)) * 1e-6        # rad/um

f_max_idx = np.argmin(np.abs(freqs - 40))   # clip at 40 GHz
k_pos     = kvals >= 0

print(f"  Time samples: {N_t}   Space cells: {N_x}")
print(f"  Freq resolution: {abs(freqs[1]-freqs[0]):.2f} GHz")
print(f"  k resolution:    {abs(kvals[1]-kvals[0]):.3f} rad/um")

# ---------------------------------------------------------------------------
# Theoretical dispersion (Kittel + exchange, no demagnetisation tensor approx)
# ω = γ μ₀ √[(Ms/2 + 2A k²/(μ₀ Ms)) × (Ms + 2A k²/(μ₀ Ms))]   (thin rod approx)
# ---------------------------------------------------------------------------
k_th  = np.linspace(0, kvals.max(), 200) * 1e6   # rad/m
f_ex  = 2 * mat.A_exchange / (mu_0 * mat.Ms) * k_th**2
H_eff_th = mat.Ms / 2 + f_ex      # simplified; fully correct dispersion needs Nxx
omega_th  = gamma * mu_0 * np.sqrt((H_eff_th) * (mat.Ms + 2*f_ex))
f_th_GHz  = omega_th / (2*np.pi) * 1e-9
k_th_um   = k_th * 1e-6   # rad/um

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. mz(x, t) heat map
ax = axes[0]
im = ax.imshow(mz, aspect="auto", origin="lower",
               extent=[0, Nx*dx*1e9, 0, t_samples[-1]*1e12],
               cmap="RdBu", vmin=-0.005, vmax=0.005)
ax.set_xlabel("Position x (nm)")
ax.set_ylabel("Time (ps)")
ax.set_title("GPU: mz(x,t)  [periodic BC, 1 ns]")
plt.colorbar(im, ax=ax, label="mz")

# 2. Dispersion S(k, f)
ax = axes[1]
# Only positive freq and positive k for clarity
f_pos = freqs[freqs >= 0]
S_pos = S[freqs >= 0][:, kvals >= 0]
k_pos_v = kvals[kvals >= 0]
clip = np.percentile(S_pos, 99.5)
ax.imshow(np.clip(S_pos, 0, clip), aspect="auto", origin="lower",
          extent=[k_pos_v[0], k_pos_v[-1], f_pos[0], f_pos[-1]],
          cmap="inferno", vmin=0, vmax=clip)
ax.plot(k_th_um, f_th_GHz, "c--", lw=1.2, label="Theory (exchange + Zeeman)")
ax.set_xlim(0, k_pos_v[-1])
ax.set_ylim(0, 35)
ax.set_xlabel("k (rad/um)")
ax.set_ylabel("f (GHz)")
ax.set_title("Dispersion S(k, f)  [GPU periodic LLG]")
ax.legend(fontsize=8)

# 3. CPU vs GPU mz at x=0 (first N_cmp snapshots)
ax = axes[2]
ax.plot(t_samples_cpu[:n_cmp]*1e12, mz_t_cpu[:n_cmp, Nx//2], "b-", lw=1,
        label=f"CPU RK4 (periodic)")
ax.plot(t_samples[:n_cmp]*1e12, mz_t_gpu[:n_cmp, Nx//2], "r--", lw=1,
        label=f"GPU RK4 (periodic)")
ax.set_xlabel("Time (ps)")
ax.set_ylabel("mz at x = L/2")
ax.set_title(f"CPU vs GPU validation\n(RMS error = {rms_err:.2e})")
ax.legend()

plt.suptitle("Notebook 19: Periodic GPU LLG Loop — Spin Wave Dispersion", fontsize=12)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "19_periodic_gpu_llg_loop.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved: {out}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"  GPU periodic LLG: {N_steps} steps in {t_gpu:.2f} s = {t_gpu/N_steps*1e3:.3f} ms/step")
print(f"  CPU periodic LLG: {N_steps_cpu} steps in {t2:.2f} s = {t2/N_steps_cpu*1e3:.3f} ms/step")
speedup_per_step = (t2/N_steps_cpu) / (t_gpu/N_steps)
print(f"  Speedup per step: {speedup_per_step:.1f}x")
print(f"  CPU vs GPU physics agreement: RMS mz err = {rms_err:.2e}")
print(f"  Fields used: DemagFieldPeriodicGPU + ExchangeFieldGPU(Periodic) + ZeemanFieldGPU")
print(f"  Integrator: RK4IntegratorGPU -- ZERO PCIe transfers per step")
