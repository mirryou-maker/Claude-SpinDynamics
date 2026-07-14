"""Notebook 13 — FMR Spectrum via Sinc Excitation

Demonstrates broadband spin-wave spectroscopy using a sinc magnetic-field
pulse (mumax3 FMR example equivalent).

Two simulations:
  A. Macrospin (1×1×1 cell) — analytical Kittel comparison, zero demag.
  B. Permalloy thin film (80×80×1 nm, 4 nm cells) — spatially uniform FMR
     mode visible in FFT of <mx>(t).

Kittel formula (no demag, bias B_bias along ẑ, LLG without α correction):
  f_FMR = γ₀ / (2π) · B_bias   [Hz],   γ₀ = 1.76e11 rad/(T·s)
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

mu_0   = 4 * np.pi * 1e-7
gamma0 = 1.76e11          # rad/(T·s)

# ===========================================================================
# A. Macrospin FMR — analytical Kittel verification
# ===========================================================================
print("=== A. Macrospin FMR ===")

mat = mm.Material.permalloy()

# Bias field: B_bias along z (μ₀ H = 50 mT)
B_bias = 50e-3            # [T]
H_bias = B_bias / mu_0   # [A/m]

# Kittel resonance (no demag): ω = γ₀ · B,  f = γ₀/(2π) · B
f_kittel = gamma0 / (2 * np.pi) * B_bias
print(f"  B_bias = {B_bias*1e3:.0f} mT -> H_bias = {H_bias:.0f} A/m")
print(f"  Kittel f_FMR = {f_kittel/1e9:.3f} GHz")

# Initial state: m along z (saturated by bias)
g1 = mm.StructuredGrid(1, 1, 1, 5e-9, 5e-9, 5e-9)
m1 = mm.uniform_mag(g1, mm.Vec3(0, 0, 1))

# Bias Zeeman (updated each step to include sinc pulse)
zeeman1 = mm.ZeemanField(mm.Vec3(0, 0, H_bias))
heff1   = mm.EffectiveFieldSum()
heff1.add(zeeman1)

dt      = 1e-12      # 1 ps
t_total = 5e-9       # 5 ns
f_max   = 30e9       # 30 GHz sinc bandwidth (well above Kittel ~1.4 GHz)
H0_sinc = H_bias * 0.001   # small transverse perturbation (0.1% of bias)

integ1 = mm.RK4Integrator(dt)

t_rec  = []
mx_rec = []
my_rec = []

t = 0.0
n_steps = int(t_total / dt)
for _ in range(n_steps + 1):
    mx, my, mz = mm.mean_magnetization(m1)
    t_rec.append(t)
    mx_rec.append(mx)
    my_rec.append(my)

    # Sinc pulse along x + DC bias along z
    pulse = mm.sinc_pulse(t, mm.Vec3(H0_sinc, 0, 0), f_max)
    zeeman1.H_ext = mm.Vec3(pulse.x, 0, H_bias)

    integ1.step(m1, mat, heff1)
    t += dt

t_rec  = np.array(t_rec)
mx_rec = np.array(mx_rec)
my_rec = np.array(my_rec)

# FFT of mx(t)
N     = len(mx_rec)
freqs = np.fft.rfftfreq(N, d=dt)
spec  = np.abs(np.fft.rfft(mx_rec - mx_rec.mean()))**2

i_peak  = np.argmax(spec[1:]) + 1
f_peak  = freqs[i_peak]
err_pct = abs(f_peak - f_kittel) / f_kittel * 100
print(f"  Simulated f_FMR = {f_peak/1e9:.3f} GHz")
print(f"  Error vs Kittel  = {err_pct:.2f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
show_ns = 2.0
mask_t  = t_rec <= show_ns * 1e-9
ax.plot(t_rec[mask_t]*1e9, mx_rec[mask_t], lw=0.8, label="mx(t)")
ax.plot(t_rec[mask_t]*1e9, my_rec[mask_t], lw=0.8, label="my(t)", alpha=0.7)
ax.set_xlabel("t  (ns)")
ax.set_ylabel("m component")
ax.set_title(f"Macrospin precession  (B_bias = {B_bias*1e3:.0f} mT)")
ax.legend()

ax = axes[1]
mask_f = freqs < 10e9
ax.semilogy(freqs[mask_f]/1e9, spec[mask_f] + 1e-20, lw=0.8, color="C0")
ax.axvline(f_kittel/1e9, color="r", ls="--", label=f"Kittel: {f_kittel/1e9:.3f} GHz")
ax.axvline(f_peak/1e9,  color="g", ls=":",  label=f"Sim:    {f_peak/1e9:.3f} GHz")
ax.set_xlabel("Frequency  (GHz)")
ax.set_ylabel("Power  (a.u.)")
ax.set_title("FMR power spectrum — macrospin")
ax.legend()
ax.set_xlim(0, 10)

plt.tight_layout()
out_a = os.path.join(os.path.dirname(__file__), "fmr_macrospin.png")
plt.savefig(out_a, dpi=150)
plt.close()
print(f"  Saved -> {out_a}")


# ===========================================================================
# B. Permalloy thin film FMR (80 × 80 × 1 nm, 4 nm cells)
#    In-plane geometry: m ∥ x, demag lowers the resonance below Kittel
# ===========================================================================
print("\n=== B. Permalloy thin film FMR ===")

nx_b, ny_b, nz_b = 20, 20, 1
dx_b = 4e-9
g2   = mm.StructuredGrid(nx_b, ny_b, nz_b, dx_b, dx_b, dx_b)
mat_b = mm.Material.permalloy()

# Initial state: m along x
m2 = mm.uniform_mag(g2, mm.Vec3(1, 0, 0))

# Bias along x
B_bias_b  = 20e-3              # [T]
H_bias_b  = B_bias_b / mu_0   # [A/m]
H0_sinc_b = H_bias_b * 0.002

zeeman_b = mm.ZeemanField(mm.Vec3(H_bias_b, 0, 0))
demag_b  = mm.DemagField(g2)
exch_b   = mm.ExchangeField(mm.BoundaryCondition.Neumann)
heff_b   = mm.EffectiveFieldSum()
heff_b.add(zeeman_b)
heff_b.add(demag_b)
heff_b.add(exch_b)

dt_b      = 5e-13      # 0.5 ps
t_total_b = 1e-9       # 1 ns
f_max_b   = 50e9       # 50 GHz sinc bandwidth

integ_b = mm.RK4Integrator(dt_b)

t_b_rec  = []
mz_b_rec = []

n_steps_b = int(t_total_b / dt_b)
print(f"  Grid: {nx_b}x{ny_b}x{nz_b}, dx = {dx_b*1e9:.0f} nm")
print(f"  B_bias = {B_bias_b*1e3:.0f} mT along x  (in-plane geometry)")
print(f"  Recording {n_steps_b} steps ...")

t_b = 0.0
for k in range(n_steps_b + 1):
    if k % max(1, n_steps_b // 5) == 0:
        mx, my, mz = mm.mean_magnetization(m2)
        print(f"    t = {t_b*1e9:.2f} ns   <mx> = {mx:.4f}")

    mx_b, my_b, mz_b = mm.mean_magnetization(m2)
    t_b_rec.append(t_b)
    mz_b_rec.append(mz_b)

    # Sinc pulse along z (transverse to in-plane m)
    pulse_b = mm.sinc_pulse(t_b, mm.Vec3(0, 0, H0_sinc_b), f_max_b)
    zeeman_b.H_ext = mm.Vec3(H_bias_b, 0, pulse_b.z)

    integ_b.step(m2, mat_b, heff_b)
    t_b += dt_b

t_b_arr  = np.array(t_b_rec)
mz_b_arr = np.array(mz_b_rec)

# FFT of mz(t)
N_b      = len(mz_b_arr)
freqs_b  = np.fft.rfftfreq(N_b, d=dt_b)
spec_b   = np.abs(np.fft.rfft(mz_b_arr - mz_b_arr.mean()))**2

mask_nz = spec_b[1:] > 0
if mask_nz.any():
    i_peak_b = np.argmax(spec_b[1:]) + 1
    f_peak_b = freqs_b[i_peak_b]
    print(f"\n  Thin-film FMR peak: {f_peak_b/1e9:.2f} GHz")
else:
    f_peak_b = 0.0
    print("\n  No clear FMR peak found")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(t_b_arr*1e9, mz_b_arr, lw=0.7, color="C2")
ax.set_xlabel("t  (ns)")
ax.set_ylabel("<mz>")
ax.set_title(f"Thin-film FMR oscillations  (80×80×1 nm Py, B_bias = {B_bias_b*1e3:.0f} mT)")

ax = axes[1]
mask_bf = freqs_b < 30e9
ax.semilogy(freqs_b[mask_bf]/1e9, spec_b[mask_bf] + 1e-20, lw=0.8, color="C2")
if f_peak_b > 0:
    ax.axvline(f_peak_b/1e9, color="r", ls="--",
               label=f"FMR: {f_peak_b/1e9:.2f} GHz")
ax.set_xlabel("Frequency  (GHz)")
ax.set_ylabel("Power  (a.u.)")
ax.set_title("FMR power spectrum — thin film (demag softens resonance)")
ax.legend()
ax.set_xlim(0, 30)

plt.tight_layout()
out_b = os.path.join(os.path.dirname(__file__), "fmr_thinfilm.png")
plt.savefig(out_b, dpi=150)
plt.close()
print(f"  Saved -> {out_b}")

print("\n=== Summary ===")
print(f"  A. Macrospin: Kittel = {f_kittel/1e9:.3f} GHz, Simulated = {f_peak/1e9:.3f} GHz  (error {err_pct:.2f}%)")
if f_peak_b > 0:
    print(f"  B. Thin film: FMR peak = {f_peak_b/1e9:.3f} GHz (demag+exchange shift)")
