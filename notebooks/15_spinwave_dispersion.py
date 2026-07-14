"""Notebook 15 — Spin-Wave Dispersion ω(k)

Computes the spin-wave dispersion relation S(k, f) for a 1D Permalloy strip via
broadband sinc-pulse excitation followed by 2D Fourier analysis.

Physics:
  Bias B₀ along x  →  equilibrium m = x̂
  Small H_y sinc pulse (flat spectrum 0..f_max)
  Record m_y(x, t)  →  2D FFT  →  S(k, f)
  Analytical: f(k) = γ₀/(2π) × (B₀ + 2A/Mₛ · k²)
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
gamma0 = 1.76e11          # rad / (T·s)

# ===========================================================================
# Grid & material
# ===========================================================================
nx   = 200
dx   = 20e-9              # 20 nm cells → L = 4 µm
g    = mm.StructuredGrid(nx, 1, 1, dx, dx, dx)
mat  = mm.Material.permalloy()    # Ms=800 kA/m, A=13 pJ/m

B_bias  = 0.1             # T  (bias along x)
H_bias  = B_bias / mu_0   # A/m

# Spin-wave dispersion constant: 2A/Ms [T·m²/rad²]
D_sw = 2 * mat.A_exchange / mat.Ms

print(f"Grid: {nx}x1x1, dx = {dx*1e9:.0f} nm  (L = {nx*dx*1e6:.1f} um)")
print(f"Py: Ms={mat.Ms/1e3:.0f} kA/m, A={mat.A_exchange*1e12:.0f} pJ/m")
print(f"B_bias = {B_bias*1e3:.0f} mT  ->  f_Kittel = {gamma0/(2*np.pi)*B_bias/1e9:.3f} GHz")
print(f"Dispersion constant 2A/Ms = {D_sw:.3e} T.m^2")

# ===========================================================================
# Fields
# ===========================================================================
zeeman_bias = mm.ZeemanField(mm.Vec3(H_bias, 0, 0))
zeeman_ac   = mm.ZeemanField(mm.Vec3(0, 0, 0))        # sinc perturbation
exch        = mm.ExchangeField(mm.BoundaryCondition.Neumann)

heff = mm.EffectiveFieldSum()
heff.add(zeeman_bias)
heff.add(zeeman_ac)
heff.add(exch)

# ===========================================================================
# Initial state: m = (1, 0, 0) + small random tilt to seed spin waves
# ===========================================================================
m = mm.uniform_mag(g, mm.Vec3(1, 0, 0))
arr_init = mm.to_numpy(m)
rng = np.random.default_rng(seed=7)
arr_init[:, :, :, 1] += rng.normal(0, 1e-3, arr_init[:, :, :, 1].shape)
arr_init[:, :, :, 2] += rng.normal(0, 1e-3, arr_init[:, :, :, 2].shape)
# Re-normalise
norms = np.linalg.norm(arr_init, axis=-1, keepdims=True)
arr_init /= norms
mm.from_numpy(m, arr_init)

# ===========================================================================
# Simulation: sinc-pulse excitation + m_y(x,t) recording
# ===========================================================================
dt_sim  = 0.5e-12      # 0.5 ps integration step
dt_save = 10e-12       # record every 10 ps
t_sim   = 2e-9         # 2 ns total → Δf = 0.5 GHz, f_max = 50 GHz
f_max   = 20e9         # sinc bandwidth
H_p     = 500.0        # A/m  (small perturbation amplitude)

integ = mm.RK4Integrator(dt_sim)

n_frames = int(t_sim / dt_save)
my_xt    = np.zeros((n_frames, nx))   # m_y(x, t) storage

print(f"\nRunning: {int(t_sim/dt_sim)} steps, saving every {int(dt_save/dt_sim)} steps ...")
print(f"  -> {n_frames} frames x {nx} cells")

t = 0.0
frame = 0
t_last_save = -dt_save

while t < t_sim:
    # Update sinc-pulse field at current time
    H_sinc = mm.sinc_pulse(t, mm.Vec3(0, H_p, 0), f_max)
    zeeman_ac.H_ext = H_sinc

    integ.step(m, mat, heff)
    t += dt_sim

    if t - t_last_save >= dt_save - 1e-15 and frame < n_frames:
        arr = mm.to_numpy(m)                 # (1, 1, nx, 3)
        my_xt[frame, :] = arr[0, 0, :, 1]   # m_y at all x
        frame += 1
        t_last_save = t

print(f"  Collected {frame} frames")

# ===========================================================================
# 2D FFT  →  S(k, f)
# ===========================================================================
kvals, freqs, S = mm.field_fft2d(my_xt, dt=dt_save, dx=dx)

# Analytical dispersion: f(k) = γ₀/(2π) × (B_bias + D_sw × k²)
k_theory = np.linspace(0, kvals.max(), 500)
f_theory = gamma0 / (2 * np.pi) * (B_bias + D_sw * k_theory**2)

print(f"\nFFT: k range [{kvals.min()/1e6:.1f}, {kvals.max()/1e6:.1f}] Mrad/m")
print(f"     f range [0, {freqs[:n_frames//2].max()/1e9:.1f}] GHz")

# ===========================================================================
# Plot
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# — Left: S(k, f) dispersion map —
ax = axes[0]
nf_half = n_frames // 2
f_GHz   = freqs[:nf_half] / 1e9
k_Mrad  = kvals / 1e6
S_plot  = S[:nf_half, :]                          # positive frequencies only

# Log-scale for visibility
S_log = np.log10(S_plot + S_plot[S_plot > 0].min() * 1e-3)

im = ax.pcolormesh(k_Mrad, f_GHz, S_log,
                   cmap="inferno", shading="auto")
plt.colorbar(im, ax=ax, label="log₁₀ |FFT|²")

# Analytical dispersion overlay (positive k only)
k_th_Mrad = k_theory / 1e6
f_th_GHz  = f_theory / 1e9
mask_vis  = f_th_GHz <= f_GHz.max()
ax.plot(k_th_Mrad[mask_vis], f_th_GHz[mask_vis],
        "w--", lw=1.5, label=r"$f = \frac{\gamma_0}{2\pi}(B_0 + \frac{2A}{M_s}k^2)$")

ax.set_xlabel("k  (Mrad/m)")
ax.set_ylabel("f  (GHz)")
ax.set_title(f"Spin-wave dispersion S(k, f)  —  Py, B₀ = {B_bias*1e3:.0f} mT")
ax.set_xlim(0, k_Mrad.max())
ax.set_ylim(0, min(30, f_GHz.max()))
ax.legend(fontsize=8)

# — Right: dispersion curve (peak frequency vs k) —
ax = axes[1]
# Find peak frequency at each k (positive half)
k_pos_idx = kvals >= 0
S_pos  = S[:nf_half, k_pos_idx]
k_pos  = kvals[k_pos_idx] / 1e6
f_peak = []
for ik in range(S_pos.shape[1]):
    col = S_pos[:, ik]
    f_peak.append(freqs[col.argmax()] / 1e9)

ax.scatter(k_pos[::2], f_peak[::2], s=8, color="royalblue",
           label="Simulation (peak f)")

k_th2 = np.linspace(0, k_pos.max() * 1e6, 300)
f_th2 = gamma0 / (2 * np.pi) * (B_bias + D_sw * k_th2**2) / 1e9
ax.plot(k_th2 / 1e6, f_th2, "r-", lw=2,
        label=r"$f(k)=\frac{\gamma_0}{2\pi}(B_0+\frac{2A}{M_s}k^2)$")

f_kit = gamma0 / (2 * np.pi) * B_bias / 1e9
ax.axhline(f_kit, color="gray", ls=":", lw=1,
           label=f"Kittel f₀ = {f_kit:.2f} GHz")

ax.set_xlabel("k  (Mrad/m)")
ax.set_ylabel("f  (GHz)")
ax.set_title("Dispersion: simulation vs analytical")
ax.set_xlim(0, k_pos.max())
ax.set_ylim(0, min(30, max(f_peak) * 1.1 + 1))
ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "spinwave_dispersion.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved -> {out}")

# ===========================================================================
# Numerical check vs Kittel
# ===========================================================================
f_kit_sim = f_peak[0]   # k ≈ 0 peak
f_kit_ana = gamma0 / (2 * np.pi) * B_bias / 1e9
print(f"\nKittel check (k~0): sim = {f_kit_sim:.3f} GHz, ana = {f_kit_ana:.3f} GHz"
      f"  (error {abs(f_kit_sim-f_kit_ana)/f_kit_ana*100:.1f}%)")

# Check a mid-k point
ik_mid = len(k_pos) // 4
k_mid  = k_pos[ik_mid] * 1e6
f_sim  = f_peak[ik_mid]
f_ana  = gamma0 / (2 * np.pi) * (B_bias + D_sw * k_mid**2) / 1e9
print(f"Mid-k check (k={k_mid/1e6:.0f} Mrad/m): sim = {f_sim:.3f} GHz,"
      f" ana = {f_ana:.3f} GHz  (error {abs(f_sim-f_ana)/max(f_ana,0.01)*100:.1f}%)")

print("\n=== Done ===")
