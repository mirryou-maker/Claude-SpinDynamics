"""
Notebook 21: GPU Skyrmion Phase Diagram (D vs K parameter sweep)

Maps topological phases as a function of interfacial DMI (D) and
uniaxial anisotropy (K) using RelaxGPU + FieldSumGPU.

Optimisation: DemagFieldGPU, ExchangeFieldGPU, UniaxialAnisotropyFieldGPU
are shared across all parameter points (only InterfacialDMIFieldGPU.set_D()
and mat.K_uniaxial change per point). This avoids repeated demag kernel
precomputation (~100 ms each).

Material base: Pt/Co-like (Ms=580 kA/m, A=15 pJ/m)
Grid: 64x64x1, dx=3 nm  (192 x 192 nm patch)

Phase classification:
  |Q| >= 0.7  ->  Skyrmion (single or multi)
  |Q| < 0.3   ->  Uniform / Ferromagnetic
  otherwise   ->  Stripe / Partial

Output: 21_skyrmion_phase_diagram_gpu.png
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc-cuda', 'python'))
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')

import numpy as np
import micromag as mm

print("Notebook 21: GPU Skyrmion Phase Diagram")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material base and grid
# ---------------------------------------------------------------------------
Ms = 580e3    # A/m   Pt/Co
A  = 15e-12   # J/m   exchange

dx = 3e-9     # 3 nm cell
Lx, Ly = 64, 64

g = mm.StructuredGrid(Lx, Ly, 1, dx, dx, dx)
print(f"Grid: {Lx}x{Ly}x1  dx={dx*1e9:.0f} nm  N={Lx*Ly}")

# ---------------------------------------------------------------------------
# Neel skyrmion initial state (R = 12 nm)
# ---------------------------------------------------------------------------
def make_neel_skyrmion(grid, R=12e-9):
    nx, ny = grid.nx, grid.ny
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    dx_ = grid.dx
    a = np.zeros((1, ny, nx, 3))
    for iy in range(ny):
        for ix in range(nx):
            rx = (ix - cx) * dx_
            ry = (iy - cy) * dx_
            r  = np.sqrt(rx**2 + ry**2)
            if r < 1e-20:
                a[0, iy, ix, 2] = -1.0
            else:
                theta = np.pi * (1.0 - r / R) if r < R else 0.0
                a[0, iy, ix, 0] = np.sin(theta) * (rx / r)
                a[0, iy, ix, 1] = np.sin(theta) * (ry / r)
                a[0, iy, ix, 2] = -np.cos(theta)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-20, 1.0, norms)
    return a / norms

a0 = make_neel_skyrmion(g, R=12e-9)
m0 = mm.VectorField3D(g)
mm.from_numpy(m0, a0)
Q_initial = mm.topological_charge_Q(m0)
print(f"Initial Q = {Q_initial:.3f}")

# ---------------------------------------------------------------------------
# Pre-build shared GPU objects (expensive DemagFieldGPU kernel only once)
# ---------------------------------------------------------------------------
demag_gpu = mm.DemagFieldGPU(g)
exch_gpu  = mm.ExchangeFieldGPU(g)
aniso_gpu = mm.UniaxialAnisotropyFieldGPU(g)   # K comes from Material at runtime
dmi_gpu   = mm.InterfacialDMIFieldGPU(g, 1e-3) # D set via .D property per point

fields = mm.FieldSumGPU()
fields.add(exch_gpu)
fields.add(aniso_gpu)
fields.add(dmi_gpu)

relax = mm.RelaxGPU(g)

opts = mm.RelaxGPUOptions()
opts.threshold   = 500.0    # A/m  — looser for speed
opts.max_steps   = 15000
opts.check_every = 500

# ---------------------------------------------------------------------------
# Phase diagram sweep: D x K
# ---------------------------------------------------------------------------
D_values = np.linspace(0.5e-3, 5.0e-3, 8)   # 0.5 to 5.0 mJ/m^2
K_values = np.linspace(0.2e6,  1.2e6,  8)   # 0.2 to 1.2 MJ/m^3

n_D = len(D_values)
n_K = len(K_values)
print(f"\nPhase diagram: {n_D} D x {n_K} K = {n_D*n_K} points")

Q_map  = np.zeros((n_K, n_D))
mz_map = np.zeros((n_K, n_D))

t0 = time.time()

for ik, K in enumerate(K_values):
    for id_, D in enumerate(D_values):

        # Update material (alpha=1 for fast damping relax)
        mat = mm.Material()
        mat.Ms         = Ms
        mat.A_exchange = A
        mat.K_uniaxial = K
        mat.easy_axis  = mm.Vec3(0, 0, 1)
        mat.alpha      = 1.0

        # Update DMI strength in-place (avoids kernel recomputation)
        dmi_gpu.D = D

        # Relax from Neel skyrmion state
        relax.upload(m0)
        relax.run(mat, demag_gpu, fields, opts)

        m_out = mm.VectorField3D(g)
        relax.download(m_out)

        # Measure Q and mean mz
        Q_map[ik, id_]  = mm.topological_charge_Q(m_out)
        a_out = mm.to_numpy(m_out)
        mz_map[ik, id_] = float(np.mean(a_out[:, :, :, 2]))

elapsed = time.time() - t0
n_pts   = n_D * n_K
print(f"\nSweep: {n_pts} points in {elapsed:.1f} s ({elapsed/n_pts*1000:.0f} ms/pt)")

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("\nTopological charge Q map (rows=K [MJ/m3], cols=D [mJ/m2]):")
hdr = "  ".join(f"{D*1e3:.1f}" for D in D_values)
print(f"  {'K\\D':>8}  {hdr}")
for ik, K in enumerate(K_values):
    row = "  ".join(f"{Q_map[ik, id_]:+.2f}" for id_ in range(n_D))
    print(f"  {K*1e-6:.2f} MJ:  {row}")

# Phase classification
def phase(Q):
    aQ = abs(Q)
    if   aQ >= 0.7: return "Skyrmion"
    elif aQ >= 0.3: return "Stripe"
    else:           return "Uniform"

n_sky = int(np.sum(np.abs(Q_map) >= 0.7))
n_str = int(np.sum((np.abs(Q_map) >= 0.3) & (np.abs(Q_map) < 0.7)))
n_uni = int(np.sum(np.abs(Q_map) < 0.3))
print(f"\nPhase counts: Skyrmion={n_sky}  Stripe={n_str}  Uniform={n_uni}  (of {n_pts})")

best_sky = np.unravel_index(np.argmin(np.abs(Q_map + 1)), Q_map.shape)  # closest to Q=-1
ik_b, id_b = best_sky
print(f"Best single skyrmion: D={D_values[id_b]*1e3:.1f} mJ/m2  K={K_values[ik_b]*1e-6:.2f} MJ/m3  Q={Q_map[ik_b, id_b]:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import get_cmap

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    D_ext = [D_values[0]*1e3, D_values[-1]*1e3]
    K_ext = [K_values[0]*1e-6, K_values[-1]*1e-6]
    ext   = [D_ext[0], D_ext[1], K_ext[0], K_ext[1]]

    # Topological charge Q map
    ax = axes[0]
    Qmax = max(1.1, np.abs(Q_map).max() * 1.05)
    im = ax.imshow(Q_map, origin='lower', extent=ext, aspect='auto',
                    cmap='RdBu_r', vmin=-Qmax, vmax=Qmax)
    plt.colorbar(im, ax=ax, label='Topological charge Q')
    D_g, K_g = np.meshgrid(D_values*1e3, K_values*1e-6)
    cs = ax.contour(D_g, K_g, np.abs(Q_map), levels=[0.3, 0.7],
                    colors=['orange', 'black'], linewidths=[1.5, 2.0], linestyles=['--', '-'])
    ax.clabel(cs, fmt={0.3: '|Q|=0.3', 0.7: '|Q|=0.7'}, fontsize=8)
    ax.set_xlabel('DMI strength D (mJ/m$^2$)')
    ax.set_ylabel('Anisotropy K (MJ/m$^3$)')
    ax.set_title('Topological Charge Q')

    # Phase map (categorical)
    ax = axes[1]
    phase_map = np.zeros_like(Q_map)
    phase_map[np.abs(Q_map) >= 0.7] = 2   # Skyrmion
    phase_map[(np.abs(Q_map) >= 0.3) & (np.abs(Q_map) < 0.7)] = 1  # Stripe
    # phase_map == 0: Uniform
    from matplotlib.colors import ListedColormap
    cmap_ph = ListedColormap(['#4fc3f7', '#ffb300', '#e53935'])
    im2 = ax.imshow(phase_map, origin='lower', extent=ext, aspect='auto',
                     cmap=cmap_ph, vmin=0, vmax=2)
    cbar2 = plt.colorbar(im2, ax=ax, ticks=[0, 1, 2])
    cbar2.ax.set_yticklabels(['Uniform', 'Stripe', 'Skyrmion'])
    ax.set_xlabel('DMI strength D (mJ/m$^2$)')
    ax.set_ylabel('Anisotropy K (MJ/m$^3$)')
    ax.set_title('Phase Classification')

    plt.suptitle(
        f'Pt/Co Skyrmion Phase Diagram (GPU, {Lx}x{Ly} grid, dx={dx*1e9:.0f} nm)\n'
        f'{n_pts} pts in {elapsed:.0f} s  Skyrmion: {n_sky}/{n_pts}',
        fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), '21_skyrmion_phase_diagram_gpu.png')
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved: {out_path}")

except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Grid: {Lx}x{Ly}x1, dx={dx*1e9:.0f} nm  ({Lx*Ly} cells)")
print(f"  Sweep: D in [{D_values[0]*1e3:.1f}, {D_values[-1]*1e3:.1f}] mJ/m2 x "
      f"K in [{K_values[0]*1e-6:.2f}, {K_values[-1]*1e-6:.2f}] MJ/m3")
print(f"  Time: {elapsed:.1f} s = {elapsed/n_pts*1000:.0f} ms/pt ({n_pts} pts)")
print(f"  Phases: Skyrmion={n_sky}  Stripe={n_str}  Uniform={n_uni}")
print(f"  Best skyrmion: D={D_values[id_b]*1e3:.1f} mJ/m2  K={K_values[ik_b]*1e-6:.2f} MJ/m3  Q={Q_map[ik_b, id_b]:.3f}")
