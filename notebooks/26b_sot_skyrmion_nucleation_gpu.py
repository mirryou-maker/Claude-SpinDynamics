"""
Notebook 26b: SOT Skyrmion Nucleation (SpinOrbitTorqueGPU + ZeemanFieldGPU)

Demonstrates current-pulse induced skyrmion nucleation in a Pt/Co disc via SOT.
SOT (sigma=+y DL torque) with in-plane bias H_x breaks symmetry, causing the
domain to nucleate a Neel skyrmion when a threshold current J_c is exceeded.

Produces:
  (A) mz snap at start, mid-pulse, end-pulse for J < J_c and J > J_c
  (B) Final Q (topological charge) vs J sweep
  (C) Time evolution of Q and mz for nucleation case

Physics:
  - DL torque: tau = a*theta_SH * [m x (m x sigma)]  pulls +z toward -y
  - In-plane bias H_x>0 breaks azimuthal symmetry -> asymmetric nucleation
  - When J > J_c: spin texture collapses to Q=-1 skyrmion (or Q=0 island)

Material: Pt/Co  Ms=580 kA/m, K=0.7 MJ/m3 (strong PMA), A=15 pJ/m, alpha=0.3
Grid: 100x100x1 disc, dx=2 nm  (200 nm diameter)
"""

import os, sys, time
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import numpy as np
import micromag as mm

print("Notebook 26: SOT Skyrmion Nucleation (GPU)")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material: Pt/Co with strong PMA and interfacial DMI
# ---------------------------------------------------------------------------
mu0   = 4e-7 * np.pi
Ms    = 580e3   # A/m
A     = 15e-12  # J/m
K     = 0.7e6   # J/m^3 (strong PMA: Q_K = 2K/mu0/Ms^2 = 3.7 > 1 -> stable +z)
D     = 2.5e-3  # J/m^2 (interfacial DMI -> Neel skyrmion)
alpha = 0.3     # high damping for faster relaxation
theta_SH = 0.15

mat = mm.Material()
mat.Ms         = Ms
mat.A_exchange = A
mat.K_uniaxial = K
mat.easy_axis  = mm.Vec3(0, 0, 1)
mat.alpha      = alpha

mu0_Heff = 2*K/Ms - mu0*Ms
print(f"\nMaterial: Ms={Ms/1e3:.0f} kA/m, K={K/1e6:.1f} MJ/m3, D={D*1e3:.1f} mJ/m2")
print(f"  mu0*Heff (PMA net) = {mu0_Heff*1e3:.0f} mT  (> 0 -> stable +z without current)")

# ---------------------------------------------------------------------------
# Grid: 100x100x1 disc, dx=2 nm
# ---------------------------------------------------------------------------
Nx, Ny = 100, 100
dx = 2e-9
d_fm = 1e-9   # FM layer thickness for SOT torque
g = mm.StructuredGrid(Nx, Ny, 1, dx, dx, dx)
print(f"\nGrid: {Nx}x{Ny}x1, dx={dx*1e9:.0f} nm, total = {Nx*dx*1e9:.0f} x {Ny*dx*1e9:.0f} nm")

# Disc mask
mask = mm.ellipse(g, Nx*dx, Ny*dx)

# ---------------------------------------------------------------------------
# GPU objects (shared across sweep)
# ---------------------------------------------------------------------------
demag_g = mm.DemagFieldGPU(g)
exch_g  = mm.ExchangeFieldGPU(g)
aniso_g = mm.UniaxialAnisotropyFieldGPU(g)
dmi_g   = mm.InterfacialDMIFieldGPU(g, D)

# In-plane bias field H_x = 50 mT to break symmetry
H_bias = 50e-3 / mu0   # A/m
zeeman_g = mm.ZeemanFieldGPU(g)
zeeman_g.H_ext = mm.Vec3(H_bias, 0, 0)

# SOT: sigma = +y (spin Hall from +y to -y, polarization along y)
sigma  = mm.Vec3(0, 1, 0)
sot_g  = mm.SpinOrbitTorqueGPU(g, 1e12, theta_SH, d_fm, sigma, 1.0, 0.0)
torques_g = mm.SpinTorqueSumGPU()
torques_g.add(sot_g)

# Build field sum (exch + aniso + DMI + zeeman)
fields_g = mm.FieldSumGPU()
fields_g.add(exch_g)
fields_g.add(aniso_g)
fields_g.add(dmi_g)
fields_g.add(zeeman_g)

# ---------------------------------------------------------------------------
# Helper: uniform +z initial state (disc)
# ---------------------------------------------------------------------------
def make_m_pz(g, mask):
    a = np.zeros((1, g.ny, g.nx, 3))
    a[..., 2] = 1.0
    m = mm.VectorField3D(g)
    mm.from_numpy(m, a)
    mm.set_geom(mask, m)
    return m

def snap_mz(m_field):
    """Return mz as 2D array [ny, nx]."""
    return mm.to_numpy(m_field)[0, :, :, 2]

def compute_Q(m_field):
    try:
        return mm.topological_charge_Q(m_field)
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Part A: Snapshots — compare J < J_c and J > J_c
# dt=2e-14 s (need small dt for high-K material),  t_pulse=200 ps
# ---------------------------------------------------------------------------
dt       = 2e-14
t_pulse  = 200e-12   # 200 ps pulse
n_pulse  = int(t_pulse / dt)

print(f"\nPulse parameters: dt={dt:.0e} s, t_pulse={t_pulse*1e12:.0f} ps, {n_pulse} steps")

J_values_A = [1.0e12, 5.0e12]
labels_A   = [f"J=1.0e12 (below?)", f"J=5.0e12 (above?)"]

snaps = []   # list of (label, mz_before, mz_mid, mz_after, Q_after)

for J_val, lbl in zip(J_values_A, labels_A):
    sot_g.J_c = J_val
    m = make_m_pz(g, mask)
    integ = mm.RK4IntegratorGPU(g, dt)
    integ.upload(m)

    # Snap before
    m_tmp = mm.VectorField3D(g)
    integ.download(m_tmp)
    mz_before = snap_mz(m_tmp)

    # Half pulse
    for _ in range(n_pulse // 2):
        integ.step(mat, demag_g, fields_g, torques_g)
    integ.download(m_tmp)
    mz_mid = snap_mz(m_tmp)

    # Rest of pulse
    for _ in range(n_pulse - n_pulse // 2):
        integ.step(mat, demag_g, fields_g, torques_g)
    integ.download(m_tmp)
    mz_after = snap_mz(m_tmp)
    Q_after  = compute_Q(m_tmp)

    snaps.append((lbl, mz_before, mz_mid, mz_after, Q_after))
    print(f"  {lbl}  Q_after={Q_after:.2f}")

# ---------------------------------------------------------------------------
# Part B: Q_final vs J sweep (6 values, 200 ps pulse)
# ---------------------------------------------------------------------------
print(f"\n--- Part B: Q vs J sweep (8 J-values, t_pulse={t_pulse*1e12:.0f} ps) ---")
J_sweep = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]) * 1e12
Q_final = []

t0 = time.time()
for J_val in J_sweep:
    sot_g.J_c = J_val
    m = make_m_pz(g, mask)
    integ = mm.RK4IntegratorGPU(g, dt)
    integ.upload(m)
    for _ in range(n_pulse):
        integ.step(mat, demag_g, fields_g, torques_g)
    m_tmp = mm.VectorField3D(g)
    integ.download(m_tmp)
    Q = compute_Q(m_tmp)
    Q_final.append(Q)
    print(f"  J={J_val/1e12:.1f}e12  Q_final={Q:.2f}")

print(f"  Sweep: {time.time()-t0:.1f} s")

# ---------------------------------------------------------------------------
# Part C: Time evolution Q(t) and mz(t) for nucleation case
# ---------------------------------------------------------------------------
J_nuc = J_sweep[int(np.argmin(np.abs(np.array(Q_final) + 1)))]
print(f"\n--- Part C: Time evolution at J_nuc={J_nuc/1e12:.1f}e12 (Q~-1) ---")

check_ev = 100
sot_g.J_c = J_nuc
m = make_m_pz(g, mask)
integ = mm.RK4IntegratorGPU(g, dt)
integ.upload(m)

Q_traj, mz_mean_traj, t_traj = [], [], []
t0 = time.time()
for step in range(0, n_pulse, check_ev):
    for _ in range(check_ev):
        integ.step(mat, demag_g, fields_g, torques_g)
    m_tmp = mm.VectorField3D(g)
    integ.download(m_tmp)
    Q_traj.append(compute_Q(m_tmp))
    mz_vals = snap_mz(m_tmp)
    mz_mean_traj.append(float(np.mean(mz_vals[mz_vals != 0.0]) if np.any(mz_vals != 0.0) else 0.0))
    t_traj.append((step + check_ev) * dt * 1e12)

print(f"  Time: {time.time()-t0:.1f} s  Q_final={Q_traj[-1]:.2f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(16, 9))

    # Part A: 2 rows (J_low, J_high) x 3 columns (before, mid, after)
    vmin, vmax = -1, 1
    extent = [0, Nx*dx*1e9, 0, Ny*dx*1e9]

    for row, (lbl, mz_b, mz_m, mz_a, Q_a) in enumerate(snaps):
        for col, (mz, title) in enumerate([(mz_b, 'Before pulse'),
                                            (mz_m, f't={t_pulse*1e12/2:.0f} ps'),
                                            (mz_a, f't={t_pulse*1e12:.0f} ps (Q={Q_a:.2f})')]):
            ax = fig.add_subplot(3, 6, row*6 + col*2 + 1)
            im = ax.imshow(mz, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                           origin='lower', extent=extent)
            ax.set_title(f'{lbl[:6]}\n{title}', fontsize=7)
            ax.set_xlabel('x (nm)', fontsize=6)
            if col == 0: ax.set_ylabel('y (nm)', fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.05)

    # Part B: Q vs J
    ax_B = fig.add_subplot(3, 3, 7)
    ax_B.plot(J_sweep/1e12, Q_final, 'o-', color='C0', lw=2, ms=7)
    ax_B.axhline(-1, color='C3', ls='--', lw=1.5, alpha=0.7, label='Q=-1 (skyrmion)')
    ax_B.axhline(0, color='C2', ls='--', lw=1.5, alpha=0.7, label='Q=0 (trivial)')
    ax_B.set_xlabel('J (1e12 A/m2)')
    ax_B.set_ylabel('Topological charge Q')
    ax_B.set_title(f'Q after {t_pulse*1e12:.0f} ps pulse vs J\n(H_x={50:.0f} mT bias)')
    ax_B.legend(fontsize=8); ax_B.grid(alpha=0.3)

    # Part C: Q(t) and mz(t)
    ax_C1 = fig.add_subplot(3, 3, 8)
    ax_C1.plot(t_traj, Q_traj, 'o-', color='C0', lw=2, ms=4)
    ax_C1.axhline(-1, color='C3', ls='--', lw=1.5, alpha=0.7)
    ax_C1.set_xlabel('Time (ps)')
    ax_C1.set_ylabel('Topological charge Q')
    ax_C1.set_title(f'Q(t) nucleation (J={J_nuc/1e12:.1f}e12)')
    ax_C1.grid(alpha=0.3)

    ax_C2 = fig.add_subplot(3, 3, 9)
    ax_C2.plot(t_traj, mz_mean_traj, '-', color='C1', lw=2)
    ax_C2.set_xlabel('Time (ps)')
    ax_C2.set_ylabel('<mz> (disc cells)')
    ax_C2.set_title(f'<mz>(t) nucleation (J={J_nuc/1e12:.1f}e12)')
    ax_C2.grid(alpha=0.3)

    plt.suptitle(
        f'SOT Skyrmion Nucleation (GPU) — Pt/Co disc {Nx*dx*1e9:.0f}x{Ny*dx*1e9:.0f}nm\n'
        f'SpinOrbitTorqueGPU: sigma=+y, theta_SH={theta_SH}, H_bias={50:.0f}mT, '
        f'D={D*1e3:.1f}mJ/m2, K={K/1e6:.1f}MJ/m3',
        fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '26b_sot_skyrmion_nucleation_gpu.png')
    plt.savefig(out, dpi=110); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Disc: {Nx}x{Ny}x1, dx={dx*1e9:.0f}nm, D={D*1e3:.1f}mJ/m2, H_bias=50mT")
print(f"  Q vs J: {[f'{q:.2f}' for q in Q_final]}")
print(f"  Nucleation current J_nuc ~ {J_nuc/1e12:.1f}e12 A/m2  (Q~-1)")
