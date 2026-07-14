"""
Notebook 26: SOT Skyrmion Motion (SpinOrbitTorqueGPU + RK4IntegratorGPU)

SOT-driven Neel skyrmion motion in a Pt/Co track.

Note: mm.neel_skyrmion(g, R, charge, pol, cx, cy) uses cx, cy as OFFSETS
from the grid center (box_cx = 0.5*Nx*dx), NOT absolute positions.
  cx=0, cy=0 -> skyrmion at grid center
  cx=-L/4, cy=0 -> skyrmion at L/4 from left (center - L/4)

Physics:
  tau_DL = a_SOT * [m x (m x sigma)],  sigma=+y
  Drives skyrmion along track with skyrmion Hall angle theta_H.

Material: Pt/Co  Ms=580 kA/m, K=0.5 MJ/m3, D=2.0 mJ/m2, alpha=0.05
Grid: 200x100x1, dx=3nm (600 x 300 nm)
"""

import os, sys, time
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
import micromag as mm

print("Notebook 26: SOT Skyrmion Motion (GPU)")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
mu0   = 4e-7 * np.pi
Ms    = 580e3;  A = 15e-12;  K = 0.5e6;  D = 2.0e-3
alpha = 0.05;   d_fm = 1e-9; theta_SH = 0.15

mat = mm.Material()
mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = K
mat.easy_axis = mm.Vec3(0,0,1); mat.alpha = alpha

l_ex = np.sqrt(2*A / (mu0*Ms**2))
mu0_Heff = 2*K/Ms - mu0*Ms
print(f"\nMaterial: Ms={Ms/1e3:.0f}kA/m, K={K/1e6:.2f}MJ/m3, D={D*1e3:.1f}mJ/m2")
print(f"  l_ex={l_ex*1e9:.1f}nm, mu0*Heff={mu0_Heff*1e3:.0f}mT, alpha={alpha}")

# ---------------------------------------------------------------------------
# Grid: 200x100x1, dx=3nm (600 x 300 nm track)
# ---------------------------------------------------------------------------
Nx, Ny = 200, 100
dx = 3e-9
g = mm.StructuredGrid(Nx, Ny, 1, dx, dx, dx)
Lx = Nx * dx;  Ly = Ny * dx
print(f"\nGrid: {Nx}x{Ny}x1, dx={dx*1e9:.0f}nm => {Lx*1e9:.0f}x{Ly*1e9:.0f}nm track")

# Skyrmion initial position: 1/4 from left, vertically centered
# cx_off = offset from grid center = -Lx/4 (places sky at Lx/4 from left)
R_sky   = 30e-9   # skyrmion radius parameter
cx_off  = -Lx / 4   # offset from center -> skyrmion at x = Lx/2 + cx_off = Lx/4
cy_off  = 0.0       # centered in y

# neel_skyrmion: cx, cy are offsets from box_center = (Lx/2, Ly/2)
m_init = mm.neel_skyrmion(g, R_sky, +1, +1, cx_off, cy_off)
Q_init = mm.topological_charge_Q(m_init)
print(f"\nInitial skyrmion: R={R_sky*1e9:.0f}nm, x0={(Lx/2+cx_off)*1e9:.0f}nm, Q={Q_init:.3f}")

# ---------------------------------------------------------------------------
# GPU objects
# ---------------------------------------------------------------------------
demag_g  = mm.DemagFieldGPU(g)
exch_g   = mm.ExchangeFieldGPU(g)
aniso_g  = mm.UniaxialAnisotropyFieldGPU(g)
dmi_g    = mm.InterfacialDMIFieldGPU(g, D)

sigma    = mm.Vec3(0, 1, 0)
sot_g    = mm.SpinOrbitTorqueGPU(g, 1e12, theta_SH, d_fm, sigma, 1.0, 0.0)

torques_g = mm.SpinTorqueSumGPU()
torques_g.add(sot_g)

fields_g = mm.FieldSumGPU()
fields_g.add(exch_g); fields_g.add(aniso_g); fields_g.add(dmi_g)

# ---------------------------------------------------------------------------
# Skyrmion position tracking via topological charge density centroid
# ---------------------------------------------------------------------------
def skyrmion_pos(m_field):
    rho_s = mm.topological_charge_density(m_field)
    r = mm.to_numpy_scalar(rho_s)[0]   # (ny, nx)
    r_abs = np.abs(r)
    total = r_abs.sum()
    if total < 1e-12:
        return (Lx/2 + cx_off), (Ly/2 + cy_off)
    xs = (np.arange(Nx) + 0.5) * dx
    ys = (np.arange(Ny) + 0.5) * dx
    xp = float(np.sum(r_abs * xs[np.newaxis, :]) / total)
    yp = float(np.sum(r_abs * ys[:, np.newaxis]) / total)
    return xp, yp

x0_sky, y0_sky = skyrmion_pos(m_init)
print(f"Tracking start: ({x0_sky*1e9:.0f}, {y0_sky*1e9:.0f}) nm")

# ---------------------------------------------------------------------------
# SOT dynamics: J = [0.5, 1.5, 3.0] x 10^12  (0.5 ns each)
# ---------------------------------------------------------------------------
dt      = 5e-14
t_run   = 0.5e-9   # 0.5 ns (shorter to avoid boundary annihilation)
n_steps = int(t_run / dt)
check_ev = 400   # every 20 ps

J_values = np.array([0.5, 1.5, 3.0]) * 1e12
colors   = ['C0', 'C1', 'C2']

print(f"\nSOT dynamics: dt={dt:.0e}s, t={t_run*1e9:.1f}ns, {n_steps} steps/J")

traj_all   = []
v_x_list   = []
v_y_list   = []
Q_fin_list = []
t0_total = time.time()

for J_val in J_values:
    sot_g.J_c = J_val

    integ = mm.RK4IntegratorGPU(g, dt)
    integ.upload(m_init)

    x_traj, y_traj, Q_traj, t_list = [x0_sky], [y0_sky], [Q_init], [0.0]
    t0 = time.time()
    ok = True

    for step in range(0, n_steps, check_ev):
        for _ in range(check_ev):
            integ.step(mat, demag_g, fields_g, torques_g)
        m_tmp = mm.VectorField3D(g)
        integ.download(m_tmp)
        xp, yp = skyrmion_pos(m_tmp)
        Qp = mm.topological_charge_Q(m_tmp)
        x_traj.append(xp); y_traj.append(yp); Q_traj.append(Qp)
        t_list.append((step + check_ev) * dt)
        if abs(Qp) < 0.2:
            print(f"    Skyrmion annihilated at t={(step+check_ev)*dt*1e12:.0f}ps (Q={Qp:.2f})")
            ok = False; break

    # Linear fit (skip first 20%)
    skip = max(1, len(t_list) // 5)
    if len(t_list) - skip >= 3:
        px = np.polyfit(t_list[skip:], x_traj[skip:], 1)
        py = np.polyfit(t_list[skip:], y_traj[skip:], 1)
        vx, vy = float(px[0]), float(py[0])
    else:
        vx = vy = 0.0

    theta_H = float(np.degrees(np.arctan2(vy, vx))) if abs(vx) > 0.1 else 0.0
    dt_run = time.time() - t0
    status = "OK" if ok else "ANNIHILATED"
    print(f"  J={J_val/1e12:.1f}e12  v_x={vx:.0f}m/s  v_y={vy:.0f}m/s  "
          f"theta_H={theta_H:.1f}deg  Q_fin={Q_traj[-1]:.2f}  {dt_run:.0f}s [{status}]")

    traj_all.append((J_val, np.array(x_traj), np.array(y_traj),
                     np.array(Q_traj), np.array(t_list)))
    v_x_list.append(vx); v_y_list.append(vy)
    Q_fin_list.append(Q_traj[-1])

print(f"\nTotal: {time.time()-t0_total:.0f} s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Trajectory plot
    ax = axes[0]
    ax.set_xlim(0, Lx*1e9); ax.set_ylim(0, Ly*1e9)
    ax.axhline(Ly/2*1e9, color='gray', ls=':', alpha=0.5)
    for (J_val, x_arr, y_arr, Q_arr, t_arr), c in zip(traj_all, colors):
        ax.plot(x_arr*1e9, y_arr*1e9, '-', color=c, lw=2,
                label=f'J={J_val/1e12:.1f}e12')
        ax.plot(x_arr[0]*1e9, y_arr[0]*1e9, 'o', color=c, ms=8)
        ax.plot(x_arr[-1]*1e9, y_arr[-1]*1e9, 's', color=c, ms=8)
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    ax.set_title(f'Skyrmion trajectory (o=start, s=end)\nt={t_run*1e9:.1f}ns, sigma=+y')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Velocity vs J
    J_arr = np.array([r[0] for r in traj_all]) / 1e12
    ax = axes[1]
    ax.plot(J_arr, v_x_list, 'o-', color='C0', lw=2, ms=7, label='v_x (longitudinal)')
    ax.plot(J_arr, v_y_list, 's-', color='C3', lw=2, ms=7, label='v_y (Hall)')
    ax.axhline(0, color='k', ls='--', lw=1, alpha=0.3)
    ax.set_xlabel('J (1e12 A/m2)'); ax.set_ylabel('Velocity (m/s)')
    ax.set_title('SOT Skyrmion Velocity vs J\n(SpinOrbitTorqueGPU: sigma=+y, theta_SH=0.15)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Q(t) stability
    ax = axes[2]
    for (J_val, x_arr, y_arr, Q_arr, t_arr), c in zip(traj_all, colors):
        ax.plot(t_arr*1e12, Q_arr, '-', color=c, lw=2,
                label=f'J={J_val/1e12:.1f}e12')
    ax.axhline(-1, color='k', ls='--', lw=1, alpha=0.5, label='Q=-1 ideal')
    ax.axhline(0, color='k', ls=':', lw=1, alpha=0.3)
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('Topological charge Q')
    ax.set_title('Skyrmion Q(t): SOT stability')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Pt/Co SOT Skyrmion Motion (GPU) — {int(Lx*1e9)}x{int(Ly*1e9)}nm track\n'
        f'SpinOrbitTorqueGPU + InterfacialDMIFieldGPU: sigma=+y, theta_SH={theta_SH}, '
        f'D={D*1e3:.1f}mJ/m2, alpha={alpha}',
        fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '26_sot_skyrmion_motion_gpu.png')
    plt.savefig(out, dpi=120); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Track: {Nx}x{Ny}x1, dx={dx*1e9:.0f}nm, D={D*1e3:.1f}mJ/m2, alpha={alpha}")
print(f"  Initial Q = {Q_init:.3f}")
for (J_val, *_), vx, vy, Qf in zip(traj_all, v_x_list, v_y_list, Q_fin_list):
    theta_H = np.degrees(np.arctan2(vy, vx)) if abs(vx) > 0.1 else 0.0
    print(f"  J={J_val/1e12:.1f}e12  v_x={vx:.0f}m/s  v_y={vy:.0f}m/s  "
          f"theta_H={theta_H:.1f}deg  Q_fin={Qf:.2f}")
