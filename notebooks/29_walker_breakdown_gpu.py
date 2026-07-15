"""
Notebook 29: Zhang-Li STT Walker Breakdown - Flat Strip (GPU)

NB27/NB24 used a 1D wire (1x1 cross-section): N_zz = N_yy -> J_W~0, all above-Walker.
A FLAT STRIP (N cells in y) has N_zz >> N_yy -> finite J_W in accessible range.

The Neel DW (core along y, in-plane) is the equilibrium for thin flat strips.
Walker breakdown occurs when STT drives the DW core to precess out-of-plane (my->mz).
Restoring force = (N_zz - N_yy) * Ms -> finite J_W.

With xi=0.5, alpha=0.05 (xi/alpha=10):
  Sub-Walker (J<J_W): v = -(xi/alpha)*u = -10*u  (10x speed enhancement!)
  Above-Walker (J>J_W): v -> -(xi/sqrt(a^2+xi^2))*u ~ -0.995*u  (approx -u)

Geometry: 200x10x1, dx=dy=dz=5nm -> 1000x50x5 nm Py flat strip
DW init: Neel wall mx=-tanh, my=sech, mz=0  (core along +y)
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

print("Notebook 29: Zhang-Li STT Walker Breakdown - Flat Strip (GPU)")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material: Py, high xi for dramatic Walker breakdown
# ---------------------------------------------------------------------------
mu0   = 4e-7 * np.pi
mu_B  = 9.274e-24
e_ch  = 1.6022e-19
gamma_0 = 1.76e11

Ms    = 860e3;  A = 13e-12;  K = 0.0
alpha = 0.05;   xi = 0.5;    P = 0.5

mat = mm.Material()
mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = 0.0
mat.easy_axis = mm.Vec3(1, 0, 0); mat.alpha = alpha

l_ex   = np.sqrt(2*A / (mu0*Ms**2))
dw_pi  = np.pi * l_ex   # DW width

v_sub_ratio  = -xi / alpha                           # -10
v_abv_ratio  = -xi / np.sqrt(alpha**2 + xi**2)      # -0.995

print(f"\nMaterial: Ms={Ms/1e3:.0f}kA/m, A={A*1e12:.0f}pJ/m, K=0, alpha={alpha}, xi={xi}")
print(f"  l_ex = {l_ex*1e9:.2f} nm,  DW width = pi*l_ex = {dw_pi*1e9:.1f} nm")
print(f"  xi/alpha = {xi/alpha:.0f}  ->  sub-Walker: v/u = {v_sub_ratio:.0f}  (10x!)")
print(f"  above-Walker: v/u -> {v_abv_ratio:.3f}")

# ---------------------------------------------------------------------------
# Grid: 200x10x1 flat strip, dx=5nm  (1000x50x5 nm)
# ---------------------------------------------------------------------------
Nx, Ny, Nz = 200, 10, 1
dx = 5e-9
g = mm.StructuredGrid(Nx, Ny, Nz, dx, dx, dx)
x0 = Nx // 2 * dx
L  = Nx * dx

print(f"\nGrid: {Nx}x{Ny}x{Nz}, dx={dx*1e9:.0f}nm -> {int(Nx*dx*1e9)}x{int(Ny*dx*1e9)}x{int(Nz*dx*1e9)} nm")
print(f"  Shape: flat strip (Ny/Nz = {Ny/Nz:.0f}) -> N_zz >> N_yy -> Neel DW preferred")

# ---------------------------------------------------------------------------
# Initial state: Neel DW at center (core along +y, in-plane)
# mx = -tanh(s),  my = sech(s),  mz = 0
# ---------------------------------------------------------------------------
a0 = np.zeros((Nz, Ny, Nx, 3))
for ix in range(Nx):
    s = ((ix + 0.5) * dx - x0) / dw_pi
    a0[:, :, ix, 0] = -np.tanh(s)
    a0[:, :, ix, 1] =  1.0 / np.cosh(s)
    a0[:, :, ix, 2] = 0.0
norms = np.linalg.norm(a0, axis=-1, keepdims=True).clip(1e-20)
a0 = a0 / norms

m0 = mm.VectorField3D(g)
mm.from_numpy(m0, a0)

# Check DW profile
mx0 = mm.to_numpy(m0)[0, 0, :, 0]   # mx along x at y=0
print(f"  DW init: mx range = [{mx0.min():.3f}, {mx0.max():.3f}]  (expected [-1, +1])")

# ---------------------------------------------------------------------------
# GPU objects (shared)
# ---------------------------------------------------------------------------
demag_g = mm.DemagFieldGPU(g)
exch_g  = mm.ExchangeFieldGPU(g)
# No explicit anisotropy (K=0 Py)
fields_g = mm.FieldSumGPU()
fields_g.add(exch_g)

def dw_pos(m_field):
    a  = mm.to_numpy(m_field)[0, :, :, 0]   # (ny, nx)
    mx = a.mean(axis=0)                       # average over y -> (nx,)
    w  = np.maximum(1.0 - mx**2, 0)
    xs = (np.arange(Nx) + 0.5) * dx
    return float(np.sum(xs * w) / w.sum()) if w.sum() > 1e-30 else x0

def dw_core_my_mz(m_field):
    a  = mm.to_numpy(m_field)[0, :, :, :]   # (ny, nx, 3)
    mx_avg = a[:, :, 0].mean(axis=0)
    idx = int(np.argmax(1 - mx_avg**2))      # cell with max |dmx/dx|
    return float(a[:, idx, 1].mean()), float(a[:, idx, 2].mean())

# ---------------------------------------------------------------------------
# Part A: DW velocity vs J sweep
# ---------------------------------------------------------------------------
J_arr = np.array([0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.8, 3.5]) * 1e12

dt_sw  = 2e-14; t_sw = 1.0e-9; n_sw = int(t_sw / dt_sw); chk = 1000

print(f"\n--- Part A: v_dw vs J sweep ({len(J_arr)} values) ---")
print(f"  dt={dt_sw:.0e}s, t_sim={t_sw*1e9:.1f}ns ({n_sw} steps/J)")

v_list, u_list, regime_list = [], [], []
t0_all = time.time()

for J_val in J_arr:
    zl   = mm.ZhangLiSTTGPU(g, mm.Vec3(J_val, 0, 0), P, xi)
    torq = mm.SpinTorqueSumGPU(); torq.add(zl)
    integ = mm.RK4IntegratorGPU(g, dt_sw)
    integ.upload(m0)

    pos_list, t_list = [], []
    for step in range(0, n_sw, chk):
        for _ in range(chk):
            integ.step(mat, demag_g, fields_g, torq)
        m_tmp = mm.VectorField3D(g)
        integ.download(m_tmp)
        pos_list.append(dw_pos(m_tmp))
        t_list.append((step + chk) * dt_sw)

    skip = max(1, len(t_list) // 5)
    p = np.polyfit(t_list[skip:], pos_list[skip:], 1)
    v_dw = float(p[0])

    u_val = P * mu_B * J_val / (e_ch * Ms)
    vu = v_dw / u_val
    v_list.append(v_dw); u_list.append(u_val)

    # Regime identification
    if abs(vu - v_sub_ratio) < abs(vu - v_abv_ratio):
        regime = "sub-Walker"
    else:
        regime = "above-Walker"
    regime_list.append(regime)

    print(f"  J={J_val/1e12:.2f}e12  u={u_val:.1f}m/s  v={v_dw:.1f}m/s  "
          f"v/u={vu:.2f}  [{regime}]")

print(f"\nPart A total: {time.time()-t0_all:.0f}s")

# Estimate J_W from transition
v_arr = np.array(v_list); u_arr = np.array(u_list)
vu_arr = v_arr / u_arr
# Walker breakdown where v/u transitions from ~v_sub to ~v_abv
# Find J that minimizes |v/u - midpoint|
midpoint = (v_sub_ratio + v_abv_ratio) / 2   # (-10 + -1)/2 = -5.5
idx_trans = int(np.argmin(np.abs(vu_arr - midpoint)))
J_W_sim = J_arr[idx_trans]
print(f"\n  Walker breakdown (estimated from simulation): J_W ~ {J_W_sim/1e12:.2f}e12 A/m2")
print(f"  v/u transitions from {v_sub_ratio:.0f} to {v_abv_ratio:.3f}")

# ---------------------------------------------------------------------------
# Part B: DW trajectories for sub/near/above Walker
# ---------------------------------------------------------------------------
print("\n--- Part B: DW trajectory + core precession ---")

J_cases = [J_W_sim * 0.3, J_W_sim, J_W_sim * 3.0]
lbl_cases = [f"sub (0.3*J_W)", f"Walker (J_W)", f"above (3*J_W)"]

dt_tr = 2e-14; t_tr = 0.6e-9; n_tr = int(t_tr / dt_tr); chk_tr = 200
traj_pos, traj_my, traj_mz, t_common = [], [], [], None

t0 = time.time()
for J_val, lbl in zip(J_cases, lbl_cases):
    zl   = mm.ZhangLiSTTGPU(g, mm.Vec3(J_val, 0, 0), P, xi)
    torq = mm.SpinTorqueSumGPU(); torq.add(zl)
    integ = mm.RK4IntegratorGPU(g, dt_tr)
    integ.upload(m0)

    pos_b, my_b, mz_b = [], [], []
    for step in range(0, n_tr, chk_tr):
        for _ in range(chk_tr):
            integ.step(mat, demag_g, fields_g, torq)
        m_tmp = mm.VectorField3D(g)
        integ.download(m_tmp)
        pos_b.append(dw_pos(m_tmp))
        my_c, mz_c = dw_core_my_mz(m_tmp)
        my_b.append(my_c); mz_b.append(mz_c)

    traj_pos.append(pos_b); traj_my.append(my_b); traj_mz.append(mz_b)
    if t_common is None:
        t_common = [(s + chk_tr) * dt_tr * 1e12 for s in range(0, n_tr, chk_tr)]
    dx_tot = (pos_b[-1] - x0) * 1e9
    print(f"  {lbl}: J={J_val/1e12:.2f}e12  delta_x={dx_tot:.0f}nm  "
          f"core my={my_b[-1]:.2f} mz={mz_b[-1]:.2f}")

print(f"Part B: {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # v vs J
    ax = axes[0]
    ax.plot(J_arr/1e12, v_arr, 'o-', color='C0', lw=2, ms=8, label='GPU sim (Neel DW)')
    ax.plot(J_arr/1e12, v_sub_ratio * u_arr, '--', color='C2', lw=2,
            label=f'Sub-Walker: {v_sub_ratio:.0f}*u')
    ax.plot(J_arr/1e12, v_abv_ratio * u_arr, ':', color='C3', lw=2,
            label=f'Above-Walker: {v_abv_ratio:.3f}*u')
    ax.axvline(J_W_sim/1e12, color='k', ls='--', lw=1.5, alpha=0.7,
               label=f'J_W ~ {J_W_sim/1e12:.2f}e12')
    ax.set_xlabel('J (1e12 A/m2)'); ax.set_ylabel('DW velocity (m/s)')
    ax.set_title(f'Walker Breakdown (flat 10x strip)\nxi/alpha={int(abs(v_sub_ratio))}  J_W~{J_W_sim/1e12:.2f}e12')
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # DW position vs time
    ax = axes[1]
    cols = ['C0', 'C1', 'C2']
    for i, (lbl, pos_b) in enumerate(zip(lbl_cases, traj_pos)):
        ax.plot(t_common, np.array(pos_b)*1e9, '-', color=cols[i], lw=2, label=lbl)
    ax.axhline(x0*1e9, color='k', ls='--', lw=1, alpha=0.4)
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('DW position (nm)')
    ax.set_title('DW position vs time\n(sub-Walker moves FASTER!)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # DW core my and mz vs time
    ax = axes[2]
    for i, (lbl, my_b, mz_b) in enumerate(zip(lbl_cases, traj_my, traj_mz)):
        ax.plot(t_common, my_b, '-', color=cols[i], lw=2, label=f'{lbl} my')
        ax.plot(t_common, mz_b, '--', color=cols[i], lw=1.5, alpha=0.7, label=f'{lbl} mz')
    ax.axhline(0, color='k', ls=':', lw=1, alpha=0.4)
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('DW core magnetization')
    ax.set_title('DW core: my (solid), mz (dashed)\nAbove-Walker: mz oscillates!')
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Py Zhang-Li STT Walker Breakdown (GPU, flat 200x10x1 strip, 5nm cells)\n'
        f'Neel DW (core along y), alpha={alpha}, xi={xi}, xi/alpha={int(abs(v_sub_ratio))}  '
        f'J_W ~ {J_W_sim/1e12:.2f}e12 A/m2',
        fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '29_walker_breakdown_gpu.png')
    plt.savefig(out, dpi=120); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Strip: {Nx}x{Ny}x{Nz}, dx={dx*1e9:.0f}nm, flat (Neel DW, core along y)")
print(f"  Py: Ms={Ms/1e3:.0f}kA/m, alpha={alpha}, xi={xi}, xi/alpha={int(abs(v_sub_ratio))}")
print(f"  Walker breakdown J_W ~ {J_W_sim/1e12:.2f}e12 A/m2")
print(f"  Sub-Walker: v/u = {v_sub_ratio:.0f}  |  Above-Walker: v/u = {v_abv_ratio:.3f}")
for J_val, v_dw, u_val, regime in zip(J_arr, v_list, u_list, regime_list):
    print(f"  J={J_val/1e12:.2f}e12  v={v_dw:.1f}m/s  v/u={v_dw/u_val:.2f}  [{regime}]")
