"""
Notebook 27: Zhang-Li DW Sub-Walker to Above-Walker Transition (GPU)

A 1x1-cell wire has N_yy = N_zz (cubic cells, isotropic self-demag), so an
easy-axis K_x alone gives NO transverse restoring force on the wall core:
the Walker threshold is ~0 and every current is above-Walker (the defect of
the original version of this notebook). The fix is an explicit TRANSVERSE
HARD-AXIS anisotropy K_perp (K < 0 along z, via a per-cell second
UniaxialAnisotropyFieldGPU): the wall core then prefers the y-axis, phi
precession costs energy, and a finite Walker current appears.

Physics (1-D collective coordinates, CS Zhang-Li convention u = J*P*muB/(e*Ms)):
  Sub-Walker  (J < J_W): steady tilt,  v = -(xi/alpha) * u   (10x enhancement)
  Above-Walker (J > J_W): precessing core; drift collapses (|v| << (xi/alpha)u)
  and approaches the -u asymptote only for J >> J_W (finite-window averages at
  J <= 3 J_W sit well below it - expected, not an error)
  Walker drift: u_W = [alpha/(xi - alpha)] * gamma_0 * Delta * mu0*H_Kperp / 2
  with Delta = sqrt(A/K_x), H_Kperp = 2*K_perp/(mu0*Ms).

Material: Py-like  Ms=860 kA/m, A=13 pJ/m, K_x=36 kJ/m3 (easy, x),
K_perp=0.5 MJ/m3 (hard, z), alpha=0.05, xi=0.5, P=0.5
Grid: 600x1x1, dx=3 nm (1.8 um strip)
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

print("Notebook 27: Zhang-Li DW Sub-Walker -> Above-Walker (GPU)")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material: Py with weak in-plane K_x > 0  and high xi
# ---------------------------------------------------------------------------
mu0     = 4e-7 * np.pi
mu_B    = 9.274e-24
e_ch    = 1.6022e-19
gamma_0 = 1.76e11

Ms    = 860e3;  A  = 13e-12;  K_x = 36e3;  K_perp = 0.5e6
alpha = 0.05;   xi = 0.5;     P   = 0.5

mat = mm.Material()
mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = K_x
mat.easy_axis = mm.Vec3(1, 0, 0); mat.alpha = alpha

l_ex     = np.sqrt(2*A / (mu0*Ms**2))
Delta    = np.sqrt(A / K_x)                 # DW width from easy-axis K_x
H_Kperp  = 2*K_perp / (mu0*Ms)              # transverse hard-axis field [A/m]
u_W      = (alpha / (xi - alpha)) * gamma_0 * Delta * mu0 * H_Kperp / 2
J_W      = u_W * e_ch * Ms / (P * mu_B)
tau_rel  = (1 + alpha**2) / (alpha * gamma_0 * mu0 * H_Kperp)  # settling time

v_sub_ratio = -xi / alpha
v_above_ratio = -xi / np.sqrt(alpha**2 + xi**2)

print(f"\nMaterial: Ms={Ms/1e3:.0f}kA/m, K_x={K_x/1e3:.0f}kJ/m3, "
      f"K_perp={K_perp/1e6:.1f}MJ/m3 (hard, z), alpha={alpha}, xi={xi}")
print(f"  Delta = sqrt(A/K_x) = {Delta*1e9:.1f} nm  (DW width)")
print(f"  mu0*H_Kperp = {mu0*H_Kperp*1e3:.0f} mT   settling tau = {tau_rel*1e9:.2f} ns")
print(f"  u_W  = {u_W:.1f} m/s  (Walker drift velocity)")
print(f"  J_W  = {J_W/1e12:.3f} e12 A/m2  (Walker current)")
print(f"  Sub-Walker:   v/u = -xi/alpha = {v_sub_ratio:.0f}  (10x enhancement!)")
print(f"  Above-Walker: v/u -> {v_above_ratio:.3f}  (approx -u)")

# ---------------------------------------------------------------------------
# Grid: 600x1x1, dx=3nm (1.8 um strip)
# ---------------------------------------------------------------------------
dx = 3e-9; Nx = 600
g = mm.StructuredGrid(Nx, 1, 1, dx, dx, dx)
x0 = Nx // 2 * dx
print(f"\nGrid: {Nx}x1x1, dx={dx*1e9:.0f}nm, L={Nx*dx*1e6:.1f}um")

# Initial: Neel DW at center (core along y: the hard-z anisotropy easy plane)
dw_w = Delta
a0 = np.zeros((1, 1, Nx, 3))
for ix in range(Nx):
    s = ((ix + 0.5) * dx - x0) / dw_w
    a0[0, 0, ix, 0] = -np.tanh(s)
    a0[0, 0, ix, 1] =  1.0 / np.cosh(s)   # core along y (easy plane)
a0 = a0 / np.linalg.norm(a0, axis=-1, keepdims=True).clip(1e-20)

m0 = mm.VectorField3D(g); mm.from_numpy(m0, a0)

# GPU objects
demag_g = mm.DemagFieldGPU(g)
exch_g  = mm.ExchangeFieldGPU(g)
aniso_g = mm.UniaxialAnisotropyFieldGPU(g)          # easy-axis K_x from mat

# Transverse hard axis: per-cell second anisotropy with K = -K_perp along z.
# (The GPU integrator path honours the material field; without this term the
# 1x1 wire has no Walker threshold at all.)
mat_hard = mm.Material()
mat_hard.Ms = Ms; mat_hard.A_exchange = A
mat_hard.K_uniaxial = -K_perp; mat_hard.easy_axis = mm.Vec3(0, 0, 1)
matf_hard = mm.MaterialField3D(g)
for i in range(Nx):
    matf_hard[i] = mat_hard
hard_g = mm.UniaxialAnisotropyFieldGPU(g)
hard_g.set_material_field(matf_hard)

fields_g = mm.FieldSumGPU()
fields_g.add(exch_g); fields_g.add(aniso_g); fields_g.add(hard_g)

def dw_pos(m_field):
    a = mm.to_numpy(m_field)[0, 0, :, 0]  # mx
    w = np.maximum(1.0 - a**2, 0)
    xs = (np.arange(Nx) + 0.5) * dx
    return float(np.sum(xs * w) / w.sum()) if w.sum() > 1e-30 else x0

def dw_mz_core(m_field):
    a = mm.to_numpy(m_field)[0, 0, :, :]   # (Nx, 3)
    mx = a[:, 0]
    idx = int(np.argmax(1 - mx**2))
    return float(a[idx, 2])

# ---------------------------------------------------------------------------
# Part A: v_dw vs J sweep  (8 values from 0.1 to 5 * J_W)
# ---------------------------------------------------------------------------
J_factors = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.3, 2.0, 3.0])
J_arr     = J_factors * J_W

dt_sw = 2e-14; chk = 500
# Regime-adapted windows: sub-Walker settles in ~3*tau (0.3 ns) but moves fast
# (strip-length limited); above-Walker drifts slowly but needs several
# precession periods to average, so it gets a longer window.
print(f"\nPart A sweep: dt={dt_sw:.0e}s, window 0.3 ns (sub) / 1.0 ns (above)")

v_list = []; u_list = []
t0_total = time.time()

for J_val in J_arr:
    t_sw = 0.3e-9 if J_val < J_W else 1.0e-9
    n_sw = int(t_sw / dt_sw)
    zl = mm.ZhangLiSTTGPU(g, mm.Vec3(J_val, 0, 0), P, xi)
    torq = mm.SpinTorqueSumGPU(); torq.add(zl)
    integ = mm.RK4IntegratorGPU(g, dt_sw)
    integ.upload(m0)
    pos_list, t_list = [], []
    for step in range(0, n_sw, chk):
        for _ in range(chk): integ.step(mat, demag_g, fields_g, torq)
        m_tmp = mm.VectorField3D(g); integ.download(m_tmp)
        pos_list.append(dw_pos(m_tmp))
        t_list.append((step + chk) * dt_sw)
    skip = len(t_list) // 2   # settling tau ~ 0.1 ns: fit steady half
    if len(t_list) - skip >= 3:
        p = np.polyfit(t_list[skip:], pos_list[skip:], 1)
        v_dw = float(p[0])
    else:
        v_dw = 0.0
    u_val = P * mu_B * J_val / (e_ch * Ms)
    v_list.append(v_dw); u_list.append(u_val)
    regime = "sub" if J_val < J_W else "above"
    print(f"  J={J_val/J_W:.1f}*J_W  u={u_val:.1f}m/s  v={v_dw:.1f}m/s  "
          f"v/u={v_dw/u_val:.2f} (exp {v_sub_ratio if J_val<J_W else v_above_ratio:.2f})  [{regime}]")

print(f"\nPart A total: {time.time()-t0_total:.0f}s")

# ---------------------------------------------------------------------------
# Part B: DW position(t) for 3 J cases: sub / near / above
# ---------------------------------------------------------------------------
print("\nPart B: trajectories for sub/near/above Walker...")
J_cases = [J_W * 0.5, J_W * 1.0, J_W * 2.5]
lbl_cases = [f"J=0.5*J_W (sub)", f"J=J_W (Walker)", f"J=2.5*J_W (above)"]

dt_tr = 2e-14; t_tr = 0.3e-9; n_tr = int(t_tr / dt_tr); chk_tr = 500
traj_pos, traj_mz_core, t_common = [], [], None

t0 = time.time()
for J_val, lbl in zip(J_cases, lbl_cases):
    zl = mm.ZhangLiSTTGPU(g, mm.Vec3(J_val, 0, 0), P, xi)
    torq = mm.SpinTorqueSumGPU(); torq.add(zl)
    integ = mm.RK4IntegratorGPU(g, dt_tr)
    integ.upload(m0)
    pos_b, mz_b = [], []
    for step in range(0, n_tr, chk_tr):
        for _ in range(chk_tr): integ.step(mat, demag_g, fields_g, torq)
        m_tmp = mm.VectorField3D(g); integ.download(m_tmp)
        pos_b.append(dw_pos(m_tmp))
        mz_b.append(dw_mz_core(m_tmp))
    traj_pos.append(pos_b); traj_mz_core.append(mz_b)
    if t_common is None:
        t_common = [(s + chk_tr) * dt_tr * 1e12 for s in range(0, n_tr, chk_tr)]
    print(f"  {lbl}: final pos={pos_b[-1]*1e9:.0f}nm  delta_x={(pos_b[-1]-x0)*1e9:.0f}nm")

print(f"Part B: {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Theory
    u_arr = np.array(u_list)
    v_sub_th  = v_sub_ratio * u_arr
    v_abv_th  = v_above_ratio * u_arr

    ax = axes[0]
    ax.plot(J_factors, np.array(v_list), 'o-', color='C0', lw=2, ms=8, label='GPU Zhang-Li (sim)')
    ax.plot(J_factors, v_sub_th, '--', color='C2', lw=2,
            label=rf'sub-Walker: $v=({v_sub_ratio:.0f})\,u$')
    ax.plot(J_factors, v_abv_th, ':', color='C3', lw=2,
            label=rf'above-Walker: $v={v_above_ratio:.3f}\,u$')
    ax.axvline(1.0, color='k', ls='--', lw=1.5, alpha=0.7, label=rf'$J_W={J_W/1e12:.2f}\times10^{{12}}$')
    ax.set_xlabel(r'$J/J_W$')
    ax.set_ylabel(r'$v_\mathrm{DW}$ (m s$^{-1}$)')
    ax.set_title(rf'Walker breakdown ($\xi/\alpha={abs(v_sub_ratio):.0f}$, hard-axis $K_\perp$)' + '\n' + rf'$J_W={J_W/1e12:.2f}\times10^{{12}}$ A m$^{{-2}}$')
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # Trajectory
    ax = axes[1]
    cols = ['C0', 'C1', 'C2']
    for i, (lbl, pos_b) in enumerate(zip(lbl_cases, traj_pos)):
        ax.plot(t_common, np.array(pos_b)*1e9, '-', color=cols[i], lw=2, label=lbl)
    ax.axhline(x0*1e9, color='k', ls='--', lw=1, alpha=0.4, label='initial')
    ax.set_xlabel(r'$t$ (ps)'); ax.set_ylabel(r'$x_\mathrm{DW}$ (nm)')
    ax.set_title('DW position vs time\n(sub-Walker: fast; above-Walker: slow)')
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # mz at core (precession marker)
    ax = axes[2]
    for i, (lbl, mz_b) in enumerate(zip(lbl_cases, traj_mz_core)):
        ax.plot(t_common, mz_b, '-', color=cols[i], lw=2, label=lbl)
    ax.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
    ax.set_xlabel(r'$t$ (ps)'); ax.set_ylabel(r'$m_z$ at DW core')
    ax.set_title('$m_z$ at DW core:\nconstant = sub-Walker, oscillating = above-Walker')
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Py Zhang-Li STT Walker Breakdown (GPU ZhangLiSTTGPU)\n'
        f'K_x={K_x/1e3:.0f}kJ/m3, alpha={alpha}, xi={xi} -> xi/alpha={xi/alpha:.0f}  '
        f'J_W={J_W/1e12:.2f}e12, u_W={u_W:.0f}m/s',
        fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '27_zhangli_subwalker_gpu.png')
    plt.savefig(out, dpi=120); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  J_W = {J_W/1e12:.3f} e12 A/m2,  u_W = {u_W:.2f} m/s")
print(f"  Sub-Walker: v/u = xi/alpha = {v_sub_ratio:.0f}  |  Above-Walker: v/u = {v_above_ratio:.3f}")
for J_val, v_dw, u_val in zip(J_arr, v_list, u_list):
    print(f"  J={J_val/J_W:.1f}*J_W  v={v_dw:.1f}m/s  u={u_val:.1f}m/s  v/u={v_dw/u_val:.2f}")
