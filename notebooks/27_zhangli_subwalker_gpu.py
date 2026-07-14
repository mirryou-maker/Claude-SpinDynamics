"""
Notebook 27: Zhang-Li DW Sub-Walker to Above-Walker Transition (GPU)

NB24 used K=0 Py (J_W ~ 0): all currents were above-Walker.
Here K_x > 0 raises J_W, revealing BOTH sub-Walker and above-Walker regimes.
High xi (=0.5) creates dramatic 10x velocity enhancement below Walker.

Physics:
  Sub-Walker  (J < J_W): v = (xi/alpha) * u        (large enhancement: xi/alpha = 10)
  Above-Walker (J > J_W): precessional; time-avg v ~ -xi/sqrt(a^2+xi^2) * u (~ -u)
  Walker velocity: u_W = gamma_0 * pi * l_ex * mu0 * H_W

Material: Py-like  Ms=860 kA/m, A=13 pJ/m, K_x=5 kJ/m3, alpha=0.05, xi=0.5, P=0.5
Grid: 600x1x1, dx=3 nm (1.8 um strip)
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

print("Notebook 27: Zhang-Li DW Sub-Walker -> Above-Walker (GPU)")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material: Py with weak in-plane K_x > 0  and high xi
# ---------------------------------------------------------------------------
mu0     = 4e-7 * np.pi
mu_B    = 9.274e-24
e_ch    = 1.6022e-19
gamma_0 = 1.76e11

Ms    = 860e3;  A  = 13e-12;  K_x = 5e3
alpha = 0.05;   xi = 0.5;     P   = 0.5

mat = mm.Material()
mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = K_x
mat.easy_axis = mm.Vec3(1, 0, 0); mat.alpha = alpha

l_ex    = np.sqrt(2*A / (mu0*Ms**2))
H_K     = 2*K_x / (mu0*Ms)
H_W     = alpha * (H_K + Ms) / (1 + xi**2)
dw_pilex = np.pi * l_ex
u_W     = gamma_0 * dw_pilex * mu0 * H_W
J_W     = u_W * e_ch * Ms / (P * mu_B)

v_sub_ratio = xi / alpha
v_above_ratio = -xi / np.sqrt(alpha**2 + xi**2)

print(f"\nMaterial: Ms={Ms/1e3:.0f}kA/m, K_x={K_x/1e3:.0f}kJ/m3, alpha={alpha}, xi={xi}")
print(f"  l_ex = {l_ex*1e9:.1f} nm,  DW width = pi*l_ex = {dw_pilex*1e9:.1f} nm")
print(f"  H_W  = mu0*{H_W:.0f} A/m = {mu0*H_W*1e3:.2f} mT  (Walker field)")
print(f"  u_W  = {u_W:.2f} m/s  (Walker drift velocity)")
print(f"  J_W  = {J_W/1e12:.3f} e12 A/m2  (Walker current)")
print(f"  Sub-Walker:   v/u = xi/alpha = {v_sub_ratio:.0f}  (10x enhancement!)")
print(f"  Above-Walker: v/u -> {v_above_ratio:.3f}  (approx -u)")

# ---------------------------------------------------------------------------
# Grid: 600x1x1, dx=3nm (1.8 um strip)
# ---------------------------------------------------------------------------
dx = 3e-9; Nx = 600
g = mm.StructuredGrid(Nx, 1, 1, dx, dx, dx)
x0 = Nx // 2 * dx
print(f"\nGrid: {Nx}x1x1, dx={dx*1e9:.0f}nm, L={Nx*dx*1e6:.1f}um")

# Initial: Neel DW at center
dw_w = dw_pilex
a0 = np.zeros((1, 1, Nx, 3))
for ix in range(Nx):
    s = ((ix + 0.5) * dx - x0) / dw_w
    a0[0, 0, ix, 0] = -np.tanh(s)
    a0[0, 0, ix, 2] =  1.0 / np.cosh(s)
a0 = a0 / np.linalg.norm(a0, axis=-1, keepdims=True).clip(1e-20)

m0 = mm.VectorField3D(g); mm.from_numpy(m0, a0)

# GPU objects
demag_g = mm.DemagFieldGPU(g)
exch_g  = mm.ExchangeFieldGPU(g)
aniso_g = mm.UniaxialAnisotropyFieldGPU(g)
fields_g = mm.FieldSumGPU()
fields_g.add(exch_g); fields_g.add(aniso_g)

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
J_factors = np.array([0.1, 0.3, 0.6, 0.9, 1.3, 2.0, 3.5, 5.0])
J_arr     = J_factors * J_W

dt_sw = 2e-14; t_sw = 0.4e-9; n_sw = int(t_sw / dt_sw); chk = 1000

print(f"\nPart A sweep: dt={dt_sw:.0e}s, t={t_sw*1e9:.1f}ns, {n_sw} steps/J")

v_list = []; u_list = []
t0_total = time.time()

for J_val in J_arr:
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
    skip = len(t_list) // 4
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
    ax.plot(J_factors, np.array(v_list), 'o-', color='C0', lw=2, ms=8, label='GPU ZhangLi (sim)')
    ax.plot(J_factors, v_sub_th, '--', color='C2', lw=2,
            label=f'Sub-Walker: xi/alpha*u = {v_sub_ratio:.0f}u')
    ax.plot(J_factors, v_abv_th, ':', color='C3', lw=2,
            label=f'Above-Walker: {v_above_ratio:.3f}u')
    ax.axvline(1.0, color='k', ls='--', lw=1.5, alpha=0.7, label=f'J_W={J_W/1e12:.2f}e12')
    ax.set_xlabel('J / J_W')
    ax.set_ylabel('DW velocity (m/s)')
    ax.set_title(f'Walker breakdown (xi/alpha = {v_sub_ratio:.0f})\nJ_W = {J_W/1e12:.2f}e12 A/m2')
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # Trajectory
    ax = axes[1]
    cols = ['C0', 'C1', 'C2']
    for i, (lbl, pos_b) in enumerate(zip(lbl_cases, traj_pos)):
        ax.plot(t_common, np.array(pos_b)*1e9, '-', color=cols[i], lw=2, label=lbl)
    ax.axhline(x0*1e9, color='k', ls='--', lw=1, alpha=0.4, label='initial')
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('DW position (nm)')
    ax.set_title('DW position vs time\n(sub-Walker: fast; above-Walker: slow!)')
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # mz at core (precession marker)
    ax = axes[2]
    for i, (lbl, mz_b) in enumerate(zip(lbl_cases, traj_mz_core)):
        ax.plot(t_common, mz_b, '-', color=cols[i], lw=2, label=lbl)
    ax.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('mz at DW core')
    ax.set_title('mz at DW core:\nconstant=sub-Walker, oscillating=above-Walker')
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
