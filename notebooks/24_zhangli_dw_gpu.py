"""
Notebook 24: GPU Zhang-Li STT Domain Wall Motion

Simulates current-driven domain wall (DW) motion via Zhang-Li spin transfer
torque using RK4IntegratorGPU + ZhangLiSTTGPU.

Sweeps current density J and measures DW velocity v_dw vs J, shows
Walker breakdown (oscillatory motion above J_W).

Physics (Zhang-Li STT):
  dm/dt += u [(j*grad) m - xi * m x (j*grad) m]
  u = P * mu_B * J / (e * Ms)
  Walker breakdown at: J_W = alpha*Ms*mu0 / (2*P*mu_B*(1+xi^2)) * H_k

Material: Permalloy strip  Ms=860 kA/m, A=13 pJ/m, alpha=0.01, K=0
          in-plane DW (Bloch or Neel along y), current along x
Grid: 200x1x1 strip, dx=5 nm (1 um strip)
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc-cuda', 'python'))
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')

import numpy as np
import micromag as mm

print("Notebook 24: GPU Zhang-Li STT Domain Wall Motion")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material: Permalloy strip (in-plane magnetization)
# ---------------------------------------------------------------------------
Ms    = 860e3   # A/m
A     = 13e-12  # J/m
K     = 0.0     # no uniaxial anisotropy (soft Py)
alpha = 0.01
P     = 0.5     # polarization
xi    = 0.05    # non-adiabaticity (typical Py)

mu0   = 4e-7 * np.pi
mu_B  = 9.274e-24   # J/T
e_ch  = 1.6022e-19

mat = mm.Material()
mat.Ms         = Ms
mat.A_exchange = A
mat.K_uniaxial = 0.0
mat.easy_axis  = mm.Vec3(1, 0, 0)
mat.alpha      = alpha

# Exchange length
l_ex = np.sqrt(2 * A / (mu0 * Ms**2))
print(f"\nExchange length l_ex = {l_ex*1e9:.1f} nm")

# Walker breakdown (for in-plane Py strip):
# H_W = alpha * Ms / (1 + xi^2)  [A/m]
# J_W = 2*e*Ms*alpha*(1+xi^2) / (P*mu_B*xi) * H_W
# Simplified: J_W ~ 2*e*Ms^2*alpha^2 / (P*mu_B*xi)
H_W   = alpha * Ms / (1 + xi**2)
u_W   = alpha * mu0 * Ms / (1 + xi**2)  # Walker velocity
J_W   = u_W * e_ch * Ms / (P * mu_B)
v_W   = xi / alpha * mu_B * P * J_W / (e_ch * Ms)  # = xi/alpha * u_W
print(f"Walker H_W = {mu0*H_W*1e3:.2f} mT  (in-plane anisotropy field)")
print(f"Walker J_W ~ {J_W/1e12:.4f} e12 A/m2  (very small for K=0 Py)")
print(f"Walker velocity ~ {v_W:.2f} m/s")
print(f"NOTE: For K=0 Py, J_W -> 0: ALL test currents are ABOVE Walker breakdown.")
print(f"  Above Walker (precessional regime): |v| ~ u (not xi/alpha*u = {xi/alpha}*u)")
print(f"  Above-Walker theory: |v| = u*xi/sqrt(alpha^2+xi^2) = {xi/np.sqrt(alpha**2+xi**2):.3f}*u")

# ---------------------------------------------------------------------------
# Grid: 200x1x1 Py strip, dx=5 nm (total length = 1 um)
# ---------------------------------------------------------------------------
dx = 5e-9
Nx = 200
g = mm.StructuredGrid(Nx, 1, 1, dx, dx, dx)
print(f"\nGrid: {Nx}x1x1, dx={dx*1e9:.0f} nm, L={Nx*dx*1e6:.1f} um")

# ---------------------------------------------------------------------------
# Initial state: Neel domain wall at center
# DW profile: mx = -tanh((x-x0)/dw), mz = sech((x-x0)/dw)
# DW width parameter dw = pi * l_ex
# ---------------------------------------------------------------------------
dw_width = np.pi * l_ex
x0 = Nx // 2 * dx   # center of strip

a0 = np.zeros((1, 1, Nx, 3))
for ix in range(Nx):
    x = (ix + 0.5) * dx - x0
    s = x / dw_width
    a0[0, 0, ix, 0] = -np.tanh(s)          # mx
    a0[0, 0, ix, 1] = 0.0                   # my = 0 (Neel wall)
    a0[0, 0, ix, 2] = 1.0 / np.cosh(s)     # mz
# Normalize
norms = np.linalg.norm(a0, axis=-1, keepdims=True)
norms = np.where(norms < 1e-20, 1.0, norms)
a0 = a0 / norms

m0_field = mm.VectorField3D(g)
mm.from_numpy(m0_field, a0)
print(f"DW width dw = pi*l_ex = {dw_width*1e9:.1f} nm")

# Initial DW position (x where mx=0)
Q0_init = mm.topological_charge_Q(m0_field)
print(f"Initial topological charge: Q = {Q0_init:.3f}")

# ---------------------------------------------------------------------------
# GPU fields (shared across J sweep)
# ---------------------------------------------------------------------------
demag_gpu = mm.DemagFieldGPU(g)
exch_gpu  = mm.ExchangeFieldGPU(g)

fields = mm.FieldSumGPU()
fields.add(exch_gpu)

# Zhang-Li STT: current J along x
J_curr = mm.Vec3(1e12, 0, 0)   # direction only; magnitude set via .J_c
zl_gpu = mm.ZhangLiSTTGPU(g, J_curr, P, xi)

torques = mm.SpinTorqueSumGPU()
torques.add(zl_gpu)

def dw_position(m_field):
    """Find DW center: x where <mx>=0 (sign change of mx weighted)."""
    a = mm.to_numpy(m_field)
    mx = a[0, 0, :, 0]   # shape (Nx,)
    # Find zero crossing near center
    x_cells = (np.arange(Nx) + 0.5) * dx
    # Weighted center by |1-mx^2| (1 near DW core)
    weight = 1.0 - mx**2
    weight = np.maximum(weight, 0)
    if weight.sum() < 1e-30:
        return x0
    return float(np.sum(x_cells * weight) / weight.sum())

# ---------------------------------------------------------------------------
# Sweep: J from 0.2 to 4.0 x 10^12 A/m2 (6 values)
# Note: ZhangLiSTTGPU takes Vec3 J with magnitude; we scale via a dummy
# field trick — actually ZhangLiSTTGPU uses |J| from the Vec3 directly.
# We recreate with updated J_vec each iteration.
# ---------------------------------------------------------------------------
J_magnitudes = np.array([0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]) * 1e12

dt       = 2e-14    # s
t_sim    = 1.5e-9   # 1.5 ns per J value
n_steps  = int(t_sim / dt)
check_ev = 1000     # every 20 ps

v_dw_list = []
print(f"\nSweep: {len(J_magnitudes)} J values, dt={dt:.0e} s, t_sim={t_sim*1e9:.1f} ns")

t0_total = time.time()

for J_mag in J_magnitudes:
    # Recreate ZhangLiSTTGPU with updated J magnitude
    J_vec = mm.Vec3(J_mag, 0, 0)
    zl_local = mm.ZhangLiSTTGPU(g, J_vec, P, xi)
    torques_local = mm.SpinTorqueSumGPU()
    torques_local.add(zl_local)

    integ = mm.RK4IntegratorGPU(g, dt)
    integ.upload(m0_field)

    pos_list = []
    t_list   = []

    for step in range(0, n_steps, check_ev):
        for _ in range(check_ev):
            integ.step(mat, demag_gpu, fields, torques_local)
        m_tmp = mm.VectorField3D(g)
        integ.download(m_tmp)
        pos_list.append(dw_position(m_tmp))
        t_list.append((step + check_ev) * dt)

    # Linear fit of DW position vs time (ignore transient first 20%)
    skip = len(t_list) // 5
    if len(t_list) - skip >= 2:
        p = np.polyfit(t_list[skip:], pos_list[skip:], 1)
        v_dw = p[0]   # m/s
    else:
        v_dw = 0.0

    v_dw_list.append(v_dw)
    u_val = P * mu_B * J_mag / (e_ch * Ms)
    print(f"  J = {J_mag/1e12:.2f}e12  u = {u_val:.1f} m/s  v_dw = {v_dw:.1f} m/s"
          f"  (v/u = {v_dw/u_val:.2f})" if u_val > 0 else "")

total_time = time.time() - t0_total
print(f"\nTotal sweep: {total_time:.1f} s")

# Theory velocity (below Walker: v = xi/alpha * u)
v_theory_below = [xi / alpha * P * mu_B * J / (e_ch * Ms) for J in J_magnitudes]
v_above_th = [-xi/np.sqrt(alpha**2+xi**2) * P*mu_B*J/(e_ch*Ms) for J in J_magnitudes]
print(f"\nAbove-Walker theory (K=0 Py): v=-xi/sqrt(a2+xi2)*u = {-xi/np.sqrt(alpha**2+xi**2):.3f}*u")
print(f"  Predicted: {[f'{v:.0f}' for v in v_above_th]} m/s")
print(f"  Simulated: {[f'{v:.0f}' for v in v_dw_list]} m/s")
print(f"  Ratio sim/theory: {[f'{v/t:.2f}' if abs(t)>0.001 else 'N/A' for v,t in zip(v_dw_list,v_above_th)]}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    J_arr = J_magnitudes / 1e12
    v_arr = np.array(v_dw_list)
    v_th  = np.array(v_theory_below)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Above-Walker theory: |v| = u * xi / sqrt(alpha^2 + xi^2)
    u_arr   = np.array([P * mu_B * J / (e_ch * Ms) for J in J_magnitudes])
    v_above = -np.array([xi / np.sqrt(alpha**2 + xi**2) * u for u in u_arr])

    # DW velocity vs J
    ax = axes[0]
    ax.plot(J_arr, v_arr, 'o-', color='C0', lw=2, ms=7, label='GPU ZhangLi STT (sim)')
    ax.plot(J_arr, v_above, '--', color='C1', lw=1.5,
            label=f'Above-Walker: v=-xi/sqrt(a2+xi2)*u={-xi/np.sqrt(alpha**2+xi**2):.3f}*u')
    ax.set_xlabel('Current density J (1e12 A/m2)')
    ax.set_ylabel('DW velocity (m/s)')
    ax.set_title('DW Velocity vs J (K=0 Py, above-Walker)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # v/u (dimensionless efficiency)
    ax = axes[1]
    ax.plot(J_arr, v_arr / u_arr, 'o-', color='C2', lw=2, ms=7, label='Simulated v/u')
    ax.axhline(-xi / np.sqrt(alpha**2 + xi**2), color='C1', ls='--', lw=1.5,
               label=f'Above-Walker: -{xi/np.sqrt(alpha**2+xi**2):.3f}')
    ax.axhline(-xi / alpha, color='C3', ls=':', lw=1.5,
               label=f'Sub-Walker (if K!=0): -xi/alpha = {-xi/alpha:.0f}')
    ax.set_xlabel('Current density J (1e12 A/m2)')
    ax.set_ylabel('DW efficiency v/u')
    ax.set_title('DW Efficiency vs J')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f'Py strip Zhang-Li STT DW motion (GPU, {Nx}x1x1, dx={dx*1e9:.0f}nm)\n'
        f'alpha={alpha}, P={P}, xi={xi}  |  K=0: J_W~0, always above-Walker (v~-u not xi/alpha*u)',
        fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), '24_zhangli_dw_gpu.png')
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved: {out_path}")

except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Strip: Py {Nx}x1x1, dx={dx*1e9:.0f} nm, l_ex={l_ex*1e9:.1f} nm, dw={dw_width*1e9:.1f} nm")
print(f"  alpha={alpha}, P={P}, xi={xi}")
print(f"  Walker J_W={J_W/1e12:.3f} e12  v_W={v_W:.1f} m/s")
print(f"  Max simulated velocity: {max(v_dw_list):.1f} m/s at J={J_magnitudes[np.argmax(v_dw_list)]/1e12:.2f}e12")
print(f"  Sweep: {total_time:.1f} s ({total_time/len(J_magnitudes)*1000:.0f} ms/pt)")
