"""
Notebook 28: muMAG Standard Problem #4 GPU Full Validation

SP#4 (Field A): Permalloy 500x125x3 nm, 200x50x1 cells (2.5x2.5x3 nm).
H_ext = (-24.6, 4.3, 0) kA/m applied instantaneously at t=0.

muMAG reference (consensus):  <mx>(1 ns) = -0.9862,  t_switch ~ 0.175 ns

This notebook:
  A) GPU RK4  (fixed dt = 5e-14 s, 20000 steps, 1 ns)
  B) GPU RK45 (adaptive DOPRI5, same physical time 1 ns)
  C) CPU  RK4 (first 500 steps = 25 ps, accuracy cross-check)
  D) Plot <mx>(t), <my>(t) for GPU RK4 + RK45 vs muMAG reference
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

print("Notebook 28: SP#4 GPU Full Validation")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Setup: SP#4 Field A
# 200x50x1, dx=dy=2.5 nm, dz=3 nm -> physical 500x125x3 nm Permalloy
# ---------------------------------------------------------------------------
Nx, Ny, Nz = 200, 50, 1
dx, dy, dz  = 2.5e-9, 2.5e-9, 3.0e-9
g = mm.StructuredGrid(Nx, Ny, Nz, dx, dy, dz)

mat = mm.Material.permalloy()          # Ms=860 kA/m, A=13 pJ/m, alpha=0.02
Hx, Hy = -24.6e3, 4.3e3               # SP#4 Field A [A/m]

print(f"\nSP#4 geometry: {Nx}x{Ny}x{Nz}, dx={dx*1e9:.1f}x{dy*1e9:.1f}x{dz*1e9:.1f} nm")
print(f"  Physical: {int(Nx*dx*1e9)}x{int(Ny*dy*1e9)}x{int(Nz*dz*1e9)} nm")
print(f"  H_ext: ({Hx/1e3:.1f}, {Hy/1e3:.1f}, 0) kA/m")
print(f"  muMAG ref: <mx>(1 ns) = -0.9862,  t_switch ~ 175 ps")

# Initial state: quasi-uniform, slightly tilted toward +y
a0 = np.zeros((Nz, Ny, Nx, 3))
a0[..., 0] = 1.0; a0[..., 1] = 0.1
n = np.linalg.norm(a0, axis=-1, keepdims=True)
a0 = a0 / n
m0 = mm.VectorField3D(g)
mm.from_numpy(m0, a0)

def mean_mx_my(m_field):
    a = mm.to_numpy(m_field)         # (nz, ny, nx, 3)
    return float(a[..., 0].mean()), float(a[..., 1].mean())

# GPU fields (shared across integrators)
demag_g  = mm.DemagFieldGPU(g)
exch_g   = mm.ExchangeFieldGPU(g)
zeeman_g = mm.ZeemanFieldGPU(g)
zeeman_g.H_ext = mm.Vec3(Hx, Hy, 0.0)

fields_g = mm.FieldSumGPU()
fields_g.add(exch_g)
fields_g.add(zeeman_g)

torques_g = mm.SpinTorqueSumGPU()    # empty (no STT)

# ---------------------------------------------------------------------------
# Part A: GPU RK4  (dt = 5e-14 s, 1 ns)
# ---------------------------------------------------------------------------
print("\n--- Part A: GPU RK4 (dt=5e-14 s, 1 ns = 20000 steps) ---")

dt_rk4   = 5e-14
n_total  = 20000          # 1 ns
log_int  = 1000           # log every 50 ps

integ_rk4 = mm.RK4IntegratorGPU(g, dt_rk4)
integ_rk4.upload(m0)

mx_rk4, my_rk4, t_rk4 = [], [], []
switched_rk4, t_switch_rk4 = False, -1.0

print(f"  {'t (ps)':>9}  {'<mx>':>10}  {'<my>':>10}")
t0 = time.time()
for k in range(1, n_total + 1):
    integ_rk4.step(mat, demag_g, fields_g, torques_g)
    if k % log_int == 0:
        m_tmp = mm.VectorField3D(g)
        integ_rk4.download(m_tmp)
        mx, my = mean_mx_my(m_tmp)
        t_ps = k * dt_rk4 * 1e12
        mx_rk4.append(mx); my_rk4.append(my); t_rk4.append(t_ps)
        if k % (log_int * 4) == 0:
            print(f"  {t_ps:>9.0f}  {mx:>10.6f}  {my:>10.6f}")
        if not switched_rk4 and mx < 0:
            switched_rk4 = True
            t_switch_rk4 = t_ps

wall_rk4 = time.time() - t0
mx_fin_rk4 = mx_rk4[-1]
print(f"  Wall: {wall_rk4:.1f} s  ({wall_rk4/n_total*1e3:.3f} ms/step)")
print(f"  <mx>(1 ns) = {mx_fin_rk4:.4f}  (muMAG: -0.9862,  err={abs(mx_fin_rk4+0.9862)*100:.2f}%)")
if switched_rk4:
    print(f"  t_switch ~ {t_switch_rk4:.0f} ps  (muMAG: 175 ps)")

# ---------------------------------------------------------------------------
# Part B: GPU RK45 (adaptive DOPRI5, 1 ns)
# ---------------------------------------------------------------------------
print("\n--- Part B: GPU RK45 adaptive (DOPRI5, 1 ns) ---")

t_end    = 1e-9
opts_rk45 = mm.RK45GPUOptions()
opts_rk45.dt_init = 1e-14
opts_rk45.rtol    = 1e-4
opts_rk45.atol    = 1e-6

integ_rk45 = mm.RK45IntegratorGPU(g, opts_rk45)
integ_rk45.upload(m0)

mx_rk45, my_rk45, t_rk45 = [], [], []
switched_rk45, t_switch_rk45 = False, -1.0

t_now = 0.0; n_steps_rk45 = 0
log_every = 100    # log every 100 accepted steps
t0 = time.time()

while t_now < t_end - 1e-18:
    integ_rk45.step(mat, demag_g, fields_g, torques_g)
    t_now += integ_rk45.dt
    n_steps_rk45 += 1
    if n_steps_rk45 % log_every == 0 or t_now >= t_end:
        m_tmp = mm.VectorField3D(g)
        integ_rk45.download(m_tmp)
        mx, my = mean_mx_my(m_tmp)
        t_ps = t_now * 1e12
        mx_rk45.append(mx); my_rk45.append(my); t_rk45.append(t_ps)
        if not switched_rk45 and mx < 0:
            switched_rk45 = True
            t_switch_rk45 = t_ps

wall_rk45 = time.time() - t0
mx_fin_rk45 = mx_rk45[-1]
print(f"  Steps taken: {n_steps_rk45}  Wall: {wall_rk45:.1f} s  ({wall_rk45/max(n_steps_rk45,1)*1e3:.3f} ms/step)")
print(f"  <mx>(1 ns) = {mx_fin_rk45:.4f}  (muMAG: -0.9862,  err={abs(mx_fin_rk45+0.9862)*100:.2f}%)")
if switched_rk45:
    print(f"  t_switch ~ {t_switch_rk45:.0f} ps  (muMAG: 175 ps)")

# ---------------------------------------------------------------------------
# Part C: CPU RK4 first 25 ps (500 steps) -- cross-check
# ---------------------------------------------------------------------------
print("\n--- Part C: CPU RK4 (first 25 ps = 500 steps) ---")

demag_c  = mm.DemagField(g)
exch_c   = mm.ExchangeField()
zeeman_c = mm.ZeemanField(mm.Vec3(Hx, Hy, 0.0))

heff_c = mm.EffectiveFieldSum()
heff_c.add(exch_c); heff_c.add(demag_c); heff_c.add(zeeman_c)

integ_cpu = mm.RK4Integrator(dt_rk4)
m_cpu = mm.VectorField3D(g)
mm.from_numpy(m_cpu, a0)

n_cpu = 1000   # 1000 steps = 50 ps  (matches first GPU log point at 50 ps)
t0 = time.time()
for _ in range(n_cpu):
    integ_cpu.step(m_cpu, mat, heff_c)
wall_cpu = time.time() - t0

mx_cpu, my_cpu = mean_mx_my(m_cpu)
print(f"  CPU 50 ps ({n_cpu} steps): <mx>={mx_cpu:.6f}  <my>={my_cpu:.6f}  ({wall_cpu:.2f} s)")

# GPU at 50 ps (first log point)
idx_50ps = min(range(len(t_rk4)), key=lambda i: abs(t_rk4[i] - 50))
print(f"  GPU RK4 ~50 ps: <mx>={mx_rk4[idx_50ps]:.6f}  (t={t_rk4[idx_50ps]:.0f} ps)")
diff = abs(mx_rk4[idx_50ps] - mx_cpu)
print(f"  CPU/GPU diff: {diff:.2e}  (should be < 1e-4)")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
mu_ref = -0.9862
print("\n=== Summary ===")
n_cells = Nx * Ny * Nz
print(f"  Grid: {Nx}x{Ny}x{Nz} = {n_cells:,} cells, {int(Nx*dx*1e9)}x{int(Ny*dy*1e9)}x{int(Nz*dz*1e9)} nm Py")
print(f"  H_ext = ({Hx/1e3:.1f}, {Hy/1e3:.1f}, 0) kA/m  (SP#4 Field A)")
print(f"")
print(f"  {'Method':<20} {'<mx>(1ns)':>12} {'err%':>8} {'t_switch(ps)':>14} {'wall(s)':>9}")
print(f"  {'-'*68}")
sw_rk4_str  = f"{int(t_switch_rk4)}" if switched_rk4 else "N/A"
sw_rk45_str = f"{int(t_switch_rk45)}" if switched_rk45 else "N/A"
print(f"  {'muMAG consensus':<20} {mu_ref:>12.4f} {'--':>8} {'~175':>14} {'--':>9}")
print(f"  {'GPU RK4':<20} {mx_fin_rk4:>12.4f} {abs(mx_fin_rk4-mu_ref)/abs(mu_ref)*100:>8.2f}"
      f" {sw_rk4_str:>14}  {wall_rk4:>8.1f}")
print(f"  {'GPU RK45':<20} {mx_fin_rk45:>12.4f} {abs(mx_fin_rk45-mu_ref)/abs(mu_ref)*100:>8.2f}"
      f" {sw_rk45_str:>14}  {wall_rk45:>8.1f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(t_rk4, mx_rk4, '-', color='C0', lw=2, label='GPU RK4 (dt=5e-14)')
    ax.plot(t_rk45, mx_rk45, '--', color='C1', lw=2,
            label=f'GPU RK45 (adaptive, {n_steps_rk45} steps)')
    ax.axhline(mu_ref, color='C3', ls=':', lw=2, alpha=0.8,
               label=f'muMAG ref = {mu_ref}')
    ax.axvline(175, color='k', ls='--', lw=1, alpha=0.5,
               label='muMAG t_switch = 175 ps')
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('<mx>')
    ax.set_title('SP#4 Field A: <mx> vs time\nPermalloy 500x125x3 nm (GPU)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    ax = axes[1]
    ax.plot(t_rk4, my_rk4, '-', color='C0', lw=2, label='GPU RK4')
    ax.plot(t_rk45, my_rk45, '--', color='C1', lw=2, label='GPU RK45')
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('<my>')
    ax.set_title('SP#4 Field A: <my> vs time')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    annot_rk4  = f"<mx>={mx_fin_rk4:.4f} (err={abs(mx_fin_rk4-mu_ref)/abs(mu_ref)*100:.2f}%)"
    annot_rk45 = f"<mx>={mx_fin_rk45:.4f} (err={abs(mx_fin_rk45-mu_ref)/abs(mu_ref)*100:.2f}%)"
    plt.suptitle(
        f'muMAG SP#4 GPU Validation\n'
        f'RK4: {annot_rk4}   RK45: {annot_rk45}\n'
        f'muMAG ref: <mx>(1ns)=-0.9862,  t_switch~175ps',
        fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '28_sp4_gpu_validation.png')
    plt.savefig(out, dpi=120); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")
