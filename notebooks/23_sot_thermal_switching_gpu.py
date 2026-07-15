"""
Notebook 23: GPU SOT Thermal-Assisted Switching (SLLG + SpinOrbitTorqueGPU)

Simulates spin-orbit torque (SOT) switching of a Pt/Co nanopillar at
finite temperature using HeunIntegratorGPU (SLLG) + SpinOrbitTorqueGPU.

Performance note:
  HeunIntegratorGPU 1x1x1 grid: ~284 us/step (sync-dominated).
  Design: dt=1e-13, t_max=0.5ns (5000 steps/trial), 5 trials/pt.

Produces:
 (a) Single trajectory: mz(t) at T=300 K, J=3.0e12 A/m2
 (b) P_switch vs T at fixed J=3.0e12 (5 T-points, 5 trials)
 (c) P_switch vs J at T=300 K (7 J-points, 5 trials)

Material: Pt/Co  Ms=580 kA/m, K=0.5 MJ/m3, alpha=0.02, d_fm=3 nm
          theta_SH=0.1, sigma=+y
"""

import os, sys, time
from pathlib import Path

def _add_micromag_to_path():
    """Locate the micromag module + its DLLs in a release package
    (../runtime-dll + ../<variant>/python for GPU, ../python for CPU) or a source
    build (../build/<preset>/python + system CUDA). Works from any working dir."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    def _adddll(_d):                      # add_dll_directory is Windows-only
        if hasattr(os, "add_dll_directory") and os.path.isdir(_d):
            os.add_dll_directory(_d)
    def _hasmod(_p):
        _pat = "_micromag*.pyd" if sys.platform == "win32" else "_micromag*.so"
        return bool(list(_p.glob(_pat)))
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"
    if rtd.is_dir():
        _adddll(str(rtd))
        for _v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            _py = root / _v / "python"
            if _hasmod(_py):
                sys.path.insert(0, str(_py)); return
    if _hasmod(root / "python"):
        _adddll(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _adddll(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
    for _p in ("windows-msvc-cuda", "windows-msvc", "linux-gcc-cuda", "linux-gcc"):
        _py = root / "build" / _p / "python"
        if _hasmod(_py):
            sys.path.insert(0, str(_py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()
import numpy as np
import micromag as mm

print("Notebook 23: GPU SOT Thermal Switching  (SLLG + SpinOrbitTorqueGPU)")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
mu0  = 4e-7 * np.pi
hbar = 1.0546e-34
e_ch = 1.6022e-19

Ms       = 580e3
A        = 15e-12
K        = 0.5e6
alpha    = 0.02
d_fm     = 3e-9
theta_SH = 0.10
eta_DL   = 1.0
eta_FL   = 0.0

mat = mm.Material()
mat.Ms         = Ms
mat.A_exchange = A
mat.K_uniaxial = K
mat.easy_axis  = mm.Vec3(0, 0, 1)
mat.alpha      = alpha

mu0_Heff   = 2 * K / Ms - mu0 * Ms           # net PMA field [T]
J_c_theory = 2 * e_ch * alpha * Ms * d_fm * mu0_Heff / (hbar * theta_SH * eta_DL)
print(f"\nTheory: mu0*Heff={mu0_Heff*1e3:.1f} mT  J_c_SOT={J_c_theory/1e12:.3f} e12 A/m2")

# ---------------------------------------------------------------------------
# Grid + shared GPU objects
# ---------------------------------------------------------------------------
g = mm.StructuredGrid(1, 1, 1, d_fm, d_fm, d_fm)

demag_gpu = mm.DemagFieldGPU(g)
exch_gpu  = mm.ExchangeFieldGPU(g)
aniso_gpu = mm.UniaxialAnisotropyFieldGPU(g)

fields = mm.FieldSumGPU()
fields.add(exch_gpu)
fields.add(aniso_gpu)

sigma  = mm.Vec3(0, 1, 0)
sot    = mm.SpinOrbitTorqueGPU(g, 1e12, theta_SH, d_fm, sigma, eta_DL, eta_FL)
torques = mm.SpinTorqueSumGPU()
torques.add(sot)

# Initial state: +z tilted 5 degrees
theta0 = np.deg2rad(5)
a0_arr = np.zeros((1, 1, 1, 3))
a0_arr[0, 0, 0] = [np.sin(theta0), 0.0, np.cos(theta0)]

def make_m0():
    m = mm.VectorField3D(g)
    mm.from_numpy(m, a0_arr)
    return m

# Simulation parameters — optimized for ~284 us/step overhead
dt       = 1e-13    # s  (larger dt → fewer steps)
t_max    = 0.5e-9   # 0.5 ns
n_steps  = int(t_max / dt)   # 5000 steps
check_ev = 100               # every 10 ps
n_trials = 5

print(f"\nParameters: dt={dt:.0e} s, t_max={t_max*1e12:.0f} ps, {n_steps} steps/trial")
print(f"  check every {check_ev} steps = {check_ev*dt*1e12:.0f} ps,  {n_trials} trials/point")

def run_trial(J_c_val, T_K, seed):
    """Run one trial, return True if switched (mz < -0.5) within t_max."""
    sot.J_c = J_c_val
    integ = mm.HeunIntegratorGPU(g, dt, seed=seed)
    integ.upload(make_m0())
    for step in range(0, n_steps, check_ev):
        for _ in range(check_ev):
            integ.step(mat, demag_gpu, fields, T_K, torques)
        m_tmp = mm.VectorField3D(g)
        integ.download(m_tmp)
        if float(mm.to_numpy(m_tmp)[0, 0, 0, 2]) < -0.5:
            return True
    return False

# ---------------------------------------------------------------------------
# Part A: Single trajectory (T=300 K, J=3.0e12)
# ---------------------------------------------------------------------------
print("\n--- Part A: Single trajectory (T=300K, J=3.0e12) ---")
J_demo = 3.0e12
T_demo = 300.0
sot.J_c = J_demo

integ_a = mm.HeunIntegratorGPU(g, dt, seed=42)
integ_a.upload(make_m0())

mz_traj, t_arr = [], []
t0 = time.time()

for step in range(0, n_steps, check_ev):
    for _ in range(check_ev):
        integ_a.step(mat, demag_gpu, fields, T_demo, torques)
    m_tmp = mm.VectorField3D(g)
    integ_a.download(m_tmp)
    mz_traj.append(float(mm.to_numpy(m_tmp)[0, 0, 0, 2]))
    t_arr.append((step + check_ev) * dt * 1e12)

t_A = time.time() - t0
mz_arr = np.array(mz_traj)
sw_idx = np.where(mz_arr < -0.5)[0]
t_sw_A = t_arr[sw_idx[0]] if len(sw_idx) > 0 else None
print(f"  t_sw = {t_sw_A:.0f} ps" if t_sw_A else "  No switch in 0.5 ns")
print(f"  Time: {t_A:.1f} s")

# ---------------------------------------------------------------------------
# Part B: P_switch vs T  (J=3.0e12, 5 T-points, n_trials trials each)
# ---------------------------------------------------------------------------
print(f"\n--- Part B: P_switch vs T (J={J_demo/1e12:.1f}e12, {n_trials} trials/pt) ---")

T_values   = np.array([0, 77, 150, 300, 500], dtype=float)
sw_prob_T  = []
t0 = time.time()

for T in T_values:
    n_sw = sum(1 for seed in range(n_trials) if run_trial(J_demo, T, seed))
    p_sw = n_sw / n_trials
    sw_prob_T.append(p_sw)
    print(f"  T={T:4.0f}K  P_sw={p_sw:.2f}  ({n_sw}/{n_trials})")

print(f"  Time: {time.time()-t0:.1f} s")

# ---------------------------------------------------------------------------
# Part C: P_switch vs J  (T=300 K, 7 J-points, n_trials trials each)
# ---------------------------------------------------------------------------
print(f"\n--- Part C: P_switch vs J (T=300K, {n_trials} trials/pt) ---")

J_sweep   = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]) * 1e12
sw_prob_J = []
t0 = time.time()

for J in J_sweep:
    n_sw = sum(1 for seed in range(n_trials) if run_trial(J, 300.0, seed * 7 + 3))
    p_sw = n_sw / n_trials
    sw_prob_J.append(p_sw)
    print(f"  J={J/1e12:.2f}e12  P_sw={p_sw:.2f}  ({n_sw}/{n_trials})")

print(f"  Time: {time.time()-t0:.1f} s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # A: mz trajectory
    ax = axes[0]
    ax.plot(t_arr, mz_arr, lw=1.2, color='C0')
    ax.axhline(-0.5, color='k', ls='--', lw=1, alpha=0.5)
    ax.axhline( 0.5, color='k', ls='--', lw=1, alpha=0.5)
    if t_sw_A:
        ax.axvline(t_sw_A, color='C3', ls='--', lw=1.5, label=f't_sw={t_sw_A:.0f} ps')
        ax.legend(fontsize=9)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('mz')
    ax.set_title(f'Single traj (T={T_demo:.0f}K, J={J_demo/1e12:.1f}e12)')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3)

    # B: P_sw vs T
    ax = axes[1]
    ax.plot(T_values, sw_prob_T, 'o-', color='C1', lw=2, ms=7)
    ax.fill_between(T_values, 0, sw_prob_T, alpha=0.15, color='C1')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('P(switch) in 0.5 ns')
    ax.set_title(f'P_switch vs T (J={J_demo/1e12:.1f}e12, {n_trials} trials)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # C: P_sw vs J
    ax = axes[2]
    ax.plot(J_sweep/1e12, sw_prob_J, 's-', color='C2', lw=2, ms=7)
    ax.axvline(J_c_theory/1e12, color='C3', ls='--', lw=1.5,
               label=f'Theory J_c={J_c_theory/1e12:.2f}')
    ax.fill_between(J_sweep/1e12, 0, sw_prob_J, alpha=0.15, color='C2')
    ax.set_xlabel('J_c (1e12 A/m2)')
    ax.set_ylabel('P(switch) in 0.5 ns')
    ax.set_title(f'P_switch vs J (T=300K, {n_trials} trials)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f'Pt/Co SOT Switching — SLLG + SpinOrbitTorqueGPU (HeunGPU)\n'
        f'theta_SH={theta_SH}, sigma=+y, alpha={alpha}, d={d_fm*1e9:.0f}nm, K={K/1e6:.1f}MJ/m3',
        fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), '23_sot_thermal_switching_gpu.png')
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved: {out_path}")

except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Theory J_c_SOT = {J_c_theory/1e12:.3f} e12 A/m2  mu0*Heff={mu0_Heff*1e3:.1f} mT")
print(f"  dt={dt:.0e}, t_max={t_max*1e12:.0f} ps, {n_steps} steps, {n_trials} trials/pt")
print(f"  P_sw vs T @J=3.0e12: {[f'{p:.2f}' for p in sw_prob_T]}")
print(f"  P_sw vs J @T=300K:   {[f'{p:.2f}' for p in sw_prob_J]}")
