"""
Notebook 30: GPU Thermal STT Switching Statistics (Neel-Brown)

Probabilistic STT switching requires thermal stability Delta = K*V/(kB*T) >> 1
so thermal fluctuations are rare without current. With J just below J_c0,
the barrier is reduced: Delta_eff = Delta*(1 - J/J_c0)^2, and switching becomes
stochastic (Neel-Brown: P_sw = 1 - exp(-t/tau) with tau ~ exp(Delta_eff)).

Material: Pt/Co PMA macrospin (1x1x1, dx=10nm)
  K=0.5MJ/m3, V=1e-24 m3, Delta(300K)=120 >> 1 (stable without current)
  J_c0(T=0) ~ 0.7e12 A/m2 (Slonczewski perpendicular switching threshold)

Three demos:
  A) N=20 ensemble at J=0.85*J_c0, T=300K: shows stochastic switching distribution
  B) P_sw vs J at T=300K: transition from P=0 to P=1 around J_c0
  C) P_sw vs T at J=0.88*J_c0: thermal enhancement of switching
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

print("Notebook 30: GPU Thermal STT Switching Statistics")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
mu0   = 4e-7 * np.pi
kB    = 1.38065e-23
hbar  = 1.0546e-34
e_ch  = 1.6022e-19

# ---------------------------------------------------------------------------
# Material: Pt/Co PMA macrospin  (dx=10nm cell)
# ---------------------------------------------------------------------------
Ms    = 580e3;   K   = 0.5e6
alpha = 0.02;    P   = 0.5
d_F   = 10e-9    # FM thickness (= cell size for macrospin)
dx    = 10e-9

mat = mm.Material()
mat.Ms = Ms; mat.A_exchange = 0.0; mat.K_uniaxial = K
mat.easy_axis = mm.Vec3(0, 0, 1); mat.alpha = alpha

V = dx**3
mu0_Heff = 2*K/Ms - mu0*Ms                  # net PMA field [T]  = 0.995 T

# Deterministic switching current (Slonczewski perpendicular, T=0):
# Torque balance: hbar*P*J/(2*e*Ms*d) = alpha * mu0*H_eff
# -> J_c0 = 2*e*alpha*Ms*d * (mu0*H_eff) / (hbar*P)
# mu0*H_eff [T] NOT H_eff [A/m] (would be off by mu0 = 1.26e-6)
J_c0_analytic = 2 * e_ch * alpha * Ms * d_F * abs(mu0_Heff) / (hbar * P)
# The analytic estimate assumes a thin film (demag Nz=1), but this macrospin is a
# 10 nm CUBE (Nz=1/3); together with the O(1) prefactor convention in
# SlonczewskiSTTGPU it underpredicts the SIMULATED deterministic threshold by ~3x.
# Calibrate J_c0 to the measured switching current (this build: mz reverses
# between 2.0 and 2.5e12 A/m² at T=0, threshold ≈ 2.35e12) so the 0.80–1.08·J_c0
# ensemble sweeps bracket the transition (Jf≈0.94) instead of sitting entirely
# below it (which gave P_sw≡0).
J_c0 = 2.5e12

# Thermal stability at 300K
Delta_300K = K * V / (kB * 300)

print(f"\nPt/Co macrospin: Ms={Ms/1e3:.0f}kA/m, K={K/1e6:.2f}MJ/m3, alpha={alpha}")
print(f"  Cell: {int(dx*1e9)}nm cube, V = {V*1e27:.0f} nm3")
print(f"  mu0*Heff = {mu0_Heff*1e3:.0f} mT  (net PMA effective field, T)")
print(f"  J_c0 (T=0 deterministic) = {J_c0/1e12:.4f} e12 A/m2")
print(f"  Delta(300K) = K*V/(kB*300) = {Delta_300K:.1f}  (>> 1 = thermally stable)")

# ---------------------------------------------------------------------------
# Grid and GPU objects
# ---------------------------------------------------------------------------
g = mm.StructuredGrid(1, 1, 1, dx, dx, dx)

# Initial state: m = +z (up)
a0 = np.zeros((1, 1, 1, 3)); a0[..., 2] = 1.0
m_up = mm.VectorField3D(g); mm.from_numpy(m_up, a0)

# GPU fields: anisotropy + demag (macrospin, no exchange)
demag_g = mm.DemagFieldGPU(g)
aniso_g = mm.UniaxialAnisotropyFieldGPU(g)
fields_g = mm.FieldSumGPU(); fields_g.add(aniso_g)

# STT: Slonczewski, p=+z
p_vec = mm.Vec3(0, 0, 1)
stt_g = mm.SlonczewskiSTTGPU(g, J_c0, P, d_F, p_vec, 0.0)
torq_g = mm.SpinTorqueSumGPU(); torq_g.add(stt_g)

def get_mz(integ):
    m_tmp = mm.VectorField3D(g)
    integ.download(m_tmp)
    return float(mm.to_numpy(m_tmp)[0, 0, 0, 2])

def is_switched(mz, thr=-0.5):
    return mz < thr

dt     = 2e-14
t_max  = 2.0e-9       # 2 ns observation window
n_max  = int(t_max / dt)
log_ev = 1000          # log every 20 ps

t_log_ps = [(s + log_ev) * dt * 1e12 for s in range(0, n_max, log_ev)]

print(f"\n  t_sim = {t_max*1e9:.1f} ns,  dt = {dt:.0e} s,  {n_max} steps per trial")
print(f"  Attempt frequency (Kittel estimate) ~ {1/dt:.0e} /s")

def run_ensemble(J_val, T_K, N_trials, label=""):
    stt_g.J = J_val
    sw_times, mz_finals, mz_trajs = [], [], []
    for trial in range(N_trials):
        seed = trial * 13 + 7
        integ = mm.HeunIntegratorGPU(g, dt, seed)
        integ.upload(m_up)
        sw_t = None; mz_t = []
        for step in range(0, n_max, log_ev):
            for _ in range(log_ev):
                integ.step(mat, demag_g, fields_g, float(T_K), torq_g)
            mz = get_mz(integ)
            mz_t.append(mz)
            if sw_t is None and is_switched(mz):
                sw_t = (step + log_ev) * dt * 1e12
        sw_times.append(sw_t)
        mz_finals.append(mz_t[-1])
        mz_trajs.append(mz_t)
    n_sw = sum(s is not None for s in sw_times)
    P_sw = n_sw / N_trials
    return P_sw, n_sw, sw_times, mz_finals, mz_trajs

# ---------------------------------------------------------------------------
# Part A: Ensemble at J=0.85*J_c0, T=300K  (N=20 trajectories)
# ---------------------------------------------------------------------------
J_A   = 0.85 * J_c0
T_A   = 300.0
N_A   = 20
Delta_eff_A = Delta_300K * (1 - J_A/J_c0)**2

print(f"\n--- Part A: Ensemble at J=0.85*J_c0={J_A/1e12:.3f}e12, T=300K (N={N_A}) ---")
print(f"  Delta_eff = Delta*(1-J/J_c0)^2 = {Delta_300K:.0f}*{(1-J_A/J_c0):.3f}^2 = {Delta_eff_A:.1f}")

t0 = time.time()
P_sw_A, n_sw_A, sw_times_A, mz_finals_A, mz_trajs_A = run_ensemble(J_A, T_A, N_A)
wall_A = time.time() - t0

for i in range(N_A):
    st = sw_times_A[i]
    if st: print(f"  Trial {i+1:2d}: SWITCHED at {st:.0f}ps  mz={mz_finals_A[i]:.2f}")
    else:  print(f"  Trial {i+1:2d}: stable             mz={mz_finals_A[i]:.2f}")

print(f"  P_sw = {P_sw_A:.2f} ({n_sw_A}/{N_A})  wall={wall_A:.1f}s")

# Mean switching time (switched trials only)
sw_valid = [t for t in sw_times_A if t is not None]
if sw_valid:
    print(f"  Mean t_switch = {np.mean(sw_valid):.0f} ps  (std = {np.std(sw_valid):.0f} ps)")

# ---------------------------------------------------------------------------
# Part B: P_sw vs J at T=300K  (10 trials each, 2 ns)
# ---------------------------------------------------------------------------
print(f"\n--- Part B: P_sw vs J at T=300K (N=10 each, t_max={t_max*1e9:.0f}ns) ---")

J_factors_B = np.array([0.80, 0.88, 0.94, 1.00, 1.08])
J_sweep_B   = J_factors_B * J_c0
N_B = 10

P_sw_B = []
t0 = time.time()
for J_f, J_val in zip(J_factors_B, J_sweep_B):
    P_sw, n_sw, *_ = run_ensemble(J_val, 300.0, N_B)
    Delta_eff = max(0, Delta_300K * (1 - J_f)**2)
    P_sw_B.append(P_sw)
    print(f"  J={J_f:.2f}*J_c0 = {J_val/1e12:.3f}e12  Delta_eff={Delta_eff:.1f}  P_sw={P_sw:.1f} ({n_sw}/{N_B})")

print(f"  Part B: {time.time()-t0:.1f} s")

# ---------------------------------------------------------------------------
# Part C: P_sw vs T at J=0.88*J_c0  (10 trials each)
# ---------------------------------------------------------------------------
print(f"\n--- Part C: P_sw vs T at J=0.88*J_c0 (N=10 each, t_max={t_max*1e9:.0f}ns) ---")

T_sweep_C = np.array([100, 200, 300, 400])
J_C = 0.88 * J_c0
N_C = 10

P_sw_C = []
t0 = time.time()
for T_val in T_sweep_C:
    Delta_T = K * V / (kB * T_val)
    Delta_eff = max(0, Delta_T * (1 - 0.88)**2)
    P_sw, n_sw, *_ = run_ensemble(J_C, float(T_val), N_C)
    P_sw_C.append(P_sw)
    print(f"  T={T_val:.0f}K  Delta={Delta_T:.0f}  Delta_eff={Delta_eff:.1f}  P_sw={P_sw:.1f} ({n_sw}/{N_C})")

print(f"  Part C: {time.time()-t0:.1f} s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    t_arr = np.array(t_log_ps)

    # Part A: mz(t) ensemble
    ax = axes[0]
    for i, mz_t in enumerate(mz_trajs_A):
        col = 'C3' if sw_times_A[i] is not None else 'C0'
        ax.plot(t_arr, mz_t, '-', color=col, lw=1.0, alpha=0.6)
    ax.axhline(-0.5, color='k', ls='--', lw=1.5, alpha=0.6, label='switch threshold')
    ax.axhline(+1, color='C2', ls=':', lw=1, alpha=0.5, label='initial +z')
    ax.axhline(-1, color='C3', ls=':', lw=1, alpha=0.5, label='final -z')
    ax.set_xlabel('Time (ps)'); ax.set_ylabel('mz')
    ax.set_title(f'mz(t) N={N_A} ensemble\nJ=0.85*J_c0, T=300K  P_sw={P_sw_A:.2f}')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_ylim(-1.15, 1.15)

    # Part B: P_sw vs J
    ax = axes[1]
    ax.plot(J_factors_B, P_sw_B, 'o-', color='C0', lw=2, ms=9)
    # Error bar (binomial)
    yerr = [np.sqrt(p*(1-p)/N_B) for p in P_sw_B]
    ax.errorbar(J_factors_B, P_sw_B, yerr=yerr, fmt='none', color='C0', capsize=5)
    ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.5, label='P=0.5')
    ax.axvline(1.0, color='C3', ls='--', lw=1, alpha=0.7, label='J=J_c0 (T=0 threshold)')
    ax.set_xlabel('J / J_c0'); ax.set_ylabel('Switching probability')
    ax.set_title(f'P_sw vs J/J_c0 (T=300K, t={t_max*1e9:.0f}ns)\nJ_c0={J_c0/1e12:.3f}e12')
    ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # Part C: P_sw vs T
    ax = axes[2]
    ax.plot(T_sweep_C, P_sw_C, 's-', color='C1', lw=2, ms=9)
    yerr_C = [np.sqrt(p*(1-p)/N_C) for p in P_sw_C]
    ax.errorbar(T_sweep_C, P_sw_C, yerr=yerr_C, fmt='none', color='C1', capsize=5)
    ax.axhline(0.5, color='k', ls='--', lw=1, alpha=0.5, label='P=0.5')
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Switching probability')
    ax.set_title(f'P_sw vs T (J=0.88*J_c0, t={t_max*1e9:.0f}ns)\nHigher T -> easier switching')
    ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Thermal STT Switching Statistics (GPU HeunIntegratorGPU + SlonczewskiSTTGPU)\n'
        f'Pt/Co PMA macrospin ({int(dx*1e9)}nm cell): Delta(300K)={Delta_300K:.0f}, '
        f'J_c0={J_c0/1e12:.3f}e12 A/m2,  alpha={alpha}',
        fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '30_thermal_stt_statistics_gpu.png')
    plt.savefig(out, dpi=120); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Pt/Co PMA: {int(dx*1e9)}nm cell, K={K/1e6:.2f}MJ/m3, Delta(300K)={Delta_300K:.1f}")
print(f"  J_c0 = {J_c0/1e12:.3f}e12 A/m2  (Slonczewski perpendicular)")
print(f"  Part A: J=0.85*J_c0, T=300K, N={N_A}: P_sw={P_sw_A:.2f}")
print(f"  Part B: " + "  ".join([f"J={f:.2f}->P={p:.1f}" for f,p in zip(J_factors_B, P_sw_B)]))
print(f"  Part C: " + "  ".join([f"T={t:.0f}K->P={p:.1f}" for t,p in zip(T_sweep_C, P_sw_C)]))
