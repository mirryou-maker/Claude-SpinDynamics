"""
Notebook 23b: Thermal-Assisted STT Switching (SLLG + SlonczewskiSTTGPU)

Shows how finite temperature (Stratonovich-Langevin LLG via HeunIntegratorGPU)
assists Slonczewski STT switching:
  - At T=0, switching requires J > J_c (deterministic threshold)
  - At T>0, thermal fluctuations allow switching at J < J_c (stochastic)

Produces:
  (a) Single SLLG trajectory at T=300K, J=0.15e12 (< J_c = 0.21e12)
  (b) P_switch vs T (5 T-values, 10 trials/pt, J=0.20e12 ~ J_c)
  (c) P_switch vs J at 3 temperatures (compare thermal vs deterministic)

Performance: 284 us/step  ->  dt=1e-13, t_max=2ns (20000 steps/trial)
  10 trials x 5 T-pts = 50 trials  x 20000 steps = ~285 s
  10 trials x 5 J-pts x 3 T-vals = 150 trials x 20000 steps = ~855 s  (long!)
  Compromise: use t_max=0.5ns (5000 steps), J range near J_c

Material: Pt/Co  Ms=580 kA/m, K=0.5 MJ/m3, alpha=0.02, d=3 nm, P=0.5
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

print("Notebook 23b: Thermal-Assisted STT Switching (SLLG + SlonczewskiSTTGPU)")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Constants and material
# ---------------------------------------------------------------------------
mu0  = 4e-7 * np.pi
hbar = 1.0546e-34
e_ch = 1.6022e-19
k_B  = 1.3806e-23

Ms    = 580e3;  A  = 15e-12;  K  = 0.5e6
alpha = 0.02;   d  = 3e-9;    P  = 0.5

mat = mm.Material()
mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = K
mat.easy_axis = mm.Vec3(0, 0, 1); mat.alpha = alpha

# Theory J_c (Slonczewski PMA macrospin)
mu0_Heff   = 2 * K / Ms - mu0 * Ms
J_c_theory = 2 * e_ch * alpha * Ms * d * mu0_Heff / (hbar * P)
E_b        = (K - 0.5 * mu0 * Ms**2) * d**3   # barrier energy [J]
print(f"\nTheory: J_c = {J_c_theory/1e12:.3f} e12 A/m2")
print(f"        E_b = {E_b/k_B/300:.2f} k_B T  (at 300 K)  = {E_b:.2e} J")

# ---------------------------------------------------------------------------
# Grid + shared GPU objects
# ---------------------------------------------------------------------------
g = mm.StructuredGrid(1, 1, 1, d, d, d)

demag_gpu = mm.DemagFieldGPU(g)
exch_gpu  = mm.ExchangeFieldGPU(g)
aniso_gpu = mm.UniaxialAnisotropyFieldGPU(g)

fields = mm.FieldSumGPU()
fields.add(exch_gpu)
fields.add(aniso_gpu)

p = mm.Vec3(0, 0, 1)  # polarization = +z
stt = mm.SlonczewskiSTTGPU(g, 1e12, P, d, p, 0.0)
torques = mm.SpinTorqueSumGPU(); torques.add(stt)

theta0 = np.deg2rad(5)
a0_arr = np.zeros((1,1,1,3)); a0_arr[0,0,0] = [np.sin(theta0), 0, np.cos(theta0)]

def make_m0():
    m = mm.VectorField3D(g); mm.from_numpy(m, a0_arr); return m

# Timing: 284 us/step
dt       = 1e-13   # s
t_max    = 1.0e-9  # 1 ns (10000 steps/trial)
n_steps  = int(t_max / dt)
check_ev = 200     # every 20 ps

def run_sllg_trial(J_val, T_K, seed):
    """SLLG trial with STT. Returns (switched_bool, t_sw_ps or None)."""
    stt.J = J_val
    integ = mm.HeunIntegratorGPU(g, dt, seed=seed)
    integ.upload(make_m0())
    for step in range(0, n_steps, check_ev):
        for _ in range(check_ev):
            integ.step(mat, demag_gpu, fields, T_K, torques)
        m_tmp = mm.VectorField3D(g)
        integ.download(m_tmp)
        if float(mm.to_numpy(m_tmp)[0,0,0,2]) < -0.5:
            return True, (step + check_ev) * dt * 1e12
    return False, None

# ---------------------------------------------------------------------------
# Part A: Single SLLG trajectory (T=300K, J = 0.15e12 < J_c)
# ---------------------------------------------------------------------------
print(f"\n--- Part A: Single SLLG trajectory (T=300K, J=0.15e12 < J_c={J_c_theory/1e12:.3f}) ---")
J_A = 0.15e12
stt.J = J_A
integ_a = mm.HeunIntegratorGPU(g, dt, seed=42)
integ_a.upload(make_m0())

mz_traj, t_arr = [], []
t0 = time.time()
for step in range(0, n_steps, check_ev):
    for _ in range(check_ev):
        integ_a.step(mat, demag_gpu, fields, 300.0, torques)
    m_tmp = mm.VectorField3D(g)
    integ_a.download(m_tmp)
    mz_traj.append(float(mm.to_numpy(m_tmp)[0,0,0,2]))
    t_arr.append((step + check_ev) * dt * 1e12)

tA = time.time() - t0
mz_arr = np.array(mz_traj)
sw_A = np.where(mz_arr < -0.5)[0]
t_sw_A = t_arr[sw_A[0]] if len(sw_A) > 0 else None
print(f"  t_sw = {t_sw_A:.0f} ps" if t_sw_A else "  No switch in 1 ns")
print(f"  Time: {tA:.1f} s")

# Part A2: Deterministic RK4 (T=0) reference at same J
print(f"  (Deterministic T=0 reference at J=0.15e12:)")
stt.J = J_A
integ_r = mm.RK4IntegratorGPU(g, dt)
integ_r.upload(make_m0())
mz_det = []
for step in range(0, n_steps, check_ev):
    for _ in range(check_ev):
        integ_r.step(mat, demag_gpu, fields, torques)
    m_tmp = mm.VectorField3D(g)
    integ_r.download(m_tmp)
    mz_det.append(float(mm.to_numpy(m_tmp)[0,0,0,2]))
sw_det = np.where(np.array(mz_det) < -0.5)[0]
print(f"  T=0: {'switched at '+str(int(t_arr[sw_det[0]]))+' ps' if len(sw_det) else 'No switch (J < J_c deterministic)'}")

# ---------------------------------------------------------------------------
# Part B: P_switch vs T (J = J_c_theory, 5 T-values, n_trials each)
# ---------------------------------------------------------------------------
n_trials_B = 10
print(f"\n--- Part B: P_switch vs T (J~J_c={J_c_theory/1e12:.3f}e12, {n_trials_B} trials/pt) ---")

T_values  = np.array([0, 100, 200, 300, 500], dtype=float)
J_B       = J_c_theory * 0.85   # slightly below J_c for interesting stochastic behavior
sw_prob_T = []
t0 = time.time()

for T in T_values:
    n_sw = 0
    t_sw_list = []
    for seed in range(n_trials_B):
        if T == 0:  # T=0: use RK4 (deterministic, no need for multiple trials)
            stt.J = J_B
            integ_b = mm.RK4IntegratorGPU(g, dt)
            integ_b.upload(make_m0())
            switched = False
            for step in range(0, n_steps, check_ev):
                for _ in range(check_ev):
                    integ_b.step(mat, demag_gpu, fields, torques)
                m_tmp = mm.VectorField3D(g)
                integ_b.download(m_tmp)
                if float(mm.to_numpy(m_tmp)[0,0,0,2]) < -0.5:
                    switched = True; break
            if switched: n_sw += 1
            if seed > 0: break  # T=0 is deterministic, one trial is enough
        else:
            switched, tsw = run_sllg_trial(J_B, T, seed)
            if switched: n_sw += 1

    # T=0: replicate result for all trials
    if T == 0: n_sw = n_sw * n_trials_B

    p_sw = n_sw / n_trials_B
    sw_prob_T.append(p_sw)
    print(f"  T={T:4.0f}K  P_sw={p_sw:.2f}  ({n_sw}/{n_trials_B})")

print(f"  Time: {time.time()-t0:.1f} s")

# ---------------------------------------------------------------------------
# Part C: P_switch vs J  (T=0 and T=300K, compare)
# ---------------------------------------------------------------------------
n_trials_C = 8
print(f"\n--- Part C: P_switch vs J  (T=0 and T=300K, {n_trials_C} trials/pt) ---")

J_sweep = np.linspace(0.05e12, 0.5e12, 7)
sw_0K   = []
sw_300K = []
t0 = time.time()

for J in J_sweep:
    # T=0 (RK4, deterministic)
    stt.J = J
    integ_c = mm.RK4IntegratorGPU(g, dt)
    integ_c.upload(make_m0())
    switched_det = False
    for step in range(0, n_steps, check_ev):
        for _ in range(check_ev):
            integ_c.step(mat, demag_gpu, fields, torques)
        m_tmp = mm.VectorField3D(g)
        integ_c.download(m_tmp)
        if float(mm.to_numpy(m_tmp)[0,0,0,2]) < -0.5:
            switched_det = True; break
    sw_0K.append(1.0 if switched_det else 0.0)

    # T=300K (SLLG, stochastic)
    n_sw = sum(1 for s in range(n_trials_C) if run_sllg_trial(J, 300.0, s)[0])
    sw_300K.append(n_sw / n_trials_C)

    print(f"  J={J/1e12:.3f}e12  T=0: {'Y' if switched_det else 'N'}  T=300K: {n_sw}/{n_trials_C}")

print(f"  Time: {time.time()-t0:.1f} s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # A: mz trajectory
    ax = axes[0]
    ax.plot(t_arr, mz_arr, lw=1.2, color='C0', label=f'T=300K')
    ax.plot(t_arr, mz_det, lw=1.2, color='C3', ls='--', label='T=0 (det.)')
    ax.axhline(-0.5, color='k', ls='--', lw=1, alpha=0.5)
    ax.axhline( 0.5, color='k', ls='--', lw=1, alpha=0.5)
    if t_sw_A:
        ax.axvline(t_sw_A, color='C0', ls=':', lw=1.5, label=f't_sw={t_sw_A:.0f} ps')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('mz')
    ax.set_title(f'SLLG vs det. (J={J_A/J_c_theory:.2f}*J_c)')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # B: P_sw vs T
    ax = axes[1]
    ax.plot(T_values, sw_prob_T, 'o-', color='C1', lw=2, ms=7)
    ax.fill_between(T_values, 0, sw_prob_T, alpha=0.15, color='C1')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(f'P(switch) in {t_max*1e12:.0f} ps')
    ax.set_title(f'Thermal assist: P vs T\n(J={J_B/1e12:.3f}e12 = {J_B/J_c_theory:.2f}*J_c)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # C: P_sw vs J for T=0 and T=300K
    ax = axes[2]
    ax.step(J_sweep/1e12, sw_0K, where='post', lw=2, color='C3', label='T=0 (det.)')
    ax.plot(J_sweep/1e12, sw_300K, 's-', lw=2, color='C0', ms=6, label='T=300K (SLLG)')
    ax.axvline(J_c_theory/1e12, color='k', ls='--', lw=1.5, label=f'J_c={J_c_theory/1e12:.3f}')
    ax.set_xlabel('J (1e12 A/m2)')
    ax.set_ylabel(f'P(switch) in {t_max*1e12:.0f} ps')
    ax.set_title('Thermal assist: P vs J\n(T=0 deterministic vs T=300K)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f'Pt/Co: Thermal-Assisted STT Switching (GPU HeunIntegratorGPU + SlonczewskiSTTGPU)\n'
        f'alpha={alpha}, P={P}, d={d*1e9:.0f}nm, K={K/1e6:.1f}MJ/m3  '
        f'J_c(theory)={J_c_theory/1e12:.3f}e12 A/m2  E_b={E_b/k_B/300:.1f}kBT(300K)',
        fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), '23b_stt_thermal_sllg_gpu.png')
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved: {out_path}")

except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  J_c (theory) = {J_c_theory/1e12:.3f} e12 A/m2")
print(f"  E_b = {E_b:.2e} J = {E_b/k_B/300:.2f} k_B*T(300K)")
print(f"  Part B: P_sw vs T @ J={J_B/1e12:.3f}e12: {[f'{p:.2f}' for p in sw_prob_T]}")
print(f"  Part C: T=0:   {['Y' if p>0.5 else 'N' for p in sw_0K]}")
print(f"  Part C: T=300: {[f'{p:.2f}' for p in sw_300K]}")
