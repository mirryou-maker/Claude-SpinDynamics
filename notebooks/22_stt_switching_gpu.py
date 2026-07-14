"""
Notebook 22: GPU Slonczewski STT Switching Demo

Simulates CPP-STT spin transfer torque switching of a Pt/Co nanopillar
using RK4IntegratorGPU + SlonczewskiSTTGPU.

Sweeps current density J to map switching time t_sw vs J and finds J_c.

Theory (PMA macrospin, Slonczewski):
  mu0*Heff = 2*K/Ms - mu0*Ms      [T, net PMA field]
  J_c = 2*e*alpha*Ms*d*mu0_Heff / (hbar*P)

Material: Pt/Co  Ms=580 kA/m, K=0.5 MJ/m3 PMA, alpha=0.02, d=3 nm
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

print("Notebook 22: GPU STT Switching Demo")
print(f"  CUDA: {mm.cuda_available()}")

# ---------------------------------------------------------------------------
# Constants and material
# ---------------------------------------------------------------------------
mu0   = 4e-7 * np.pi
hbar  = 1.0546e-34
e_ch  = 1.6022e-19

Ms    = 580e3   # A/m
A     = 15e-12  # J/m
K     = 0.5e6   # J/m3  (PMA)
alpha = 0.02
d_fm  = 3e-9    # free layer thickness (m)
P     = 0.5     # spin polarization

mat = mm.Material()
mat.Ms         = Ms
mat.A_exchange = A
mat.K_uniaxial = K
mat.easy_axis  = mm.Vec3(0, 0, 1)
mat.alpha      = alpha

# ---------------------------------------------------------------------------
# Theory: Slonczewski critical current (PMA macrospin)
#   mu0*Heff [T] = 2K/Ms - mu0*Ms   (net PMA after demag subtraction)
#   J_c = 2*e*alpha*Ms*d * mu0*Heff / (hbar*P)
# ---------------------------------------------------------------------------
mu0_Heff  = 2 * K / Ms - mu0 * Ms   # [T]
J_c_theory = 2 * e_ch * alpha * Ms * d_fm * mu0_Heff / (hbar * P)
print(f"\nTheory:  mu0*Heff = {mu0_Heff*1e3:.1f} mT")
print(f"         J_c      = {J_c_theory/1e12:.3f} e12 A/m2")

# ---------------------------------------------------------------------------
# Grid: 1x1x1 macrospin
# ---------------------------------------------------------------------------
g = mm.StructuredGrid(1, 1, 1, d_fm, d_fm, d_fm)

demag_gpu = mm.DemagFieldGPU(g)
exch_gpu  = mm.ExchangeFieldGPU(g)
aniso_gpu = mm.UniaxialAnisotropyFieldGPU(g)

fields = mm.FieldSumGPU()
fields.add(exch_gpu)
fields.add(aniso_gpu)

p = mm.Vec3(0, 0, 1)
stt = mm.SlonczewskiSTTGPU(g, 1e12, P, d_fm, p, 0.0)

torques = mm.SpinTorqueSumGPU()
torques.add(stt)

# ---------------------------------------------------------------------------
# Initial state: +z tilted 5 degrees (break symmetry)
# ---------------------------------------------------------------------------
theta0 = np.deg2rad(5)
m0 = mm.VectorField3D(g)
a0 = np.zeros((1, 1, 1, 3))
a0[0, 0, 0] = [np.sin(theta0), 0.0, np.cos(theta0)]
mm.from_numpy(m0, a0)

# ---------------------------------------------------------------------------
# Sweep: J from 0.05 to 1.5 x 10^12 A/m2 (covers J_c ~ 0.2e12)
# ---------------------------------------------------------------------------
J_values  = np.linspace(0.05e12, 1.5e12, 16)
dt        = 5e-14   # s
t_max     = 2e-9    # 2 ns
n_steps   = int(t_max / dt)
check_ev  = 500     # check every 500 steps = 25 ps

t_sw_list = []
final_mz  = []

print(f"\nSweep: {len(J_values)} J pts, dt={dt:.0e} s, t_max={t_max*1e9:.0f} ns, check every {check_ev*dt*1e12:.0f} ps")
t0 = time.time()

for J in J_values:
    stt.J = J

    integ = mm.RK4IntegratorGPU(g, dt)
    integ.upload(m0)

    t_sw = None
    for step in range(0, n_steps, check_ev):
        for _ in range(check_ev):
            integ.step(mat, demag_gpu, fields, torques)

        m_tmp = mm.VectorField3D(g)
        integ.download(m_tmp)
        mz = float(mm.to_numpy(m_tmp)[0, 0, 0, 2])
        t_now = (step + check_ev) * dt

        if mz < -0.5 and t_sw is None:
            t_sw = t_now
            break

    m_final = mm.VectorField3D(g)
    integ.download(m_final)
    mz_fin = float(mm.to_numpy(m_final)[0, 0, 0, 2])

    t_sw_list.append(t_sw)
    final_mz.append(mz_fin)
    sw_str = f"{t_sw*1e12:.0f} ps" if t_sw else "NO SWITCH"
    print(f"  J = {J/1e12:.3f}e12  t_sw = {sw_str:>12}  mz_fin = {mz_fin:+.3f}")

elapsed = time.time() - t0
print(f"\nTime: {elapsed:.1f} s  ({elapsed/len(J_values)*1000:.0f} ms/pt)")

switched = [i for i, t in enumerate(t_sw_list) if t is not None]
if switched:
    J_c_sim = J_values[switched[0]]
    J_c_sim_upper = J_values[switched[0]]
    print(f"Simulated J_c < {J_c_sim/1e12:.3f} e12 A/m2")
    print(f"Theory    J_c = {J_c_theory/1e12:.3f} e12 A/m2")
else:
    J_c_sim = None
    print("No switching in 2 ns (J_c > sweep range)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    J_arr   = J_values / 1e12
    t_sw_ps = [t * 1e12 if t else np.nan for t in t_sw_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: switching time vs J
    ax = axes[0]
    mask = ~np.isnan(t_sw_ps)
    ax.semilogy(J_arr[mask], np.array(t_sw_ps)[mask], 'o-', color='C0', lw=2, ms=6, label='Simulated')
    ax.axvline(J_c_theory / 1e12, color='C1', ls='--', lw=1.5,
               label=f'Theory J_c={J_c_theory/1e12:.3f}')
    ax.set_xlabel('J (1e12 A/m2)')
    ax.set_ylabel('Switching time (ps, log scale)')
    ax.set_title('STT Switching Time vs J')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    # Right: final mz (switched / not switched)
    ax = axes[1]
    dJ = (J_values[1] - J_values[0]) / 1e12 * 0.7
    colors = ['#e53935' if mz < -0.5 else '#1565c0' for mz in final_mz]
    ax.bar(J_arr, final_mz, width=dJ, color=colors, edgecolor='k', lw=0.5)
    ax.axhline(-0.5, color='k', ls=':', lw=1, alpha=0.7)
    ax.axhline( 0.5, color='k', ls=':', lw=1, alpha=0.7)
    ax.axvline(J_c_theory / 1e12, color='C1', ls='--', lw=1.5,
               label=f'Theory J_c={J_c_theory/1e12:.3f}')
    ax.set_xlabel('J (1e12 A/m2)')
    ax.set_ylabel('Final mz (after 2 ns)')
    ax.set_title('Final Magnetization State')
    ax.set_ylim(-1.15, 1.15)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(handles=[Patch(color='#e53935', label='Switched (-z)'),
                       Patch(color='#1565c0', label='Not switched (+z)')],
              loc='upper right')

    n_sw = sum(1 for t in t_sw_list if t)
    plt.suptitle(
        f'Pt/Co CPP-STT Switching (GPU macrospin, d={d_fm*1e9:.0f} nm, alpha={alpha}, P={P})\n'
        f'{n_sw}/{len(J_values)} points switched  |  Theory J_c={J_c_theory/1e12:.3f} e12 A/m2',
        fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), '22_stt_switching_gpu.png')
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved: {out_path}")

except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
print(f"  Material: Pt/Co, Ms={Ms/1e3:.0f} kA/m, K={K/1e6:.1f} MJ/m3, d={d_fm*1e9:.0f} nm")
print(f"  alpha={alpha}, P={P}, p=+z")
print(f"  mu0*Heff = {mu0_Heff*1e3:.1f} mT (net PMA after demag)")
print(f"  Theory J_c = {J_c_theory/1e12:.3f} e12 A/m2")
if J_c_sim:
    print(f"  Simulated: first switch at J={J_c_sim/1e12:.3f} e12 A/m2")
print(f"  Sweep: {elapsed:.1f} s  ({elapsed/len(J_values)*1000:.0f} ms/pt)")
