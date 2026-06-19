"""Notebook 17 -- Antiferromagnetic Coupling & Zhang-Li STT

Demonstrates two Phase J features:

A. Antiferromagnetic exchange via checkerboard_regions + set_inter_exchange(-A):
   - 1D chain of 20 Py cells partitioned into 0/1 alternating sublattices
   - J_IEC = +A (FM coupling): final state uniform
   - J_IEC = -A (AFM coupling): final state alternating (Neel order)
   - Plots mx per cell for both cases

B. Zhang-Li STT domain-wall motion:
   - 1D Py strip (100 cells, dx=5nm), PMA geometry -> Neel wall in mz
   - Apply J_x = 5e12 A/m2, P=0.5, xi=0.04 via zhang_li_from_current()
   - Measure domain-wall drift velocity vs time using domain_wall_pos()
   - Save mz(x) profile CSV via save_profile()
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc', 'python'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import micromag as mm

mu_0 = 4 * np.pi * 1e-7

# ===========================================================================
# A. Antiferromagnetic coupling via checkerboard_regions
# ===========================================================================
print("=== A. Antiferromagnetic Exchange (checkerboard) ===")

nx_afm = 20
dx_afm = 5e-9
g_afm  = mm.StructuredGrid(nx_afm, 1, 1, dx_afm, dx_afm, dx_afm)
mat_afm = mm.Material.permalloy()

# Create 0/1 alternating region map
rmap_afm = mm.checkerboard_regions(g_afm)
A_Py = mat_afm.A_exchange

# Initial state: all spins along +x
m_afm = mm.uniform_mag(g_afm, mm.Vec3(1, 0, 0))

def run_case(A_iec, label):
    """Run exchange coupling case and return mx per cell.

    For AFM (A_iec < 0): initialize with Neel state (+x/-x alternating),
    since starting from a uniform state gives zero torque (degenerate).
    """
    arr_init = mm.to_numpy(mm.uniform_mag(g_afm, mm.Vec3(1, 0, 0)))
    if A_iec < 0:
        # Neel-order initial state: alternate sign by (ix+iy+iz)%2
        arr0 = np.zeros_like(arr_init)
        for ix in range(nx_afm):
            sign = +1 if (ix % 2 == 0) else -1
            arr0[0, 0, ix, 0] = sign   # mx = +1 or -1
        m = mm.VectorField3D(g_afm)
        mm.from_numpy(m, arr0)
    else:
        m = mm.uniform_mag(g_afm, mm.Vec3(1, 0, 0))
    exch = mm.ExchangeField(mm.BoundaryCondition.Neumann)
    exch.set_region_map(rmap_afm)
    exch.set_inter_exchange(0, 1, A_iec)
    heff = mm.EffectiveFieldSum()
    heff.add(exch)
    mm.relax(m, mat_afm, heff,
             mm.RelaxOptions(threshold=0.1, max_steps=500_000))
    arr = mm.to_numpy(m)
    return arr[0, 0, :, 0]   # mx per cell along x

print("  Relaxing FM coupling (J_IEC = +A_Py) ...")
mx_fm  = run_case(+A_Py, "FM")

print("  Relaxing AFM coupling (J_IEC = -A_Py) ...")
mx_afm = run_case(-A_Py, "AFM")

print(f"  FM:  mean|mx| = {np.abs(mx_fm).mean():.3f}  (expected ~1)")
print(f"  AFM: mean|mx| = {np.abs(mx_afm).mean():.3f}  (expected ~1, alternating sign)")
print(f"  AFM: pattern = {['+ ' if v>0 else '- ' for v in mx_afm[:8]]}...")

# ===========================================================================
# B. Zhang-Li STT domain-wall motion
# ===========================================================================
print("\n=== B. Zhang-Li STT Domain-Wall Motion ===")

nx_dw = 200
dx_dw = 5e-9                     # 5 nm cells -> 1000 nm strip
g_dw  = mm.StructuredGrid(nx_dw, 1, 1, dx_dw, dx_dw, dx_dw)

# Permalloy with moderate PMA (Bloch wall width ~30 nm, clear crossing)
mat_dw = mm.Material()
mat_dw.Ms         = 800e3
mat_dw.A_exchange = 13e-12
mat_dw.K_uniaxial = 1.5e5         # Delta = pi*sqrt(A/K) ~ 29 nm (well-resolved)
mat_dw.easy_axis  = mm.Vec3(0, 0, 1)
mat_dw.alpha      = 0.02

# Initial state: analytical Bloch wall profile
#   mz = tanh((x-x0)/Delta), my = sech((x-x0)/Delta), mx = 0
#   Delta = pi*sqrt(A/K) (wall half-width parameter)
Delta = np.pi * np.sqrt(mat_dw.A_exchange / mat_dw.K_uniaxial)  # ~29 nm
x_cells = (np.arange(nx_dw) + 0.5) * dx_dw - 0.5 * nx_dw * dx_dw  # box-centred
xi_dw   = x_cells / Delta
arr_bloch = np.zeros((1, 1, nx_dw, 3))
arr_bloch[0, 0, :, 0] = 0.0                     # mx
arr_bloch[0, 0, :, 1] = 1.0 / np.cosh(xi_dw)   # my = sech(xi)
arr_bloch[0, 0, :, 2] = np.tanh(xi_dw)          # mz = tanh(xi)  (+1 right, -1 left)
# Reverse to get +z on left, -z on right
arr_bloch[0, 0, :, 2] = -arr_bloch[0, 0, :, 2]

m_dw = mm.VectorField3D(g_dw)
mm.from_numpy(m_dw, arr_bloch)

exch_dw  = mm.ExchangeField(mm.BoundaryCondition.Neumann)
aniso_dw = mm.UniaxialAnisotropyField()
heff_dw  = mm.EffectiveFieldSum()
heff_dw.add(aniso_dw)
heff_dw.add(exch_dw)

# Short relax to remove numerical artefacts
print(f"  Bloch wall width Delta = {Delta*1e9:.1f} nm")
print("  Relaxing domain wall ...")
mm.relax(m_dw, mat_dw, heff_dw, mm.RelaxOptions(threshold=1.0, max_steps=50_000))
arr0 = mm.to_numpy(m_dw)
mz0  = arr0[0, 0, :, 2]
dw0  = mm.domain_wall_pos(m_dw, component=2, threshold=0.0, axis=0)
print(f"  Initial wall position: x = {dw0*1e9:.1f} nm")

# Save initial profile CSV
csv_path = os.path.join(os.path.dirname(__file__), "dw_profile_initial.csv")
mm.save_profile(m_dw, csv_path, component="z", axis=0)
print(f"  Saved profile -> {csv_path}")

# Zhang-Li STT: J_x = 1e12 A/m2 along +x (moderate current)
j_amp = 1e12
stt   = mm.zhang_li_from_current(j_amp, direction=(1, 0, 0), Ms=mat_dw.Ms, P=0.5, xi=0.04)
stt_sum = mm.SpinTorqueSum()
stt_sum.add(stt)

u_velocity = stt.u(mat_dw.Ms)
xi_par = 0.04
print(f"  u = {u_velocity:.3f} m/s  (spin drift velocity)")
print(f"  Expected DW velocity ~ xi*u/alpha = {xi_par*u_velocity/mat_dw.alpha:.0f} m/s")

# Run STT dynamics: track wall position every 0.2 ns
integ_stt = mm.RK4Integrator(1e-13)     # 0.1 ps
t_run     = 5e-9                         # 5 ns
dt_record = 0.2e-9
t_vals, xwall_vals = [], []

def record_cb(t, m_field):
    x = mm.domain_wall_pos(m_field, component=2, threshold=0.0, axis=0)
    t_vals.append(t)
    xwall_vals.append(x)

print(f"  Running STT dynamics for {t_run*1e9:.0f} ns ...")
mm.run(integ_stt, m_dw, mat_dw, heff_dw, t_total=t_run,
       stt=stt_sum, callback=record_cb, callback_dt=dt_record)

arr1 = mm.to_numpy(m_dw)
mz1  = arr1[0, 0, :, 2]
dw1  = xwall_vals[-1] if xwall_vals else float("nan")
print(f"  Final wall position: x = {dw1*1e9:.1f} nm")
print(f"  Wall displacement: {(dw1-dw0)*1e9:.1f} nm in {t_run*1e9:.0f} ns")
v_sim = (dw1 - dw0) / t_run
print(f"  Simulated DW velocity: {v_sim:.1f} m/s")

# Save final profile
csv_path2 = os.path.join(os.path.dirname(__file__), "dw_profile_final.csv")
mm.save_profile(m_dw, csv_path2, component="z", axis=0)

# rotate_mag demo: tilt m by 10 degrees around z
mm.rotate_mag(m_dw, theta=10*np.pi/180, axis=(0, 0, 1))
arr_rot = mm.to_numpy(m_dw)
print(f"\n  rotate_mag(10 deg around z):"
      f"  mx after = {arr_rot[0,0,:,0].mean():.3f}"
      f"  (expected ~sin(10deg)*mz_before)")

# ===========================================================================
# C. Plotting
# ===========================================================================
x_afm_nm = (np.arange(nx_afm) + 0.5) * dx_afm * 1e9
x_dw_nm  = (np.arange(nx_dw) + 0.5) * dx_dw * 1e9

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# AFM vs FM
ax = axes[0]
ax.bar(x_afm_nm - 0.4, mx_fm,  width=0.7, label="FM  (J>0)", color="steelblue", alpha=0.7)
ax.bar(x_afm_nm + 0.4, mx_afm, width=0.7, label="AFM (J<0)", color="tomato",    alpha=0.7)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("x (nm)"); ax.set_ylabel("mx")
ax.set_title("AFM vs FM exchange via checkerboard_regions")
ax.legend(fontsize=8)

# DW profiles
ax = axes[1]
ax.plot(x_dw_nm, mz0, "royalblue", lw=1.5, label="initial")
ax.plot(x_dw_nm, mz1, "tomato",    lw=1.5, ls="--", label=f"after STT {t_run*1e9:.0f} ns")
ax.axvline(dw0*1e9, color="royalblue", ls=":", lw=1)
ax.axvline(dw1*1e9, color="tomato",    ls=":", lw=1)
ax.set_xlabel("x (nm)"); ax.set_ylabel("mz")
ax.set_title(f"Zhang-Li DW motion (J={j_amp:.0e} A/m2, P=0.5, xi=0.04)")
ax.legend(fontsize=8); ax.set_ylim(-1.2, 1.2)

# Wall position vs time
ax = axes[2]
ax.plot(np.array(t_vals)*1e9, np.array(xwall_vals)*1e9, "k-o", ms=4)
if len(t_vals) >= 2:
    # Linear fit
    t_arr = np.array(t_vals)
    x_arr = np.array(xwall_vals)
    coeffs = np.polyfit(t_arr, x_arr, 1)
    ax.plot(np.array(t_vals)*1e9,
            (coeffs[0]*t_arr + coeffs[1])*1e9, "r--", lw=1,
            label=f"v = {coeffs[0]:.1f} m/s")
ax.set_xlabel("t (ns)"); ax.set_ylabel("x_wall (nm)")
ax.set_title("Domain-wall position vs time")
ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "afm_and_zhangli.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved -> {out}")
print("\n=== Done ===")
