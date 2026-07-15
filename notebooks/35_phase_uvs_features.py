"""Notebook 35 - Phase U + V + S feature showcase.

Demonstrates:
  (U) hysteresis_loop      -- automated H sweep + relax
  (V) MagnetoelasticField per-cell strain (set_exx_field / ScalarField3D)
  (S) MagnetoelasticFieldGPU / SurfaceAnisotropyFieldGPU (GPU build only)
"""
import sys, os
_repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import numpy as np
import micromag as mm
import math

mu0 = 4e-7 * math.pi

print("=== Notebook 35: Phase U + V + S Features ===\n")

# ---------------------------------------------------------------------------
# (U) hysteresis_loop — SP#3-style hysteresis on small Py strip
# ---------------------------------------------------------------------------
print("--- (U) hysteresis_loop ---")

# 8x8x1 disk: sweep Zeeman field and verify mx correlates with H
nx, ny, nz = 8, 8, 1
dx = 5e-9
g   = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)
mat = mm.Material.permalloy()

zee   = mm.ZeemanField(mm.Vec3(0.0, 0.0, 0.0))
exch  = mm.ExchangeField()
demag = mm.DemagField(g)
heff  = mm.EffectiveFieldSum()
heff.add(exch)
heff.add(demag)
heff.add(zee)

integ = mm.RK4Integrator(dt=5e-13)

# Sweep from large +H to large -H (no reset: continuous evolution)
# Start from +x saturation
m = mm.uniform_mag(g, mm.Vec3(1.0, 0.0, 0.0))
H_sat = 300e-3 / mu0   # 300 mT — strong enough to saturate
H_sweep = np.linspace(H_sat, -H_sat, 9)   # +300 to -300 mT, 9 points

res = mm.hysteresis_loop(
    m, mat, heff, integ,
    H_list=H_sweep, zee=zee, axis='x',
    tol_deg=2.0, max_steps=40_000, check_interval=100,
    verbose=False
)

print(f"  H sweep: {len(H_sweep)} points  ({H_sweep.max()*mu0*1e3:.0f} mT down to {H_sweep.min()*mu0*1e3:.0f} mT)")
print(f"  Return keys: {sorted(res.keys())}")
print(f"  mx values: {[f'{v:.3f}' for v in res['mx']]}")

assert sorted(res.keys()) == sorted(["H", "Hvec", "mx", "my", "mz", "E_total"])
assert res["H"].shape == (9,)
assert res["mx"].shape == (9,)
# At +300mT (first point), mx should be positive
assert res["mx"][0] > 0.5, f"mx at +300mT too low: {res['mx'][0]}"
# At -300mT (last point), mx should be negative
assert res["mx"][-1] < -0.5, f"mx at -300mT too high: {res['mx'][-1]}"
# mx decreasing trend overall (Pearson correlation with H should be positive)
corr = float(np.corrcoef(res["H"], res["mx"])[0, 1])
print(f"  Pearson corr(H, mx) = {corr:.3f}  (expect > 0.8)")
assert corr > 0.8, f"mx not correlated with H: corr={corr}"
print("  hysteresis_loop OK")

# ---------------------------------------------------------------------------
# (U2) hysteresis_loop with Vec3 H_list (full vector sweep)
# ---------------------------------------------------------------------------
print("\n--- (U2) hysteresis_loop Vec3 sweep ---")

# Sweep at 45 deg in x-y plane
H_angle = np.array([[H_sat*math.cos(math.pi/4), H_sat*math.sin(math.pi/4), 0],
                     [0.0, 0.0, 0.0],
                     [-H_sat, 0.0, 0.0]], dtype=float)
m2 = mm.uniform_mag(g, mm.Vec3(1.0, 0.0, 0.0))

res2 = mm.hysteresis_loop(m2, mat, heff, integ, H_list=H_angle, zee=zee,
                           tol_deg=5.0, max_steps=10_000, check_interval=100)
print(f"  Vec3 sweep: {len(H_angle)} points, Hvec shape = {res2['Hvec'].shape}")
assert res2["Hvec"].shape == (3, 3), f"Hvec shape wrong: {res2['Hvec'].shape}"
print("  Vec3 sweep OK")

# ---------------------------------------------------------------------------
# (V) MagnetoelasticField per-cell strain via ScalarField3D
# ---------------------------------------------------------------------------
print("\n--- (V) MagnetoelasticField per-cell strain ---")

# 4x1x1 strip: sinusoidal exx = A * cos(pi * ix / (nx-1))
g_v = mm.StructuredGrid(4, 1, 1, 5e-9, 5e-9, 5e-9)
mat_v = mm.Material()
mat_v.Ms = 8.6e5   # Fe-like

B1_Fe = 6.96e6   # J/m3
me_v  = mm.MagnetoelasticField(B1=B1_Fe)

# Build per-cell exx ScalarField3D
exx_sf = mm.ScalarField3D(g_v)
A_strain = 1e-3
exx_vals = [A_strain * math.cos(math.pi * ix / 3.0) for ix in range(4)]
for ix in range(4):
    exx_sf[ix] = exx_vals[ix]   # ScalarField3D supports [] indexing

me_v.set_exx_field(exx_sf)
print(f"  has_spatial_strain = {me_v.has_spatial_strain}  (expect True)")
assert me_v.has_spatial_strain

# m = +x: H_x = -(2B1/mu0Ms) * exx(ix)
m_v = mm.uniform_mag(g_v, mm.Vec3(1.0, 0.0, 0.0))
H_v = mm.VectorField3D(g_v)
me_v.accumulate(m_v, mat_v, H_v)
H_v_np = np.asarray(mm.to_numpy(H_v))   # (1,1,4,3)

prefac_expected = -2.0 * B1_Fe / (mu0 * mat_v.Ms)
for ix in range(4):
    Hx_computed = H_v_np[0, 0, ix, 0]
    Hx_expected = prefac_expected * exx_vals[ix]
    err = abs(Hx_computed - Hx_expected) / (abs(Hx_expected) + 1e-30)
    if err > 1e-9:
        print(f"  ERROR ix={ix}: expected={Hx_expected:.4e}, got={Hx_computed:.4e}")
    print(f"  ix={ix}: exx={exx_vals[ix]:+.5f}, H_x={Hx_computed:+.4e} A/m  (expected {Hx_expected:+.4e})")

# Verify all 4 cells match
Hx_arr = H_v_np[0, 0, :, 0]
Hx_exp = np.array([prefac_expected * e for e in exx_vals])
max_err = np.max(np.abs(Hx_arr - Hx_exp) / (np.abs(Hx_exp) + 1e-30))
print(f"  Max relative error across cells: {max_err:.2e}  (expect < 1e-9)")
assert max_err < 1e-9, f"Per-cell strain error: {max_err}"

# Clear and verify revert to scalar
me_v.clear_strain_fields()
assert not me_v.has_spatial_strain, "clear_strain_fields failed"
print("  Per-cell strain OK")

# ---------------------------------------------------------------------------
# (V2) Mixed mode: per-cell exx + uniform eyy
# ---------------------------------------------------------------------------
print("\n--- (V2) Mixed: per-cell exx + uniform eyy ---")
me_v2 = mm.MagnetoelasticField(B1=B1_Fe)
me_v2.set_exx_field(exx_sf)    # per-cell exx
me_v2.eyy = 0.0005             # uniform eyy
assert me_v2.has_spatial_strain
print("  Mixed mode set OK")

# m = +y: H_y = -(2B1/mu0Ms) * eyy_uniform  (exx doesn't contribute for m_y in H_y direction)
m_y = mm.uniform_mag(g_v, mm.Vec3(0.0, 1.0, 0.0))
H_v2 = mm.VectorField3D(g_v)
me_v2.accumulate(m_y, mat_v, H_v2)
H_v2_np = np.asarray(mm.to_numpy(H_v2))
Hy_computed = H_v2_np[0, 0, 0, 1]
Hy_expected = prefac_expected * 0.0005   # B1*my*eyy, my=1, eyy=0.0005
print(f"  H_y uniform (from eyy): computed={Hy_computed:.4e}, expected={Hy_expected:.4e}")
err_hy = abs(Hy_computed - Hy_expected) / (abs(Hy_expected) + 1e-30)
assert err_hy < 1e-9, f"Mixed mode H_y error: {err_hy}"
print("  Mixed mode OK")

# ---------------------------------------------------------------------------
# (S) GPU fields — only if CUDA build available
# ---------------------------------------------------------------------------
print("\n--- (S) GPU fields ---")
if not mm.cuda_available():
    print("  [SKIP] CUDA build not available - skipping GPU field tests")
    print("  (Build with: cmake --preset windows-msvc-cuda)")
else:
    print("  CUDA available -- testing GPU fields")

    g_gpu = mm.StructuredGrid(20, 20, 2, 5e-9, 5e-9, 1e-9)
    mat_gpu = mm.Material.cobalt()

    # MagnetoelasticFieldGPU
    me_gpu = mm.MagnetoelasticFieldGPU(B1=-62.4e6, B2=-27.1e6, grid=g_gpu)
    me_gpu.exx = 0.001
    assert me_gpu.name == "MagnetoelasticFieldGPU"
    assert abs(me_gpu.B1 + 62.4e6) < 1e-3
    assert abs(me_gpu.exx - 0.001) < 1e-15

    m_gpu = mm.uniform_mag(g_gpu, mm.Vec3(1.0, 0.0, 0.0))
    H_gpu_out = mm.VectorField3D(g_gpu)
    me_gpu.accumulate(m_gpu, mat_gpu, H_gpu_out)
    H_gpu_np = np.asarray(mm.to_numpy(H_gpu_out))
    print(f"  MagnetoelasticFieldGPU: H_x[0,0,0] = {H_gpu_np[0,0,0,0]:.4e} A/m")
    # Compare against CPU
    me_cpu = mm.MagnetoelasticField(B1=-62.4e6, B2=-27.1e6)
    me_cpu.exx = 0.001
    H_cpu_out = mm.VectorField3D(g_gpu)
    me_cpu.accumulate(m_gpu, mat_gpu, H_cpu_out)
    H_cpu_np = np.asarray(mm.to_numpy(H_cpu_out))
    err_me = np.max(np.abs(H_gpu_np - H_cpu_np)) / (np.max(np.abs(H_cpu_np)) + 1e-30)
    print(f"  GPU vs CPU max err: {err_me:.2e}  (expect < 1e-10)")
    assert err_me < 1e-10, f"MagnetoelasticFieldGPU error: {err_me}"
    print("  MagnetoelasticFieldGPU OK")

    # SurfaceAnisotropyFieldGPU
    sa_gpu = mm.SurfaceAnisotropyFieldGPU(Ks=1.2e-3, grid=g_gpu)
    assert sa_gpu.name == "SurfaceAnisotropyFieldGPU"
    assert abs(sa_gpu.Ks - 1.2e-3) < 1e-15

    m_z = mm.uniform_mag(g_gpu, mm.Vec3(0.0, 0.0, 1.0))
    H_sa_gpu = mm.VectorField3D(g_gpu)
    sa_gpu.accumulate(m_z, mat_gpu, H_sa_gpu)
    H_sa_np = np.asarray(mm.to_numpy(H_sa_gpu))
    # Compare CPU SurfaceAnisotropyField
    sa_cpu = mm.SurfaceAnisotropyField(Ks=1.2e-3)
    H_sa_cpu_out = mm.VectorField3D(g_gpu)
    sa_cpu.accumulate(m_z, mat_gpu, H_sa_cpu_out)
    H_sa_cpu_np = np.asarray(mm.to_numpy(H_sa_cpu_out))
    err_sa = np.max(np.abs(H_sa_np - H_sa_cpu_np)) / (np.max(np.abs(H_sa_cpu_np)) + 1e-30)
    print(f"  SurfaceAnisotropyFieldGPU vs CPU max err: {err_sa:.2e}  (expect < 1e-10)")
    assert err_sa < 1e-10, f"SurfaceAnisotropyFieldGPU error: {err_sa}"
    print("  SurfaceAnisotropyFieldGPU OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Phase U + V + S API Summary ===")
print("  mm.hysteresis_loop(m, mat, heff, integ, H_list, zee,")
print("      axis='x', tol_deg=1.0, reset_m=None, verbose=False)")
print("    -> dict: H, Hvec, mx, my, mz, E_total  [all numpy arrays]")
print("  MagnetoelasticField.set_exx_field(ScalarField3D)  -- per-cell strain")
print("  MagnetoelasticField.has_spatial_strain            -- True if field attached")
print("  MagnetoelasticField.clear_strain_fields()         -- revert to scalar")
print("  mm.MagnetoelasticFieldGPU(B1, B2, grid)           -- GPU kernel version")
print("  mm.SurfaceAnisotropyFieldGPU(Ks, grid, n_hat)     -- GPU surface mask kernel")
print("\nAll Phase U + V features verified OK.")
print("Phase S GPU: tested if CUDA build available.")
