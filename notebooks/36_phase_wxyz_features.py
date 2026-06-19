"""Notebook 36 - Phase W + X + Y + Z feature showcase.

(W) MagnetoelasticFieldGPU / SurfaceAnisotropyFieldGPU -> IEffectiveFieldGPU
    (can be added to FieldSumGPU for zero-PCIe GPU integrator path)
(X) gpu_hysteresis_loop + run_until_converged_gpu
    (GPU-accelerated H sweep with convergence-based relaxation)
(Y) bilayer / trilayer / saf_stack
    (MaterialField3D builders for multi-layer stacks)
(Z) mfm_overlap_integral
    (FFT-based MFM contrast via pole propagation + tip transfer function)
"""
import sys, os
_repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_cpu_path = os.path.join(_repo, 'build', 'windows-msvc', 'python')
_gpu_path = os.path.join(_repo, 'build', 'windows-msvc-cuda', 'python')

import numpy as np
import math

mu0 = 4e-7 * math.pi

print("=== Notebook 36: Phase W + X + Y + Z Features ===\n")

# ---------------------------------------------------------------------------
# (W) IEffectiveFieldGPU inheritance — Phase S GPU fields in FieldSumGPU
# ---------------------------------------------------------------------------
print("--- (W) Phase S GPU fields as IEffectiveFieldGPU ---")
_gpu_available = os.path.isfile(os.path.join(_gpu_path, '_micromag.cp313-win_amd64.pyd'))
if _gpu_available:
    sys.path.insert(0, _gpu_path)
    os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')
else:
    sys.path.insert(0, _cpu_path)

import micromag as mm

if not mm.cuda_available():
    print("  [INFO] CUDA not available - W/X GPU tests will be skipped")
else:
    g = mm.StructuredGrid(20, 20, 2, 5e-9, 5e-9, 1e-9)
    mat = mm.Material.cobalt()

    me_gpu = mm.MagnetoelasticFieldGPU(B1=-62.4e6, B2=-27.1e6, grid=g)
    me_gpu.exx = 0.001
    sa_gpu = mm.SurfaceAnisotropyFieldGPU(Ks=1.2e-3, grid=g)

    assert isinstance(me_gpu, mm.IEffectiveFieldGPU), "MagnetoelasticFieldGPU not IEffectiveFieldGPU"
    assert isinstance(sa_gpu, mm.IEffectiveFieldGPU), "SurfaceAnisotropyFieldGPU not IEffectiveFieldGPU"

    exch_g = mm.ExchangeFieldGPU(g)
    zee_g  = mm.ZeemanFieldGPU(g, mm.Vec3(50e3, 0, 0))
    fsum   = mm.FieldSumGPU()
    fsum.add(exch_g)
    fsum.add(zee_g)
    fsum.add(me_gpu)
    fsum.add(sa_gpu)
    assert len(fsum) == 4

    m0 = mm.uniform_mag(g, mm.Vec3(1.0, 0.01, 0.0))
    demag_g = mm.DemagFieldGPU(g)
    integ_g = mm.RK4IntegratorGPU(g, 5e-13)
    integ_g.upload(m0)
    for _ in range(50):
        integ_g.step(mat, demag_g, fsum)
    m_out = mm.VectorField3D(g)
    integ_g.download(m_out)
    mx, my, mz = mm.mean_magnetization(m_out)
    print(f"  FieldSumGPU(4 fields) + RK4IntegratorGPU: mx={mx:.4f} my={my:.4f} mz={mz:.4f}")
    print("  Phase W OK")

# ---------------------------------------------------------------------------
# (X) run_until_converged_gpu + gpu_hysteresis_loop
# ---------------------------------------------------------------------------
print("\n--- (X) gpu_hysteresis_loop ---")
if not mm.cuda_available():
    print("  [SKIP] CUDA not available")
else:
    g2 = mm.StructuredGrid(8, 8, 1, 5e-9, 5e-9, 5e-9)
    mat2 = mm.Material.permalloy()

    # Set up GPU field stack
    exch2  = mm.ExchangeFieldGPU(g2)
    zee2   = mm.ZeemanFieldGPU(g2, mm.Vec3(0, 0, 0))
    fsum2  = mm.FieldSumGPU()
    fsum2.add(exch2)
    fsum2.add(zee2)
    demag2 = mm.DemagFieldGPU(g2)

    m0_2 = mm.uniform_mag(g2, mm.Vec3(1.0, 0.0, 0.0))
    integ2 = mm.RK4IntegratorGPU(g2, 5e-13)
    integ2.upload(m0_2)

    m_cpu2 = mm.VectorField3D(g2)

    # Sweep +300 mT -> -300 mT
    H_sat = 300e-3 / mu0
    H_sweep = np.linspace(H_sat, -H_sat, 7)

    res = mm.gpu_hysteresis_loop(integ2, mat2, demag2, fsum2, zee2, H_sweep,
                                  m_cpu2,
                                  axis='x',
                                  tol_deg=2.0,
                                  max_steps=30_000,
                                  check_interval=200,
                                  verbose=True)

    print(f"\n  H sweep: {len(H_sweep)} points")
    print(f"  Return keys: {sorted(res.keys())}")
    print(f"  mx values: {[f'{v:.3f}' for v in res['mx']]}")
    assert sorted(res.keys()) == sorted(["H", "Hvec", "mx", "my", "mz"])
    assert res["mx"][0] > 0.5, f"mx at +300mT: {res['mx'][0]}"
    assert res["mx"][-1] < -0.5, f"mx at -300mT: {res['mx'][-1]}"
    corr = float(np.corrcoef(res["H"], res["mx"])[0, 1])
    print(f"  Pearson corr(H,mx) = {corr:.3f}  (expect > 0.8)")
    assert corr > 0.8
    print("  gpu_hysteresis_loop OK")

    # run_until_converged_gpu standalone — strong H to guarantee convergence
    print("\n  run_until_converged_gpu standalone:")
    m0_3 = mm.uniform_mag(g2, mm.Vec3(0.9, 0.1, 0.0))
    zee2.H_ext = mm.Vec3(400e3, 0, 0)   # 400 kA/m = 500 mT >> shape anisotropy
    integ2.upload(m0_3)
    m_cpu3 = mm.VectorField3D(g2)
    info = mm.run_until_converged_gpu(integ2, mat2, demag2, fsum2, m_cpu3,
                                      tol_deg=2.0, max_steps=100_000,
                                      check_interval=200, verbose=False)
    print(f"  converged={info['converged']}  steps={info['steps']}"
          f"  max_angle={info['max_angle']:.3f} deg")
    assert info['converged'], f"GPU convergence failed: max_angle={info['max_angle']:.3f}"
    print("  run_until_converged_gpu OK")

# ---------------------------------------------------------------------------
# (Y) bilayer / trilayer / saf_stack
# ---------------------------------------------------------------------------
print("\n--- (Y) bilayer / trilayer / saf_stack ---")
# Use CPU build for Y tests (no GPU needed)
sys.path.insert(0, _cpu_path)
import importlib
mm_cpu = importlib.import_module('micromag') if mm.cuda_available() else mm

# bilayer: 4 cells z, each 5nm. Top 2 cells = cobalt, bottom 2 = permalloy
g_y = mm.StructuredGrid(2, 2, 4, 5e-9, 5e-9, 5e-9)  # 4 z-layers, 5nm each
mat_py = mm.Material.permalloy()
mat_co = mm.Material.cobalt()

mf_bi = mm.bilayer(g_y, mat_top=mat_co, mat_bot=mat_py, t_top=10e-9)  # top 2 layers = Co
assert mf_bi.size == 16   # 2x2x4 = 16 cells
# iz=0,1 should be permalloy (Ms~860k), iz=2,3 should be cobalt (Ms~1440k)
for iz in range(4):
    idx = 0 + 2 * (0 + 2 * iz)  # ix=0,iy=0
    ms = mf_bi.Ms_field[idx]
    if iz < 2:
        assert abs(ms - mat_py.Ms) < 1e-3, f"iz={iz} should be Py, got Ms={ms}"
    else:
        assert abs(ms - mat_co.Ms) < 1e-3, f"iz={iz} should be Co, got Ms={ms}"
print("  bilayer: Py(bot 2) / Co(top 2)  (Ms: bot={:.1e} top={:.1e})".format(
    mf_bi.Ms_field[0], mf_bi.Ms_field[2*4]))

# trilayer: 6 z-layers at 5nm. top 2 = Fe-like, mid 2 = Co, bot 2 = Py
g_tri = mm.StructuredGrid(2, 2, 6, 5e-9, 5e-9, 5e-9)
mat_fe = mm.Material()
mat_fe.Ms = 1.71e6; mat_fe.A_exchange = 21e-12

mf_tri = mm.trilayer(g_tri, mat_top=mat_fe, mat_mid=mat_co, mat_bot=mat_py,
                      t_top=10e-9, t_mid=10e-9)
for iz in range(6):
    idx = 0 + 2*(0 + 2*iz)
    ms = mf_tri.Ms_field[idx]
    if iz < 2:
        assert abs(ms - mat_py.Ms) < 1e-3, f"iz={iz} trilayer bot fail"
    elif iz < 4:
        assert abs(ms - mat_co.Ms) < 1e-3, f"iz={iz} trilayer mid fail"
    else:
        assert abs(ms - mat_fe.Ms) < 1e-3, f"iz={iz} trilayer top fail"
print("  trilayer: Py(bot)/Co(mid)/Fe(top) verified")

# saf_stack: 6 layers; RL=2 layers (Py), spacer=2 (Ms=0), FL=2 (Co)
mf_saf = mm.saf_stack(g_tri, mat_fl=mat_co, mat_rl=mat_py,
                       t_fl=10e-9, t_rl=10e-9)
for iz in range(6):
    idx = 0 + 2*(0 + 2*iz)
    ms = mf_saf.Ms_field[idx]
    if iz < 2:
        assert abs(ms - mat_py.Ms) < 1e-3, f"iz={iz} SAF RL fail"
    elif iz < 4:
        assert ms == 0.0, f"iz={iz} SAF spacer should be Ms=0, got {ms}"
    else:
        assert abs(ms - mat_co.Ms) < 1e-3, f"iz={iz} SAF FL fail"
print("  saf_stack: RL(Py)/spacer(Ms=0)/FL(Co) verified")
print("  Phase Y OK")

# ---------------------------------------------------------------------------
# (Z) mfm_overlap_integral
# ---------------------------------------------------------------------------
print("\n--- (Z) mfm_overlap_integral ---")
# Use a simple Neel-wall domain pattern in xy
g_z = mm.StructuredGrid(32, 32, 2, 10e-9, 10e-9, 5e-9)
mat_z = mm.Material.permalloy()
m_z = mm.VectorField3D(g_z)

# Create left/right domain pattern: left half mx=+1, right half mx=-1
arr_z = np.zeros((2, 32, 32, 3), dtype=float)
arr_z[:, :, :16, 0] = +1.0   # left: +x
arr_z[:, :, 16:, 0] = -1.0   # right: -x
mm.from_numpy(m_z, arr_z)

lift = 30e-9   # 30 nm lift height
sig_dipole  = mm.mfm_overlap_integral(m_z, mat_z, lift_m=lift,
                                       tip_mode='dipole')
sig_monopole = mm.mfm_overlap_integral(m_z, mat_z, lift_m=lift,
                                        tip_mode='monopole')
sig_blurred  = mm.mfm_overlap_integral(m_z, mat_z, lift_m=lift,
                                        tip_sigma=20e-9, tip_mode='dipole')

print(f"  dipole  signal: shape={sig_dipole.shape} range=[{sig_dipole.min():.3e}, {sig_dipole.max():.3e}]")
print(f"  monopole signal: shape={sig_monopole.shape} range=[{sig_monopole.min():.3e}, {sig_monopole.max():.3e}]")
print(f"  blurred dipole: shape={sig_blurred.shape} range=[{sig_blurred.min():.3e}, {sig_blurred.max():.3e}]")

assert sig_dipole.shape == (32, 32), f"Wrong shape: {sig_dipole.shape}"
assert sig_monopole.shape == (32, 32)

# Dipole contrast: peaks at domain wall (ix=15/16 boundary)
# Signal should be antisymmetric (positive/negative about centre)
wall_col_left  = sig_dipole[:, 14].mean()
wall_col_right = sig_dipole[:, 16].mean()
domain_centre  = sig_dipole[:, 8].mean()
print(f"  dipole at wall left={wall_col_left:.3e}  right={wall_col_right:.3e}"
      f"  domain centre={domain_centre:.3e}")

# The MFM signal should have significant contrast at the wall
wall_contrast = max(abs(wall_col_left), abs(wall_col_right))
domain_bg     = abs(domain_centre)
print(f"  wall/bg ratio: {wall_contrast/(domain_bg+1e-40):.1f}  (expect > 1)")

# Blurred tip reduces peak signal
blur_max = abs(sig_blurred).max()
dipole_max = abs(sig_dipole).max()
print(f"  blurred peak / unblurred peak = {blur_max/dipole_max:.3f}  (expect < 1.0)")
assert blur_max < dipole_max, "Blurring should reduce peak signal"

print("  mfm_overlap_integral OK")
print("  Phase Z OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Phase W + X + Y + Z API Summary ===")
print("  W: MagnetoelasticFieldGPU / SurfaceAnisotropyFieldGPU -> IEffectiveFieldGPU")
print("     fsum.add(me_gpu)  # now valid for FieldSumGPU")
print("  X: mm.gpu_hysteresis_loop(integ, mat, demag, fsum, zee_gpu, H_list, m_cpu)")
print("     mm.run_until_converged_gpu(integ, mat, demag, fsum, m_cpu)")
print("  Y: mm.bilayer(grid, mat_top, mat_bot, t_top)")
print("     mm.trilayer(grid, mat_top, mat_mid, mat_bot, t_top, t_mid)")
print("     mm.saf_stack(grid, mat_fl, mat_rl, t_fl, t_rl)")
print("  Z: mm.mfm_overlap_integral(m, mat, lift_m, tip_sigma, tip_mode)")
print("     -> (ny,nx) FFT-propagated MFM frequency shift map")
