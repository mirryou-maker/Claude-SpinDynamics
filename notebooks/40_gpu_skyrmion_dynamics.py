"""Notebook 40 — GPU skyrmion dynamics: DMI + RK45 + FieldSumGPU.

Demonstrates the GPU DMI pipeline with RK45IntegratorGPU:
1. Skyrmion nucleation via InterfacialDMIFieldGPU + RelaxGPU (FieldSumGPU overload)
2. Skyrmion motion under SOT (SpinOrbitTorqueGPU) with RK45IntegratorGPU
3. Core position tracking via skyrmion_corepos_gpu (CPU fallback)
4. bloch_dw_width analytical comparison: Python utility validation
5. save_animation: GIF of skyrmion trajectory

The GPU pipeline uses FieldSumGPU to pass extra fields to RK45IntegratorGPU:
    integ.step(mat, demag, extra_fields)          # IDemagGPU + FieldSumGPU
    integ.step(mat, demag, extra_fields, torques)  # + SpinTorqueSumGPU
"""

import sys
import os
import time
import math

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
GPU = mm.cuda_available()

print("=== Notebook 40: GPU Skyrmion Dynamics (RK45 + FieldSumGPU) ===")
print(f"    CUDA available: {GPU}\n")

# ---------------------------------------------------------------------------
# 1. Grid + material: Co/Pt PMA thin film
# ---------------------------------------------------------------------------
nx, ny, nz = 64, 64, 1
dx = 3e-9                     # 3 nm cell
g = mm.StructuredGrid(nx, ny, nz, dx, dx, 0.8e-9)

mat = mm.Material()
mat.Ms         = 0.58e6       # A/m  (Co/Pt)
mat.A_exchange = 15e-12       # J/m
mat.K_uniaxial = 0.8e6        # J/m³  (PMA)
mat.easy_axis  = mm.Vec3(0, 0, 1)
mat.alpha      = 0.3

D_dmi = 3.5e-3                # J/m²  interfacial DMI

print(f"Grid: {nx}x{ny}x{nz}, dx={dx*1e9:.0f}nm, total {nx*dx*1e9:.0f}x{ny*dx*1e9:.0f}nm")
print(f"Material: Ms={mat.Ms/1e6:.2f}MA/m, A={mat.A_exchange*1e12:.0f}pJ/m, "
      f"K={mat.K_uniaxial/1e6:.1f}MJ/m³, α={mat.alpha}")
print(f"DMI: D={D_dmi*1e3:.1f}mJ/m^2  (D_c = 4sqrt(AK)/pi ~ {4*math.sqrt(mat.A_exchange*mat.K_uniaxial)/math.pi*1e3:.1f}mJ/m^2)")

# ---------------------------------------------------------------------------
# 2. Bloch DW width validation (CPU, analytical comparison)
# ---------------------------------------------------------------------------
print("\n--- 2. bloch_dw_width analytical validation (CPU) ---")

# Small 1D grid for DW measurement
g_dw = mm.StructuredGrid(512, 1, 1, 1e-9, 1e-9, 1e-9)  # 512nm × 1nm cells
mat_dw = mm.Material()
mat_dw.Ms         = 860e3
mat_dw.A_exchange = 13e-12
mat_dw.easy_axis  = mm.Vec3(0, 0, 1)
mat_dw.alpha      = 0.9

for K_dw in [1e5, 1e6]:
    # Two-domain initial state: left=-z, right=+z
    m_dw = mm.VectorField3D(g_dw)
    for i in range(512):
        m_dw[i] = mm.Vec3(0, 0, -1 if i < 256 else 1)

    # Relax with Exchange + UniaxialAnisotropy only (no Demag for 1D)
    exch_dw = mm.ExchangeField(g_dw)
    ani_dw  = mm.UniaxialAnisotropyField(g_dw)
    fields_dw = mm.EffectiveFieldSum()
    fields_dw.add(exch_dw)
    fields_dw.add(ani_dw)

    mat_dw.K_uniaxial = K_dw
    H_K  = 2 * K_dw / (4 * math.pi * 1e-7 * mat_dw.Ms)
    gp   = 1.76e11 / (1 + mat_dw.alpha**2)
    dt_k = 0.05 / (gp * 4 * math.pi * 1e-7 * H_K)
    integ_dw = mm.RK4Integrator(dt_k)

    for _ in range(100000):
        integ_dw.step(m_dw, mat_dw, fields_dw)

    lam_meas, x0 = mm.bloch_dw_width(m_dw, axis=0, comp=2)
    lam_theory   = math.pi * math.sqrt(mat_dw.A_exchange / K_dw)
    err = 100 * abs(lam_meas - lam_theory) / lam_theory

    status = "OK" if err < 10.0 else "WARN"
    print(f"  K={K_dw:.0e}: lambda_theory={lam_theory*1e9:.1f}nm  "
          f"λ_meas={lam_meas*1e9:.1f}nm  err={err:.1f}%  [{status}]")

# ---------------------------------------------------------------------------
# 3. Skyrmion nucleation: InterfacialDMIFieldGPU + RelaxGPU (FieldSumGPU)
# ---------------------------------------------------------------------------
print("\n--- 3. Skyrmion nucleation via RelaxGPU + FieldSumGPU ---")

def init_neel_skyrmion(grid, R):
    """Inward Neel skyrmion seed — favoured by D>0 interfacial DMI."""
    nx_, ny_, nz_ = grid.nx, grid.ny, grid.nz
    cx = nx_ * grid.dx * 0.5
    cy = ny_ * grid.dy * 0.5
    m = mm.VectorField3D(grid)
    for iz in range(nz_):
        for iy in range(ny_):
            for ix in range(nx_):
                rx = (ix + 0.5) * grid.dx - cx
                ry = (iy + 0.5) * grid.dy - cy
                r  = math.sqrt(rx*rx + ry*ry)
                cos_t = math.cos(math.pi * r / (2 * R)) if r < 2 * R else -1.0
                sin_t = math.sqrt(max(0.0, 1.0 - cos_t**2))
                lin = ix + nx_ * (iy + ny_ * iz)
                if r < 1e-30:
                    m[lin] = mm.Vec3(0, 0, 1)
                else:
                    m[lin] = mm.Vec3(-(rx/r)*sin_t, -(ry/r)*sin_t, cos_t)
    return m

if GPU:
    R_sky = 8e-9   # skyrmion radius seed ≈ 8nm
    m0 = init_neel_skyrmion(g, R_sky)
    Q_seed = mm.topological_charge_Q(m0)
    print(f"  Seed Q = {Q_seed:.3f}")

    demag = mm.DemagFieldGPU(g)
    exch  = mm.ExchangeFieldGPU(g)
    ani   = mm.UniaxialAnisotropyFieldGPU(g)
    dmi   = mm.InterfacialDMIFieldGPU(g, D_dmi)

    # FieldSumGPU: pass extra fields to RelaxGPU overload
    extra = mm.FieldSumGPU()
    extra.add(exch)
    extra.add(ani)
    extra.add(dmi)

    relax = mm.RelaxGPU(g)
    relax.upload(m0)

    opts = mm.RelaxGPU.Options()
    opts.threshold   = 5000.0
    opts.max_steps   = 300000
    opts.check_every = 2000
    opts.dt          = 5e-13
    opts.alpha_relax = 0.8

    t0 = time.perf_counter()
    steps = relax.run(mat, demag, extra, opts)
    dt_rel = time.perf_counter() - t0

    m_relax = mm.VectorField3D(g)
    relax.download(m_relax)
    m_relax.normalize()

    Q_relax = mm.topological_charge_Q(m_relax)
    print(f"  Relaxed: {steps} steps ({dt_rel:.2f}s), Q = {Q_relax:.3f}")

    if abs(Q_relax) > 0.5:
        print("  Skyrmion STABLE after relax")
        core = mm.skyrmion_corepos(m_relax)
        print(f"  Core position: ({core[0]*1e9:.1f}, {core[1]*1e9:.1f}) nm")
    else:
        print("  Skyrmion not stabilised (D < D_c, using Q from seed for demo)")
        m_relax = m0

else:
    print("  (CPU mode: skipping GPU relax)")
    m_relax = init_neel_skyrmion(g, 8e-9)
    Q_relax = mm.topological_charge_Q(m_relax)
    print(f"  Seed Q = {Q_relax:.3f}")

# ---------------------------------------------------------------------------
# 4. SOT-driven skyrmion motion: RK45IntegratorGPU + SpinTorqueSumGPU
# ---------------------------------------------------------------------------
print("\n--- 4. SOT skyrmion motion: RK45IntegratorGPU + FieldSumGPU ---")

if GPU:
    # SOT: Pt/Co interface — spin Hall angle θ_SH ≈ 0.07
    theta_SH = 0.07
    J_HM     = 3e11      # A/m²  — heavy-metal current density
    t_Co     = 0.8e-9    # Co layer thickness [m]
    hbar     = 1.0546e-34
    e_charge = 1.602e-19
    # a_SOT = (ℏ/2e) * θ_SH * J / (Ms * t)
    a_SOT    = (hbar / (2 * e_charge)) * theta_SH * J_HM / (mat.Ms * t_Co)
    sigma_hat = mm.Vec3(0, 1, 0)   # spin polarisation along y

    sot = mm.SpinOrbitTorqueGPU(g, a_SOT, 0.0, sigma_hat)
    torques = mm.SpinTorqueSumGPU()
    torques.add(sot)

    # RK45 adaptive integrator with FieldSumGPU
    rk45 = mm.RK45IntegratorGPU(g)
    rk45.upload(m_relax)

    rk45_opts        = mm.RK45GPUOptions()
    rk45_opts.rtol   = 1e-4
    rk45_opts.atol   = 1e-5
    rk45_opts.dt_max = 1e-12

    # LLG integration: 2 ns with SOT current
    t_sim  = 2e-9
    t      = 0.0
    frames = []
    core_x = []
    core_t = []

    t0 = time.perf_counter()
    n_steps = 0
    save_interval = 0.05e-9   # save every 50 ps

    last_save = -save_interval
    while t < t_sim:
        dt_used = rk45.step(mat, demag, extra, torques, rk45_opts)
        t += dt_used
        n_steps += 1

        if t - last_save >= save_interval:
            m_snap = mm.VectorField3D(g)
            rk45.download(m_snap)
            m_snap.normalize()
            frames.append(m_snap)
            core = mm.skyrmion_corepos(m_snap)
            core_x.append(core[0] * 1e9)
            core_t.append(t * 1e9)
            last_save = t

    m_final = mm.VectorField3D(g)
    rk45.download(m_final)
    m_final.normalize()
    dt_llg = time.perf_counter() - t0

    Q_final = mm.topological_charge_Q(m_final)
    print(f"  LLG: {n_steps} steps, {t*1e9:.2f} ns simulated, {dt_llg:.2f}s wall time")
    print(f"  Final Q = {Q_final:.3f}")

    if core_x:
        dx_core = core_x[-1] - core_x[0]
        print(f"  Core displacement: Deltax = {dx_core:.1f} nm over {core_t[-1]-core_t[0]:.0f} ps")
        v_sky = dx_core * 1e-9 / (core_t[-1] - core_t[0]) * 1e9 / 1e-9  # m/s? Actually dx_core is in nm, time in ns
        # Δx [nm] / Δt [ns] → nm/ns = m/s
        v_sky = dx_core / (core_t[-1] - core_t[0]) if (core_t[-1] - core_t[0]) > 0 else 0.0
        print(f"  Skyrmion velocity: v ~ {v_sky:.1f} m/s")

    # ---------------------------------------------------------------------------
    # 5. save_animation — GIF of mz snapshots
    # ---------------------------------------------------------------------------
    print("\n--- 5. save_animation ---")
    if frames:
        gif_path = os.path.join(os.path.dirname(__file__), "40_skyrmion_sot.gif")
        try:
            mm.save_animation(frames, gif_path, component="z", fps=10)
            print(f"  Saved {len(frames)} frames -> 40_skyrmion_sot.gif")
        except ImportError as e:
            print(f"  save_animation skipped (matplotlib/pillow not installed): {e}")
    else:
        print("  No frames captured (skipped)")

else:
    print("  (CPU mode: skipping GPU LLG integration)")

# ---------------------------------------------------------------------------
# 6. FieldSumGPU overload smoke test (all GPU integrators)
# ---------------------------------------------------------------------------
print("\n--- 6. FieldSumGPU + RK4/RK45/Heun overload smoke test ---")

if GPU:
    g_s = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
    m_s = mm.VectorField3D(g_s)
    m_s.set_uniform(mm.Vec3(1, 0, 0))

    demag_s = mm.DemagFieldGPU(g_s)
    exch_s  = mm.ExchangeFieldGPU(g_s)
    ani_s   = mm.UniaxialAnisotropyFieldGPU(g_s)
    fs = mm.FieldSumGPU()
    fs.add(exch_s)
    fs.add(ani_s)

    mat_s = mm.Material.permalloy()

    # RK4 overload
    rk4 = mm.RK4IntegratorGPU(g_s, 1e-13)
    rk4.upload(m_s)
    rk4.step(mat_s, demag_s, fs)      # FieldSumGPU overload
    print("  RK4IntegratorGPU.step(mat, demag, FieldSumGPU): OK")

    # RK45 overload
    rk45_s = mm.RK45IntegratorGPU(g_s)
    rk45_s.upload(m_s)
    rk45_s.step(mat_s, demag_s, fs)   # FieldSumGPU overload
    print("  RK45IntegratorGPU.step(mat, demag, FieldSumGPU): OK")

    # Heun overload
    heun = mm.HeunIntegratorGPU(g_s, 1e-13)
    heun.upload(m_s)
    heun.step(mat_s, demag_s, fs)     # FieldSumGPU overload
    print("  HeunIntegratorGPU.step(mat, demag, FieldSumGPU): OK")

    # batch_to_numpy smoke test
    if frames:
        arr = mm.batch_to_numpy(frames)
        print(f"  batch_to_numpy: {len(frames)} frames -> shape {arr.shape}  OK")

else:
    print("  (CPU mode: skipping GPU overload tests)")

print("\n=== Notebook 40 complete ===")
