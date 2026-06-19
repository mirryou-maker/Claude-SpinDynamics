"""Notebook 38 — Per-cell GPU fields + parameter_sweep demo.

Demonstrates:
1. voronoi_grains polycrystalline texture (per-cell exchange/anisotropy)
2. InterfacialDMIFieldGPU.set_D_field() — per-cell DMI
3. RKKYFieldGPU coupling across spacer
4. parameter_sweep() — Cartesian D×K sweep on skyrmion stability
"""

import sys
import os
import time

_repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_gpu_path = os.path.join(_repo, "build", "windows-msvc-cuda", "python")
_cpu_path = os.path.join(_repo, "build", "windows-msvc", "python")
_gpu_pyd = os.path.join(_gpu_path, "_micromag.cp313-win_amd64.pyd")
if os.path.isfile(_gpu_pyd):
    sys.path.insert(0, _gpu_path)
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
    GPU = True
else:
    sys.path.insert(0, _cpu_path)
    GPU = False

import numpy as np
import math
import micromag as mm

print("=== Notebook 38: Per-cell GPU Fields + parameter_sweep ===")
print(f"    CUDA available: {GPU}\n")

# ---------------------------------------------------------------------------
# 1. voronoi_grains — polycrystalline texture on a Co/Pt thin film
# ---------------------------------------------------------------------------
print("--- 1. voronoi_grains polycrystalline texture ---")

nx, ny, nz = 64, 64, 1
dx = 5e-9
g = mm.StructuredGrid(nx, ny, nz, dx, dx, 3e-9)

# Generate random polycrystalline grains (12 grains, seed=42)
grain_id = mm.voronoi_grains(g, n_grains=12, seed=42)

# Build MaterialField3D: each grain gets a random easy-axis tilt ±30° from z
rng = np.random.default_rng(42)
N = nx * ny * nz
matf = mm.MaterialField3D(g)
mat_co = mm.Material.cobalt()
grain_ids_arr = np.asarray(grain_id, dtype=np.int32)
num_grains = int(grain_ids_arr.max()) + 1
grain_axes = []
for _ in range(num_grains):
    theta = rng.uniform(0, 30 * math.pi / 180)  # tilt angle from z
    phi   = rng.uniform(0, 2 * math.pi)
    ax = math.sin(theta) * math.cos(phi)
    ay = math.sin(theta) * math.sin(phi)
    az = math.cos(theta)
    grain_axes.append((ax, ay, az))

for i in range(N):
    gid = int(grain_ids_arr[i])
    ax, ay, az = grain_axes[gid]
    c = mm.Material.cobalt()
    c.easy_axis = mm.Vec3(ax, ay, az)
    matf[i] = c

print(f"  Grid: {nx}×{ny}×{nz}, {num_grains} grains")

unique, counts = np.unique(grain_ids_arr, return_counts=True)
avg_size = counts.mean() * dx * dx * 1e18  # nm²
print(f"  Average grain area: {avg_size:.1f} nm²")

# ---------------------------------------------------------------------------
# 2. ExchangeFieldGPU.set_material_field — per-cell exchange
# ---------------------------------------------------------------------------
if GPU:
    print("\n--- 2. ExchangeFieldGPU per-cell (harmonic mean at grain boundary) ---")

    exch = mm.ExchangeFieldGPU(g)
    exch.set_material_field(matf)
    print(f"  has_material_field: {exch.has_material_field}")

    ani = mm.UniaxialAnisotropyFieldGPU(g)
    ani.set_material_field(matf)
    print(f"  ani has_material_field: {ani.has_material_field}")

    # Initial state: all up (+z)
    m0 = mm.VectorField3D(g)
    m0.set_uniform(mm.Vec3(0, 0, 1))

    demag = mm.DemagFieldGPU(g)

    relax = mm.RelaxGPU(g)
    relax.upload(m0)

    fields = mm.FieldSumGPU()
    fields.add(exch)
    fields.add(ani)

    opts = mm.RelaxGPU.Options()
    opts.threshold  = 2000.0
    opts.max_steps  = 100000
    opts.check_every = 500

    t0 = time.perf_counter()
    steps = relax.run(mat_co, demag, fields, opts)
    dt = time.perf_counter() - t0

    m_out = mm.VectorField3D(g)
    relax.download(m_out)

    arr = np.asarray(mm.to_numpy(m_out)).reshape(N, 3)
    avg_mz = arr[:, 2].mean()
    print(f"  Relaxed in {steps} steps ({dt:.2f}s), avg_mz = {avg_mz:.4f}")

    # Per-cell should produce grain-texture variation
    mz_std = arr[:, 2].std()
    print(f"  mz std dev (grain texture): {mz_std:.4f}")
    assert mz_std < 0.2, "Expected mostly aligned (within ±30° tilt)"

# ---------------------------------------------------------------------------
# 3. InterfacialDMIFieldGPU.set_D_field — per-cell DMI
# ---------------------------------------------------------------------------
if GPU:
    print("\n--- 3. InterfacialDMIFieldGPU per-cell D ---")

    # Grains alternately have D=3e-3 or D=2e-3 J/m²
    D_arr = np.where(grain_ids_arr % 2 == 0, 3e-3, 2e-3).astype(np.float64)
    Ms_arr = np.full(N, mat_co.Ms, dtype=np.float64)

    D_sf  = mm.ScalarField3D(g)
    Ms_sf = mm.ScalarField3D(g)
    for i in range(N):
        D_sf[i]  = D_arr[i]
        Ms_sf[i] = Ms_arr[i]

    dmi = mm.InterfacialDMIFieldGPU(g, 3e-3)
    dmi.set_D_field(D_sf, Ms_sf)
    print(f"  has_D_field: {dmi.has_D_field}")

    # Skyrmion initial state
    m_sky = mm.VectorField3D(g)
    m_sky.set_vortex(nx // 2 * dx, ny // 2 * dx, 4.0)
    m_sky.normalize()
    cx_idx = nx // 2 + nx * (ny // 2)
    m_arr = np.asarray(mm.to_numpy(m_sky)).reshape(N, 3)
    for i in range(N):
        r2 = (m_arr[i, 0]**2 + m_arr[i, 1]**2)
        m_arr[i, 2] = 1.0 if r2 < 0.09 else -1.0
    # Write back
    for i in range(N):
        m_sky[i] = mm.Vec3(float(m_arr[i, 0]), float(m_arr[i, 1]), float(m_arr[i, 2]))
    m_sky.normalize()

    mat_sky = mm.Material()
    mat_sky.Ms         = 0.58e6
    mat_sky.A_exchange = 15e-12
    mat_sky.K_uniaxial = 0.8e6
    mat_sky.easy_axis  = mm.Vec3(0, 0, 1)
    mat_sky.alpha      = 0.5

    demag2 = mm.DemagFieldGPU(g)
    exch2  = mm.ExchangeFieldGPU(g)
    ani2   = mm.UniaxialAnisotropyFieldGPU(g)

    sky_fields = mm.FieldSumGPU()
    sky_fields.add(exch2)
    sky_fields.add(ani2)
    sky_fields.add(dmi)

    relax2 = mm.RelaxGPU(g)
    relax2.upload(m_sky)

    opts2 = mm.RelaxGPU.Options()
    opts2.threshold  = 5000.0
    opts2.max_steps  = 200000
    opts2.check_every = 1000

    t0 = time.perf_counter()
    steps2 = relax2.run(mat_sky, demag2, sky_fields, opts2)
    dt2 = time.perf_counter() - t0

    m_sky_out = mm.VectorField3D(g)
    relax2.download(m_sky_out)

    Q_pair = mm.topological_charge(m_sky_out)
    Q = Q_pair[0]
    print(f"  Skyrmion relax: {steps2} steps ({dt2:.2f}s), Q = {Q:.3f}")
    if abs(Q) > 0.5:
        print("  Skyrmion STABILISED")
    else:
        print("  Skyrmion not stabilised (may need finer grid or longer relax)")

# ---------------------------------------------------------------------------
# 4. RKKYFieldGPU — AFM coupling across spacer
# ---------------------------------------------------------------------------
if GPU:
    print("\n--- 4. RKKYFieldGPU coupling ---")

    g_rkky = mm.StructuredGrid(8, 8, 1, 5e-9, 5e-9, 5e-9)
    N_rkky = 8 * 8

    # Reference layer: all +z
    m_ref = mm.VectorField3D(g_rkky)
    m_ref.set_uniform(mm.Vec3(0, 0, 1))

    # AFM coupling: J < 0 → reference layer pushes m toward -z
    J_RKKY   = -2e-4    # J/m²
    d_spacer = 1e-9     # 1 nm spacer

    rkky = mm.RKKYFieldGPU(g_rkky, J_RKKY, d_spacer)
    rkky.set_ref(m_ref)
    print(f"  J = {rkky.J:.2e} J/m², d = {rkky.d:.1e} m")

    mat_py = mm.Material.permalloy()

    # Test field on uniformly +z state → should push toward -z (AFM)
    m_test = mm.VectorField3D(g_rkky)
    m_test.set_uniform(mm.Vec3(0, 0, 1))
    H_rkky = mm.VectorField3D(g_rkky)
    rkky.accumulate(m_test, mat_py, H_rkky)

    H_arr = np.asarray(mm.to_numpy(H_rkky)).reshape(N_rkky, 3)
    Hz_mean = H_arr[:, 2].mean()
    print(f"  H_z mean (AFM, expect <0): {Hz_mean:.3e} A/m")
    assert Hz_mean < 0, "RKKY AFM coupling should give H_z < 0"

# ---------------------------------------------------------------------------
# 5. parameter_sweep — D×K skyrmion stability phase diagram
# ---------------------------------------------------------------------------
if GPU:
    print("\n--- 5. parameter_sweep: D×K skyrmion stability ---")

    def run_skyrmion(D, K):
        """Returns topological charge Q for given DMI D [J/m²] and PMA K [J/m³]."""
        g_s = mm.StructuredGrid(32, 32, 1, 3e-9, 3e-9, 1e-9)
        N_s = 32 * 32

        mat_s = mm.Material()
        mat_s.Ms         = 0.58e6
        mat_s.A_exchange = 15e-12
        mat_s.K_uniaxial = K
        mat_s.easy_axis  = mm.Vec3(0, 0, 1)
        mat_s.alpha      = 0.5

        m_s = mm.VectorField3D(g_s)
        m_s.set_vortex(16 * 3e-9, 16 * 3e-9, 4.0)
        m_s.normalize()
        m_arr_s = np.asarray(mm.to_numpy(m_s)).reshape(N_s, 3)
        for i in range(N_s):
            r2 = m_arr_s[i, 0]**2 + m_arr_s[i, 1]**2
            m_arr_s[i, 2] = 1.0 if r2 < 0.09 else -1.0
        for i in range(N_s):
            m_s[i] = mm.Vec3(float(m_arr_s[i, 0]), float(m_arr_s[i, 1]), float(m_arr_s[i, 2]))
        m_s.normalize()

        demag_s = mm.DemagFieldGPU(g_s)
        exch_s  = mm.ExchangeFieldGPU(g_s)
        ani_s   = mm.UniaxialAnisotropyFieldGPU(g_s)
        dmi_s   = mm.InterfacialDMIFieldGPU(g_s, D)

        fs = mm.FieldSumGPU()
        fs.add(exch_s); fs.add(ani_s); fs.add(dmi_s)

        rel = mm.RelaxGPU(g_s)
        rel.upload(m_s)

        opts_s = mm.RelaxGPU.Options()
        opts_s.threshold  = 5000.0
        opts_s.max_steps  = 200000
        opts_s.check_every = 1000
        rel.run(mat_s, demag_s, fs, opts_s)

        m_final = mm.VectorField3D(g_s)
        rel.download(m_final)

        Q_p = mm.topological_charge(m_final)
        return {"Q": Q_p[0]}

    # Coarse 3×3 sweep: D in [2,3,4] mJ/m², K in [0.5,0.8,1.1] MJ/m³
    D_vals = [2e-3, 3e-3, 4e-3]
    K_vals = [0.5e6, 0.8e6, 1.1e6]

    t0 = time.perf_counter()
    results = mm.parameter_sweep(
        run_skyrmion,
        {"D": D_vals, "K": K_vals},
        progress=True,
        n_jobs=1,
    )
    dt_sweep = time.perf_counter() - t0

    print(f"\n  Sweep done in {dt_sweep:.1f}s ({len(results)} cases)")
    print(f"  {'D (mJ/m²)':>12} {'K (MJ/m³)':>12} {'Q':>8}")
    for r in results:
        print(f"  {r['D']*1e3:>12.1f} {r['K']/1e6:>12.2f} {r['Q']:>8.3f}")

    # Count skyrmions (|Q| > 0.5)
    n_sky = sum(1 for r in results if abs(r["Q"]) > 0.5)
    print(f"\n  Skyrmion states: {n_sky}/{len(results)}")

print("\n=== Notebook 38 complete ===")
