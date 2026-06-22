"""Claude-SD side of the fair-comparison benchmark.

Runs the same 5 scenarios as run_mumax_bench.py using the CS Python API.
Tests three builds: cuFFT f64, cuFFT f32, VkFFT f32.

Usage:
    py -3.13 run_cs_bench.py [build_root]

build_root defaults to ../../build (relative to this script).
The script auto-detects which GPU builds are available.

Output: fair_comparison_cs.json
"""
import os, sys, time, json, pathlib
import numpy as np

HERE       = pathlib.Path(__file__).parent
BUILD_ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "build"

# ── build detection ────────────────────────────────────────────────────────────
BUILDS = {
    "cuFFT_f64":  BUILD_ROOT / "windows-msvc-cuda"        / "python",
    "cuFFT_f32":  BUILD_ROOT / "windows-msvc-cuda-f32"    / "python",
    "VkFFT_f32":  BUILD_ROOT / "windows-msvc-cuda-vkfft-f32" / "python",
}

# ── scenario definitions ───────────────────────────────────────────────────────
# (label, nx, ny, nz, dx, dy, dz, scenario_type)
# type: "fixed_rk4" | "adaptive_dp45"
SCENARIOS = [
    # (label, nx, ny, nz, dx, dy, dz, type, t_end_s [adaptive only])
    ("S1 pow2 128×128×4",         128, 128,  4, 4e-9, 4e-9, 4e-9, "fixed_rk4",    None),
    ("S2 nonpow2 200×50×1",       200,  50,  1, 5e-9, 5e-9,20e-9, "fixed_rk4",    None),
    ("S3 nonpow2 300×300×6",      300, 300,  6, 3e-9, 3e-9, 4e-9, "fixed_rk4",    None),
    ("S4 SP#4 switch 1ns",        200,  50,  1, 5e-9, 5e-9,20e-9, "adaptive_sp4", 1e-9),
    ("S5 large 500×500×10",       500, 500, 10, 2e-9, 2e-9, 3e-9, "fixed_rk4",    None),
    ("S6 Medium switch 0.03ns",   300, 300,  6, 3e-9, 3e-9, 4e-9, "adaptive_sp4", 3e-11),
    ("S7 DW motion 2ns",          400,  20,  1, 4e-9, 4e-9,20e-9, "adaptive_dw",  2e-9),
    ("S8 Precession 1ns α=5e-3",  200,  50,  1, 5e-9, 5e-9,20e-9, "adaptive_prec",1e-9),
]

# ── fixed-step RK4 params ──────────────────────────────────────────────────────
DT       = 5e-14           # same as mumax3 FixDt
N_WARM   = 10
N1, N2   = 200, 600        # step counts for subtraction timing
N1_L, N2_L = 5, 20        # for Large (S5)

# ── adaptive DP45 params ───────────────────────────────────────────────────────
T_END     = 1e-9           # 1 ns
RTOL      = 1e-4           # matches mumax3 MaxErr=1e-4
ATOL      = 1e-6
# SP#4 Field A (in A/m): μ₀H = (-24.6mT, 4.3mT, 0) → H = B/μ₀
MU0       = 1.2566370614e-6
H_SP4     = (-24.6e-3 / MU0, 4.3e-3 / MU0, 0.0)

EVALS_RK4  = 4
EVALS_RK45 = 5   # FSAL: 7 stages but k7→k1, effective ~5 unique evals/step


# ── helpers ────────────────────────────────────────────────────────────────────
def load_mm(python_path: pathlib.Path):
    """Import micromag from a specific build path."""
    import importlib, sys as _sys
    key = str(python_path)
    if key not in _sys.path:
        _sys.path.insert(0, key)
    # Also add CUDA DLL dir for CUDA 13.2
    for cuda_bin in [
        pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"),
        pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"),
    ]:
        if cuda_bin.exists():
            os.add_dll_directory(str(cuda_bin))
    if "micromag" in _sys.modules:
        del _sys.modules["micromag"]
    import micromag as mm
    return mm


def make_permalloy(mm):
    m = mm.Material()
    m.Ms = 860e3; m.A_exchange = 13e-12; m.K_uniaxial = 0.0; m.alpha = 0.02
    return m


def make_state(mm, nx, ny, nz, dx, dy, dz):
    grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
    mat  = make_permalloy(mm)
    m0   = mm.VectorField3D(grid)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m0[i, j, k] = (1.0, 0.05, 0.0)
    # normalise
    arr = np.array(m0)
    nrm = np.linalg.norm(arr, axis=-1, keepdims=True)
    arr /= nrm
    m0[:] = arr
    return grid, mat, m0


def bench_fixed_rk4(mm, nx, ny, nz, dx, dy, dz, large=False) -> dict:
    n1 = N1_L if large else N1
    n2 = N2_L if large else N2

    grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
    mat  = make_permalloy(mm)
    m0   = mm.VectorField3D(grid)
    # uniform init (skip explicit loop for speed; rely on default or set quickly)
    arr = np.ones((nz, ny, nx, 3), dtype=np.float64)
    arr[..., 1] = 0.05; arr[..., 2] = 0.0
    arr /= np.linalg.norm(arr, axis=-1, keepdims=True)

    demag  = mm.DemagFieldGPU(grid)
    exch   = mm.ExchangeFieldGPU(grid)
    zeeman = mm.ZeemanFieldGPU(grid, mm.Vec3(0, 0, 0))

    rk4 = mm.RK4IntegratorGPU(grid, DT)
    rk4.upload(m0)

    # warmup
    for _ in range(N_WARM):
        rk4.step(mat, demag, exch, zeeman)

    # timed run n1
    t0 = time.perf_counter()
    for _ in range(n1):
        rk4.step(mat, demag, exch, zeeman)
    rk4.download(m0)     # force sync
    t1 = time.perf_counter()

    # timed run n2
    t0b = time.perf_counter()
    for _ in range(n2):
        rk4.step(mat, demag, exch, zeeman)
    rk4.download(m0)
    t2 = time.perf_counter()

    ms_step = ((t2 - t0b) - (t1 - t0)) / (n2 - n1) * 1e3
    return {
        "cells":   nx * ny * nz,
        "ms_step": ms_step,
        "ms_eval": ms_step / EVALS_RK4,
    }


def _make_rk45(mm, grid):
    opts = mm.RK45GPUOptions()
    opts.rtol = RTOL
    opts.atol = ATOL
    return mm.RK45IntegratorGPU(grid, opts)


def _run_rk45(mm, grid, mat, m0, zeeman, t_end_s) -> dict:
    """Common RK45 loop — returns timing dict."""
    demag = mm.DemagFieldGPU(grid)
    exch  = mm.ExchangeFieldGPU(grid)
    rk45  = _make_rk45(mm, grid)
    rk45.upload(m0)

    t0 = time.perf_counter()
    t_sim, n_steps = 0.0, 0
    while t_sim < t_end_s:
        t_sim += rk45.step(mat, demag, exch, zeeman)
        n_steps += 1

    rk45.download(m0)
    wall_ms = (time.perf_counter() - t0) * 1e3
    t_ns = t_end_s * 1e9
    return {
        "cells":     grid.nx * grid.ny * grid.nz,
        "wall_ms":   wall_ms,
        "ms_per_ns": wall_ms / t_ns,
        "n_steps":   n_steps,
        "ms_eval":   wall_ms / (n_steps * EVALS_RK45),
    }


def _uniform_m0(mm, grid, mx, my, mz):
    arr = np.array([mx, my, mz], dtype=np.float64)
    arr /= np.linalg.norm(arr)
    m0 = mm.VectorField3D(grid)
    m0.set_uniform(mm.Vec3(float(arr[0]), float(arr[1]), float(arr[2])))
    return m0


def bench_adaptive_sp4(mm, nx, ny, nz, dx, dy, dz, t_end_s) -> dict:
    """S4/S6: SP#4 field reversal (Field A)."""
    grid   = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
    mat    = make_permalloy(mm)
    m0     = _uniform_m0(mm, grid, 1.0, 0.1, 0.01)
    zeeman = mm.ZeemanFieldGPU(grid, mm.Vec3(*H_SP4))
    return _run_rk45(mm, grid, mat, m0, zeeman, t_end_s)


def bench_adaptive_dw(mm, nx, ny, nz, dx, dy, dz, t_end_s) -> dict:
    """S7: Domain wall motion — two-domain initial state, in-plane drive field."""
    grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
    mat = mm.Material()
    mat.Ms = 860e3; mat.A_exchange = 13e-12; mat.K_uniaxial = 1e4; mat.alpha = 0.02
    mat.easy_axis = mm.Vec3(0, 0, 1)

    m0  = mm.VectorField3D(grid)
    mid = nx // 2
    left  = mm.Vec3(1, 0, 0)
    right = mm.Vec3(-1, 0, 0)
    for idx in range(nx * ny * nz):
        m0[idx] = left if (idx % nx) < mid else right

    demag  = mm.DemagFieldGPU(grid)
    exch   = mm.ExchangeFieldGPU(grid)
    aniso  = mm.UniaxialAnisotropyFieldGPU(grid)
    Hx     = 0.6e-3 / MU0
    zeeman = mm.ZeemanFieldGPU(grid, mm.Vec3(Hx, 0, 0))

    # FieldSumGPU overload: exch + aniso + zeeman all in fields
    fields = mm.FieldSumGPU()
    fields.add(exch); fields.add(aniso); fields.add(zeeman)

    rk45 = _make_rk45(mm, grid)
    rk45.upload(m0)

    t0 = time.perf_counter()
    t_sim, n_steps = 0.0, 0
    while t_sim < t_end_s:
        t_sim += rk45.step(mat, demag, fields)
        n_steps += 1

    rk45.download(m0)
    wall_ms = (time.perf_counter() - t0) * 1e3
    return {
        "cells":     nx * ny * nz,
        "wall_ms":   wall_ms,
        "ms_per_ns": wall_ms / (t_end_s * 1e9),
        "n_steps":   n_steps,
        "ms_eval":   wall_ms / (n_steps * EVALS_RK45),
    }


def bench_adaptive_prec(mm, nx, ny, nz, dx, dy, dz, t_end_s) -> dict:
    """S8: Low-damping uniform precession (alpha=0.005, Hz=0.5T/mu0)."""
    grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)
    mat  = mm.Material()
    mat.Ms = 860e3; mat.A_exchange = 13e-12; mat.K_uniaxial = 0.0; mat.alpha = 0.005
    m0   = _uniform_m0(mm, grid, 0.99, 0.0, 0.141)   # tilted 8° from xy
    Hz   = 0.5 / MU0                                  # 0.5 T → A/m
    zeeman = mm.ZeemanFieldGPU(grid, mm.Vec3(0, 0, Hz))
    return _run_rk45(mm, grid, mat, m0, zeeman, t_end_s)


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}

    ADAPTIVE_DISPATCH = {
        "adaptive_sp4":  bench_adaptive_sp4,
        "adaptive_dw":   bench_adaptive_dw,
        "adaptive_prec": bench_adaptive_prec,
    }

    for build_label, py_path in BUILDS.items():
        if not py_path.exists():
            print(f"\n[SKIP] {build_label}: {py_path} not found")
            continue
        try:
            mm = load_mm(py_path)
        except Exception as e:
            print(f"\n[SKIP] {build_label}: import error — {e}")
            continue

        print(f"\n{'='*72}")
        print(f"Build: {build_label}  (CUDA: {mm.cuda_available()})")

        build_results = {}

        # ── fixed RK4 ──
        print(f"\n{'Scenario':<32} {'Cells':>9} {'ms/step':>9} {'ms/eval':>9}  note")
        print("-" * 70)
        for label, nx, ny, nz, dx, dy, dz, stype, _t in SCENARIOS:
            if stype != "fixed_rk4":
                continue
            large = (nx * ny * nz > 1_000_000)
            try:
                r = bench_fixed_rk4(mm, nx, ny, nz, dx, dy, dz, large)
                build_results[label] = r
                print(f"{label:<32} {r['cells']:>9,}  {r['ms_step']:>8.3f}  {r['ms_eval']:>8.3f}  RK4")
            except Exception as e:
                print(f"{label:<32}  ERROR: {e}")
                build_results[label] = {"error": str(e)}

        # ── adaptive DP45 ──
        print(f"\n{'Scenario':<32} {'Cells':>9} {'wall_ms':>9} {'ms/ns':>8} {'steps':>7}  note")
        print("-" * 76)
        for label, nx, ny, nz, dx, dy, dz, stype, t_end in SCENARIOS:
            if stype not in ADAPTIVE_DISPATCH:
                continue
            fn = ADAPTIVE_DISPATCH[stype]
            try:
                r = fn(mm, nx, ny, nz, dx, dy, dz, t_end)
                build_results[label] = r
                print(f"{label:<32} {r['cells']:>9,}  {r['wall_ms']:>8.1f}  "
                      f"{r['ms_per_ns']:>7.1f}  {r['n_steps']:>7}  DP45 rtol=1e-4")
            except Exception as e:
                print(f"{label:<32}  ERROR: {e}")
                build_results[label] = {"error": str(e)}

        all_results[build_label] = build_results

    out_path = HERE / "fair_comparison_cs.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out_path}")
