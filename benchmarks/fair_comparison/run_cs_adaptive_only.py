"""CS adaptive-only bench — runs S4,S6,S7,S8 for all 3 builds.
Fixed results are merged from existing fair_comparison_cs.json.
"""
import os, sys, time, json, pathlib
import numpy as np

HERE       = pathlib.Path(__file__).parent
BUILD_ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "build"

BUILDS = {
    "cuFFT_f64":  BUILD_ROOT / "windows-msvc-cuda"           / "python",
    "cuFFT_f32":  BUILD_ROOT / "windows-msvc-cuda-f32"       / "python",
    "VkFFT_f32":  BUILD_ROOT / "windows-msvc-cuda-vkfft-f32" / "python",
}

MU0   = 1.2566370614e-6
H_SP4 = (-24.6e-3 / MU0, 4.3e-3 / MU0, 0.0)
RTOL, ATOL = 1e-4, 1e-6
EVALS_RK45 = 5   # FSAL effective


def load_mm(python_path: pathlib.Path):
    import importlib, sys as _sys
    key = str(python_path)
    if key not in _sys.path:
        _sys.path.insert(0, key)
    for p in [
        pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"),
        pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"),
    ]:
        if p.exists():
            os.add_dll_directory(str(p))
    if "micromag" in _sys.modules:
        del _sys.modules["micromag"]
    import micromag as mm
    return mm


def _make_rk45(mm, grid):
    opts = mm.RK45GPUOptions()
    opts.rtol = RTOL; opts.atol = ATOL
    return mm.RK45IntegratorGPU(grid, opts)


def _uniform_m0(mm, grid, mx, my, mz):
    arr = np.array([mx, my, mz], dtype=np.float64)
    arr /= np.linalg.norm(arr)
    m0 = mm.VectorField3D(grid)
    m0.set_uniform(mm.Vec3(float(arr[0]), float(arr[1]), float(arr[2])))
    return m0


def _rk45_loop(mm, grid, mat, m0, extra_fields, demag, t_end_s):
    rk45 = _make_rk45(mm, grid)
    rk45.upload(m0)
    t0 = time.perf_counter()
    t_sim, n = 0.0, 0
    while t_sim < t_end_s:
        t_sim += rk45.step(mat, demag, extra_fields)
        n += 1
    rk45.download(m0)
    wall_ms = (time.perf_counter() - t0) * 1e3
    return wall_ms, n


def bench_s4(mm, t_end=1e-9):
    g = mm.StructuredGrid(200, 50, 1, 5e-9, 5e-9, 20e-9)
    mat = mm.Material(); mat.Ms=860e3; mat.A_exchange=13e-12; mat.K_uniaxial=0; mat.alpha=0.02
    m0 = _uniform_m0(mm, g, 1.0, 0.1, 0.01)
    demag  = mm.DemagFieldGPU(g)
    exch   = mm.ExchangeFieldGPU(g)
    zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(*H_SP4))
    fields = mm.FieldSumGPU(); fields.add(exch); fields.add(zeeman)
    wall_ms, n = _rk45_loop(mm, g, mat, m0, fields, demag, t_end)
    t_ns = t_end * 1e9
    return {"cells": 200*50*1, "wall_ms": wall_ms, "ms_per_ns": wall_ms/t_ns,
            "n_steps": n, "ms_eval": wall_ms/(n*EVALS_RK45)}


def bench_s6(mm, t_end=3e-11):
    g = mm.StructuredGrid(300, 300, 6, 3e-9, 3e-9, 4e-9)
    mat = mm.Material(); mat.Ms=860e3; mat.A_exchange=13e-12; mat.K_uniaxial=0; mat.alpha=0.02
    m0 = _uniform_m0(mm, g, 1.0, 0.1, 0.01)
    demag  = mm.DemagFieldGPU(g)
    exch   = mm.ExchangeFieldGPU(g)
    zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(*H_SP4))
    fields = mm.FieldSumGPU(); fields.add(exch); fields.add(zeeman)
    wall_ms, n = _rk45_loop(mm, g, mat, m0, fields, demag, t_end)
    t_ns = t_end * 1e9
    return {"cells": 300*300*6, "wall_ms": wall_ms, "ms_per_ns": wall_ms/t_ns,
            "n_steps": n, "ms_eval": wall_ms/(n*EVALS_RK45)}


def bench_s7(mm, t_end=2e-9):
    g   = mm.StructuredGrid(400, 20, 1, 4e-9, 4e-9, 20e-9)
    mat = mm.Material(); mat.Ms=860e3; mat.A_exchange=13e-12; mat.K_uniaxial=1e4; mat.alpha=0.02
    mat.easy_axis = mm.Vec3(0, 0, 1)
    m0  = mm.VectorField3D(g)
    mid = g.nx // 2
    for idx in range(g.nx * g.ny * g.nz):
        m0[idx] = mm.Vec3(1,0,0) if (idx % g.nx) < mid else mm.Vec3(-1,0,0)
    demag  = mm.DemagFieldGPU(g)
    exch   = mm.ExchangeFieldGPU(g)
    aniso  = mm.UniaxialAnisotropyFieldGPU(g)
    zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(0.6e-3/MU0, 0, 0))
    fields = mm.FieldSumGPU(); fields.add(exch); fields.add(aniso); fields.add(zeeman)
    wall_ms, n = _rk45_loop(mm, g, mat, m0, fields, demag, t_end)
    t_ns = t_end * 1e9
    return {"cells": 400*20*1, "wall_ms": wall_ms, "ms_per_ns": wall_ms/t_ns,
            "n_steps": n, "ms_eval": wall_ms/(n*EVALS_RK45)}


def bench_s8(mm, t_end=1e-9):
    g   = mm.StructuredGrid(200, 50, 1, 5e-9, 5e-9, 20e-9)
    mat = mm.Material(); mat.Ms=860e3; mat.A_exchange=13e-12; mat.K_uniaxial=0; mat.alpha=0.005
    m0  = _uniform_m0(mm, g, 0.99, 0.0, 0.141)
    demag  = mm.DemagFieldGPU(g)
    exch   = mm.ExchangeFieldGPU(g)
    zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 0.5/MU0))
    fields = mm.FieldSumGPU(); fields.add(exch); fields.add(zeeman)
    wall_ms, n = _rk45_loop(mm, g, mat, m0, fields, demag, t_end)
    t_ns = t_end * 1e9
    return {"cells": 200*50*1, "wall_ms": wall_ms, "ms_per_ns": wall_ms/t_ns,
            "n_steps": n, "ms_eval": wall_ms/(n*EVALS_RK45)}


BENCH = {
    "S4 SP#4 switch 1ns":      (bench_s4, 1e-9),
    "S6 Medium switch 0.03ns": (bench_s6, 3e-11),
    "S7 DW motion 2ns":        (bench_s7, 2e-9),
    "S8 Precession 1ns α=5e-3":(bench_s8, 1e-9),
}

if __name__ == "__main__":
    # Load existing fixed results
    cs_json = HERE / "fair_comparison_cs.json"
    if cs_json.exists():
        all_results = json.loads(cs_json.read_text())
    else:
        all_results = {}

    for build_label, py_path in BUILDS.items():
        if not py_path.exists():
            print(f"\n[SKIP] {build_label}: not found"); continue
        try:
            mm = load_mm(py_path)
        except Exception as e:
            print(f"\n[SKIP] {build_label}: {e}"); continue

        print(f"\n{'='*72}")
        print(f"Build: {build_label}  (CUDA: {mm.cuda_available()})")
        print(f"{'Scenario':<32} {'Cells':>9} {'wall_ms':>9} {'ms/ns':>8} {'steps':>7}  ms/eval")
        print("-" * 76)

        build_res = all_results.get(build_label, {})
        for label, (fn, t_end) in BENCH.items():
            try:
                r = fn(mm, t_end)
                build_res[label] = r
                print(f"{label:<32} {r['cells']:>9,}  {r['wall_ms']:>8.1f}  "
                      f"{r['ms_per_ns']:>7.1f}  {r['n_steps']:>7}  {r['ms_eval']:.3f}")
            except Exception as e:
                print(f"{label:<32}  ERROR: {e}")
                build_res[label] = {"error": str(e)}

        all_results[build_label] = build_res

    cs_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {cs_json}")
