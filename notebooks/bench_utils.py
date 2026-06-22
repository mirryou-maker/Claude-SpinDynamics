"""Shared utilities for build-comparison notebooks (41-45).

Provides:
  load_mm(build)  -- load micromag module from a specific build
  BUILDS          -- ordered dict of available build paths
  MU0             -- 4pi * 1e-7
  run_mumax3(mx3_path, t_timeout_s) -- run mumax3 and parse output
"""
import os, sys, pathlib, subprocess, time
import numpy as np

HERE    = pathlib.Path(__file__).parent
ROOT    = HERE.parent
BUILD   = ROOT / "build"
MU0     = 1.2566370614e-6

CUDA_DLL_DIRS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin",
]

BUILDS = {
    "cuFFT_f64": BUILD / "windows-msvc-cuda"           / "python",
    "cuFFT_f32": BUILD / "windows-msvc-cuda-f32"       / "python",
    "VkFFT_f32": BUILD / "windows-msvc-cuda-vkfft-f32" / "python",
}

_MUMAX3 = None


def _find_mumax3():
    global _MUMAX3
    if _MUMAX3:
        return _MUMAX3
    candidates = [
        r"D:\Mumax3\mumax3.exe",
        r"C:\mumax3\mumax3.exe",
        r"D:\mumax3\mumax3.exe",
        r"C:\Program Files\mumax3\mumax3.exe",
    ]
    for c in candidates:
        if pathlib.Path(c).exists():
            _MUMAX3 = c; return c
    try:
        r = subprocess.run(["where", "mumax3"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _MUMAX3 = r.stdout.strip().split("\n")[0]
            return _MUMAX3
    except Exception:
        pass
    return None


def load_mm(build_label: str):
    """Import micromag from the given build label. Returns mm module."""
    py_path = BUILDS.get(build_label)
    if py_path is None or not py_path.exists():
        raise RuntimeError(f"Build not found: {build_label} -> {BUILDS.get(build_label)}")
    key = str(py_path)
    if key not in sys.path:
        sys.path.insert(0, key)
    for d in CUDA_DLL_DIRS:
        if pathlib.Path(d).exists():
            os.add_dll_directory(d)
    if "micromag" in sys.modules:
        del sys.modules["micromag"]
    import micromag as mm
    return mm


def run_mumax3(mx3_path, timeout_s: int = 300) -> dict:
    """Run mumax3 on an .mx3 script.  Returns dict with 'ok', 'stdout', 'wall_ms'."""
    exe = _find_mumax3()
    if not exe:
        return {"ok": False, "error": "mumax3 not found", "stdout": "", "wall_ms": None}
    mx3_path = pathlib.Path(mx3_path)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            [exe, str(mx3_path)],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(mx3_path.parent),
        )
        wall_ms = (time.perf_counter() - t0) * 1e3
        return {"ok": r.returncode == 0, "stdout": r.stdout + r.stderr,
                "wall_ms": wall_ms, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "stdout": "", "wall_ms": None}
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": "", "wall_ms": None}


def parse_mumax3_table(outdir: pathlib.Path):
    """Parse mumax3 table.txt -> numpy array (rows = time steps)."""
    tbl = pathlib.Path(outdir) / "table.txt"
    if not tbl.exists():
        return None
    try:
        data = np.loadtxt(str(tbl), comments="#")
        return data
    except Exception:
        return None


def uniform_m0(mm, grid, mx, my, mz):
    arr = np.array([mx, my, mz], dtype=np.float64)
    arr /= np.linalg.norm(arr)
    m0 = mm.VectorField3D(grid)
    m0.set_uniform(mm.Vec3(float(arr[0]), float(arr[1]), float(arr[2])))
    return m0


def run_mumaxplus(scenario_fn, timeout_s: int = 600) -> dict:
    """Run a mumax+ scenario.
    scenario_fn(mxp) -> dict with physics observables + optional 'wall_ms'.
    Returns dict with 'label', 'ok', 'wall_ms', and observable keys.

    mumax+ array convention: (3, nz, ny, nx) — component-first.
    mumax+ sign convention: jcur_z > 0 stabilises +z; use jcur_z = -|J| to switch +z -> -z.
    """
    try:
        import mumaxplus as mxp
        t0 = time.perf_counter()
        result = scenario_fn(mxp)
        wall_ms = (time.perf_counter() - t0) * 1e3
        return {"label": "mumax+", "ok": True,
                "wall_ms": result.pop("wall_ms", wall_ms), **result}
    except Exception as e:
        return {"label": "mumax+", "ok": False, "error": str(e), "wall_ms": None}


def mxp_topological_charge(m_eval):
    """Compute topological charge Q from mumax+ magnetization eval().
    m_eval: (3, nz, ny, nx) numpy array.
    Returns Q (float), summed over z-slices.
    Uses finite-difference solid-angle method (same physics as CS mm.topological_charge_Q).
    """
    # Reshape to (ny, nx, 3) for 2D slice (nz=1 assumed)
    m = m_eval[:, 0, :, :].transpose(1, 2, 0)  # (ny, nx, 3)
    ny, nx, _ = m.shape
    Q = 0.0
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = m[iy,   ix]
            b = m[iy,   ix+1]
            c = m[iy+1, ix]
            d = m[iy+1, ix+1]
            # Two triangles per cell
            for t in [(a, b, c), (b, d, c)]:
                m0, m1, m2 = t
                denom = (1 + np.dot(m0, m1)) * (1 + np.dot(m1, m2)) * (1 + np.dot(m2, m0))
                if abs(denom) < 1e-30:
                    continue
                num = np.dot(m0, np.cross(m1, m2))
                Q += 2.0 * np.arctan2(num, denom)
    return Q / (4.0 * np.pi)


def print_table(rows, headers, col_widths=None):
    """Pretty-print a table to stdout."""
    if col_widths is None:
        all_rows = [headers] + rows
        col_widths = [max(len(str(row[i])) for row in all_rows)
                      for i in range(len(headers))]
    header = "  ".join(f"{str(h):<{w}}" for h, w in zip(headers, col_widths))
    sep    = "  ".join("-" * w for w in col_widths)
    print(header); print(sep)
    for row in rows:
        print("  ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)))


def auto_integrator(mm, grid, mat, T_K=0.0, goal="dynamics", dt=None,
                    B_eff_T=0.1, t_end=None, seed=42, verbose=True):
    """Self-configuring integrator selection for a benchmark scenario.

    Calls mm.recommend_integrator() and instantiates the recommended GPU
    integrator, so a notebook is reproducibly self-configuring and the choice
    (+ reason) is logged for the paper's auto-selection table.

    Returns (integ, choice) where choice is the recommend_integrator() dict
    augmented with the matched mumax3 SetSolver index for fair comparison.
    """
    rec = mm.recommend_integrator(mat, T_K=T_K, goal=goal, dt=dt,
                                  B_eff_T=B_eff_T, t_end=t_end, verbose=False)
    name = rec["integrator"]
    if name == "HeunIntegratorGPU":
        integ = mm.HeunIntegratorGPU(grid, dt if dt else 5e-14, seed)
        mumax_solver = 2          # mumax3 SetSolver(2) = Heun
    elif name == "RK45IntegratorGPU":
        integ = mm.RK45IntegratorGPU(grid)
        mumax_solver = 5          # mumax3 SetSolver(5) = Dormand-Prince RK45
    else:  # RK4IntegratorGPU
        integ = mm.RK4IntegratorGPU(grid, dt if dt else 5e-14)
        mumax_solver = 4          # mumax3 SetSolver(4) = RK4
    choice = dict(rec)
    choice["mumax_solver"] = mumax_solver
    if verbose:
        print(f"[auto_integrator] -> {name}  (mumax3 SetSolver({mumax_solver}))  "
              f"reason: {rec.get('reason','')}")
    return integ, choice
