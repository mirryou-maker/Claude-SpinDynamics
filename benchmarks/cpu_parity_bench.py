"""CPU cross-platform parity benchmark for Claude-SpinDynamics.

Measures CPU RK4 ms/step for the full effective field (Demag + Exchange +
Anisotropy + Zeeman) across a few grid sizes, and OpenMP thread scaling on the
SP#4 grid. The identical harness is meant to be run on both Windows (MSVC +
vcpkg FFTW) and Linux (g++ + system FFTW) so the two builds can be compared on
the same hardware — a *port-quality* check, not a competitive claim.

Run:
    # thread count via OMP_NUM_THREADS (FFTW + OpenMP loops honour it)
    OMP_NUM_THREADS=8 python benchmarks/cpu_parity_bench.py            # Linux
    $env:OMP_NUM_THREADS=8; python benchmarks/cpu_parity_bench.py      # Windows

    # sweep threads and emit JSON for the parity table
    python benchmarks/cpu_parity_bench.py --threads 1,2,4,8 --json out.json

Output: a human table on stdout, plus optional JSON with every (grid, threads)
ms/step sample for regeneration of benchmarks/linux_cpu_parity.md.
"""
import os
import sys
import json
import time
import argparse
import platform
from pathlib import Path


def _add_micromag_to_path():
    """Locate the micromag module (release package or source build), any OS."""
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
    # CPU build first: this is a CPU benchmark, prefer the CPU module if both exist
    for _p in ("windows-msvc", "linux-gcc", "windows-msvc-cuda", "linux-gcc-cuda"):
        _py = root / "build" / _p / "python"
        if _hasmod(_py):
            sys.path.insert(0, str(_py)); return
    raise RuntimeError("micromag module not found (release package or source build).")


_add_micromag_to_path()
import micromag as mm   # noqa: E402


# Grid presets: (label, nx, ny, nz).  dx = dy = dz = 5 nm; SP#4-like geometry.
GRIDS = [
    ("SP#4  (200x50x1)",   200,  50, 1),   # ~10 K cells
    ("Med   (200x200x1)",  200, 200, 1),   # ~40 K cells
    ("Med2  (200x200x5)",  200, 200, 5),   # ~200 K cells
]
DX = 5e-9


def build_case(nx, ny, nz):
    """Full-physics CPU effective field + relaxed-ish vortex initial state."""
    g = mm.StructuredGrid(nx, ny, nz, DX, DX, DX)
    m = mm.VectorField3D(g)
    m.set_vortex(cx=nx * DX * 0.5, cy=ny * DX * 0.5, core_radius=8e-9)

    mat = mm.Material.permalloy()
    mat.alpha = 0.02
    mat.K_uniaxial = 5e4
    mat.easy_axis = mm.Vec3(0, 0, 1)

    heff = mm.EffectiveFieldSum()
    heff.add(mm.DemagField(g))
    heff.add(mm.ExchangeField(mm.BoundaryCondition.Neumann))
    heff.add(mm.UniaxialAnisotropyField())
    heff.add(mm.ZeemanField(mm.Vec3(0, 0, 1e4)))
    return g, m, mat, heff


def time_steps(nx, ny, nz, n_warm=5, n_meas=40):
    g, m, mat, heff = build_case(nx, ny, nz)
    rk4 = mm.RK4Integrator(5e-14)
    for _ in range(n_warm):               # warm plans / caches
        rk4.step(m, mat, heff)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        rk4.step(m, mat, heff)
    dt = time.perf_counter() - t0
    return dt / n_meas * 1e3              # ms/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default=os.environ.get("OMP_NUM_THREADS", ""),
                    help="comma list, e.g. 1,2,4,8 (spawns subprocesses). "
                         "Empty = single run at current OMP_NUM_THREADS.")
    ap.add_argument("--json", default="")
    ap.add_argument("--_single", action="store_true",
                    help=argparse.SUPPRESS)   # internal: one thread-count run
    args = ap.parse_args()

    osname = f"{platform.system()} {platform.machine()}"
    pyver = platform.python_version()
    cuda = mm.cuda_available()

    # -- single-run mode (used directly, or spawned per thread count) ---------
    if args._single or not args.threads or "," not in args.threads:
        nthr = os.environ.get("OMP_NUM_THREADS", "default")
        rows = []
        print(f"# {osname}  py{pyver}  OMP_NUM_THREADS={nthr}  cuda={cuda}")
        print(f"{'grid':<20}{'cells':>10}{'ms/step':>12}")
        for label, nx, ny, nz in GRIDS:
            ms = time_steps(nx, ny, nz)
            cells = nx * ny * nz
            rows.append({"grid": label, "cells": cells, "ms_step": ms,
                         "threads": nthr})
            print(f"{label:<20}{cells:>10}{ms:>12.3f}")
        if args.json:
            Path(args.json).write_text(json.dumps(
                {"os": osname, "python": pyver, "cuda": cuda, "rows": rows},
                indent=2))
            print(f"wrote {args.json}")
        return

    # -- sweep mode: re-spawn self per thread count (clean FFTW/OMP init) -----
    import subprocess
    all_rows = []
    for t in args.threads.split(","):
        t = t.strip()
        env = dict(os.environ, OMP_NUM_THREADS=t)
        r = subprocess.run([sys.executable, __file__, "--_single"],
                           env=env, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise SystemExit(f"thread={t} run failed")
        # parse the printed table
        for line in r.stdout.splitlines():
            for label, nx, ny, nz in GRIDS:
                if line.startswith(label):
                    ms = float(line.split()[-1])
                    all_rows.append({"grid": label, "cells": nx * ny * nz,
                                     "ms_step": ms, "threads": t})
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"os": osname, "python": pyver, "cuda": cuda, "rows": all_rows},
            indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
