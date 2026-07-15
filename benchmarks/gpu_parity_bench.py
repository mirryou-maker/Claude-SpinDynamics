"""GPU cross-platform parity benchmark for Claude-SpinDynamics.

Measures GPU RK4 ms/step for the full effective field (Demag + Exchange +
Uniaxial) across the same three grids used in the CLAUDE.md GPU table, so the
Linux GPU build (e.g. AWS g6.xlarge, NVIDIA L4) can be compared with the Windows
GPU numbers on identical CUDA source — a *port-quality* check.

Requires a CUDA build:  build/linux-gcc-cuda/python  (or a GPU release package).

Run:
    python benchmarks/gpu_parity_bench.py --json benchmarks/_parity_gpu_linux.json
    python benchmarks/gpu_parity_bench.py --sizes sp4,medium         # subset
    python benchmarks/gpu_parity_bench.py --steps 400                # more steps

Timing note: RK4IntegratorGPU.step() is asynchronous (CUDA-graph replay); the
following download() forces a device sync, so we time N steps + one download and
divide by N.
"""
import os
import sys
import json
import time
import argparse
import platform
from pathlib import Path


def _add_micromag_to_path():
    """Locate the micromag module (GPU release package or source build), any OS."""
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
    # CUDA build first: this is a GPU benchmark
    for _p in ("linux-gcc-cuda", "windows-msvc-cuda", "linux-gcc", "windows-msvc"):
        _py = root / "build" / _p / "python"
        if _hasmod(_py):
            sys.path.insert(0, str(_py)); return
    raise RuntimeError("micromag module not found (release package or source build).")


_add_micromag_to_path()
import micromag as mm   # noqa: E402


# (key, label, nx, ny, nz) — matches the CLAUDE.md GPU perf table geometry.
GRIDS = [
    ("sp4",    "SP#4  (200x50x1)",   200,  50,  1),   # 10 K   (Win GPU ~1.51 ms)
    ("medium", "Med   (200x200x5)",  200, 200,  5),   # 200 K  (Win GPU ~21.9 ms)
    ("large",  "Large (500x500x10)", 500, 500, 10),   # 2.5 M  (Win GPU ~290 ms)
]
DX = 5e-9
# Windows GPU baseline (ms/step) from CLAUDE.md, for the printed ratio column.
WIN_GPU_MS = {"sp4": 1.51, "medium": 21.89, "large": 290.0}


def build_case(nx, ny, nz):
    g = mm.StructuredGrid(nx, ny, nz, DX, DX, DX)
    m = mm.VectorField3D(g)
    m.set_uniform(mm.Vec3(0.1, 0.0, 1.0))
    m.normalize()

    mat = mm.Material.permalloy()
    mat.alpha = 0.02
    mat.K_uniaxial = 5e4
    mat.easy_axis = mm.Vec3(0, 0, 1)

    demag = mm.DemagFieldGPU(g)
    fields = mm.FieldSumGPU()
    fields.add(mm.ExchangeFieldGPU(g))
    fields.add(mm.UniaxialAnisotropyFieldGPU(g))
    return g, m, mat, demag, fields


def time_steps(nx, ny, nz, n_warm, n_meas):
    g, m, mat, demag, fields = build_case(nx, ny, nz)
    integ = mm.RK4IntegratorGPU(g, 5e-14)
    integ.upload(m)
    for _ in range(n_warm):               # warm CUDA-graph capture / plans
        integ.step(mat, demag, fields)
    integ.download(m)                     # sync
    t0 = time.perf_counter()
    for _ in range(n_meas):
        integ.step(mat, demag, fields)
    integ.download(m)                     # forces device sync
    dt = time.perf_counter() - t0
    return dt / n_meas * 1e3              # ms/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="sp4,medium,large",
                    help="comma list of grid keys: sp4,medium,large")
    ap.add_argument("--steps", type=int, default=300, help="measured steps")
    ap.add_argument("--warm", type=int, default=20, help="warm-up steps")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not mm.cuda_available():
        sys.exit("ERROR: this is a CUDA build check but mm.cuda_available() is "
                 "False. Point at a GPU build (build/linux-gcc-cuda/python).")

    want = [s.strip() for s in args.sizes.split(",")]
    osname = f"{platform.system()} {platform.machine()}"
    print(f"# {osname}  py{platform.python_version()}  cuda={mm.cuda_available()}")
    print(f"{'grid':<22}{'cells':>10}{'ms/step':>12}{'Win ms':>10}{'Lin/Win':>10}")
    rows = []
    for key, label, nx, ny, nz in GRIDS:
        if key not in want:
            continue
        ms = time_steps(nx, ny, nz, args.warm, args.steps)
        cells = nx * ny * nz
        win = WIN_GPU_MS.get(key)
        ratio = (ms / win) if win else float("nan")
        rows.append({"key": key, "grid": label, "cells": cells,
                     "ms_step": ms, "win_ms": win, "ratio_lin_win": ratio})
        print(f"{label:<22}{cells:>10}{ms:>12.3f}"
              f"{(win if win else 0):>10.2f}{ratio:>10.2f}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"os": osname, "python": platform.python_version(),
             "gpu_note": "set via nvidia-smi at run time", "rows": rows},
            indent=2))
        print(f"wrote {args.json}")

    print("\nVERDICT GUIDE: ratio ~1.0 => Linux GPU on par with Windows. The f64 "
          "path should match closely; f32/Blackwell-specific speedups will NOT "
          "reproduce on L4/A10G (no Blackwell Tensor-Core FFT).")


if __name__ == "__main__":
    main()
