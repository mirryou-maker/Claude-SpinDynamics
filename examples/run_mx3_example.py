"""Run a mumax3-syntax .mx3 script on the Claude-SD engine via micromag.mx3.

`micromag.mx3` is an *interpreter*: it parses a mumax3-syntax script and executes
it on the Claude-SD (micromag) engine — it does NOT call the mumax3 program, and
mumax3 does not need to be installed. See docs/USER_GUIDE.md Appendix A for the
supported/unsupported command coverage.

Usage:
    python examples/run_mx3_example.py            # runs examples/mx3/sp4.mx3
    python examples/run_mx3_example.py my.mx3     # runs your own script
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Prefer a GPU build if present (falls back to the CPU build otherwise).
GPU_PY = ROOT / "build" / "windows-msvc-cuda" / "python"
CPU_PY = ROOT / "build" / "windows-msvc" / "python"
if GPU_PY.exists():
    # GPU module needs the CUDA runtime DLLs on the search path before import.
    cuda_bin = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    sys.path.insert(0, str(GPU_PY))
else:
    sys.path.insert(0, str(CPU_PY))

import micromag as mm


def main():
    script = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples" / "mx3" / "sp4.mx3"
    outdir = ROOT / "mx3_out"

    print(f"backend      : {'GPU (CUDA)' if mm.cuda_available() else 'CPU'}")
    print(f"running .mx3 : {script}")
    print("(mumax3 syntax, executed on the Claude-SD engine — mumax3 is not invoked)\n")

    eng = mm.run_mx3(str(script), outdir=str(outdir))   # CPU/GPU auto-selected

    mx, my, mz = mm.mean_magnetization(eng.m)
    print(f"\nfinal <m> = ({mx:+.4f}, {my:+.4f}, {mz:+.4f})")
    print(f"outputs written to: {outdir}")
    # For sp4.mx3 (µMAG SP#4 field A) expect <mx> heading toward ~ -0.98.


if __name__ == "__main__":
    main()
