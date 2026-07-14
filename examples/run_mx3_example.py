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

def _add_micromag_to_path():
    """Find the micromag module + its DLLs in either a downloaded release package
    (../runtime-dll + ../<variant>/python, or ../python for the CPU package) or a
    source build (../build/<preset>/python + the system CUDA toolkit)."""
    # numpy (MKL) and the bundled module both ship an OpenMP runtime; allow both
    # to load instead of aborting with "OMP: Error #15".
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"                          # 1) GPU release package
    if rtd.is_dir():
        os.add_dll_directory(str(rtd))
        for v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            py = root / v / "python"
            if list(py.glob("_micromag*.pyd")):
                sys.path.insert(0, str(py)); return
    if list((root / "python").glob("_micromag*.pyd")):  # 2) CPU release package
        os.add_dll_directory(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    cuda_bin = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"
    if os.path.isdir(cuda_bin):                          # 3) source build tree
        os.add_dll_directory(cuda_bin)
    for preset in ("windows-msvc-cuda", "windows-msvc"):
        py = root / "build" / preset / "python"
        if py.is_dir():
            sys.path.insert(0, str(py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()
import micromag as mm


def main():
    here = Path(__file__).resolve().parent          # examples/ (repo or package)
    script = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "mx3" / "sp4.mx3"
    outdir = Path.cwd() / "mx3_out"

    print(f"backend      : {'GPU (CUDA)' if mm.cuda_available() else 'CPU'}")
    print(f"running .mx3 : {script}")
    print("(mumax3 syntax, executed on the Claude-SD engine; mumax3 is not invoked)\n")

    eng = mm.run_mx3(str(script), outdir=str(outdir))   # CPU/GPU auto-selected

    mx, my, mz = mm.mean_magnetization(eng.m)
    print(f"\nfinal <m> = ({mx:+.4f}, {my:+.4f}, {mz:+.4f})")
    print(f"outputs written to: {outdir}")
    # For sp4.mx3 (µMAG SP#4 field A) expect <mx> heading toward ~ -0.98.


if __name__ == "__main__":
    main()
