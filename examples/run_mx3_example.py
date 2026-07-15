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
    def _adddll(_d):                      # add_dll_directory is Windows-only
        if hasattr(os, "add_dll_directory") and os.path.isdir(_d):
            os.add_dll_directory(_d)
    def _hasmod(_p):
        _pat = "_micromag*.pyd" if sys.platform == "win32" else "_micromag*.so"
        return bool(list(_p.glob(_pat)))
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"                          # 1) GPU release package
    if rtd.is_dir():
        _adddll(str(rtd))
        for v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            py = root / v / "python"
            if _hasmod(py):
                sys.path.insert(0, str(py)); return
    if _hasmod(root / "python"):                        # 2) CPU release package
        _adddll(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _adddll(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
    for preset in ("windows-msvc-cuda", "windows-msvc", "linux-gcc-cuda", "linux-gcc"):
        py = root / "build" / preset / "python"
        if _hasmod(py):
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
