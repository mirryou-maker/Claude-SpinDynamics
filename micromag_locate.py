"""Single source of truth for locating the micromag module + its DLLs.

Every example/notebook/benchmark script bootstraps with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from micromag_locate import add_micromag_to_path
    add_micromag_to_path()

so search-path changes are made HERE, once — not in 50+ copies.

Layouts handled (any OS):
  1. Source build:      <root>/build/<preset>/python/_micromag*.{pyd,so}
  2. GPU release pkg:   <root>/runtime-dll + <root>/<variant>/python/
  3. CPU release pkg:   <root>/python/_micromag*.pyd
NOTE for packagers: copy this file into the release-package root so the bundled
examples keep working (they resolve it via parent.parent).
"""
import os
import sys
from pathlib import Path

_GPU_VARIANTS = ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32")
_PRESETS_GPU_FIRST = ("windows-msvc-cuda", "windows-msvc", "linux-gcc-cuda", "linux-gcc")
_PRESETS_CPU_FIRST = ("windows-msvc", "linux-gcc", "windows-msvc-cuda", "linux-gcc-cuda")
_CUDA_BIN = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"


def _adddll(d):
    """os.add_dll_directory is Windows-only; no-op elsewhere/if missing."""
    if hasattr(os, "add_dll_directory") and os.path.isdir(d):
        os.add_dll_directory(d)


def _hasmod(p):
    pat = "_micromag*.pyd" if sys.platform == "win32" else "_micromag*.so"
    return bool(list(Path(p).glob(pat)))


def add_micromag_to_path(prefer="gpu", root=None):
    """Locate micromag (release package or source build) and prepend it to
    sys.path. `prefer` orders the source-build preset search ("gpu"|"cpu").
    Returns the directory used. Raises RuntimeError if nothing is found."""
    # numpy (MKL) and the bundled module both ship an OpenMP runtime; allow
    # both to load instead of aborting with "OMP: Error #15".
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    root = Path(root) if root else Path(__file__).resolve().parent

    rtd = root / "runtime-dll"                     # 1) GPU release package
    if rtd.is_dir():
        _adddll(str(rtd))
        for v in _GPU_VARIANTS:
            py = root / v / "python"
            if _hasmod(py):
                sys.path.insert(0, str(py)); return str(py)
    if _hasmod(root / "python"):                   # 2) CPU release package
        _adddll(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return str(root / "python")
    _adddll(_CUDA_BIN)                             # 3) source build tree
    presets = _PRESETS_CPU_FIRST if prefer == "cpu" else _PRESETS_GPU_FIRST
    for preset in presets:
        py = root / "build" / preset / "python"
        if _hasmod(py):
            sys.path.insert(0, str(py)); return str(py)
    raise RuntimeError(
        "micromag module not found. Build it first (cmake --preset "
        "windows-msvc[-cuda] / linux-gcc[-cuda]) or run from a release package.")
