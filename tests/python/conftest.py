"""Locate the micromag module for the Python binding tests (any OS, CPU or GPU
build). Prefers the CPU build — every test here must run without a GPU."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# If the caller already put a build on PYTHONPATH (e.g. CI's build/ci-windows),
# honour it; otherwise auto-locate a known build/release layout. Verify the
# import is actually OUR package — an unrelated project installing a module
# also named `micromag` (e.g. FEM-SpinDynamics) must not shadow these tests.
try:
    import micromag
    if not hasattr(micromag, "StructuredGrid"):
        raise ImportError(f"foreign 'micromag' on sys.path: {micromag.__file__}")
except ImportError:
    sys.modules.pop("micromag", None)
    from micromag_locate import add_micromag_to_path
    add_micromag_to_path(prefer="cpu")
