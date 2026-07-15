"""Locate the micromag module for the Python binding tests (any OS, CPU or GPU
build). Prefers the CPU build — every test here must run without a GPU."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# If the caller already put a build on PYTHONPATH (e.g. CI's build/ci-windows),
# honour it; otherwise auto-locate a known build/release layout.
try:
    import micromag  # noqa: F401
except ImportError:
    from micromag_locate import add_micromag_to_path
    add_micromag_to_path(prefer="cpu")
