"""Locate the micromag module for the Python binding tests (any OS, CPU or GPU
build). Prefers the CPU build — every test here must run without a GPU."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from micromag_locate import add_micromag_to_path  # noqa: E402

add_micromag_to_path(prefer="cpu")
