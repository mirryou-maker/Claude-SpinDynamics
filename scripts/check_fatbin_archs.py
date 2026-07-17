"""Release gate: verify a CUDA binary embeds kernels for every supported arch.

The v1.0.0 GPU package shipped with kernels for sm_120 only and crashed on all
other GPUs (claude-sd-gpu-arch-issue_JYCho.md). This check makes that class of
mistake impossible to ship again: run it on the built _micromag*.pyd (or any
GPU exe) and it fails unless every architecture in --require is present.

Usage:
    python scripts/check_fatbin_archs.py build/windows-msvc-cuda/python/_micromag.cp313-win_amd64.pyd
    python scripts/check_fatbin_archs.py --require 75,80,86,89,90,100,120 --ptx 120 <binary>
    python scripts/check_fatbin_archs.py --require native <binary>   # dev build: just list

Needs cuobjdump (ships with the CUDA toolkit) on PATH or in the default
toolkit location.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REQUIRE = "75,80,86,89,90,100,120"
DEFAULT_PTX = "120"
CUOBJDUMP_HINTS = [
    "cuobjdump",
    r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/cuobjdump.exe",
    "/usr/local/cuda/bin/cuobjdump",
]


def find_cuobjdump():
    for c in CUOBJDUMP_HINTS:
        if shutil.which(c) or Path(c).exists():
            return c
    sys.exit("ERROR: cuobjdump not found (install the CUDA toolkit or add it to PATH).")


def embedded_archs(binary, cuobjdump):
    """Return (sm_set, ptx_set) of architectures embedded in the binary."""
    sm, ptx = set(), set()
    for flag, bucket in (("--list-elf", sm), ("--list-ptx", ptx)):
        p = subprocess.run([cuobjdump, flag, str(binary)],
                           capture_output=True, text=True, errors="replace")
        # lines like: ELF file    1: _micromag.1.sm_89.cubin  /  PTX file 1: x.compute_120.ptx
        bucket.update(re.findall(r"\.(?:sm|compute)_(\d+)\.", p.stdout))
    return sm, ptx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", help=".pyd/.dll/.exe/.so containing CUDA fatbins")
    ap.add_argument("--require", default=DEFAULT_REQUIRE,
                    help=f"comma list of required sm_ archs (default {DEFAULT_REQUIRE}); "
                         "'native' = report only, no gate")
    ap.add_argument("--ptx", default=DEFAULT_PTX,
                    help=f"comma list of required PTX (compute_) archs (default {DEFAULT_PTX}); '' to skip")
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        sys.exit(f"ERROR: {binary} does not exist")

    sm, ptx = embedded_archs(binary, find_cuobjdump())
    print(f"{binary.name}:")
    print(f"  ELF (sm_)      : {sorted(sm, key=int) or 'NONE'}")
    print(f"  PTX (compute_) : {sorted(ptx, key=int) or 'NONE'}")

    if args.require.strip().lower() == "native":
        print("  (report-only mode)")
        return

    missing_sm = [a for a in args.require.split(",") if a.strip() and a.strip() not in sm]
    missing_ptx = [a for a in args.ptx.split(",") if a.strip() and a.strip() not in ptx]
    if missing_sm or missing_ptx:
        if missing_sm:
            print(f"  MISSING sm_ archs : {missing_sm}")
        if missing_ptx:
            print(f"  MISSING PTX archs : {missing_ptx}")
        sys.exit("FAIL: this binary would crash on GPUs of the missing architectures. "
                 "Rebuild with -DMICROMAG_CUDA_ARCHS=release.")
    print("  PASS: all required architectures embedded.")


if __name__ == "__main__":
    main()
