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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
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
