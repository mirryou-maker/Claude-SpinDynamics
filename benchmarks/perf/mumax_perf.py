"""mumax3 / MuMax-CO fixed-step performance reference.

For each grid: write a Heun (2 evals/step) FixDt script, run it twice with
different step counts, and take ms/step = (t2 - t1)/(N2 - N1) so the mumax3
startup/compile cost cancels out.  Reports ms/step AND ms/eval so it can be
compared fairly against Claude-SD's RK4 (4 evals/step).

Usage:  py -3.13 mumax_perf.py <path-to-mumax3.exe> [label]
"""
import os, sys, subprocess, time

EXE = sys.argv[1] if len(sys.argv) > 1 else r"D:/Mumax3/mumax3.exe"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "mumax3"
HERE = os.path.dirname(os.path.abspath(__file__))

GRIDS = [(256, 64, 1), (512, 128, 1), (1024, 256, 1), (1024, 512, 1), (2048, 512, 1)]
DT = 1e-14
# A warmup run first compiles+caches the demag kernel so the two timed runs
# share an identical (cached) startup that cancels in (t2 - t1).  Large step
# counts make stepping time dominate mumax3's ~2 s GUI startup variance.
N_WARM = 20
N1, N2 = 3000, 30000       # step counts; difference removes startup cost
EVALS = 2                  # Heun = 2 field evals/step


def script(nx, ny, nz, nsteps):
    return f"""SetGridSize({nx}, {ny}, {nz})
SetCellSize(2e-9, 2e-9, 3e-9)
Msat = 800e3
Aex  = 13e-12
alpha = 0.02
setsolver(2)
FixDt = {DT}
m = uniform(1, 0.05, 0)
B_ext = vector(1e3*mu0, 0, 0)
steps({nsteps})
"""


def timed_run(nx, ny, nz, nsteps):
    path = os.path.join(HERE, "_mxperf.mx3")
    with open(path, "w") as fh:
        fh.write(script(nx, ny, nz, nsteps))
    t0 = time.perf_counter()
    p = subprocess.run([EXE, "-f", "-o", os.path.join(HERE, "_mxperf.out"), path],
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    ok = "panic" not in (p.stdout + p.stderr).lower() and "error" not in p.stderr.lower()
    return wall, ok


if __name__ == "__main__":
    print(f"[{LABEL}] fixed-step Heun (2 evals/step) perf, FixDt={DT:.0e}")
    print(f"{'grid':>16}{'cells':>10}{'ms/step':>10}{'ms/eval':>10}{'Mcell-st/s':>12}")
    for nx, ny, nz in GRIDS:
        timed_run(nx, ny, nz, N_WARM)          # warmup: compile + cache kernel
        t1, ok1 = timed_run(nx, ny, nz, N1)
        t2, ok2 = timed_run(nx, ny, nz, N2)
        if not (ok1 and ok2):
            print(f"{f'{nx}x{ny}x{nz}':>16}{nx*ny*nz:>10}   FAILED")
            continue
        ms_step = (t2 - t1) / (N2 - N1) * 1e3
        cells = nx * ny * nz
        thr = cells * 1000.0 / ms_step / 1e6
        print(f"{f'{nx}x{ny}x{nz}':>16}{cells:>10}{ms_step:>10.3f}"
              f"{ms_step/EVALS:>10.3f}{thr:>12.1f}", flush=True)
