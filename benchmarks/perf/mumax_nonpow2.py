"""mumax3 reference timing on our actual benchmark grids (non-pow2).

Same Heun / FixDt approach as mumax_perf.py: two-run subtraction
to cancel mumax3 startup + GPU compile cost.

Usage:  py -3.13 mumax_nonpow2.py [path-to-mumax3.exe]
"""
import os, sys, subprocess, time

EXE = sys.argv[1] if len(sys.argv) > 1 else r"D:/Mumax3/mumax3.exe"
HERE = os.path.dirname(os.path.abspath(__file__))

# Our exact benchmark grids — non-pow2, matches sp4_full_gpu_bench / llg_large_bench
GRIDS = [
    (200,  50, 1),    # SP#4  — 10K cells
    (200, 200, 5),    # Medium — 200K cells (3D)
    (500, 500, 10),   # Large  — 2.5M cells (3D)
]
DT   = 5e-14
EVALS = 2  # Heun
# Step counts chosen so (N2-N1) dominates startup (~2 s) by ≥20×
N_WARM = 20
N1_SMALL, N2_SMALL = 3000, 30000   # thin-film (fast steps)
N1_MED,   N2_MED   = 100,  1000    # 200K cells
N1_LARGE, N2_LARGE = 10,   100     # 2.5M cells

def step_counts(ncells):
    if ncells <= 50_000:
        return N1_SMALL, N2_SMALL
    elif ncells <= 500_000:
        return N1_MED, N2_MED
    else:
        return N1_LARGE, N2_LARGE

def script(nx, ny, nz, nsteps):
    return f"""SetGridSize({nx}, {ny}, {nz})
SetCellSize(2e-9, 2e-9, 2e-9)
Msat  = 800e3
Aex   = 13e-12
alpha = 0.02
setsolver(2)
FixDt = {DT}
m = uniform(1, 0.1, 0)
B_ext = vector(-24.6e3*mu0, 4.3e3*mu0, 0)
run({nsteps}*{DT})
"""

def timed_run(nx, ny, nz, nsteps, outdir):
    path = os.path.join(HERE, "_mxnp.mx3")
    with open(path, "w") as fh:
        fh.write(script(nx, ny, nz, nsteps))
    t0 = time.perf_counter()
    p = subprocess.run(
        [EXE, "-f", "-o", outdir, path],
        capture_output=True, text=True, timeout=600)
    wall = time.perf_counter() - t0
    ok = ("panic" not in (p.stdout + p.stderr).lower() and
          "error"  not in p.stderr.lower())
    return wall, ok

if __name__ == "__main__":
    print(f"[mumax3] Heun FixDt={DT:.0e}  exe={EXE}")
    print(f"{'grid':>14}{'cells':>10}{'ms/step':>10}{'ms/eval':>10}{'Mcell-st/s':>13}")
    outdir = os.path.join(HERE, "_mxnp.out")
    for nx, ny, nz in GRIDS:
        cells = nx * ny * nz
        N1, N2 = step_counts(cells)
        # warmup: compile + cache kernel
        timed_run(nx, ny, nz, N_WARM, outdir)
        t1, ok1 = timed_run(nx, ny, nz, N1, outdir)
        t2, ok2 = timed_run(nx, ny, nz, N2, outdir)
        if not (ok1 and ok2):
            print(f"{f'{nx}x{ny}x{nz}':>14}{cells:>10}   FAILED")
            continue
        ms_step = (t2 - t1) / (N2 - N1) * 1e3
        thr = cells * 1000.0 / ms_step / 1e6
        print(f"{f'{nx}x{ny}x{nz}':>14}{cells:>10}"
              f"{ms_step:>10.3f}{ms_step/EVALS:>10.3f}{thr:>13.1f}", flush=True)
