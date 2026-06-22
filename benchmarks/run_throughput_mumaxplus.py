"""mumax+ (mumaxplus) throughput runner — Python API, not .mx3.

mumax+ has NO fixed-step RK4. Its solvers: Heun, BogackiShampine(RK23),
CashKarp, Fehlberg, DormandPrince(RK45-DP, default). For a fair cross-solver
comparison we time DormandPrince with a FIXED timestep (adaptive off) and report
ms per FIELD-EVAL (the method-independent unit of work): DOPRI5 = 6 evals/step.

Writes records (solver="mumax+", integrator="RK45-DP") to all_solvers.json.

Usage:  py -3.13 benchmarks/run_throughput_mumaxplus.py [S1 S2 S3 S5]
"""
import os, sys, time, statistics, pathlib

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "results"))
import results_io as rio  # noqa: E402

import mumaxplus as mxp
from mumaxplus import World, Grid, Ferromagnet

BUILD = "f64" if str(getattr(mxp, "FP_PRECISION", "")).lower().find("double") >= 0 else "f32"

SCENARIOS = {
    "S1": (128, 128,  4, "3D"),
    "S2": (200,  50,  1, "2D"),
    "S3": (300, 300,  6, "3D"),
    "S5": (500, 500, 10, "3D"),
}
REPEATS = 5
EVALS_DP = 6


def step_counts(cells):
    if cells < 50_000:    return 20, 200
    if cells < 300_000:   return 10, 100
    if cells < 1_000_000: return 5, 40
    return 3, 15


def bench(nx, ny, nz):
    dx = 3.0e-9
    w = World(cellsize=(dx, dx, dx))
    m = Ferromagnet(w, Grid((nx, ny, nz)))
    m.msat = 8e5; m.aex = 13e-12; m.alpha = 0.02
    m.magnetization = (1, 0.1, 0)
    m.bias_magnetic_field = (0, 0, 0.05)
    ts = w.timesolver
    ts.set_method("DormandPrince")
    ts.adaptive_timestep = False
    ts.timestep = 5e-14

    cells = nx * ny * nz
    nwarm, n = step_counts(cells)
    ts.steps(nwarm); _ = m.magnetization.eval()  # warmup + sync

    ms_list = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        ts.steps(n)
        _ = m.magnetization.eval()   # force GPU sync
        ms_list.append((time.perf_counter() - t0) / n * 1e3)
    med = statistics.median(ms_list)
    if len(ms_list) >= 4:
        srt = sorted(ms_list); iqr = srt[(3*len(srt))//4] - srt[len(srt)//4]
    else:
        iqr = max(ms_list) - min(ms_list)
    return med, iqr


if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SCENARIOS)
    print(f"mumax+ {mxp.__version__}  precision={BUILD}  (DormandPrince fixed-dt, {EVALS_DP} evals/step)")
    print(f"{'scenario':<8}{'cells':>10}{'ms/step':>10}{'ms/eval':>10}{'IQR':>9}")
    print("-" * 50)
    for sid in want:
        if sid not in SCENARIOS:
            continue
        nx, ny, nz, dim = SCENARIOS[sid]
        try:
            med, iqr = bench(nx, ny, nz)
        except Exception as e:
            print(f"{sid:<8}{nx*ny*nz:>10}   FAILED {str(e)[:40]}")
            continue
        rio.append(rio.make_record(
            f"{sid}_throughput", "mumax+", BUILD, "RK45-DP", dim, nx*ny*nz,
            ms_step=med, metric="throughput", repeats=REPEATS, ms_step_iqr=iqr,
            notes="mumax+ DormandPrince fixed-dt; 6 evals/step; ms/eval is the fair metric"))
        print(f"{sid:<8}{nx*ny*nz:>10}{med:>10.3f}{med/EVALS_DP:>10.3f}{iqr:>9.3f}", flush=True)
    print(f"\nappended to {rio.STORE}")
