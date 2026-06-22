"""Unified throughput driver for the mumax3-compatible family (mumax3 + MuMax-CO).

Both take the same .mx3 scripts; only the exe differs. Hardened timing:
  - warmup run (kernel compile / cache)
  - REPEATS independent two-run subtractions  (ms_step = (t(N2)-t(N1))/(N2-N1))
  - report median ms/step + IQR across repeats
Writes records to benchmarks/results/all_solvers.json via results_io.

Usage:
  py -3.13 benchmarks/run_throughput_mumax.py [scenario_ids...]   (default: all)
  e.g.  py -3.13 benchmarks/run_throughput_mumax.py S2 S5
"""
import os, sys, time, shutil, pathlib, statistics, subprocess

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "results"))
import results_io as rio  # noqa: E402

MX3DIR = HERE / "fair_comparison" / "mx3"
OUT    = HERE / "_throughput_out"

EXES = {
    "mumax3":   r"D:/Mumax3/mumax3.exe",
    "mumax-co": r"D:/Claude-Code-R/MuMax-CO/mumax3-src/mumax3.exe",
}

# (id, mx3_stem, nx, ny, nz, dim, large)
SCENARIOS = {
    "S1": ("s1_pow2_rk4",       128, 128,  4, "3D", False),
    "S2": ("s2_nonpow2_2d_rk4", 200,  50,  1, "2D", False),
    "S3": ("s3_nonpow2_3d_rk4", 300, 300,  6, "3D", False),
    "S5": ("s5_large3d_rk4",    500, 500, 10, "3D", True),
}

N_WARM = 10
REPEATS = 5
EVALS_RK4 = 4


def step_counts(cells):
    """Size-tiered (N1, N2) so (N2-N1)*ms_step >> subprocess startup jitter (~1s).
    Targets several seconds of pure compute in the two-run difference."""
    if cells < 50_000:      return 2000, 10000   # ~0.5 ms/step -> 4 s signal
    if cells < 300_000:     return 1000, 5000
    if cells < 1_000_000:   return 300, 1500
    return 50, 250                                # large -> ~50 ms/step, 10 s signal


def write_script(stem, nsteps):
    src = (MX3DIR / f"{stem}.mx3").read_text(encoding="utf-8")
    src = src.replace("NSTEPS_PLACEHOLDER", str(nsteps))
    dst = HERE / f"_tp_{stem}_{nsteps}.mx3"
    dst.write_text(src, encoding="utf-8")
    return dst


def run_mx3(exe, mx3_path):
    out_dir = OUT / mx3_path.stem
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    t0 = time.perf_counter()
    p = subprocess.run([exe, "-f", "-o", str(out_dir), str(mx3_path)],
                       capture_output=True, text=True, timeout=900)
    wall = time.perf_counter() - t0
    low = (p.stdout + p.stderr).lower()
    ok = "panic" not in low and ("error" not in low or "no error" in low)
    return wall, ok


def bench_one(exe, stem, cells):
    n1, n2 = step_counts(cells)
    # warmup
    pw = write_script(stem, N_WARM); run_mx3(exe, pw)
    ms_list = []
    for _ in range(REPEATS):
        p1 = write_script(stem, n1); t1, ok1 = run_mx3(exe, p1)
        p2 = write_script(stem, n2); t2, ok2 = run_mx3(exe, p2)
        if ok1 and ok2:
            ms_list.append((t2 - t1) / (n2 - n1) * 1e3)
    for f in HERE.glob("_tp_*.mx3"):
        try: f.unlink()
        except Exception: pass
    if not ms_list:
        return None
    med = statistics.median(ms_list)
    # IQR across repeats (robust spread)
    if len(ms_list) >= 4:
        srt = sorted(ms_list)
        q1 = srt[len(srt)//4]; q3 = srt[(3*len(srt))//4]; iqr = q3 - q1
    else:
        iqr = max(ms_list) - min(ms_list)
    return med, iqr, ms_list


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    want = [s.upper() for s in sys.argv[1:]] or list(SCENARIOS)
    print(f"{'solver':<10}{'scenario':<8}{'cells':>9}{'ms/step':>10}{'ms/eval':>10}{'IQR':>9}  (RK4 f32, {REPEATS} reps)")
    print("-" * 70)
    for sid in want:
        if sid not in SCENARIOS:
            continue
        stem, nx, ny, nz, dim, large = SCENARIOS[sid]
        cells = nx * ny * nz
        for solver, exe in EXES.items():
            r = bench_one(exe, stem, cells)
            if r is None:
                print(f"{solver:<10}{sid:<8}{cells:>9}   FAILED")
                continue
            med, iqr, _ = r
            rio.append(rio.make_record(
                f"{sid}_{stem}", solver, "f32", "RK4", dim, cells,
                ms_step=med, metric="throughput", repeats=REPEATS, ms_step_iqr=iqr,
                notes=f"mumax-family two-run subtraction; exe={exe}"))
            print(f"{solver:<10}{sid:<8}{cells:>9}{med:>10.3f}{med/EVALS_RK4:>10.3f}{iqr:>9.3f}", flush=True)
    print(f"\nappended to {rio.STORE}")
