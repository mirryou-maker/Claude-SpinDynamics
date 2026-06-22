"""mumax3 side of the fair-comparison benchmark.

Scenarios S1–S3, S5 (fixed RK4): two-run subtraction to cancel startup cost.
Scenario S4 (adaptive DP45):     single run measuring total wall time for 1 ns.

Usage:
    py -3.13 run_mumax_bench.py [path/to/mumax3.exe]

Output: fair_comparison_mumax.tsv
"""
import os, sys, re, subprocess, time, json, shutil, pathlib

HERE   = pathlib.Path(__file__).parent
MX3DIR = HERE / "mx3"
OUT    = HERE / "_mxout"
EXE    = sys.argv[1] if len(sys.argv) > 1 else r"D:/Mumax3/mumax3.exe"

# ── step counts ────────────────────────────────────────────────────────────────
N_WARM = 10       # warmup steps (compile + cache CUDA kernels)
N1     = 200      # first timed run
N2     = 600      # second timed run  (ms/step = (t2-t1)/(N2-N1)*1e3)
# For S5 (Large 2.5M cells) use fewer steps due to cost:
N1_L, N2_L = 5, 20

EVALS_RK4 = 4     # RK4 = 4 field evaluations per step

# ── scenario definitions ───────────────────────────────────────────────────────
# (label, mx3_stem, nx, ny, nz, is_adaptive, use_large_N)
FIXED_SCENARIOS = [
    ("S1 pow2 128×128×4",       "s1_pow2_rk4",      128, 128,  4, False),
    ("S2 nonpow2 200×50×1",     "s2_nonpow2_2d_rk4",200,  50,  1, False),
    ("S3 nonpow2 300×300×6",    "s3_nonpow2_3d_rk4",300, 300,  6, False),
    ("S5 large 500×500×10",     "s5_large3d_rk4",   500, 500, 10, True),
]
ADAPTIVE_SCENARIOS = [
    # (label, mx3_stem, nx, ny, nz, t_sim_ns)
    ("S4 SP#4 switch 1ns",       "s4_sp4_adaptive",       200,  50,  1,  1.0),
    ("S6 Medium switch 0.3ns",   "s6_medium_adaptive",    300, 300,  6,  0.3),
    ("S7 DW motion 2ns",         "s7_dw_motion_adaptive", 400,  20,  1,  2.0),
    ("S8 Precession 1ns α=5e-3", "s8_precession_adaptive",200,  50,  1,  1.0),
]

# ── helpers ────────────────────────────────────────────────────────────────────
def write_script(stem: str, nsteps: int) -> pathlib.Path:
    """Substitute NSTEPS_PLACEHOLDER and write a temp .mx3 file."""
    src = (MX3DIR / f"{stem}.mx3").read_text(encoding="utf-8")
    src = src.replace("NSTEPS_PLACEHOLDER", str(nsteps))
    dst = HERE / f"_run_{stem}_{nsteps}.mx3"
    dst.write_text(src, encoding="utf-8")
    return dst

def run_mx3(mx3_path: pathlib.Path) -> tuple[float, bool]:
    out_dir = OUT / mx3_path.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.perf_counter()
    p  = subprocess.run(
        [EXE, "-f", "-o", str(out_dir), str(mx3_path)],
        capture_output=True, text=True, timeout=600
    )
    wall = time.perf_counter() - t0
    combined = (p.stdout + p.stderr).lower()
    ok = "panic" not in combined and ("error" not in combined or "no error" in combined)
    return wall, ok

def bench_fixed(stem: str, nx: int, ny: int, nz: int, large: bool) -> dict | None:
    n1 = N1_L if large else N1
    n2 = N2_L if large else N2

    # warmup
    p = write_script(stem, N_WARM)
    run_mx3(p)

    # timed runs
    p1 = write_script(stem, n1)
    t1, ok1 = run_mx3(p1)
    p2 = write_script(stem, n2)
    t2, ok2 = run_mx3(p2)

    for p in (write_script(stem, N_WARM), p1, p2):
        try: p.unlink()
        except: pass

    if not (ok1 and ok2):
        return None

    ms_step = (t2 - t1) / (n2 - n1) * 1e3
    return {
        "cells":   nx * ny * nz,
        "ms_step": ms_step,
        "ms_eval": ms_step / EVALS_RK4,
    }

def bench_adaptive(stem: str, t_sim_ns: float = 1.0) -> dict | None:
    """Run the adaptive scenario once; wall time = metric."""
    mx3_path = MX3DIR / f"{stem}.mx3"
    out_dir  = OUT / stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.perf_counter()
    p  = subprocess.run(
        [EXE, "-f", "-o", str(out_dir), str(mx3_path)],
        capture_output=True, text=True, timeout=600
    )
    wall_ms  = (time.perf_counter() - t0) * 1e3
    combined = (p.stdout + p.stderr).lower()
    ok = "panic" not in combined and "error" not in p.stderr.lower()

    # Count accepted steps from log (mumax3 prints "step XX" or similar)
    # Also try the table row count as an approximation
    n_steps = None
    candidates = list((out_dir).rglob("table.txt"))
    if candidates:
        lines = candidates[0].read_text().splitlines()
        data  = [l for l in lines if not l.startswith("#") and l.strip()]
        n_steps = len(data)

    return {
        "wall_ms":   wall_ms,
        "ms_per_ns": wall_ms / t_sim_ns,
        "ok":        ok,
        "n_steps":   n_steps,
        "t_sim_ns":  t_sim_ns,
    }

# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    results = {}

    print(f"\n{'Scenario':<28} {'Cells':>9} {'ms/step':>9} {'ms/eval':>9}  note")
    print("-" * 68)

    for label, stem, nx, ny, nz, large in FIXED_SCENARIOS:
        r = bench_fixed(stem, nx, ny, nz, large)
        results[label] = r
        if r:
            cells_k = nx*ny*nz / 1e3
            print(f"{label:<28} {nx*ny*nz:>9,}  {r['ms_step']:>8.3f}  {r['ms_eval']:>8.3f}  RK4 f32")
        else:
            print(f"{label:<28}  FAILED")

    print()
    print(f"{'Scenario':<32} {'Cells':>9} {'wall_ms':>9} {'ms/ns_sim':>10} {'steps':>7}  note")
    print("-" * 76)
    for label, stem, nx, ny, nz, t_ns in ADAPTIVE_SCENARIOS:
        r = bench_adaptive(stem, t_ns)
        results[label] = r
        if r and r.get("ok"):
            steps_str = str(r.get("n_steps", "?"))
            print(f"{label:<32} {nx*ny*nz:>9,}  {r['wall_ms']:>8.1f}  {r['ms_per_ns']:>9.1f}  {steps_str:>7}  DP45 MaxErr=1e-4")
        else:
            print(f"{label:<32}  FAILED")

    (HERE / "fair_comparison_mumax.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {HERE}/fair_comparison_mumax.json")
