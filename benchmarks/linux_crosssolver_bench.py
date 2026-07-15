"""Native-Linux cross-solver throughput: Claude-SD (f64/f32) vs mumax3 (f32).

Same-host comparison to build Linux competitive evidence (Ada regime) — see
benchmarks/linux_competitive_claim.md. Physics is matched to the Windows
fair-comparison campaign (Msat=860e3, Aex=13e-12, alpha=0.02, demag+exchange,
fixed-step, dt=5e-14). Throughput is dominated by the demag FFT + exchange, so a
zero vs non-zero applied field does not affect the timing.

Methodology (mirrors run_throughput_cs.py / run_throughput_mumax.py):
  - Claude-SD: in-process RK4IntegratorGPU, download() forces a device sync;
    median of REPEATS runs of N steps.  One subprocess per build (two _micromag
    modules cannot share an interpreter).
  - mumax3: subprocess on a generated .mx3; two-run subtraction
    ms_step = (t(N2)-t(N1))/(N2-N1) to cancel process startup + kernel compile;
    median of REPEATS.

Usage (on the Linux GPU host):
  python3 benchmarks/linux_crosssolver_bench.py \
      --mumax ~/mumax3/mumax3.12_linux_cuda12.9/mumax3 \
      --cs-f64 build/linux-gcc-cuda/python \
      --cs-f32 build/linux-gcc-cuda-f32/python \
      --json benchmarks/_crosssolver_linux_L4.json

  # extra grids (label:nx,ny,nz,dim):
  python3 ... --grids "S5:500,500,10,3D L1M:256,256,16,3D L4M:512,512,16,3D"

Internal re-exec:  --cs-worker <build_dir> <grids_json>
"""
import os
import sys
import json
import time
import shutil
import pathlib
import argparse
import statistics
import subprocess

HERE = pathlib.Path(__file__).resolve().parent

# Campaign fair-comparison grids (id, nx, ny, nz, dim).
DEFAULT_GRIDS = [
    ("S2", 200,  50,  1, "2D"),   # 10 K   — small, launch-bound (CS's winning regime)
    ("S1", 128, 128,  4, "3D"),   # 65 K   — CS beats mumax3 on Windows here
    ("S3", 300, 300,  6, "3D"),   # 540 K  — mid
    ("S5", 500, 500, 10, "3D"),   # 2.5 M  — large, mumax3 FFT pulls ahead on Windows
]

# Matched physics (mumax3 fair-comparison .mx3).
MSAT, AEX, ALPHA, DT = 860e3, 13e-12, 0.02, 5e-14
REPEATS = 5
EVALS = 4  # eval/step used for ms/eval column


# ---------------------------------------------------------------------------
# Claude-SD side (runs inside one build's module via --cs-worker re-exec)
# ---------------------------------------------------------------------------
def cs_worker(build_dir, grids):
    if hasattr(os, "add_dll_directory"):
        pass  # Linux: no-op
    sys.path.insert(0, build_dir)
    import micromag as mm

    def step_counts(cells):
        if cells < 50_000:    return 200, 3000
        if cells < 300_000:   return 100, 1500
        if cells < 1_000_000: return 50, 600
        return 20, 200

    def time_grid(nx, ny, nz):
        dx = 3.0e-9
        g = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)
        mat = mm.Material.permalloy()
        mat.Ms = MSAT
        mat.A_exchange = AEX
        mat.alpha = ALPHA
        demag = mm.DemagFieldGPU(g)
        exch  = mm.ExchangeFieldGPU(g)
        zee   = mm.ZeemanFieldGPU(g, mm.Vec3(0.0, 0.0, 0.0))
        fs = mm.FieldSumGPU(); fs.add(exch); fs.add(zee)
        rk = mm.RK4IntegratorGPU(g, DT)
        m0 = mm.VectorField3D(g); m0.set_uniform(mm.Vec3(1.0, 0.05, 0.0)); m0.normalize()
        rk.upload(m0)
        m_cpu = mm.VectorField3D(g)
        cells = nx * ny * nz
        nwarm, n = step_counts(cells)
        for _ in range(nwarm):
            rk.step(mat, demag, fs)
        rk.download(m_cpu)                       # sync
        ms = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            for _ in range(n):
                rk.step(mat, demag, fs)
            rk.download(m_cpu)                    # sync
            ms.append((time.perf_counter() - t0) / n * 1e3)
        med = statistics.median(ms)
        srt = sorted(ms)
        iqr = srt[(3*len(srt))//4] - srt[len(srt)//4] if len(ms) >= 4 else max(ms)-min(ms)
        return med, iqr

    out = []
    for sid, nx, ny, nz, dim in grids:
        try:
            med, iqr = time_grid(nx, ny, nz)
            out.append({"sid": sid, "cells": nx*ny*nz, "ms_step": med, "iqr": iqr, "ok": True})
        except Exception as e:
            out.append({"sid": sid, "cells": nx*ny*nz, "ok": False, "err": str(e)})
    print("RESULT_JSON " + json.dumps(out))


def run_cs_build(py, build_dir, grids):
    arg = json.dumps(grids)
    p = subprocess.run([py, "-u", str(HERE / "linux_crosssolver_bench.py"),
                        "--cs-worker", build_dir, arg],
                       capture_output=True, text=True, timeout=2400)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    sys.stderr.write(f"CS worker ({build_dir}) failed:\n{p.stdout[-800:]}\n{p.stderr[-800:]}\n")
    return None


# ---------------------------------------------------------------------------
# mumax3 side (subprocess, two-run subtraction)
# ---------------------------------------------------------------------------
def mx3_text(nx, ny, nz, nsteps):
    dz = 20e-9 if nz == 1 else 5e-9
    return (f"SetGridSize({nx}, {ny}, {nz})\n"
            f"SetCellSize(5e-9, 5e-9, {dz})\n"
            f"Msat  = {MSAT}\nAex = {AEX}\nalpha = {ALPHA}\n"
            f"setsolver(4)\nFixDt = {DT}\n"
            f"m = uniform(1, 0.05, 0)\nB_ext = vector(0,0,0)\n"
            f"steps({nsteps})\n")


def step_counts_mx(cells):
    if cells < 50_000:    return 2000, 10000
    if cells < 300_000:   return 1000, 5000
    if cells < 1_000_000: return 300, 1500
    return 50, 250


def run_mumax(exe, nx, ny, nz, nsteps, tmp):
    mx = tmp / f"_x_{nx}_{ny}_{nz}_{nsteps}.mx3"
    mx.write_text(mx3_text(nx, ny, nz, nsteps))
    out = tmp / f"{mx.stem}.out"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    t0 = time.perf_counter()
    p = subprocess.run([exe, "-f", "-o", str(out), str(mx)],
                       capture_output=True, text=True, timeout=1800)
    wall = time.perf_counter() - t0
    low = (p.stdout + p.stderr).lower()
    ok = "panic" not in low and p.returncode == 0
    mx.unlink(missing_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    return wall, ok


def bench_mumax(exe, nx, ny, nz, tmp):
    cells = nx * ny * nz
    n1, n2 = step_counts_mx(cells)
    run_mumax(exe, nx, ny, nz, 10, tmp)          # warmup
    ms = []
    for _ in range(REPEATS):
        t1, ok1 = run_mumax(exe, nx, ny, nz, n1, tmp)
        t2, ok2 = run_mumax(exe, nx, ny, nz, n2, tmp)
        if ok1 and ok2:
            ms.append((t2 - t1) / (n2 - n1) * 1e3)
    if not ms:
        return None
    med = statistics.median(ms)
    srt = sorted(ms)
    iqr = srt[(3*len(srt))//4] - srt[len(srt)//4] if len(ms) >= 4 else max(ms)-min(ms)
    return med, iqr


# ---------------------------------------------------------------------------
def parse_grids(spec):
    grids = []
    for tok in spec.split():
        label, rest = tok.split(":")
        nx, ny, nz, dim = rest.split(",")
        grids.append((label, int(nx), int(ny), int(nz), dim))
    return grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs-worker", nargs=2, default=None,
                    help=argparse.SUPPRESS)   # <build_dir> <grids_json>
    ap.add_argument("--mumax", default="")
    ap.add_argument("--cs-f64", default="")
    ap.add_argument("--cs-f32", default="")
    ap.add_argument("--grids", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--py", default=sys.executable)
    args = ap.parse_args()

    if args.cs_worker:
        build_dir, grids_json = args.cs_worker
        cs_worker(build_dir, json.loads(grids_json))
        return

    grids = parse_grids(args.grids) if args.grids else DEFAULT_GRIDS
    tmp = HERE / "_xsolver_tmp"; tmp.mkdir(exist_ok=True)

    # collect ms/step per (solver, sid)
    data = {}   # sid -> {solver: (ms, iqr)}
    for sid, nx, ny, nz, dim in grids:
        data[sid] = {"cells": nx*ny*nz, "dim": dim}

    cs_builds = [("CS_f64", args.cs_f64), ("CS_f32", args.cs_f32)]
    for name, bd in cs_builds:
        if not bd or not pathlib.Path(bd).exists():
            print(f"# skip {name}: build dir missing ({bd})")
            continue
        res = run_cs_build(args.py, bd, [list(g) for g in grids])
        if res:
            for r in res:
                if r.get("ok"):
                    data[r["sid"]][name] = (r["ms_step"], r["iqr"])
                else:
                    print(f"# {name} {r['sid']} FAILED: {r.get('err','')[:80]}")

    if args.mumax and pathlib.Path(args.mumax).expanduser().exists():
        exe = str(pathlib.Path(args.mumax).expanduser())
        for sid, nx, ny, nz, dim in grids:
            r = bench_mumax(exe, nx, ny, nz, tmp)
            if r:
                data[sid]["mumax3_f32"] = r
            else:
                print(f"# mumax3 {sid} FAILED")
    else:
        print(f"# skip mumax3: exe missing ({args.mumax})")

    # ---- report -----------------------------------------------------------
    print(f"\n{'grid':<6}{'cells':>10}{'dim':>5}"
          f"{'CS_f64':>10}{'CS_f32':>10}{'mumax3':>10}"
          f"{'f32/mmx':>9}{'f64/mmx':>9}")
    print("-" * 79)
    for sid, nx, ny, nz, dim in grids:
        d = data[sid]
        f64 = d.get("CS_f64", (None,))[0]
        f32 = d.get("CS_f32", (None,))[0]
        mmx = d.get("mumax3_f32", (None,))[0]
        def s(v): return f"{v:.3f}" if v is not None else "  --"
        r32 = f"{f32/mmx:.2f}" if (f32 and mmx) else " --"
        r64 = f"{f64/mmx:.2f}" if (f64 and mmx) else " --"
        print(f"{sid:<6}{d['cells']:>10}{dim:>5}{s(f64):>10}{s(f32):>10}{s(mmx):>10}"
              f"{r32:>9}{r64:>9}")
    print("\nratio < 1.0 => Claude-SD faster than mumax3 (same grid, same host).")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"\nwrote {args.json}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
