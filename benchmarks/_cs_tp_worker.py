"""Claude-SD throughput worker — runs INSIDE one build's python module.

Invoked by run_throughput_cs.py as a subprocess (one per build, since two
_micromag modules cannot coexist in one interpreter). Times RK4IntegratorGPU
ms/step for a list of grids and prints JSON to stdout.

Usage:
  python _cs_tp_worker.py <build_python_dir> <cuda_bin_dir> <scenarios_json>
where scenarios_json = '[["S2",200,50,1],["S5",500,500,10], ...]'
"""
import os, sys, json, time, statistics

build_dir = sys.argv[1]
cuda_bin  = sys.argv[2]
scenarios = json.loads(sys.argv[3])

os.add_dll_directory(cuda_bin)
sys.path.insert(0, build_dir)
import micromag as mm

REPEATS = 5


def step_counts(cells):
    if cells < 50_000:    return 200, 3000
    if cells < 300_000:   return 100, 1500
    if cells < 1_000_000: return 50, 600
    return 20, 200


def time_grid(nx, ny, nz):
    dx = 3.0e-9
    g = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)
    mat = mm.Material.permalloy()
    demag = mm.DemagFieldGPU(g)
    exch  = mm.ExchangeFieldGPU(g)
    zee   = mm.ZeemanFieldGPU(g, mm.Vec3(-24.6e3, 4.3e3, 0.0))
    fs = mm.FieldSumGPU(); fs.add(exch); fs.add(zee)
    rk = mm.RK4IntegratorGPU(g, 5e-14)
    m0 = mm.VectorField3D(g); m0.set_uniform(mm.Vec3(1.0, 0.1, 0.0)); m0.normalize()
    rk.upload(m0)
    m_cpu = mm.VectorField3D(g)

    cells = nx * ny * nz
    nwarm, n = step_counts(cells)
    # warmup (also triggers CUDA-graph capture)
    for _ in range(nwarm):
        rk.step(mat, demag, fs)
    rk.download(m_cpu)   # sync

    ms_list = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(n):
            rk.step(mat, demag, fs)
        rk.download(m_cpu)   # sync (download blocks until GPU done)
        ms_list.append((time.perf_counter() - t0) / n * 1e3)
    med = statistics.median(ms_list)
    if len(ms_list) >= 4:
        srt = sorted(ms_list); iqr = srt[(3*len(srt))//4] - srt[len(srt)//4]
    else:
        iqr = max(ms_list) - min(ms_list)
    return med, iqr


out = []
for sid, nx, ny, nz in scenarios:
    try:
        med, iqr = time_grid(nx, ny, nz)
        out.append({"sid": sid, "cells": nx*ny*nz, "ms_step": med, "iqr": iqr, "ok": True})
    except Exception as e:
        out.append({"sid": sid, "cells": nx*ny*nz, "ok": False, "err": str(e)})

print("RESULT_JSON " + json.dumps(out))
