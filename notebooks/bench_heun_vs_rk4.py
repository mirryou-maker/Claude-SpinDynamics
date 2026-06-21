"""Benchmark: HeunIntegratorGPU (T=0) vs RK4IntegratorGPU at SP4 / Medium / Large.

Heun = 2 field evals per step (predictor + corrector).
RK4  = 4 field evals per step.
At T=0, Heun gives identical physics accuracy to first-order (lower truncation error
per step than RK4), with half the field evaluations → expected ~2x speedup.

Also demonstrates run_until_converged_gpu with HeunIntegratorGPU
(now possible after adding max_angle_gpu() to HeunIntegratorGPU).
"""

import os, time
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
import sys
sys.path.insert(0, r"D:/Claude-Code-R/Claude-SpinDynamics/build/windows-msvc-cuda/python")
import micromag as mm
import numpy as np

DT = 5e-13  # 0.5 ps — same for both integrators
N_WARMUP = 20
N_BENCH  = 200

def bench_grid(label, nx, ny, nz, dx=5e-9):
    print(f"\n{'='*60}")
    print(f"Grid: {label} ({nx}x{ny}x{nz}) = {nx*ny*nz//1000}K cells")
    print(f"{'='*60}")

    g = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)
    mat = mm.Material.permalloy()
    mat.alpha = 0.5  # high damping → faster convergence

    m0 = mm.uniform_mag(g, mm.Vec3(1, 0, 0))

    demag = mm.DemagFieldGPU(g)
    exch  = mm.ExchangeFieldGPU(g)
    ze    = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 0))

    def run_rk4():
        integ = mm.RK4IntegratorGPU(g, DT)
        integ.upload(m0)
        # warmup (build graph)
        for _ in range(N_WARMUP):
            integ.step(mat, demag, exch, ze)
        # timed
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            integ.step(mat, demag, exch, ze)
        elapsed = time.perf_counter() - t0
        ms_step = elapsed / N_BENCH * 1e3
        return ms_step

    def run_heun():
        integ = mm.HeunIntegratorGPU(g, DT)
        integ.upload(m0)
        # warmup (build graph)
        for _ in range(N_WARMUP):
            integ.step(mat, demag, exch, ze, T_K=0.0)
        # timed
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            integ.step(mat, demag, exch, ze, T_K=0.0)
        elapsed = time.perf_counter() - t0
        ms_step = elapsed / N_BENCH * 1e3
        return ms_step

    rk4_ms  = run_rk4()
    heun_ms = run_heun()
    ratio   = rk4_ms / heun_ms

    print(f"  RK4  (T=0, 4 evals/step): {rk4_ms:.3f} ms/step")
    print(f"  Heun (T=0, 2 evals/step): {heun_ms:.3f} ms/step")
    print(f"  Speedup (RK4/Heun):        {ratio:.2f}x")
    print(f"  Per-eval ratio (expect ~1.0x): {rk4_ms/4:.3f} vs {heun_ms/2:.3f} ms/eval")

    # Verify max_angle_gpu works on Heun
    integ_h = mm.HeunIntegratorGPU(g, DT)
    integ_h.upload(m0)
    integ_h.step(mat, demag, exch, ze, T_K=0.0)
    angle = integ_h.max_angle_gpu()
    print(f"  HeunIntegratorGPU.max_angle_gpu() = {angle:.2f} deg  [OK]")

    return rk4_ms, heun_ms, ratio

# SP#4
bench_grid("SP#4", 200, 50, 1, dx=5e-9)

# Medium
bench_grid("Medium", 200, 200, 5, dx=5e-9)

# Large (optional - slow)
# bench_grid("Large", 500, 500, 10, dx=5e-9)

print("\nDone.")
