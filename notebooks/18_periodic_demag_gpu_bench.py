"""Notebook 14 — DemagFieldPeriodicGPU vs DemagFieldGPU: Performance Benchmark

Measures wall-clock time per LLG step for periodic-BC (DemagFieldPeriodicGPU)
versus open-BC (DemagFieldGPU) on the same grid.

Periodic BC: FFT size = N (grid size).
Open BC:     FFT size = 2N (zero-padded) → 8× larger 3D FFT.

Expected: periodic demag ~2–4× faster than open BC on same grid.

Test grids:
  Small:  32×32×4   (4K cells)
  Medium: 64×64×8   (32K cells)
  Large: 128×128×16 (262K cells)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc-cuda', 'python'))
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import micromag as mm

if not mm.cuda_available():
    print("CUDA not available — skipping GPU benchmark")
    sys.exit(0)

mu_0 = 4 * np.pi * 1e-7
mat  = mm.Material.permalloy()

# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def bench_grid(nx, ny, nz, n_warmup=5, n_steps=20):
    cell_nm = 5e-9
    g  = mm.StructuredGrid(nx, ny, nz, cell_nm, cell_nm, cell_nm)

    # Initial state: vortex-like tilt
    m  = mm.VectorField3D(g)
    arr = mm.to_numpy(m)
    arr[..., 0] = 1.0; arr[..., 1] = 0.0; arr[..., 2] = 0.0
    mm.from_numpy(m, arr)

    exch   = mm.ExchangeFieldGPU(g)
    zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 80e3))
    dt     = 1e-13

    results = {}
    for label, demag_cls in [("Open BC (DemagFieldGPU)",     mm.DemagFieldGPU),
                              ("Periodic BC (DemagFieldPeriodicGPU)", mm.DemagFieldPeriodicGPU)]:
        demag = demag_cls(g)
        integ = mm.RK4IntegratorGPU(g, dt)
        integ.upload(m)

        # Warmup
        for _ in range(n_warmup):
            integ.step(mat, demag, exch, zeeman)

        # Timed run
        t0 = time.perf_counter()
        for _ in range(n_steps):
            integ.step(mat, demag, exch, zeeman)
        t1 = time.perf_counter()

        ms_per_step = (t1 - t0) / n_steps * 1e3
        results[label] = ms_per_step
        print(f"  {label}: {ms_per_step:.3f} ms/step")

    speedup = results["Open BC (DemagFieldGPU)"] / results["Periodic BC (DemagFieldPeriodicGPU)"]
    return results, speedup

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
grids = [
    ("Small  32×32×4",   32,  32,  4),
    ("Medium 64×64×8",   64,  64,  8),
    ("Large  128×128×16",128, 128, 16),
]

labels      = []
open_ms     = []
periodic_ms = []
speedups    = []

for name, nx, ny, nz in grids:
    cells = nx * ny * nz
    print(f"\n{name} ({cells:,} cells)")
    res, spd = bench_grid(nx, ny, nz)
    labels.append(f"{name}\n({cells:,} cells)")
    open_ms.append(res["Open BC (DemagFieldGPU)"])
    periodic_ms.append(res["Periodic BC (DemagFieldPeriodicGPU)"])
    speedups.append(spd)
    print(f"  Speedup (Periodic / Open): {spd:.2f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(labels))
w = 0.35
ax1.bar(x - w/2, open_ms,     w, label="Open BC (DemagFieldGPU)",     color="steelblue")
ax1.bar(x + w/2, periodic_ms, w, label="Periodic BC (DemagFieldPeriodicGPU)", color="coral")
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("ms / RK4 step")
ax1.set_title("GPU Demag: Open vs Periodic BC — Wall Time")
ax1.legend()

bars = ax2.bar(x, speedups, color="seagreen", width=0.5)
ax2.axhline(1.0, ls="--", color="gray", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Speedup (Open / Periodic)")
ax2.set_title("Speedup: Open BC ÷ Periodic BC\n(8× smaller FFT expected)")
for bar, s in zip(bars, speedups):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{s:.2f}×", ha="center", va="bottom")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "14_periodic_demag_gpu_bench.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved: {out}")
