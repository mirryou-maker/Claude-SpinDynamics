"""
Notebook 25: GPU STT Performance Benchmark

Measures GPU speedup for SlonczewskiSTTGPU vs CPU SlonczewskiSTT
across grid sizes. Uses RK4 integrators on both sides.

Grids tested: 1x1x1, 10x10x1, 32x32x1, 64x64x1, 128x128x1
Each: 1000 steps, measures ms/step on GPU and CPU.
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc-cuda', 'python'))
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')

import numpy as np
import micromag as mm

print("Notebook 25: GPU STT Performance Benchmark")
print(f"  CUDA: {mm.cuda_available()}")

Ms = 580e3; A = 15e-12; K = 0.5e6
d  = 3e-9;  P = 0.5; dt = 5e-14

def make_mat():
    mat = mm.Material()
    mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = K
    mat.easy_axis = mm.Vec3(0,0,1); mat.alpha = 0.02
    return mat

def init_m(g):
    a = np.zeros((1, g.ny, g.nx, 3))
    a[..., 0] = np.sin(np.deg2rad(5))
    a[..., 2] = np.cos(np.deg2rad(5))
    m = mm.VectorField3D(g); mm.from_numpy(m, a); return m

NSTEPS = 1000
WARMUP = 50
p_vec  = mm.Vec3(0, 0, 1)

grid_sizes = [(1,1,1), (10,10,1), (32,32,1), (64,64,1), (128,128,1)]

results = []
print(f"\n{'Grid':>14}  {'N':>7}  {'CPU ms/step':>12}  {'GPU ms/step':>12}  {'Speedup':>8}")
print("-" * 62)

for (nx, ny, nz) in grid_sizes:
    g   = mm.StructuredGrid(nx, ny, nz, d, d, d)
    mat = make_mat()
    m0  = init_m(g)
    N   = nx * ny * nz

    # ------ CPU ------
    demag_c = mm.DemagField(g)
    exch_c  = mm.ExchangeField()
    aniso_c = mm.UniaxialAnisotropyField()
    stt_c   = mm.SlonczewskiSTT(1e12, P, d, p_vec, 0.0)

    heff_c = mm.EffectiveFieldSum()
    heff_c.add(demag_c); heff_c.add(exch_c); heff_c.add(aniso_c)

    stt_sum_c = mm.SpinTorqueSum()
    stt_sum_c.add(stt_c)

    integ_c = mm.RK4Integrator(dt)

    # Warmup CPU
    for _ in range(WARMUP):
        integ_c.step(m0, mat, heff_c, stt_sum_c)

    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        integ_c.step(m0, mat, heff_c, stt_sum_c)
    cpu_ms = (time.perf_counter() - t0) / NSTEPS * 1e3

    # ------ GPU ------
    demag_g = mm.DemagFieldGPU(g)
    exch_g  = mm.ExchangeFieldGPU(g)
    aniso_g = mm.UniaxialAnisotropyFieldGPU(g)
    stt_g   = mm.SlonczewskiSTTGPU(g, 1e12, P, d, p_vec, 0.0)

    fields_g = mm.FieldSumGPU()
    fields_g.add(exch_g); fields_g.add(aniso_g)

    torques_g = mm.SpinTorqueSumGPU()
    torques_g.add(stt_g)

    integ_g = mm.RK4IntegratorGPU(g, dt)
    integ_g.upload(m0)

    # Warmup GPU
    for _ in range(WARMUP):
        integ_g.step(mat, demag_g, fields_g, torques_g)

    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        integ_g.step(mat, demag_g, fields_g, torques_g)
    gpu_ms = (time.perf_counter() - t0) / NSTEPS * 1e3

    speedup = cpu_ms / gpu_ms
    results.append((nx, ny, nz, N, cpu_ms, gpu_ms, speedup))
    print(f"  {nx}x{ny}x{nz}  {N:>7,}  {cpu_ms:>12.3f}  {gpu_ms:>12.3f}  {speedup:>8.1f}x")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N_arr   = [r[3] for r in results]
    cpu_arr = [r[4] for r in results]
    gpu_arr = [r[5] for r in results]
    spd_arr = [r[6] for r in results]
    labels  = [f"{r[0]}x{r[1]}x{r[2]}" for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.loglog(N_arr, cpu_arr, 'o-', color='C3', lw=2, ms=7, label='CPU RK4 + STT')
    ax.loglog(N_arr, gpu_arr, 's-', color='C0', lw=2, ms=7, label='GPU RK4 + SlonczewskiSTTGPU')
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (N_arr[i], gpu_arr[i]),
                    textcoords='offset points', xytext=(5,-10), fontsize=7)
    ax.set_xlabel('Grid cells N')
    ax.set_ylabel('Time per step (ms, log scale)')
    ax.set_title('CPU vs GPU: ms/step with STT')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.semilogx(N_arr, spd_arr, 'D-', color='C2', lw=2, ms=8)
    for i, lbl in enumerate(labels):
        ax.annotate(f"  {spd_arr[i]:.1f}x", (N_arr[i], spd_arr[i]), fontsize=8)
    ax.axhline(1, color='k', ls='--', lw=1, alpha=0.4)
    ax.set_xlabel('Grid cells N')
    ax.set_ylabel('GPU Speedup vs CPU')
    ax.set_title('GPU Speedup with SlonczewskiSTTGPU')
    ax.grid(alpha=0.3, which='both')
    ax.set_ylim(0, max(spd_arr) * 1.4)

    plt.suptitle(f'GPU STT Benchmark ({NSTEPS} steps each, Pt/Co PMA, d={d*1e9:.0f}nm)', fontsize=10)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '25_stt_gpu_bench.png')
    plt.savefig(out, dpi=120); print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"Plot error: {e}")

print("\n=== Summary ===")
for r in results:
    print(f"  {r[0]}x{r[1]}x{r[2]}  N={r[3]:,}  CPU={r[4]:.3f}ms  GPU={r[5]:.3f}ms  {r[6]:.1f}x")
