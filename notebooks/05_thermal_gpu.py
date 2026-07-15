# %% [markdown]
# # GPU SLLG — HeunIntegratorGPU at Finite Temperature
# **Stratonovich Heun on GPU: full thermal SP#4 switching, zero PCIe per step**
#
# Compares three integrators on SP#4 Field A at T = 300 K:
#   - CPU HeunIntegrator (reference)
#   - GPU HeunIntegratorGPU (T=300 K — stochastic)
#   - GPU RK4IntegratorGPU  (T=0 K — deterministic, for trajectory comparison)
#
# Requires: build with `cmake --preset windows-msvc-cuda`

# %%
import os, sys, time
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import micromag as mm
import numpy as np
import matplotlib.pyplot as plt

print('CUDA available:', mm.cuda_available())
print('HeunIntegratorGPU:', hasattr(mm, 'HeunIntegratorGPU'))

# %% [markdown]
# ## 1. Setup

# %%
grid  = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
mat   = mm.Material.permalloy()
H_ext = mm.Vec3(-24.6e3, 4.3e3, 0.0)
T_K   = 300.0
dt    = 1e-13     # 0.1 ps — fixed step for Heun/SLLG
n_steps  = 5000   # 0.5 ns
log_every = 50    # record every 5 ps

print(f'Grid : {grid.nx}x{grid.ny}x{grid.nz} = {grid.size} cells')
print(f'T    : {T_K} K,  dt = {dt*1e15:.0f} fs,  t_sim = {n_steps*dt*1e9:.2f} ns')

def make_m0():
    m = mm.VectorField3D(grid)
    m.set_uniform(mm.Vec3(1.0, 0.1, 0.0))
    m.normalize()
    return m

# %% [markdown]
# ## 2. GPU SLLG — HeunIntegratorGPU (T = 300 K)

# %%
demag  = mm.DemagFieldGPU(grid)
exch   = mm.ExchangeFieldGPU(grid)
zeeman = mm.ZeemanFieldGPU(grid, H_ext)

m0 = make_m0()
heun_gpu = mm.HeunIntegratorGPU(grid, dt, seed=42)
heun_gpu.upload(m0)

ts_gpu, mxs_gpu, mys_gpu = [], [], []

t0 = time.perf_counter()
for step in range(n_steps):
    heun_gpu.step(mat, demag, exch, zeeman, T_K)
    if step % log_every == 0:
        m_tmp = make_m0()
        heun_gpu.download(m_tmp)
        mx, my, mz = mm.mean_magnetization(m_tmp)
        ts_gpu.append(step * dt * 1e9)
        mxs_gpu.append(mx); mys_gpu.append(my)
wall_gpu_heun = time.perf_counter() - t0

m_final_gpu = make_m0()
heun_gpu.download(m_final_gpu)
mx_f, my_f, mz_f = mm.mean_magnetization(m_final_gpu)
print(f'GPU Heun (T=300K) : {wall_gpu_heun:.2f} s  ({wall_gpu_heun*1000/n_steps:.3f} ms/step)')
print(f'  Final <m> = ({mx_f:.4f}, {my_f:.4f}, {mz_f:.4f})')

# %% [markdown]
# ## 3. GPU Deterministic — RK4IntegratorGPU (T = 0 K, reference)

# %%
m0 = make_m0()
rk4_gpu = mm.RK4IntegratorGPU(grid, dt)
rk4_gpu.upload(m0)

ts_rk4, mxs_rk4 = [], []

t0 = time.perf_counter()
for step in range(n_steps):
    rk4_gpu.step(mat, demag, exch, zeeman)
    if step % log_every == 0:
        m_tmp = make_m0()
        rk4_gpu.download(m_tmp)
        mx, *_ = mm.mean_magnetization(m_tmp)
        ts_rk4.append(step * dt * 1e9)
        mxs_rk4.append(mx)
wall_gpu_rk4 = time.perf_counter() - t0
print(f'GPU RK4  (T=0 K)  : {wall_gpu_rk4:.2f} s  ({wall_gpu_rk4*1000/n_steps:.3f} ms/step)')

# %% [markdown]
# ## 4. CPU SLLG Reference — HeunIntegrator (T = 300 K)

# %%
m0 = make_m0()
heff_cpu = mm.EffectiveFieldSum()
heff_cpu.add(mm.DemagField(grid))
heff_cpu.add(mm.ExchangeField())
heff_cpu.add(mm.ZeemanField(H_ext))
thermal_cpu = mm.ThermalField(grid, T_K=T_K, dt=dt, seed=42)
heun_cpu = mm.HeunIntegrator(dt)

ts_cpu, mxs_cpu = [], []

t0 = time.perf_counter()
for step in range(n_steps):
    heun_cpu.step(m0, mat, heff_cpu, thermal_cpu)
    if step % log_every == 0:
        mx, *_ = mm.mean_magnetization(m0)
        ts_cpu.append(step * dt * 1e9)
        mxs_cpu.append(mx)
wall_cpu = time.perf_counter() - t0
print(f'CPU Heun (T=300K) : {wall_cpu:.2f} s  ({wall_cpu*1000/n_steps:.3f} ms/step)')

# %% [markdown]
# ## 5. Performance Summary

# %%
print()
print('=== Performance (ms/step) ===')
print(f'CPU HeunIntegrator     : {wall_cpu*1000/n_steps:.3f} ms/step')
print(f'GPU HeunIntegratorGPU  : {wall_gpu_heun*1000/n_steps:.3f} ms/step   ({wall_cpu/wall_gpu_heun:.1f}x speedup)')
print(f'GPU RK4IntegratorGPU   : {wall_gpu_rk4*1000/n_steps:.3f} ms/step   ({wall_cpu/wall_gpu_rk4:.1f}x speedup vs CPU Heun)')

# %% [markdown]
# ## 6. Switching Dynamics Comparison

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))

ax.plot(ts_rk4, mxs_rk4, color='steelblue', lw=2.0, zorder=5,
        label='T = 0 K  — GPU RK4IntegratorGPU')
ax.plot(ts_gpu, mxs_gpu, color='crimson', lw=1.2, alpha=0.9,
        label=f'T = {T_K:.0f} K — GPU HeunIntegratorGPU (seed=42)')
ax.plot(ts_cpu, mxs_cpu, color='darkorange', lw=1.0, alpha=0.7, ls='--',
        label=f'T = {T_K:.0f} K — CPU HeunIntegrator (seed=42)')
ax.axhline(-0.9862, color='k', ls=':', lw=0.8, alpha=0.5, label='µMAG ref −0.9862')

ax.set_xlabel('t  (ns)')
ax.set_ylabel('⟨mx⟩')
ax.set_title(f'SP#4 Field A — GPU HeunIntegratorGPU at T = {T_K:.0f} K'
             f'  ({wall_gpu_heun/wall_cpu:.2f}× wall time vs CPU)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig('sp4_thermal_gpu.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Multiple Thermal Realizations on GPU

# %%
seeds = [42, 43, 44, 45]
colors = ['crimson', 'tomato', 'salmon', 'lightsalmon']

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(ts_rk4, mxs_rk4, 'steelblue', lw=2.0, zorder=5, label='T = 0 K (GPU RK4)')

for seed, col in zip(seeds, colors):
    m0 = make_m0()
    ig = mm.HeunIntegratorGPU(grid, dt, seed=seed)
    ig.upload(m0)
    ts_s, mxs_s = [], []
    for step in range(n_steps):
        ig.step(mat, demag, exch, zeeman, T_K)
        if step % log_every == 0:
            m_tmp = make_m0()
            ig.download(m_tmp)
            ts_s.append(step * dt * 1e9)
            mxs_s.append(mm.mean_magnetization(m_tmp)[0])
    ax.plot(ts_s, mxs_s, color=col, lw=0.9, alpha=0.75,
            label=f'T=300K seed={seed}')
    print(f'seed {seed}: <mx> final = {mxs_s[-1]:.4f}')

ax.axhline(-0.9862, color='k', ls=':', lw=0.8, alpha=0.4)
ax.set_xlabel('t  (ns)')
ax.set_ylabel('⟨mx⟩')
ax.set_title(f'SP#4 Field A — 4 GPU SLLG realizations at T = {T_K:.0f} K')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig('sp4_thermal_gpu_realizations.png', dpi=150, bbox_inches='tight')
plt.show()
