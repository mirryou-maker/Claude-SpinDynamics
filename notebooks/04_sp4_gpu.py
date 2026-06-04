# %% [markdown]
# # SP#4 with GPU RK4IntegratorGPU
# **Full-GPU LLG: Exchange + Demag + Zeeman, zero PCIe per step**
#
# Requires: build with `cmake --preset windows-msvc-cuda` (CUDA build)
# CUDA DLLs must be in PATH (see setup below).

# %%
import os, sys, time
# Windows: CUDA DLLs live in CUDA toolkit bin directory
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')

sys.path.insert(0, '../build/windows-msvc-cuda/python')
import _micromag as mm
import numpy as np
import matplotlib.pyplot as plt

print('CUDA available:', mm.cuda_available())

# %% [markdown]
# ## 1. Setup — SP#4 Field A

# %%
grid  = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
mat   = mm.Material.permalloy()
H_ext = mm.Vec3(-24.6e3, 4.3e3, 0.0)

print(f'Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.size} cells')
print(f'Ms = {mat.Ms/1e3:.0f} kA/m,  A = {mat.A_exchange*1e12:.0f} pJ/m,  alpha = {mat.alpha}')

m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(1.0, 0.1, 0.0))
m.normalize()

# %% [markdown]
# ## 2. GPU Field Objects

# %%
demag  = mm.DemagFieldGPU(grid)
exch   = mm.ExchangeFieldGPU(grid)
zeeman = mm.ZeemanFieldGPU(grid, H_ext)

print('Fields:', demag.name, '/', exch.name, '/', zeeman.name)

# %% [markdown]
# ## 3. GPU Simulation — 0.3 ns (6000 steps x 5e-14 s)

# %%
dt      = 5e-14    # 50 fs fixed step
n_steps = 6000     # 0.3 ns

integ = mm.RK4IntegratorGPU(grid, dt)
integ.upload(m)

ts, mxs, mys = [], [], []
log_every = 200    # record every 10 ps

t0 = time.perf_counter()
for step in range(n_steps):
    integ.step(mat, demag, exch, zeeman)
    if step % log_every == 0:
        integ.download(m)
        mx, my, mz = mm.mean_magnetization(m)
        ts.append(step * dt * 1e9)
        mxs.append(mx); mys.append(my)

wall = time.perf_counter() - t0
integ.download(m)
mx_f, my_f, mz_f = mm.mean_magnetization(m)

print(f'Wall time : {wall:.2f} s  ({wall*1000/n_steps:.2f} ms/step)')
print(f'Final <m> : ({mx_f:.4f}, {my_f:.4f}, {mz_f:.4f})')

# %% [markdown]
# ## 4. Switching Curve

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ts, mxs, lw=1.5, label='<mx>')
ax.plot(ts, mys, lw=1.5, label='<my>', alpha=0.8)
ax.axhline(-0.9862, color='k', ls='--', lw=0.9, alpha=0.5, label='muMAG ref -0.9862')
ax.set_xlabel('t (ns)'); ax.set_ylabel('<m>')
ax.set_title(f'SP#4 Field A - GPU RK4IntegratorGPU  ({wall:.1f} s wall)')
ax.legend(); ax.grid(True, alpha=0.25); ax.set_ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig('sp4_gpu_switching.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Final Magnetization Map (numpy)

# %%
m_np = mm.to_numpy(m)          # (1, 50, 200, 3)
mx2d = m_np[0, :, :, 0]
my2d = m_np[0, :, :, 1]

X = np.arange(grid.nx) * grid.dx * 1e9
Y = np.arange(grid.ny) * grid.dy * 1e9

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
im = axes[0].imshow(mx2d, origin='lower', cmap='RdBu_r',
                    extent=[0, X[-1], 0, Y[-1]], vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[0], label='mx')
axes[0].set_title(f'mx  (t = {n_steps*dt*1e9:.1f} ns)')
axes[0].set_xlabel('x (nm)'); axes[0].set_ylabel('y (nm)')

s = 5
axes[1].quiver(X[::s], Y[::s], mx2d[::s,::s], my2d[::s,::s],
               np.sqrt(mx2d[::s,::s]**2 + my2d[::s,::s]**2),
               cmap='plasma', scale=20, width=0.004)
axes[1].set_title('Magnetization vectors (GPU result)')
axes[1].set_xlabel('x (nm)'); axes[1].set_ylabel('y (nm)')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.savefig('sp4_gpu_magnetization.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Performance Comparison

# %%
# Quick CPU reference timing (100 steps)
heff_cpu = mm.EffectiveFieldSum()
heff_cpu.add(mm.DemagField(grid))
heff_cpu.add(mm.ExchangeField())
heff_cpu.add(mm.ZeemanField(H_ext))

m_cpu = mm.VectorField3D(grid)
m_cpu.set_uniform(mm.Vec3(1.0, 0.1, 0.0)); m_cpu.normalize()
cpu = mm.RK4Integrator(dt)

t0 = time.perf_counter()
for _ in range(100): cpu.step(m_cpu, mat, heff_cpu)
cpu_time = (time.perf_counter() - t0) / 100 * 1000   # ms/step
gpu_time = wall * 1000 / n_steps

print(f'CPU RK4:   {cpu_time:.2f} ms/step')
print(f'GPU RK4:   {gpu_time:.2f} ms/step')
print(f'Speedup:   {cpu_time/gpu_time:.1f}x')
