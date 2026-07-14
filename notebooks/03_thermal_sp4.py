# %% [markdown]
# # SP#4 at Finite Temperature (T = 300 K)
# **Stochastic LLG — Heun integrator + ThermalField**
#
# Compares deterministic (T = 0 K, RK45) and stochastic (T = 300 K, Heun)
# SP#4 switching dynamics. At finite temperature, thermal fluctuations
# add noise to the switching trajectory.
#
# **Important**: RK45 (adaptive dt) CANNOT be used with ThermalField.
# The noise amplitude σ ∝ 1/√Δt — changing Δt changes the physics.
# Heun (fixed Δt) is the correct integrator for SLLG.

# %%
import sys, time
import os
from pathlib import Path

def _add_micromag_to_path():
    """Locate the micromag module + its DLLs in a release package
    (../runtime-dll + ../<variant>/python for GPU, ../python for CPU) or a source
    build (../build/<preset>/python + system CUDA). Works from any working dir."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"
    if rtd.is_dir():
        os.add_dll_directory(str(rtd))
        for _v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            _py = root / _v / "python"
            if list(_py.glob("_micromag*.pyd")):
                sys.path.insert(0, str(_py)); return
    if list((root / "python").glob("_micromag*.pyd")):
        os.add_dll_directory(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _cuda = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _p in ("windows-msvc-cuda", "windows-msvc"):
        _py = root / "build" / _p / "python"
        if _py.is_dir():
            sys.path.insert(0, str(_py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()
import micromag as mm
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Common Setup

# %%
grid = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
mat  = mm.Material.permalloy()
H_ext = mm.Vec3(-24.6e3, 4.3e3, 0.0)

def make_m0(grid):
    m = mm.VectorField3D(grid)
    m.set_uniform(mm.Vec3(1.0, 0.1, 0.0))
    m.normalize()
    return m

print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.size} cells")
print(f"H_ext = ({H_ext.x/1e3:.1f}, {H_ext.y/1e3:.1f}) kA/m")

# %% [markdown]
# ## 2. T = 0 K Reference Run (RK45)

# %%
m0 = make_m0(grid)
heff0 = mm.EffectiveFieldSum()
heff0.add(mm.DemagField(grid))
heff0.add(mm.ExchangeField())
heff0.add(mm.ZeemanField(H_ext))

integ_rk = mm.RK45Integrator()
t_end = 5e-10   # 0.5 ns (captures the switching event)

ts_rk, mxs_rk = [], []
t = 0.0
t0_w = time.perf_counter()
while t < t_end:
    dt = integ_rk.step(m0, mat, heff0)
    t += dt
    mxs_rk.append(mm.mean_magnetization(m0)[0])
    ts_rk.append(t * 1e9)

print(f"T=0 K (RK45):  {len(ts_rk)} steps,  {time.perf_counter()-t0_w:.1f} s")
print(f"  Final <mx> = {mxs_rk[-1]:.4f}")

# %% [markdown]
# ## 3. T = 300 K Stochastic Run (Heun, fixed Δt)

# %%
dt_heun = 1e-13   # 0.1 ps — fixed step for Heun/SLLG

m_th = make_m0(grid)
heff_th = mm.EffectiveFieldSum()
heff_th.add(mm.DemagField(grid))
heff_th.add(mm.ExchangeField())
heff_th.add(mm.ZeemanField(H_ext))

# ThermalField generates Langevin noise with σ ∝ √(αk_BT / μ₀Ms γ₀ V Δt)
thermal = mm.ThermalField(grid, T_K=300.0, dt=dt_heun, seed=42)

integ_heun = mm.HeunIntegrator(dt_heun)
n_steps = int(t_end / dt_heun)

ts_th, mxs_th = [], []
t0_w = time.perf_counter()
for step in range(n_steps):
    integ_heun.step(m_th, mat, heff_th, thermal)
    if step % 100 == 0:   # record every 100 steps (every 10 ps)
        mxs_th.append(mm.mean_magnetization(m_th)[0])
        ts_th.append(step * dt_heun * 1e9)

print(f"T=300K (Heun): {n_steps} steps,  {time.perf_counter()-t0_w:.1f} s")
print(f"  Final <mx> = {mxs_th[-1]:.4f}")

# %% [markdown]
# ## 4. Comparison Plot

# %%
fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(ts_rk, mxs_rk, color='steelblue', lw=1.5, label='T = 0 K  (RK45)')
ax.plot(ts_th, mxs_th, color='crimson',   lw=1.0, alpha=0.8, label='T = 300 K  (Heun)')
ax.axhline(-0.9862, color='k', ls='--', lw=0.8, alpha=0.5, label='µMAG ref −0.9862')
ax.axhline(0, color='gray', ls=':', lw=0.6)
ax.set_xlabel('t  (ns)')
ax.set_ylabel('⟨mx⟩')
ax.set_title('SP#4 Field A:  T = 0 K vs T = 300 K')
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig('sp4_thermal_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Multiple Thermal Realizations (seeds 42–44)
# Due to stochastic noise, each run has a slightly different trajectory.

# %%
seeds = [42, 43, 44]
colors = ['crimson', 'tomato', 'salmon']
ts_multi, mxs_multi = [], []

fig, ax = plt.subplots(figsize=(9, 4))

# T=0 reference
ax.plot(ts_rk, mxs_rk, 'steelblue', lw=2, label='T = 0 K', zorder=5)

for seed, col in zip(seeds, colors):
    m_s = make_m0(grid)
    heff_s = mm.EffectiveFieldSum()
    heff_s.add(mm.DemagField(grid))
    heff_s.add(mm.ExchangeField())
    heff_s.add(mm.ZeemanField(H_ext))
    tf_s = mm.ThermalField(grid, 300.0, dt_heun, seed)
    ig_s = mm.HeunIntegrator(dt_heun)

    ts_s, mxs_s = [], []
    t0_w = time.perf_counter()
    for step in range(n_steps):
        ig_s.step(m_s, mat, heff_s, tf_s)
        if step % 100 == 0:
            mxs_s.append(mm.mean_magnetization(m_s)[0])
            ts_s.append(step * dt_heun * 1e9)

    ax.plot(ts_s, mxs_s, color=col, lw=0.9, alpha=0.7, label=f'T=300K seed={seed}')
    print(f"seed {seed}: <mx> final = {mxs_s[-1]:.4f}  ({time.perf_counter()-t0_w:.0f} s)")

ax.axhline(-0.9862, color='k', ls='--', lw=0.8, alpha=0.4)
ax.set_xlabel('t  (ns)')
ax.set_ylabel('⟨mx⟩')
ax.set_title('SP#4 Field A:  3 stochastic realizations at T = 300 K')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig('sp4_thermal_realizations.png', dpi=150, bbox_inches='tight')
plt.show()
