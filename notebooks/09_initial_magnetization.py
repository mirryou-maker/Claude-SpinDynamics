# %% [markdown]
# # Initial Magnetization States
# **Replication of mumax3 "Initial Magnetization" example**
#
# mumax3 shows: uniform, vortex, TwoDomain, RandomMag, Skyrmion, VortexWall
# We implement: uniform, vortex, two-domain (manual), random.
# Skyrmion and VortexWall require advanced initialization not yet in our code.
#
# Each state is relaxed to a local energy minimum (RK45, α=0.5).
# Energies and magnetization maps are compared.

# %%
import sys, time, math
import matplotlib
matplotlib.use('Agg')
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
import matplotlib.gridspec as gridspec

# %% [markdown]
# ## 1. Setup — 500 × 500 × 10 nm Permalloy square

# %%
grid = mm.StructuredGrid(100, 100, 2, 5e-9, 5e-9, 5e-9)
mat  = mm.Material.permalloy()
mat.alpha = 0.5    # overdamped for fast convergence

print(f"Grid  : {grid.nx}x{grid.ny}x{grid.nz} = {grid.size} cells")
print(f"Size  : {grid.nx*grid.dx*1e9:.0f}x{grid.ny*grid.dy*1e9:.0f}x{grid.nz*grid.dz*1e9:.0f} nm")

print("Constructing DemagField...", end=" ", flush=True)
t0 = time.perf_counter()
demag = mm.DemagField(grid)
print(f"{(time.perf_counter()-t0)*1e3:.0f} ms")

heff = mm.EffectiveFieldSum()
heff.add(mm.ExchangeField())
heff.add(demag)

# %% [markdown]
# ## 2. Relaxation Helper

# %%
def relax_state(m, mat, heff, label, t_max=3e-9):
    """Relax m to local minimum, return energy (J)."""
    opts = mm.RK45Options()
    opts.dt_init = 5e-14
    opts.dt_max  = 5e-12
    integ = mm.RK45Integrator(opts)
    t = 0.0
    t0 = time.perf_counter()
    while t < t_max:
        t += integ.step(m, mat, heff)
    E = heff.total_energy(m, mat)
    wall = time.perf_counter() - t0
    mx, my, mz = mm.mean_magnetization(m)
    print(f"  [{label:12s}]  E={E*1e18:.4f} aJ  |<m>|={math.sqrt(mx**2+my**2+mz**2):.3f}"
          f"  ({wall:.1f} s)")
    return E

# %% [markdown]
# ## 3. Initial States

# %%
states = {}

# --- Uniform +x (mumax: uniform(1,0,0)) ---
m_uni = mm.VectorField3D(grid)
m_uni.set_uniform(mm.Vec3(1.0, 0.01, 0.0)); m_uni.normalize()
states['Uniform +x'] = (m_uni, '#5b9bd5')

# --- Vortex CCW (mumax: vortex(1,1)) ---
m_vortex = mm.VectorField3D(grid)
cx = grid.nx * grid.dx / 2
cy = grid.ny * grid.dy / 2
m_vortex.set_vortex(cx, cy, 5e-9)
states['Vortex CCW'] = (m_vortex, '#ed7d31')

# --- Two-domain (mumax: TwoDomain(x,-x,x)) --- left +x, right -x
m_twod = mm.VectorField3D(grid)
m_np_init = mm.to_numpy(m_twod)
m_np_init[:] = 0
for iz in range(grid.nz):
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            idx = ix + grid.nx*(iy + grid.ny*iz)
            if ix < grid.nx // 2:
                m_np_init[iz, iy, ix, :] = [1, 0, 0]
            else:
                m_np_init[iz, iy, ix, :] = [-1, 0, 0]
mm.from_numpy(m_twod, m_np_init)
states['Two-domain'] = (m_twod, '#a9d18e')

# --- Random (mumax: RandomMag()) --- uniform random unit vectors
rng = np.random.default_rng(42)
m_rand = mm.VectorField3D(grid)
m_rand_np = rng.standard_normal((grid.nz, grid.ny, grid.nx, 3))
m_rand_np /= np.linalg.norm(m_rand_np, axis=-1, keepdims=True)
mm.from_numpy(m_rand, m_rand_np.astype(np.float64))
states['Random'] = (m_rand, '#ff99cc')

# %% [markdown]
# ## 4. Relax Each State

# %%
print("Relaxing initial states...")
energies = {}
for name, (m, color) in states.items():
    E = relax_state(m, mat, heff, name)
    energies[name] = E

# %% [markdown]
# ## 5. Comparison Plot

# %%
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.3)

state_names = list(states.keys())
colors_map  = [states[n][1] for n in state_names]

for col, (name, color) in enumerate(zip(state_names, colors_map)):
    m_state, _ = states[name]
    m_np = mm.to_numpy(m_state)   # (nz, ny, nx, 3)
    # Average over z layers
    mx2d = m_np[:, :, :, 0].mean(axis=0)
    my2d = m_np[:, :, :, 1].mean(axis=0)
    X = np.arange(grid.nx) * grid.dx * 1e9
    Y = np.arange(grid.ny) * grid.dy * 1e9
    E_aJ = energies[name] * 1e18

    # mx colour map
    ax_top = fig.add_subplot(gs[0, col])
    im = ax_top.imshow(mx2d, origin='lower', cmap='RdBu_r',
                       extent=[0, X[-1], 0, Y[-1]], vmin=-1, vmax=1)
    ax_top.set_title(f'{name}\nE = {E_aJ:.3f} aJ', fontsize=10)
    ax_top.set_xlabel('x (nm)', fontsize=8)
    ax_top.set_ylabel('y (nm)', fontsize=8)
    plt.colorbar(im, ax=ax_top, fraction=0.046, label='mx')

    # quiver
    ax_bot = fig.add_subplot(gs[1, col])
    s = 6
    ax_bot.quiver(X[::s], Y[::s], mx2d[::s, ::s], my2d[::s, ::s],
                  np.sqrt(mx2d[::s,::s]**2 + my2d[::s,::s]**2),
                  cmap='plasma', scale=15, width=0.006)
    ax_bot.set_title(f'Magnetization vectors', fontsize=9)
    ax_bot.set_xlabel('x (nm)', fontsize=8)
    ax_bot.set_ylabel('y (nm)', fontsize=8)
    ax_bot.set_aspect('equal')

plt.suptitle('µMAG Initial Magnetization States — 500×500×10 nm Permalloy\n'
             '(mumax3 "Initial Magnetization" example replicated)',
             fontsize=12, y=1.02)
plt.savefig('initial_magnetization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: initial_magnetization.png")

# %% [markdown]
# ## 6. Energy Comparison Bar Chart

# %%
fig, ax = plt.subplots(figsize=(8, 4))
names = list(energies.keys())
E_vals = [energies[n]*1e18 for n in names]
bars = ax.bar(names, E_vals, color=[states[n][1] for n in names],
              edgecolor='k', linewidth=0.8)
ax.set_ylabel('Total energy  (aJ = 10⁻¹⁸ J)')
ax.set_title('Energy of relaxed states — 500×500×10 nm Permalloy\n'
             '(lower = more stable)')
for bar, E in zip(bars, E_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{E:.3f}', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('initial_magnetization_energies.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: initial_magnetization_energies.png")

# %% [markdown]
# ## 7. Summary
print("\n=== Summary: Energy comparison ===")
E_sorted = sorted(energies.items(), key=lambda x: x[1])
for rank, (name, E) in enumerate(E_sorted, 1):
    print(f"  {rank}. {name:15s}  E = {E*1e18:.4f} aJ")
print(f"\nGround state: {E_sorted[0][0]}")
