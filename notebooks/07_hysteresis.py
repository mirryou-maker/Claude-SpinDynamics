# %% [markdown]
# # µMAG Hysteresis Loop
# **Replication of mumax3 "hysteresis" example**
#
# mumax3 original:
#   SetGridSize(128, 32, 1); SetCellSize(4e-9, 4e-9, 4e-9)
#   Msat=800e3; Aex=13e-12
#   Sweep B from +100 mT to -100 mT using Minimize() at each step.
#
# Our equivalent: RK45 with α=0.5 (overdamped) plays the role of
# mumax's conjugate-gradient Minimize().

# %%
import sys, time
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, '../build/windows-msvc/python')
import _micromag as mm
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Setup — 512 × 128 × 4 nm Permalloy strip

# %%
# Identical geometry to mumax3 hysteresis example
grid = mm.StructuredGrid(128, 32, 1, 4e-9, 4e-9, 4e-9)
mat  = mm.Material.permalloy()
mat.alpha = 0.5         # overdamped -> fast convergence (Minimize equivalent)

mu_0        = 4 * np.pi * 1e-7
mT_to_Am    = 1e-3 / mu_0

print(f"Grid : {grid.nx}x{grid.ny}x{grid.nz} = {grid.size} cells")
print(f"Size : {grid.nx*grid.dx*1e9:.0f} x {grid.ny*grid.dy*1e9:.0f} x {grid.nz*grid.dz*1e9:.0f} nm")
print(f"Mat  : Ms={mat.Ms/1e3:.0f} kA/m  A={mat.A_exchange*1e12:.0f} pJ/m  alpha={mat.alpha}")

# %% [markdown]
# ## 2. Build Fields

# %%
print("Constructing DemagField...", end=" ", flush=True)
t0 = time.perf_counter()
demag = mm.DemagField(grid)
print(f"{(time.perf_counter()-t0)*1e3:.0f} ms")

zeeman_field = mm.ZeemanField(mm.Vec3(0, 0, 0))
heff = mm.EffectiveFieldSum()
heff.add(mm.ExchangeField())
heff.add(demag)
heff.add(zeeman_field)

# %% [markdown]
# ## 3. Field Sweep: +100 mT → −100 mT

# %%
def relax(m, mat, heff, tol_Am=500.0, t_max=2e-9):
    """Relax with RK45 (high α → fast convergence)."""
    opts = mm.RK45Options()
    opts.dt_max  = 5e-12
    opts.dt_init = 5e-14
    integ = mm.RK45Integrator(opts)
    t = 0.0
    while t < t_max:
        t += integ.step(m, mat, heff)
    return integ

H_fields_mT = np.arange(100, -105, -5)   # 100 → -100 mT in 5 mT steps (41 pts)

# Initialise saturated in +x at high field, pre-relax
m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(1.0, 0.01, 0.0)); m.normalize()
zeeman_field.H_ext = mm.Vec3(100 * mT_to_Am, 0, 0)
relax(m, mat, heff)

results = []
t_start = time.perf_counter()

for H_mT in H_fields_mT:
    zeeman_field.H_ext = mm.Vec3(H_mT * mT_to_Am, 0, 0)
    relax(m, mat, heff)
    mx, my, mz = mm.mean_magnetization(m)
    E = heff.total_energy(m, mat) * 1e18   # aJ
    results.append((H_mT, mx, my, mz, E))
    print(f"  H={H_mT:+5.0f} mT  <mx>={mx:+.4f}  E={E:.3f} aJ")

wall = time.perf_counter() - t_start
print(f"\nTotal: {wall:.1f} s")

# %% [markdown]
# ## 4. Hysteresis Loop Plot

# %%
data = np.array(results)
H_mT_arr = data[:, 0]
mx_arr   = data[:, 1]
E_arr    = data[:, 4]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(H_mT_arr, mx_arr, 'o-', markersize=4, lw=1.5, color='steelblue')
axes[0].axhline(0, color='k', lw=0.6, ls=':')
axes[0].axvline(0, color='k', lw=0.6, ls=':')
axes[0].set_xlabel('Applied field  H  (mT)')
axes[0].set_ylabel('⟨mx⟩')
axes[0].set_title(f'Hysteresis Loop — 512×128×4 nm Permalloy\n'
                   f'α={mat.alpha}, mumax3 example replicated')
axes[0].grid(True, alpha=0.2)
axes[0].set_xlim(H_mT_arr[-1], H_mT_arr[0])
axes[0].set_ylim(-1.15, 1.15)

axes[1].plot(H_mT_arr, E_arr, 's-', markersize=4, lw=1.5, color='darkorange')
axes[1].set_xlabel('Applied field  H  (mT)')
axes[1].set_ylabel('Total energy  (aJ = 10⁻¹⁸ J)')
axes[1].set_title('Total energy vs field')
axes[1].grid(True, alpha=0.2)
axes[1].set_xlim(H_mT_arr[-1], H_mT_arr[0])

plt.tight_layout()
plt.savefig('hysteresis_loop.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: hysteresis_loop.png")

# %% [markdown]
# ## 5. Magnetization Map at Remanence (H=0)

# %%
# Get state at H=0 (index where H_mT=0)
idx0 = np.argmin(np.abs(H_mT_arr))
print(f"Remanence (H~0): <mx>={mx_arr[idx0]:.4f}")

# Relax at H=0 to get the state
zeeman_field.H_ext = mm.Vec3(0, 0, 0)
relax(m, mat, heff)
m_np = mm.to_numpy(m)    # (1, 32, 128, 3)
mx2d = m_np[0, :, :, 0]
my2d = m_np[0, :, :, 1]

X = np.arange(128) * 4    # nm
Y = np.arange(32) * 4     # nm

fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

im = axes[0].imshow(mx2d, origin='lower', cmap='RdBu_r',
                    extent=[0, X[-1], 0, Y[-1]], vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[0], label='mx')
axes[0].set_title('mx  at remanence (H = 0)')
axes[0].set_xlabel('x (nm)'); axes[0].set_ylabel('y (nm)')

s = 4
axes[1].quiver(X[::s], Y[::s], mx2d[::s, ::s], my2d[::s, ::s],
               np.sqrt(mx2d[::s,::s]**2 + my2d[::s,::s]**2),
               cmap='plasma', scale=18, width=0.005)
axes[1].set_title('Magnetization vectors at remanence')
axes[1].set_xlabel('x (nm)'); axes[1].set_ylabel('y (nm)')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.savefig('hysteresis_remanence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: hysteresis_remanence.png")

# %% [markdown]
# ## 6. Save CSV

# %%
import os
np.savetxt('hysteresis_data.csv',
           np.column_stack([H_mT_arr, mx_arr, data[:,2], data[:,3], E_arr]),
           header='H_mT,mx,my,mz,E_aJ', delimiter=',', comments='')
print("Saved: hysteresis_data.csv")
print(f"\nmumax3 equivalent: Minimize() -> RK45 alpha={mat.alpha} (2 ns max per field)")
