# %% [markdown]
# # µMAG Standard Problem #4 — Field A
# **Permalloy 500×125×3 nm, RK45 adaptive integrator**
#
# Reference: Miltat et al. (2002); µMAG expected `<mx> = −0.9862` at equilibrium.
#
# This notebook runs the full switching simulation from Python using the
# Claude-SpinDynamics C++ backend via pybind11 bindings.

# %%
import os, sys, time
from pathlib import Path

def _add_micromag_to_path():
    """Find the micromag module + its DLLs in either a downloaded release package
    (../python for the CPU package, or ../runtime-dll + ../<variant>/python for the
    GPU package) or a source build (../build/<preset>/python)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # numpy MKL + module OpenMP
    def _adddll(_d):                      # add_dll_directory is Windows-only
        if hasattr(os, "add_dll_directory") and os.path.isdir(_d):
            os.add_dll_directory(_d)
    def _hasmod(_p):
        _pat = "_micromag*.pyd" if sys.platform == "win32" else "_micromag*.so"
        return bool(list(_p.glob(_pat)))
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"                          # 1) GPU release package
    if rtd.is_dir():
        _adddll(str(rtd))
        for v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            py = root / v / "python"
            if _hasmod(py):
                sys.path.insert(0, str(py)); return
    if _hasmod(root / "python"):                        # 2) CPU release package
        _adddll(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _adddll(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
    for preset in ("windows-msvc-cuda", "windows-msvc", "linux-gcc-cuda", "linux-gcc"):
        py = root / "build" / preset / "python"
        if _hasmod(py):
            sys.path.insert(0, str(py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()
import micromag as mm
import numpy as np
import matplotlib.pyplot as plt

print(f"_micromag loaded. Available: DemagField, RK45Integrator, to_numpy, ...")

# %% [markdown]
# ## 1. Grid and Material

# %%
# SP#4: 500 nm × 125 nm × 3 nm Permalloy, 2.5 nm cells
grid = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
mat  = mm.Material.permalloy()

print(f"Grid  : {grid.nx}x{grid.ny}x{grid.nz} = {grid.size} cells")
print(f"Size  : {grid.nx*grid.dx*1e9:.0f}x{grid.ny*grid.dy*1e9:.0f}x{grid.nz*grid.dz*1e9:.0f} nm")
print(f"Ms    = {mat.Ms/1e3:.0f} kA/m   A = {mat.A_exchange*1e12:.0f} pJ/m   alpha = {mat.alpha}")

# %% [markdown]
# ## 2. Initial State
# Saturated along +x with a small y-tilt (breaks symmetry, allows switching).

# %%
m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(1.0, 0.1, 0.0))
m.normalize()
print("Initial <m> =", mm.mean_magnetization(m))

# %% [markdown]
# ## 3. Effective Field: Exchange + Demag + Zeeman
# Field A: θ = 170° from +x → H = (−24.6, +4.3, 0) kA/m

# %%
H_ext = mm.Vec3(-24.6e3, 4.3e3, 0.0)   # A/m

heff = mm.EffectiveFieldSum()
heff.add(mm.DemagField(grid))
heff.add(mm.ExchangeField())
heff.add(mm.ZeemanField(H_ext))

print(f"H_ext = ({H_ext.x/1e3:.1f}, {H_ext.y/1e3:.1f}, {H_ext.z:.0f}) kA/m  (theta = 170deg)")

# %% [markdown]
# ## 4. RK45 Simulation to 1 ns
# RK45 uses adaptive time-stepping: large steps when m moves slowly,
# small steps near the switching event.

# %%
integ = mm.RK45Integrator()
t_end = 1.0e-9    # 1 ns

ts, mxs, mys, mzs = [], [], [], []
t = 0.0
t_wall = time.perf_counter()

while t < t_end:
    dt = integ.step(m, mat, heff)
    t += dt
    mx, my, mz = mm.mean_magnetization(m)
    ts.append(t * 1e9)           # store in ns
    mxs.append(mx); mys.append(my); mzs.append(mz)

wall = time.perf_counter() - t_wall
ts  = np.array(ts)
mxs = np.array(mxs)
mys = np.array(mys)

print(f"Steps : {len(ts)}")
print(f"t_sim : {ts[-1]:.3f} ns    wall : {wall:.1f} s")
print(f"Final <m> = ({mxs[-1]:.5f}, {mys[-1]:.5f}, {mzs[-1]:.5f})")
print(f"uMAG ref  = (-0.98620,  0.00263,  0.00000)")
print(f"Error     = {abs(mxs[-1]+0.9862)*100:.3f}%")

# %% [markdown]
# ## 5. Switching Curve

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ts, mxs, lw=1.5, label='⟨mx⟩')
ax.plot(ts, mys, lw=1.5, label='⟨my⟩', alpha=0.8)
ax.axhline(-0.9862, color='k', ls='--', lw=0.9, alpha=0.5, label='µMAG ref −0.9862')
ax.set_xlabel('t  (ns)')
ax.set_ylabel('⟨m⟩')
ax.set_title('SP#4 Field A — Permalloy 500×125×3 nm  (RK45)')
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_ylim(-1.15, 1.15)
plt.tight_layout()
plt.savefig('sp4_switching.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Magnetization Map (final state)

# %%
m_np = mm.to_numpy(m)           # shape (nz=1, ny=50, nx=200, 3)
mx2d = m_np[0, :, :, 0]        # (50, 200)
my2d = m_np[0, :, :, 1]
mag  = np.sqrt(mx2d**2 + my2d**2)

# Coordinates in nm
X = np.arange(grid.nx) * grid.dx * 1e9
Y = np.arange(grid.ny) * grid.dy * 1e9

fig, axes = plt.subplots(1, 2, figsize=(13, 3))

# Left: mx colour map
im = axes[0].imshow(mx2d, origin='lower', cmap='RdBu_r',
                    extent=[0, X[-1], 0, Y[-1]], vmin=-1, vmax=1)
plt.colorbar(im, ax=axes[0], label='mx')
axes[0].set_title(f'mx   (t = {ts[-1]:.2f} ns)')
axes[0].set_xlabel('x (nm)'); axes[0].set_ylabel('y (nm)')

# Right: quiver (subsampled)
s = 5
axes[1].quiver(X[::s], Y[::s], mx2d[::s, ::s], my2d[::s, ::s],
               mag[::s, ::s], cmap='plasma', scale=20, width=0.004)
axes[1].set_title('Magnetization vectors (subsampled ×5)')
axes[1].set_xlabel('x (nm)'); axes[1].set_ylabel('y (nm)')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('sp4_magnetization.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Switching Time Detection

# %%
# Find the time when <mx> crosses 0
cross_idx = np.where(np.diff(np.sign(mxs)))[0]
if len(cross_idx):
    t_cross = ts[cross_idx[0]]
    print(f"<mx> zero-crossing (rough switching time): {t_cross:.3f} ns")
    print(f"uMAG reference t_switch ~ 0.175 ns")
else:
    print("No zero-crossing found in 1 ns")

print(f"\nSummary:")
print(f"  Final <mx> = {mxs[-1]:.5f}  (ref -0.9862,  err {abs(mxs[-1]+0.9862)*100:.3f}%)")
print(f"  Total steps = {len(ts)}")
print(f"  Avg dt      = {ts[-1]/len(ts)*1e3:.3f} ps")
