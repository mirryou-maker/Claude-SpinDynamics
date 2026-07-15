# %% [markdown]
# # Slonczewski Spin-Transfer Torque Switching
# **Replication of mumax3 "Slonczewski STT" example**
#
# mumax3 original (slonczewski.mx3):
#   Geometry: ellipse 160×80 nm (we use rectangle — no shape API)
#   SetGridSize(40, 20, 1); SetCellSize(4e-9, 4e-9, 4e-9)
#   Msat=860e3; Aex=13e-12; Alpha=0.01
#   FixedLayer = uniform(1,0,0); Pol=0.5; Lambda=1
#   J = 3e12 A/m²  →  switch from -x to +x
#
# We replicate the physics exactly.  Rectangle instead of ellipse.

# %%
import sys, time
import matplotlib
matplotlib.use('Agg')
import os
from pathlib import Path

def _add_micromag_to_path():
    """Locate the micromag module + its DLLs in a release package
    (../runtime-dll + ../<variant>/python for GPU, ../python for CPU) or a source
    build (../build/<preset>/python + system CUDA). Works from any working dir."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    def _adddll(_d):                      # add_dll_directory is Windows-only
        if hasattr(os, "add_dll_directory") and os.path.isdir(_d):
            os.add_dll_directory(_d)
    def _hasmod(_p):
        _pat = "_micromag*.pyd" if sys.platform == "win32" else "_micromag*.so"
        return bool(list(_p.glob(_pat)))
    root = Path(__file__).resolve().parent.parent
    rtd = root / "runtime-dll"
    if rtd.is_dir():
        _adddll(str(rtd))
        for _v in ("cuFFT-f64", "cuFFT-f32", "VkFFT-f64", "VkFFT-f32"):
            _py = root / _v / "python"
            if _hasmod(_py):
                sys.path.insert(0, str(_py)); return
    if _hasmod(root / "python"):
        _adddll(str(root / "python"))
        sys.path.insert(0, str(root / "python")); return
    _adddll(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
    for _p in ("windows-msvc-cuda", "windows-msvc", "linux-gcc-cuda", "linux-gcc"):
        _py = root / "build" / _p / "python"
        if _hasmod(_py):
            sys.path.insert(0, str(_py)); return
    raise RuntimeError("micromag module not found (release package or source build).")

_add_micromag_to_path()
import micromag as mm
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Setup — 160 × 80 × 4 nm Free Layer

# %%
# mumax3 geometry (rectangle, not ellipse — we lack shape API)
grid = mm.StructuredGrid(40, 20, 1, 4e-9, 4e-9, 4e-9)

mat       = mm.Material.permalloy()
mat.Ms    = 860e3    # A/m  (mumax3: Msat=860e3)
mat.A_exchange = 13e-12
mat.alpha = 0.01     # low damping → precessional switching (mumax3: Alpha=0.01)

# STT parameters (mumax3: Pol=0.5, Lambda=1, FixedLayer=(1,0,0))
# Sign convention: J < 0 = electrons from fixed to free layer
#   -> a_J < 0 -> anti-damping torque drives -x to +x switching
J_c = -3e12         # A/m²  negative: switches antiparallel(-x) -> parallel(+x)
P   = 0.5           # spin polarisation
d   = 4e-9          # free layer thickness [m]
p   = mm.Vec3(1, 0, 0)   # fixed layer direction (+x)

print(f"Grid  : {grid.nx}x{grid.ny}x{grid.nz} = {grid.size} cells")
print(f"Size  : {grid.nx*grid.dx*1e9:.0f}x{grid.ny*grid.dy*1e9:.0f}x{grid.nz*grid.dz*1e9:.0f} nm")
print(f"alpha={mat.alpha}, Ms={mat.Ms/1e3:.0f} kA/m")

# Compute a_J to verify it's strong enough for switching
import math
gamma = 1.760859630e11
hbar  = 1.054571817e-34
e_c   = 1.602176634e-19
a_J = gamma * hbar * J_c * P / (2 * e_c * mat.Ms * d)
mu0_Ms = 4*math.pi*1e-7 * mat.Ms
precession_freq = gamma * mu0_Ms
print(f"a_J   = {a_J:.3e} rad/s  ({a_J/precession_freq*100:.1f}% of precession frequency)")

# %% [markdown]
# ## 2. Build Fields + STT

# %%
print("Constructing DemagField...", end=" ", flush=True)
t0 = time.perf_counter()
demag  = mm.DemagField(grid)
print(f"{(time.perf_counter()-t0)*1e3:.0f} ms")

exch   = mm.ExchangeField()
zeeman = mm.ZeemanField(mm.Vec3(0, 0, 0))   # no applied field
stt    = mm.SlonczewskiSTT(J_c, P, d, p)

heff = mm.EffectiveFieldSum()
heff.add(exch); heff.add(demag); heff.add(zeeman)

stt_sum = mm.SpinTorqueSum()
stt_sum.add(stt)

print(f"a_J(Ms) = {stt.a_J(mat.Ms):.3e} rad/s  (from SlonczewskiSTT)")

# %% [markdown]
# ## 3. STT Switching Dynamics (1 ns at J = 3×10¹² A/m²)

# %%
# Initial state: antiparallel to fixed layer (−x) with tiny tilt for symmetry breaking
m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(-1.0, 0.05, 0.0)); m.normalize()

dt      = 5e-14    # 50 fs fixed step (stable for low-α Permalloy at 4 nm)
n_steps = 20000    # 1 ns
log_every = 100    # every 5 ps

integ = mm.RK4Integrator(dt)
ts, mxs, mys, mzs = [], [], [], []

t0 = time.perf_counter()
for step in range(n_steps):
    integ.step(m, mat, heff, stt_sum)
    if step % log_every == 0:
        mx, my, mz = mm.mean_magnetization(m)
        ts.append(step * dt * 1e9)
        mxs.append(mx); mys.append(my); mzs.append(mz)

wall = time.perf_counter() - t0
mx_f, my_f, mz_f = mm.mean_magnetization(m)

print(f"Simulation: {n_steps} steps x dt={dt:.0e} s = {n_steps*dt*1e9:.1f} ns")
print(f"Wall time : {wall:.1f} s  ({wall*1000/n_steps:.2f} ms/step)")
print(f"Final <m> : ({mx_f:.4f}, {my_f:.4f}, {mz_f:.4f})")
switched = mx_f > 0.5
print(f"Switching : {'YES (switched!)' if switched else 'NO (increase |J|)'}")

# %% [markdown]
# ## 4. Switching Trajectory Plot

# %%
ts_arr  = np.array(ts)
mxs_arr = np.array(mxs)
mys_arr = np.array(mys)
mzs_arr = np.array(mzs)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].plot(ts_arr, mxs_arr, lw=1.5, color='steelblue',   label='⟨mx⟩')
axes[0].plot(ts_arr, mys_arr, lw=1.2, color='darkorange',  label='⟨my⟩', alpha=0.8)
axes[0].plot(ts_arr, mzs_arr, lw=1.0, color='forestgreen', label='⟨mz⟩', alpha=0.6)
axes[0].axhline( 1, color='k', ls='--', lw=0.7, alpha=0.4)
axes[0].axhline(-1, color='k', ls='--', lw=0.7, alpha=0.4)
axes[0].axhline( 0, color='k', ls=':',  lw=0.5)
axes[0].set_ylabel('⟨m⟩')
axes[0].set_title(f'Slonczewski STT Switching — 160×80×4 nm Permalloy\n'
                   f'J={J_c:.0e} A/m²  P={P}  α={mat.alpha}  Fixed layer: +x')
axes[0].legend(); axes[0].grid(True, alpha=0.2)
axes[0].set_ylim(-1.2, 1.2)

# Phase space: mx vs my
axes[1].plot(mxs_arr, mys_arr, lw=1.0, color='purple', alpha=0.8)
axes[1].plot(mxs_arr[0], mys_arr[0], 'go', markersize=8, label='start (−x)')
axes[1].plot(mxs_arr[-1], mys_arr[-1], 'r*', markersize=10, label='end')
axes[1].set_xlabel('⟨mx⟩'); axes[1].set_ylabel('⟨my⟩')
axes[1].set_title('Phase space trajectory (mx − my)')
axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.2)
axes[1].legend()

plt.tight_layout()
plt.savefig('stt_switching.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stt_switching.png")

# %% [markdown]
# ## 5. Critical Current Sweep

# %%
print("Sweeping current density to find switching threshold...")
# Negative J = antiparallel->parallel switching; test -0.1 to -3 x 10^12
J_values = -np.arange(0.1e12, 3.5e12, 0.2e12)
n_steps_sweep = 10000  # 0.5 ns per J value

switch_results = []
for J_test in J_values:
    stt_test = mm.SlonczewskiSTT(J_test, P, d, p)
    stt_test_sum = mm.SpinTorqueSum(); stt_test_sum.add(stt_test)

    m_test = mm.VectorField3D(grid)
    m_test.set_uniform(mm.Vec3(-1.0, 0.05, 0.0)); m_test.normalize()

    integ_s = mm.RK4Integrator(dt)
    for _ in range(n_steps_sweep):
        integ_s.step(m_test, mat, heff, stt_test_sum)

    mx_end = mm.mean_magnetization(m_test)[0]
    switched_s = mx_end > 0.5
    switch_results.append((J_test/1e12, mx_end, switched_s))
    status = "switched" if switched_s else "no switch"
    print(f"  J={J_test/1e12:.2f}e12 A/m2  <mx>={mx_end:+.3f}  {status}")

# Critical current
J_critical = None
for (J_t, mx_e, sw) in switch_results:
    if sw:
        J_critical = J_t
        break

fig, ax = plt.subplots(figsize=(7, 4))
J_arr = [r[0] for r in switch_results]
mx_arr_s = [r[1] for r in switch_results]
colors_s = ['forestgreen' if r[2] else 'crimson' for r in switch_results]
bars = ax.bar(J_arr, mx_arr_s, width=0.4, color=colors_s, alpha=0.8)
ax.axhline(0, color='k', lw=0.8)
if J_critical:
    ax.axvline(J_critical, color='k', ls='--', lw=1.2,
               label=f'J_c ≈ {J_critical:.1f}×10¹² A/m²')
ax.set_xlabel('Current density  J  (×10¹² A/m²)')
ax.set_ylabel('Final ⟨mx⟩  (after 0.5 ns)')
ax.set_title('Critical switching current\n(green = switched, red = not switched)')
ax.legend(); ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('stt_critical_current.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stt_critical_current.png")

# %% [markdown]
# ## 6. Save CSV

# %%
np.savetxt('stt_switching_data.csv',
           np.column_stack([ts_arr, mxs_arr, mys_arr, mzs_arr]),
           header='t_ns,mx,my,mz', delimiter=',', comments='')
np.savetxt('stt_critical_current.csv',
           np.array(switch_results),
           header='J_1e12_Am2,mx_final,switched', delimiter=',', comments='')
print("Saved: stt_switching_data.csv, stt_critical_current.csv")
