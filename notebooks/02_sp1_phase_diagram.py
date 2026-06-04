# %% [markdown]
# # µMAG Standard Problem #1 — Phase Diagram
# **Vortex nucleation critical size L_c in Permalloy squares**
#
# Finds the element size at which vortex becomes energetically favourable
# over the single-domain (S-state). Runs entirely from Python.
#
# Reference: Cowburn et al., PRL 83, 1042 (1999)

# %%
import sys, time
sys.path.insert(0, '../build/windows-msvc/python')
import _micromag as mm
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Pre-computed Phase Diagram (from sp1_phase.exe)
# Quick look at the already-measured data before running any live simulations.

# %%
# Results from apps/sp1_phase.cpp  (L×L×10 nm, 5 nm cells, α=0.5)
data = {
    'L_nm':      [ 60,   80,  100,  110,  120,  130,  140,  160,  180,  200,  250,  300],
    'E_unif_aJ': [1.83, 2.60, 3.36, 3.74, 4.13, 4.50, 4.88, 5.63, 6.37, 7.11, 8.90, 10.66],
    'E_vort_aJ': [2.86, 3.30, 3.69, 3.87, 4.04, 4.20, 4.36, 4.66, 4.94, 5.20, 5.82,  6.37],
}
L    = np.array(data['L_nm'])
Eu   = np.array(data['E_unif_aJ'])
Ev   = np.array(data['E_vort_aJ'])
dE   = (Ev - Eu) / Eu * 100    # %

# Find crossover by linear interpolation
idx = np.where(np.diff(np.sign(dE)))[0][0]
frac = -dE[idx] / (dE[idx+1] - dE[idx])
Lc   = L[idx] + frac * (L[idx+1] - L[idx])
print(f"Critical size from pre-computed data:  L_c ≈ {Lc:.1f} nm")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: energies vs L
axes[0].plot(L, Eu, 'o-', color='steelblue',  label='E  (S-state init)', lw=1.5)
axes[0].plot(L, Ev, 's-', color='darkorange', label='E  (vortex init)',  lw=1.5)
axes[0].axvline(Lc, color='gray', ls='--', lw=1, label=f'L_c = {Lc:.0f} nm')
axes[0].set_xlabel('L (nm)')
axes[0].set_ylabel('E  (aJ = 10⁻¹⁸ J)')
axes[0].set_title('Total energy vs element size (t = 10 nm)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.25)

# Right: ΔE/E_uniform
axes[1].bar(L, dE, width=8, color=['steelblue' if d >= 0 else 'darkorange' for d in dE])
axes[1].axhline(0, color='k', lw=1)
axes[1].axvline(Lc, color='gray', ls='--', lw=1, label=f'L_c = {Lc:.0f} nm')
axes[1].set_xlabel('L (nm)')
axes[1].set_ylabel('ΔE / E_uniform  (%)')
axes[1].set_title('(E_vortex − E_uniform) / E_uniform')
axes[1].legend()
axes[1].grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig('sp1_phase_diagram.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 2. Thickness Dependence — Pre-computed Results
# From `apps/sp1_thickness.cpp`: L_c vs element thickness t.

# %%
t_nm  = np.array([ 5.0,   10.0,   20.0,   40.0])
Lc_nm = np.array([250.7, 116.0,   63.2,   50.0])
lex   = 5.686  # nm, exchange length for Permalloy (Ms=800 kA/m, A=13 pJ/m)

# Power-law fit on log-log scale
log_t  = np.log(t_nm)
log_Lc = np.log(Lc_nm)
beta, log_a = np.polyfit(log_t, log_Lc, 1)
a = np.exp(log_a)
print(f"Power-law fit:  L_c = {a:.0f} × t^({beta:.3f}) nm")

t_fit  = np.linspace(4, 45, 200)
Lc_fit = a * t_fit**beta

fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(t_nm, Lc_nm, 'o', markersize=8, color='steelblue', label='Simulation')
ax.loglog(t_fit, Lc_fit, '--', color='steelblue', alpha=0.6,
          label=f'Fit: L_c = {a:.0f}·t^{beta:.2f}')
ax.set_xlabel('Thickness  t  (nm)')
ax.set_ylabel('Critical size  L_c  (nm)')
ax.set_title('Vortex nucleation: L_c vs thickness\n(Permalloy squares, Exchange + Demag)')
ax.legend()
ax.grid(True, which='both', alpha=0.2)
# Annotate data points
for ti, lci in zip(t_nm, Lc_nm):
    ax.annotate(f' {ti:.0f} nm', (ti, lci), fontsize=9)
plt.tight_layout()
plt.savefig('sp1_thickness_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPhysical interpretation:")
print(f"  β = {beta:.3f} (negative → L_c decreases with thickness)")
print(f"  Thicker elements: vortex core distributed over more volume → cheaper")
print(f"  Cowburn (1999) experiment: β ≈ +0.33–0.5 (different geometry/conditions)")

# %% [markdown]
# ## 3. Live Simulation: Compare Vortex vs S-state for L = 120 nm
# Run one case live to show the Python workflow.

# %%
def relax_to_energy(grid, mat, use_vortex: bool, t_max=3e-9) -> tuple:
    """Relax magnetization and return (energy_J, mean_m_tuple)."""
    m = mm.VectorField3D(grid)
    if use_vortex:
        cx = grid.nx * grid.dx / 2
        cy = grid.ny * grid.dy / 2
        m.set_vortex(cx, cy, 5e-9)
    else:
        m.set_uniform(mm.Vec3(1.0, 0.05, 0.0))
    m.normalize()

    heff = mm.EffectiveFieldSum()
    heff.add(mm.ExchangeField())
    heff.add(mm.DemagField(grid))

    integ = mm.RK45Integrator()
    t = 0.0
    while t < t_max:
        dt = integ.step(m, mat, heff)
        t += dt

    E = heff.total_energy(m, mat)
    avg = mm.mean_magnetization(m)
    return E, avg, mm.to_numpy(m)

# %%
# 120×120×10 nm — right at the vortex/S-state boundary
L_nm   = 120
d      = 5e-9
n_cell = int(round(L_nm * 1e-9 / d))
nz     = 2     # 10 nm thick

grid_test = mm.StructuredGrid(n_cell, n_cell, nz, d, d, d)
mat_test  = mm.Material.permalloy(); mat_test.alpha = 0.5

print(f"Running L = {L_nm} nm ({n_cell}×{n_cell}×{nz} = {n_cell*n_cell*nz} cells) ...")
t0 = time.perf_counter()
E_u, avg_u, m_u_np = relax_to_energy(grid_test, mat_test, use_vortex=False)
E_v, avg_v, m_v_np = relax_to_energy(grid_test, mat_test, use_vortex=True)
wall = time.perf_counter() - t0

print(f"  S-state: E = {E_u*1e18:.4f} aJ,  |<m>| = {np.linalg.norm(avg_u):.3f}")
print(f"  Vortex:  E = {E_v*1e18:.4f} aJ,  |<m>| = {np.linalg.norm(avg_v):.3f}")
dE_pct = (E_v - E_u) / abs(E_u) * 100
gs = "Vortex" if E_v < E_u else "S-state"
print(f"  ΔE/E_u = {dE_pct:+.2f}%  →  Ground state: {gs}")
print(f"  wall = {wall:.1f} s")

# %% [markdown]
# ### Magnetization maps side-by-side

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
titles = ['S-state init (uniform +x)', 'Vortex init']
arrays = [m_u_np, m_v_np]

for ax, arr, title in zip(axes, arrays, titles):
    mx = arr[0, :, :, 0]
    my = arr[0, :, :, 1]
    s  = 2
    X  = np.arange(n_cell) * d * 1e9
    Y  = np.arange(n_cell) * d * 1e9
    speed = np.sqrt(mx**2 + my**2)
    im = ax.imshow(mx, origin='lower', cmap='RdBu_r',
                   extent=[0, X[-1], 0, Y[-1]], vmin=-1, vmax=1)
    ax.quiver(X[::s], Y[::s], mx[::s, ::s], my[::s, ::s],
              scale=15, width=0.006, color='k', alpha=0.6)
    ax.set_title(f'{title}\nE = {(E_u if "S" in title else E_v)*1e18:.3f} aJ')
    ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    plt.colorbar(im, ax=ax, label='mx')

plt.suptitle(f'L = {L_nm} nm × {L_nm} nm × 10 nm Permalloy', y=1.02)
plt.tight_layout()
plt.savefig('sp1_states.png', dpi=150, bbox_inches='tight')
plt.show()
