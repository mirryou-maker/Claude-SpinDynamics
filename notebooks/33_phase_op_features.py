"""Notebook 33 - Phase O + P feature showcase.

Demonstrates:
  (O1) max_angle      -- convergence criterion (mumax3 MaxAngle)
  (O3) energy_table   -- per-term energy breakdown
  (O4) Table.add_column -- custom column registration
  (O2) B_eff          -- effective flux density [T]
  (P3) def_region / new_region_map -- RegionMap from geometry shapes
  (P1) FrozenIntegrator -- pinned spin dynamics
  (P2) SurfaceAnisotropyField -- PMA interface anisotropy
"""
import sys, os
_repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import numpy as np
import micromag as mm

print("=== Notebook 33: Phase O + P Features ===\n")

# ---------------------------------------------------------------------------
# Shared setup: permalloy strip 200x50x1, 10 nm cells
# ---------------------------------------------------------------------------
nx, ny, nz = 20, 10, 1
dx = 10e-9
g  = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)
mat = mm.Material.permalloy()

m = mm.uniform_mag(g, mm.Vec3(1.0, 0.0, 0.0))   # +x

exch  = mm.ExchangeField()
zee   = mm.ZeemanField(mm.Vec3(0.0, 0.0, 0.0))
demag = mm.DemagField(g)
heff  = mm.EffectiveFieldSum()
heff.add(exch)
heff.add(zee)
heff.add(demag)

# ---------------------------------------------------------------------------
# (O1) max_angle
# ---------------------------------------------------------------------------
print("--- (O1) max_angle ---")
angle_uniform = mm.max_angle(m)
print(f"  Uniform +x state:   max_angle = {angle_uniform:.4f} deg  (expect 0.0)")

m_rand = mm.random_mag(g, seed=7)
angle_rand = mm.max_angle(m_rand)
print(f"  Random state:       max_angle = {angle_rand:.2f} deg  (expect > 0)")
assert angle_uniform < 1e-6, f"Expected ~0 for uniform state, got {angle_uniform}"
assert angle_rand > 1.0,    f"Expected > 1 for random state, got {angle_rand}"
print("  max_angle OK")

# ---------------------------------------------------------------------------
# (O2) B_eff
# ---------------------------------------------------------------------------
print("\n--- (O2) B_eff ---")
m_u = mm.uniform_mag(g, mm.Vec3(0.0, 0.0, 1.0))
zee_z = mm.ZeemanField(mm.Vec3(0.0, 0.0, 1e5))   # 100 kA/m = 0.1257 T
heff2 = mm.EffectiveFieldSum()
heff2.add(zee_z)

B = mm.B_eff(m_u, mat, heff2)
B_np = np.asarray(mm.to_numpy(B))    # (nz, ny, nx, 3)
mu0 = 4e-7 * np.pi
expected_Bz = mu0 * 1e5    # ~0.1257 T
print(f"  Applied H_z = 1e5 A/m  =>  mu0*H = {expected_Bz:.5f} T")
print(f"  B_eff[0,0,0,2] = {B_np[0,0,0,2]:.5f} T   (expect {expected_Bz:.5f} T)")
err = abs(B_np[0,0,0,2] - expected_Bz)
assert err < 1e-6, f"B_eff mismatch: {err}"
print("  B_eff OK")

# ---------------------------------------------------------------------------
# (O3) energy_table
# ---------------------------------------------------------------------------
print("\n--- (O3) energy_table ---")
m3 = mm.uniform_mag(g, mm.Vec3(1.0, 0.0, 0.0))
E  = mm.energy_table(m3, mat, heff)
print("  Energy breakdown (J):")
for k, v in E.items():
    print(f"    {k:22s} = {v:+.4e} J")
assert any("Exchange" in k for k in E), f"Missing Exchange in energy_table: {list(E.keys())}"
assert any("Zeeman"   in k for k in E), f"Missing Zeeman in energy_table: {list(E.keys())}"
assert any("Demag"    in k for k in E), f"Missing Demag in energy_table: {list(E.keys())}"
assert "total"    in E,                  "Missing total in energy_table"
# Zeeman = 0 (no applied field), exchange ~ 0 (uniform)
zee_key  = next(k for k in E if "Zeeman" in k)
exch_key = next(k for k in E if "Exchange" in k)
print(f"  Zeeman energy = {E[zee_key]:.2e} J  (should be ~0 for H=0)")
assert abs(E[zee_key]) < 1e-30, f"Zeeman energy nonzero: {E[zee_key]}"
print("  energy_table OK")

# ---------------------------------------------------------------------------
# (O4) Table.add_column
# ---------------------------------------------------------------------------
print("\n--- (O4) Table.add_column ---")
tbl = mm.Table()
tbl.add_column("max_angle", lambda t, m: mm.max_angle(m))
tbl.add_column("Q", lambda t, m: float(mm.topological_charge_Q(m)))

# Short RK4 run: 3 steps, collect table rows
integ = mm.RK4Integrator(dt=1e-14)
m4 = mm.uniform_mag(g, mm.Vec3(1.0, 0.0, 0.0))
zee4 = mm.ZeemanField(mm.Vec3(0.0, 0.0, -5e4))   # slightly off-axis
heff4 = mm.EffectiveFieldSum()
heff4.add(mm.ExchangeField())
heff4.add(zee4)

for step in range(3):
    t = step * 1e-14
    tbl.add_row(t, m4, mat=mat, heff=heff4)
    integ.step(m4, mat, heff4)

print(f"  Table rows: {len(tbl)}")
print(f"  Columns: {tbl.columns}")
assert "max_angle" in tbl.columns, "Custom column 'max_angle' missing"
assert "Q"         in tbl.columns, "Custom column 'Q' missing"
# All rows should have all columns
arr = tbl.to_numpy()
print(f"  Table shape: {arr.shape}  (expect 3 rows x {len(tbl.columns)} cols)")
assert arr.shape[0] == 3, f"Expected 3 rows, got {arr.shape[0]}"
assert arr.shape[1] == len(tbl.columns), "Column count mismatch"
print("  Table.add_column + to_numpy OK")

# ---------------------------------------------------------------------------
# (P3) def_region / new_region_map
# ---------------------------------------------------------------------------
print("\n--- (P3) def_region / new_region_map ---")
g5 = mm.StructuredGrid(20, 10, 1, 10e-9, 10e-9, 10e-9)

left  = mm.x_range(g5, -100e-9, 0.0)    # left half
right = mm.x_range(g5,  0.0, 100e-9)    # right half

rm = mm.new_region_map(g5, (1, left), (2, right))
ids = [rm[i] for i in range(20*10)]
n1 = ids.count(1)
n2 = ids.count(2)
print(f"  Region 1 (left):  {n1} cells  (expect 100)")
print(f"  Region 2 (right): {n2} cells  (expect 100)")
assert n1 == 100, f"Expected 100 left cells, got {n1}"
assert n2 == 100, f"Expected 100 right cells, got {n2}"

# def_region: partial overwrite
rm2 = mm.RegionMap(g5)
mm.def_region(rm2, 3, mm.circle(g5, 40e-9))
n3 = sum(1 for i in range(20*10) if rm2[i] == 3)
print(f"  def_region circle r=40nm: {n3} cells inside circle")
assert n3 > 0, "def_region set 0 cells"
print("  def_region / new_region_map OK")

# ---------------------------------------------------------------------------
# (P1) FrozenIntegrator
# ---------------------------------------------------------------------------
print("\n--- (P1) FrozenIntegrator ---")
g6 = mm.StructuredGrid(10, 5, 1, 10e-9, 10e-9, 10e-9)
m6 = mm.random_mag(g6, seed=42)

# Pin left 3 columns
pin_mask = mm.x_range(g6, -50e-9, -20e-9)   # ix = 0, 1, 2  (cells 0..2)

# Save initial state as numpy to check pinned cells
m6_init_np = np.asarray(mm.to_numpy(m6)).copy()   # (nz, ny, nx, 3)
nx6 = g6.nx; ny6 = g6.ny; nz6 = g6.nz

# Identify pinned (ix, iy, iz) from mask
pinned_cells = []
for iz in range(nz6):
    for iy in range(ny6):
        for ix in range(nx6):
            lin = ix + nx6*(iy + ny6*iz)
            if pin_mask[lin] > 0.5:
                pinned_cells.append((iz, iy, ix))

integ6 = mm.RK4Integrator(dt=1e-13)
frozen = mm.FrozenIntegrator(integ6, pin_mask, m6)

zee6 = mm.ZeemanField(mm.Vec3(0.0, 0.0, 5e4))
heff6 = mm.EffectiveFieldSum()
heff6.add(mm.ExchangeField())
heff6.add(zee6)

# Run 5 steps
for _ in range(5):
    frozen.step(m6, mat, heff6)

# Verify pinned cells are unchanged
n_pin = len(pinned_cells)
m6_after_np = np.asarray(mm.to_numpy(m6))
errors = []
for (iz, iy, ix) in pinned_cells:
    diff = np.abs(m6_after_np[iz, iy, ix, :] - m6_init_np[iz, iy, ix, :]).sum()
    if diff > 1e-12:
        errors.append(((ix, iy, iz), diff))

print(f"  Pinned cells: {n_pin}")
print(f"  After 5 steps, frozen cells changed: {len(errors)}  (expect 0)")
if errors:
    print(f"  ERROR: {errors[:3]}")
assert len(errors) == 0, f"FrozenIntegrator: {len(errors)} pinned cells changed"
print("  FrozenIntegrator OK")

# ---------------------------------------------------------------------------
# (P2) SurfaceAnisotropyField
# ---------------------------------------------------------------------------
print("\n--- (P2) SurfaceAnisotropyField ---")

# 1-layer slab: all cells are surface cells (iz=0 AND iz=nz-1 = same layer)
g7 = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 1e-9)  # 1 nm thick
m7 = mm.uniform_mag(g7, mm.Vec3(0.0, 0.0, 1.0))     # +z = along n_hat

Ks  = 1.2e-3   # J/m^2 (typical Co/Pt PMA)
mat7 = mm.Material.cobalt()
sa = mm.SurfaceAnisotropyField(Ks=Ks)

H_out = mm.VectorField3D(g7)
sa.accumulate(m7, mat7, H_out)
H_np = np.asarray(mm.to_numpy(H_out))   # (nz, ny, nx, 3)

mu0 = 4e-7 * np.pi
t_cell = g7.dz
Ms = mat7.Ms
H_s_expected = 2 * Ks / (mu0 * Ms * t_cell)   # H in A/m
print(f"  Ks = {Ks:.2e} J/m2, t_cell = {t_cell*1e9:.0f} nm, Ms = {Ms:.2e} A/m")
print(f"  H_s expected = {H_s_expected:.4e} A/m  (along z)")
print(f"  H_s computed = {H_np[0,0,0,2]:.4e} A/m")
err_sa = abs(H_np[0,0,0,2] - H_s_expected) / H_s_expected
print(f"  Relative error = {err_sa:.2e}  (should be < 1e-9)")
assert err_sa < 1e-9, f"SurfaceAnisotropyField error too large: {err_sa}"

# Energy check: E = -Ks * (m.n)^2 * (dV/t) * N_cells
dV = g7.dx * g7.dy * g7.dz
n_cells = g7.nx * g7.ny * g7.nz
E_sa = sa.energy(m7, mat7)
E_expected = -Ks * 1.0**2 * (dV / t_cell) * n_cells
print(f"  Energy expected = {E_expected:.4e} J")
print(f"  Energy computed = {E_sa:.4e} J")
err_E = abs(E_sa - E_expected) / abs(E_expected)
print(f"  Relative energy error = {err_E:.2e}  (should be < 1e-9)")
assert err_E < 1e-9, f"SurfaceAnisotropyField energy error: {err_E}"

# Verify name
print(f"  name = '{sa.name}'   (expect 'SurfaceAnisotropy')")
assert sa.name == "SurfaceAnisotropy", f"name mismatch: {sa.name}"

# Test properties
sa.Ks = 0.8e-3
sa.n_hat = mm.Vec3(1.0, 0.0, 0.0)
print(f"  After set: Ks = {sa.Ks:.2e}, n_hat = ({sa.n_hat.x:.1f},{sa.n_hat.y:.1f},{sa.n_hat.z:.1f})")
assert abs(sa.Ks - 0.8e-3) < 1e-15
print("  SurfaceAnisotropyField OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Phase O + P API Summary ===")
print("  mm.max_angle(m)                    -- max inter-cell angle [deg] (convergence)")
print("  mm.B_eff(m, mat, heff)             -- mu0 * H_eff [T]  (mumax3 B_eff)")
print("  mm.energy_table(m, mat, heff)      -- {name: E_J, ..., 'total': E_J}")
print("  Table.add_column(name, fn)         -- custom lambda column before add_row")
print("  Table.to_numpy()                   -- (n_rows, n_cols) float array")
print("  mm.def_region(rm, id, geom)        -- assign region id to mask > 0.5")
print("  mm.new_region_map(g, (1,m1), ...) -- build RegionMap from shape pairs")
print("  mm.FrozenIntegrator(integ, mask, m0) -- pin cells; wraps any CPU integ")
print("  mm.SurfaceAnisotropyField(Ks, n_hat) -- PMA interface anisotropy [J/m2]")
print("\nAll Phase O + P features verified OK.")
