"""Notebook 34 - Phase Q + R feature showcase.

Demonstrates:
  (Q) MagnetoelasticField -- B1/B2 magnetostrictive coupling
  (R) run_until_converged  -- convergence-based adaptive relaxation
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
import math

print("=== Notebook 34: Phase Q + R Features ===\n")

mu0 = 4e-7 * math.pi

# ---------------------------------------------------------------------------
# (Q1) MagnetoelasticField — field verification (Ni, uniaxial exx)
# ---------------------------------------------------------------------------
print("--- (Q1) MagnetoelasticField: analytic field check ---")

# 1-cell system for exact analytic check
g1 = mm.StructuredGrid(1, 1, 1, 5e-9, 5e-9, 5e-9)
mat_ni = mm.Material()
mat_ni.Ms = 4.85e5     # Ni Ms [A/m]
mat_ni.A_exchange = 9e-12   # [J/m]

B1_ni = -62.4e6   # J/m^3  (Ni)
B2_ni = -27.1e6
exx   =  0.001    # uniaxial strain along x

me = mm.MagnetoelasticField(B1=B1_ni, B2=B2_ni)
me.exx = exx

# m = +x: only H_x should be nonzero
m1 = mm.uniform_mag(g1, mm.Vec3(1.0, 0.0, 0.0))
H_out = mm.VectorField3D(g1)
me.accumulate(m1, mat_ni, H_out)
H_np = np.asarray(mm.to_numpy(H_out))   # (1, 1, 1, 3)
Hx_me = H_np[0, 0, 0, 0]
Hy_me = H_np[0, 0, 0, 1]
Hz_me = H_np[0, 0, 0, 2]

# Analytic: H_x = -(2/mu0*Ms) * B1*mx*exx  (mx=1, exy=exz=0)
Hx_expected = -2.0 * B1_ni * exx / (mu0 * mat_ni.Ms)
print(f"  B1={B1_ni:.2e} J/m3, exx={exx}, Ms={mat_ni.Ms:.2e} A/m")
print(f"  H_x expected = {Hx_expected:.4e} A/m")
print(f"  H_x computed = {Hx_me:.4e} A/m")
print(f"  H_y = {Hy_me:.2e}, H_z = {Hz_me:.2e}  (expect 0)")
err = abs(Hx_me - Hx_expected) / abs(Hx_expected)
assert err < 1e-9, f"H_x error too large: {err}"
assert abs(Hy_me) < 1e-15 * abs(Hx_expected), f"H_y nonzero: {Hy_me}"
assert abs(Hz_me) < 1e-15 * abs(Hx_expected), f"H_z nonzero: {Hz_me}"
print("  Field check (uniaxial exx) OK")

# ---------------------------------------------------------------------------
# (Q2) MagnetoelasticField — off-diagonal strain (B2 coupling)
# ---------------------------------------------------------------------------
print("\n--- (Q2) MagnetoelasticField: off-diagonal strain exy ---")

exy = 0.002
me2 = mm.MagnetoelasticField(B1=0.0, B2=B2_ni)
me2.exy = exy

# m = (1/sqrt2)(x+y): should couple B2 via mx*my*exy
m_xy = mm.uniform_mag(g1, mm.Vec3(1.0/math.sqrt(2), 1.0/math.sqrt(2), 0.0))
H_out2 = mm.VectorField3D(g1)
me2.accumulate(m_xy, mat_ni, H_out2)
H2_np = np.asarray(mm.to_numpy(H_out2))
Hx2 = H2_np[0, 0, 0, 0]
Hy2 = H2_np[0, 0, 0, 1]
mx = 1.0/math.sqrt(2); my = 1.0/math.sqrt(2)
# H_x = -(2/mu0Ms) * B2 * my * exy
Hx2_exp = -2.0 * B2_ni * my * exy / (mu0 * mat_ni.Ms)
Hy2_exp = -2.0 * B2_ni * mx * exy / (mu0 * mat_ni.Ms)
print(f"  H_x expected = {Hx2_exp:.4e}, computed = {Hx2:.4e}")
print(f"  H_y expected = {Hy2_exp:.4e}, computed = {Hy2:.4e}")
assert abs(Hx2 - Hx2_exp) / abs(Hx2_exp) < 1e-9, f"H_x error: {Hx2-Hx2_exp}"
assert abs(Hy2 - Hy2_exp) / abs(Hy2_exp) < 1e-9, f"H_y error: {Hy2-Hy2_exp}"
print("  Off-diagonal exy OK")

# ---------------------------------------------------------------------------
# (Q3) MagnetoelasticField — energy check
# ---------------------------------------------------------------------------
print("\n--- (Q3) MagnetoelasticField: energy ---")

g3 = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
me3 = mm.MagnetoelasticField(B1=B1_ni)
me3.set_strain(exx=0.001, eyy=-0.0005, ezz=-0.0005)

# Uniform m = +z: E = B1*mz^2*ezz * N_cells * dV
m3 = mm.uniform_mag(g3, mm.Vec3(0.0, 0.0, 1.0))
E3 = me3.energy(m3, mat_ni)
dV = (5e-9)**3
N  = 4 * 4 * 1
E3_exp = B1_ni * 1.0**2 * (-0.0005) * N * dV   # mz=1, ezz=-0.0005
print(f"  Uniform +z: E expected = {E3_exp:.4e} J, computed = {E3:.4e} J")
err_E = abs(E3 - E3_exp) / abs(E3_exp)
assert err_E < 1e-9, f"Energy error: {err_E}"
print("  Energy check OK")

# ---------------------------------------------------------------------------
# (Q4) MagnetoelasticField — property access and name
# ---------------------------------------------------------------------------
print("\n--- (Q4) MagnetoelasticField properties ---")
me4 = mm.MagnetoelasticField(B1=1e6, B2=2e6)
me4.exx = 0.005; me4.exy = 0.001; me4.eyz = -0.002
print(f"  name = '{me4.name}'   (expect 'MagnetoelasticField')")
print(f"  B1={me4.B1:.2e}, B2={me4.B2:.2e}")
print(f"  exx={me4.exx}, exy={me4.exy}, eyz={me4.eyz}")
assert me4.name == "MagnetoelasticField"
assert abs(me4.exx - 0.005) < 1e-15
assert abs(me4.exy - 0.001) < 1e-15
assert abs(me4.eyz + 0.002) < 1e-15
print("  Properties OK")

# ---------------------------------------------------------------------------
# (Q5) MagnetoelasticField — physical demo: Ni anisotropy from SAW strain
# ---------------------------------------------------------------------------
print("\n--- (Q5) MagnetoelasticField: Ni effective field from SAW ---")
# Surface acoustic wave produces oscillating strain exx ~ 1e-4
# This shifts the effective anisotropy field felt by magnetization
B1_Ni = -62.4e6
Ms_Ni =  4.85e5
exx_SAW = 1e-4   # typical SAW amplitude

H_ME = 2 * abs(B1_Ni) * exx_SAW / (mu0 * Ms_Ni)   # magnitude
print(f"  Ni SAW-induced H_ME = {H_ME:.1f} A/m  (~{H_ME/Ms_Ni*1000:.1f} mT eq.)")
print(f"  (Competes with shape anisotropy Hk ~ {Ms_Ni/2:.0f} A/m for Ni sphere)")
print("  MagnetoelasticField (Phase Q) complete.")

# ---------------------------------------------------------------------------
# (R1) run_until_converged — basic test
# ---------------------------------------------------------------------------
print("\n--- (R1) run_until_converged: basic convergence ---")

# Small permalloy disk, relax from random state
g_r = mm.StructuredGrid(16, 16, 1, 5e-9, 5e-9, 5e-9)
mat_py = mm.Material.permalloy()
m_r = mm.random_mag(g_r, seed=42)

exch_r = mm.ExchangeField()
demag_r = mm.DemagField(g_r)
heff_r  = mm.EffectiveFieldSum()
heff_r.add(exch_r)
heff_r.add(demag_r)

integ_r = mm.RK4Integrator(dt=1e-13)

ang_before = mm.max_angle(m_r)
print(f"  Initial max_angle = {ang_before:.2f} deg  (random state)")

result = mm.run_until_converged(
    integ_r, m_r, mat_py, heff_r,
    tol_deg=5.0,
    max_steps=50_000,
    check_interval=200,
    verbose=False
)

print(f"  Converged: {result['converged']}")
print(f"  Steps: {result['steps']}")
print(f"  Final max_angle: {result['max_angle']:.4f} deg  (tol=5.0 deg)")
print(f"  Simulated time: {result['t_sim']*1e12:.1f} ps")

assert result["converged"], f"run_until_converged failed to converge: {result}"
assert result["max_angle"] < 5.0, f"max_angle {result['max_angle']} >= tol 5.0"
assert result["steps"] > 0, "No steps taken"
print("  run_until_converged OK")

# ---------------------------------------------------------------------------
# (R2) run_until_converged — max_steps guard
# ---------------------------------------------------------------------------
print("\n--- (R2) run_until_converged: max_steps guard ---")

g_r2 = mm.StructuredGrid(4, 4, 1, 5e-9, 5e-9, 5e-9)
m_r2 = mm.random_mag(g_r2, seed=99)
heff_r2 = mm.EffectiveFieldSum()
heff_r2.add(mm.ZeemanField(mm.Vec3(0.0, 0.0, 0.0)))  # no driving field
integ_r2 = mm.RK4Integrator(dt=1e-13)

# Tiny max_steps with impossible tol: should hit step limit
res2 = mm.run_until_converged(
    integ_r2, m_r2, mat_py, heff_r2,
    tol_deg=0.0001,     # essentially unreachable
    max_steps=50,       # hit this quickly
    check_interval=10
)
assert not res2["converged"], "Should NOT converge with max_steps=50, tol=0.0001"
assert res2["steps"] == 50, f"Expected exactly 50 steps, got {res2['steps']}"
print(f"  max_steps guard: steps={res2['steps']}, converged={res2['converged']}  OK")

# ---------------------------------------------------------------------------
# (R3) run_until_converged — return dict structure
# ---------------------------------------------------------------------------
print("\n--- (R3) run_until_converged: return dict keys ---")
keys = set(result.keys())
expected_keys = {"converged", "steps", "max_angle", "t_sim"}
assert keys == expected_keys, f"Missing keys: {expected_keys - keys}"
print(f"  Keys: {sorted(keys)}  OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Phase Q + R API Summary ===")
print("  mm.MagnetoelasticField(B1, B2)        -- cubic magnetostrictive coupling")
print("    .set_strain(exx, eyy, ...)           -- set uniform strain tensor")
print("    .exx/.eyy/.ezz/.exy/.exz/.eyz       -- individual strain components")
print("    .B1 / .B2                            -- coupling constants [J/m3]")
print("  mm.run_until_converged(integ, m, mat, heff,")
print("      tol_deg=1.0, max_steps=1e6,        -- mumax3 RunWhile(MaxAngle)")
print("      check_interval=100) -> dict")
print("  Returns: {'converged', 'steps', 'max_angle', 't_sim'}")
print("\nAll Phase Q + R features verified OK.")
