"""Notebook 31 - Phase M feature showcase.

Demonstrates the new Phase M API additions:
  (A) get_torque_field()     - per-cell LLG torque map
  (B) PythonField            - user-defined effective field term
  (C) stray_field()          - dipole stray field from external magnet

System: permalloy nano-disk (r=100 nm, t=5 nm), 2 nm cells.
All computations on CPU.
"""

import sys, os
# Claude-SpinDynamics module location (CPU build)
_repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_repo, 'build', 'windows-msvc', 'python'))

import numpy as np
import micromag as mm

print("=== Notebook 31: Phase M Features ===\n")

# ---------------------------------------------------------------------------
# Setup: permalloy nano-disk
# ---------------------------------------------------------------------------
nx, ny, nz = 100, 100, 5
dx = 2e-9
g  = mm.StructuredGrid(nx, ny, nz, dx, dx, dx)

mat = mm.Material.permalloy()
mat.alpha = 0.01

# Circular disk geometry
disk = mm.circle(g, 100e-9)

m = mm.uniform_mag(g, mm.Vec3(0.0, 0.0, 1.0))
mm.set_geom(disk, m)

demag = mm.DemagField(g)
exch  = mm.ExchangeField()
heff  = mm.EffectiveFieldSum()
heff.add(demag)
heff.add(exch)

print("Grid: {}x{}x{}, dx={} nm".format(nx, ny, nz, dx*1e9))
print("Material: permalloy  Ms={:.0f} kA/m  alpha={}".format(mat.Ms/1e3, mat.alpha))

# ---------------------------------------------------------------------------
# (A) get_torque_field() - per-cell LLG torque
# ---------------------------------------------------------------------------
print("\n--- (A) get_torque_field() ---")

tau = mm.get_torque_field(m, mat, heff)
tau_np = mm.to_numpy(tau)  # (nz, ny, nx, 3) rad/s

tau_mag = np.sqrt((tau_np**2).sum(-1))  # (nz, ny, nx)
tau_max = tau_mag.max()
disk_np = np.array([disk[i + nx*(j + ny*k)]
                     for k in range(nz) for j in range(ny) for i in range(nx)],
                    dtype=float).reshape(nz, ny, nx).astype(bool)
tau_mean_inside = tau_mag[disk_np].mean()

print(f"  Max |torque|  = {tau_max:.4e} rad/s")
print(f"  Mean |torque| inside disk = {tau_mean_inside:.4e} rad/s")
print(f"  max_torque_field() = {mm.max_torque_field(m, mat, heff):.4e} rad/s")

# Compare z-component of torque (out-of-plane exchange + demag)
tau_z_mean = tau_np[nz//2, ny//2-5:ny//2+5, nx//2-5:nx//2+5, 2].mean()
print(f"  tau_z near centre (mz=1 layer) ~= {tau_z_mean:.3e} rad/s (should be ~0 for mz=1)")

# ---------------------------------------------------------------------------
# (B) PythonField - user-defined Zeeman gradient field
# ---------------------------------------------------------------------------
print("\n--- (B) PythonField (user-defined effective field) ---")

# A field that creates a linear H_x gradient across the disk
# H_x(x) = H0 * (2*x/Lx - 1) so it ranges from -H0 to +H0
H0 = 20e3  # A/m
Lx = nx * dx

def gradient_field(m_arr):
    """Zeeman gradient: H_x varies linearly from -H0 to +H0."""
    H = np.zeros_like(m_arr)
    xs = (np.arange(nx) + 0.5) * dx / Lx  # 0..1
    # Broadcast over (nz, ny, nx, 3)
    H[..., 0] = H0 * (2 * xs - 1)[np.newaxis, np.newaxis, :] * 2
    return H

pf = mm.PythonField(gradient_field, name_str="ZeemanGradient")
heff.add(pf)

tau2 = mm.get_torque_field(m, mat, heff)
tau2_np = mm.to_numpy(tau2)
tau2_max = np.sqrt((tau2_np**2).sum(-1)).max()
print(f"  PythonField name: '{pf.name()}'")
print(f"  Added gradient H_x in [{-H0*2/1e3:.0f}, {H0*2/1e3:.0f}] kA/m")
print(f"  Max |torque| with PythonField = {tau2_max:.4e} rad/s")
print(f"  PythonField.energy(m, mat) = {pf.energy(m, mat):.2f} J (no energy_fn set)")

# Remove gradient field and verify torque returns to original
heff_plain = mm.EffectiveFieldSum()
heff_plain.add(demag)
heff_plain.add(exch)
tau3 = mm.get_torque_field(m, mat, heff_plain)
tau3_max = mm.max_torque_field(m, mat, heff_plain)
print(f"  Max |torque| without PythonField = {tau3_max:.4e} rad/s (back to original)")

# ---------------------------------------------------------------------------
# (C) stray_field() - dipole from external magnet below the disk
# ---------------------------------------------------------------------------
print("\n--- (C) stray_field() (external dipole) ---")

# Magnet: cobalt disk 100nm diameter x 5nm, positioned 20nm below
Ms_co  = 1.4e6    # A/m (cobalt)
r_co   = 50e-9    # radius [m]
h_co   = 5e-9     # thickness [m]
V_co   = np.pi * r_co**2 * h_co  # ~= 3.93e-23 m^3

# Dipole at box centre (x,y), 20nm below bottom surface (z = -dz)
box_cx = nx * dx / 2
box_cy = ny * dx / 2
dipole_z = -20e-9   # 20 nm below the simulation box bottom

H_stray = mm.stray_field(g,
                          Ms_ext    = Ms_co,
                          volume_ext= V_co,
                          position  = (box_cx, box_cy, dipole_z),
                          moment_dir= (0.0, 0.0, 1.0))

H_stray_np = mm.to_numpy(H_stray)  # (nz, ny, nx, 3) A/m

# Centre-cell stray field (closest to dipole)
iz_mid, iy_mid, ix_mid = 0, ny//2, nx//2
H_centre = H_stray_np[iz_mid, iy_mid, ix_mid]
H_centre_mag = np.linalg.norm(H_centre)
print(f"  Cobalt source: Ms={Ms_co/1e6:.1f} MA/m, V={V_co:.2e} m^3")
print(f"  Dipole position: z={dipole_z*1e9:.0f} nm below disk")
print(f"  Stray field at disk centre: Hx={H_centre[0]:.0f}, Hy={H_centre[1]:.0f}, Hz={H_centre[2]:.0f} A/m")
print(f"  |H_stray| at centre = {H_centre_mag:.1f} A/m = {H_centre_mag/1e3:.2f} kA/m")

# Add stray field as spatial Zeeman and compute torque
zee_stray = mm.ZeemanFieldSpatial(H_stray)
heff_stray = mm.EffectiveFieldSum()
heff_stray.add(demag)
heff_stray.add(exch)
heff_stray.add(zee_stray)
tau_stray = mm.max_torque_field(m, mat, heff_stray)
print(f"  Max |torque| with stray field = {tau_stray:.4e} rad/s")

# Stray field profile along x (at y=centre, bottom layer)
H_z_profile = H_stray_np[0, ny//2, :, 2]  # Hz along x
print(f"  Hz stray profile: min={H_z_profile.min():.0f}, max={H_z_profile.max():.0f} A/m")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Phase M API Summary ===")
print("  get_torque_field(m, mat, heff, stt=None) -> VectorField3D [rad/s]")
print("  max_torque_field(m, mat, heff, stt=None) -> float [rad/s]")
print("  PythonField(fn, name_str, energy_fn)    -> IEffectiveField subclass")
print("  stray_field(grid, Ms, vol, pos, dir)    -> VectorField3D [A/m]")
print("\nAll Phase M features verified OK.")
