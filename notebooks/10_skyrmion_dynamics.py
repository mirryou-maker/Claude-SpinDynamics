"""
10_skyrmion_dynamics.py — Néel skyrmion stabilisation + SOT-driven motion

Physics
-------
- 200×200×1 Py film, 5 nm cells, R=50 nm Néel skyrmion initial state
- PMA: K_uniaxial = 0.8e5 J/m³ (z easy-axis, stabilises skyrmion)
- SOT drives the skyrmion across the film
- Topological charge Q tracked every 10 ps

Output
------
- skyrmion_trajectory.png  (x,y centre vs time)
- skyrmion_topological_Q.png  (Q vs time)
- table_skyrmion.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'windows-msvc', 'python'))

import micromag as mm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Grid + material (Py with PMA)
# ---------------------------------------------------------------------------
Lx, Ly, Lz = 200e-9, 200e-9, 5e-9
nx, ny, nz  = 40, 40, 1
g = mm.StructuredGrid(nx, ny, nz, Lx/nx, Ly/ny, Lz/nz)

mat = mm.Material()
mat.Ms          = 8e5
mat.A_exchange  = 1.3e-11
mat.K_uniaxial  = 0.8e5        # PMA
mat.easy_axis   = mm.Vec3(0, 0, 1)
mat.alpha       = 0.05

# ---------------------------------------------------------------------------
# Initial state: Néel skyrmion r=15nm at centre
# ---------------------------------------------------------------------------
m = mm.neel_skyrmion(g, 15e-9, charge=1, pol=-1)
m.apply_mask(mm.rect(g, Lx, Ly))

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
demag  = mm.DemagField(g)
exch   = mm.ExchangeField(mm.BoundaryCondition.Neumann)
aniso  = mm.UniaxialAnisotropyField()
zeeman = mm.ZeemanField(mm.Vec3(0, 0, 0))

heff = mm.EffectiveFieldSum()
heff.add(demag)
heff.add(exch)
heff.add(aniso)
heff.add(zeeman)

# SOT: J_c along x, spin polarisation along y (drives skyrmion in x)
sot = mm.SpinOrbitTorque(
    J_c=1e12,            # current density A/m²
    theta_SH=0.1,        # spin Hall angle
    d_fm=Lz,
    sigma=mm.Vec3(0, 1, 0),  # spin polarisation along y
    eta_DL=1.0,
    eta_FL=0.0
)
stt_sum = mm.SpinTorqueSum()
stt_sum.add(sot)

# ---------------------------------------------------------------------------
# First: relax skyrmion (no SOT)
# ---------------------------------------------------------------------------
print("Relaxing skyrmion...")
integ_relax = mm.RK4Integrator(dt=5e-13)
t_relax = mm.run(integ_relax, m, mat, heff, t_total=1e-9)
Q_init = mm.topological_charge_Q(m)
print(f"  Q after relax = {Q_init:.3f}   (expect ≈ -1 for charge=1, pol=-1)")

# ---------------------------------------------------------------------------
# SOT dynamics
# ---------------------------------------------------------------------------
print("SOT dynamics...")
tbl  = mm.Table()
integ = mm.RK4Integrator(dt=1e-13)

t_total = 2e-9
t_snap  = 0.0
dt_out  = 10e-12   # output every 10 ps

centres_x, centres_y, times_ns = [], [], []

def callback(t, mv):
    global t_snap
    if t - t_snap < dt_out - 1e-15:
        return
    t_snap = t
    # Find skyrmion centre: minimum mz
    arr = mm.to_numpy(mv)[:, :, :, 2]  # shape nz,ny,nx
    iz = 0
    idx = np.argmin(arr[iz])
    iy, ix = np.unravel_index(idx, (ny, nx))
    cx = (ix + 0.5) * g.dx - Lx/2
    cy = (iy + 0.5) * g.dy - Ly/2
    Q  = mm.topological_charge_Q(mv)
    centres_x.append(cx * 1e9)
    centres_y.append(cy * 1e9)
    times_ns.append(t * 1e9)
    tbl.add_row(t, mv, mat=mat, heff=heff,
                extra={"Q": Q, "cx_nm": cx*1e9, "cy_nm": cy*1e9})

t_sim = mm.run(integ, m, mat, heff, t_total=t_total,
               stt=stt_sum, callback=callback, callback_dt=dt_out)

# ---------------------------------------------------------------------------
# Save table
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(__file__)
tbl.save(os.path.join(out_dir, "table_skyrmion.csv"))
print(f"  Saved table_skyrmion.csv  ({len(tbl)} rows)")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(times_ns, centres_x, label="cx")
axes[0].plot(times_ns, centres_y, label="cy")
axes[0].set_xlabel("time (ns)")
axes[0].set_ylabel("position (nm)")
axes[0].set_title("Skyrmion trajectory (SOT-driven)")
axes[0].legend()

# topological charge from table
t_vals = [float(r) for r in open(os.path.join(out_dir, "table_skyrmion.csv")).readlines()[1:] if r.strip()]
import csv
with open(os.path.join(out_dir, "table_skyrmion.csv")) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
t_arr = [float(r["t"])*1e9 for r in rows]
Q_arr = [float(r["Q"]) for r in rows]

axes[1].plot(t_arr, Q_arr)
axes[1].axhline(-1, ls='--', color='gray', label='Q=-1 (ideal)')
axes[1].set_xlabel("time (ns)")
axes[1].set_ylabel("topological charge Q")
axes[1].set_title("Q vs time (SOT dynamics)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "skyrmion_trajectory.png"), dpi=150)
print("  Saved skyrmion_trajectory.png")
plt.close()
