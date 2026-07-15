"""Skyrmion Hall effect — a skyrmion driven by SOT translates without
annihilating and deflects transversely (skyrmion-Hall effect). GPU.

Produces (in paraview_demo/skyrmion_hall/):
  * run.pvd + run_NNNN.vtk        ParaView time series
  * snap_NNNN.png                 individual m_z + arrow snapshots
  * hall_trajectory.png           COMPOSITE: skyrmion outline at every time
                                  overlaid + core path, coloured by time
  * hall_curves.png               x(t), y(t), Q(t), Hall angle

    python examples/skyrmion_hall.py
"""
import os, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
import os
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
import micromag as mm
assert mm.cuda_available(), "GPU build required"

OUT = ROOT / "paraview_demo" / "skyrmion_hall"; OUT.mkdir(parents=True, exist_ok=True)
INK = "#0b0b0b"

# ---- geometry: long, tall track so the Hall deflection has room -------------
nx, ny, nz = 480, 240, 1
dx = 2.5e-9
Lx, Ly = nx * dx, ny * dx                    # 1200 x 600 nm
g = mm.StructuredGrid(nx, ny, nz, dx, dx, 1e-9)

mat = mm.Material()
mat.Ms = 5.8e5; mat.A_exchange = 1.5e-11
mat.K_uniaxial = 0.8e6; mat.easy_axis = mm.Vec3(0, 0, 1)
mat.alpha = 0.30                              # higher damping -> ~diagonal Hall path

# compact skyrmion, lower-left (room to move up-right diagonally)
m = mm.neel_skyrmion(g, 28e-9, charge=1, pol=-1, cx=-0.29 * Lx, cy=-0.25 * Ly)

demag = mm.DemagFieldGPU(g)
fields = mm.FieldSumGPU()
fields.add(mm.ExchangeFieldGPU(g))
fields.add(mm.UniaxialAnisotropyFieldGPU(g))
fields.add(mm.InterfacialDMIFieldGPU(g, 3.0e-3))

# ---- relax to a clean, compact skyrmion (energy minimizer, robust for DMI) --
print("relax (MinimizeGPU) ...")
mini = mm.MinimizeGPU(g); opts = mm.MinimizeGPUOptions()
opts.threshold = 1e-5; opts.max_steps = 60000
mini.upload(m); mini.run(mat, demag, fields, opts); mini.download(m)
print(f"  Q after relax = {mm.topological_charge_Q(m):+.3f}")

# ---- SOT dynamics (moderate current -> steady motion, no annihilation) ------
sot = mm.SpinOrbitTorqueGPU(g, 2.0e11, 0.30, 1e-9, mm.Vec3(0, 1, 0), 1.0, 0.0)
torques = mm.SpinTorqueSumGPU(); torques.add(sot)

dt = 1e-13
t_total = 8.0e-9
snap_every = 0.4e-9
nsteps = int(round(t_total / dt))
snap_stride = int(round(snap_every / dt))

integ = mm.RK4IntegratorGPU(g, dt); integ.upload(m)
frames, times, cx_nm, cy_nm, Qs, area = [], [], [], [], [], []


def capture(t):
    integ.download(m)
    arr = np.array(mm.to_numpy(m), copy=True)[0]           # (ny,nx,3)
    mz = arr[:, :, 2]; core = mz < -0.3                    # core-down skyrmion (robust)
    frames.append(arr); times.append(t * 1e9)
    if core.sum():
        yy, xx = np.where(core)
        cx_nm.append(xx.mean() * dx * 1e9); cy_nm.append(yy.mean() * dx * 1e9)
        area.append(core.sum() * (dx * 1e9) ** 2)
    else:
        cx_nm.append(np.nan); cy_nm.append(np.nan); area.append(0.0)
    Qs.append(mm.topological_charge_Q(m))


print("SOT dynamics (GPU) ...")
capture(0.0)
for k in range(1, nsteps + 1):
    integ.step(mat, demag, fields, torques)
    if k % snap_stride == 0:
        capture(k * dt)
        print(f"  t={k*dt*1e9:4.1f} ns  core=({cx_nm[-1]:.0f},{cy_nm[-1]:.0f})nm  Q={Qs[-1]:+.2f}", flush=True)
print(f"  Q: {Qs[0]:+.2f} -> {Qs[-1]:+.2f}  (should stay ~const = no annihilation)")

# ---- ParaView series + individual snapshots --------------------------------
mm.save_paraview_series([f[None] for f in frames], str(OUT / "run"), spacing=dx, dt=snap_every)
for i, arr in enumerate(frames):
    fig, ax = plt.subplots(figsize=(8, 8 * ny / nx))
    mz = arr[:, :, 2]; mxc = arr[:, :, 0]; myc = arr[:, :, 1]
    ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
              extent=[0, Lx * 1e9, 0, Ly * 1e9], aspect="equal")
    st = max(1, nx // 40); ys, xs = np.mgrid[0:ny:st, 0:nx:st]
    ax.quiver((xs + .5) * dx * 1e9, (ys + .5) * dx * 1e9, mxc[::st, ::st], myc[::st, ::st],
              color=INK, scale=34, width=0.003, pivot="mid")
    ax.set_title(f"t = {times[i]:.1f} ns   Q = {Qs[i]:+.2f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    fig.tight_layout(); fig.savefig(OUT / f"snap_{i:04d}.png", dpi=110, facecolor="white")
    plt.close(fig)

# ---- COMPOSITE trajectory figure -------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12 * Ly / Lx + 0.6))
tmax = max(times) if times else 1.0
norm = plt.Normalize(0, tmax)
cmap = cm.viridis
xg = (np.arange(nx) + 0.5) * dx * 1e9
yg = (np.arange(ny) + 0.5) * dx * 1e9
# skyrmion outline (mz=0 contour) at each snapshot, coloured by time
for i, arr in enumerate(frames):
    ax.contour(xg, yg, arr[:, :, 2], levels=[0.0], colors=[cmap(norm(times[i]))],
               linewidths=1.6)
# core path
pts = np.array([cx_nm, cy_nm]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.5, zorder=5)
lc.set_array(np.array(times[:-1])); ax.add_collection(lc)
ax.plot(cx_nm, cy_nm, "o", ms=4, color="k", zorder=6)
# Hall angle from net displacement
good = ~np.isnan(cx_nm)
xc = np.array(cx_nm)[good]; yc = np.array(cy_nm)[good]
if len(xc) > 1:
    dxs, dys = xc[-1] - xc[0], yc[-1] - yc[0]
    hall = np.degrees(np.arctan2(dys, dxs))
    ax.annotate(f"skyrmion-Hall angle ≈ {hall:+.0f}°\n(core moved Δx={dxs:+.0f}, Δy={dys:+.0f} nm)",
                xy=(xc[-1], yc[-1]), xytext=(0.55, 0.12), textcoords="axes fraction",
                fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color="#555"))
ax.set_xlim(0, Lx * 1e9); ax.set_ylim(0, Ly * 1e9); ax.set_aspect("equal")
ax.set_xlabel("x (nm)", fontsize=12); ax.set_ylabel("y (nm)", fontsize=12)
ax.set_title("Skyrmion-Hall trajectory — SOT-driven skyrmion outline (m_z=0) vs time",
             fontsize=13, fontweight="bold")
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02); cb.set_label("time (ns)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "hall_trajectory.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote hall_trajectory.png")

# ---- curves -----------------------------------------------------------------
fig2, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(cx_nm, cy_nm, "-o", ms=3); ax[0].set_xlabel("x core (nm)"); ax[0].set_ylabel("y core (nm)")
ax[0].set_title("core trajectory"); ax[0].set_aspect("equal", "datalim"); ax[0].grid(alpha=.3)
ax[1].plot(times, cx_nm, "-o", ms=3, label="x"); ax[1].plot(times, cy_nm, "-s", ms=3, label="y")
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("core position (nm)"); ax[1].set_title("x, y vs time")
ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].plot(times, Qs, "-o", ms=3); ax[2].axhline(Qs[0], ls="--", color="gray")
ax[2].set_xlabel("t (ns)"); ax[2].set_ylabel("Q"); ax[2].set_title("topological charge (constant)")
ax[2].grid(alpha=.3)
fig2.tight_layout(); fig2.savefig(OUT / "hall_curves.png", dpi=140, bbox_inches="tight", facecolor="white")
print("wrote hall_curves.png")
print(f"\nParaView: open {OUT / 'run.pvd'}")
