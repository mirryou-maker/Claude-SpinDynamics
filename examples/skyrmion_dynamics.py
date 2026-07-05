"""Skyrmion DYNAMICS -> ParaView snapshots.

A DMI-stabilized Néel skyrmion in a thin track is driven by a spin-orbit torque
(SOT). It translates along +x, deflects transversely (skyrmion Hall effect), and
deforms (breathing / elongation, compression against the edge). We capture the
magnetization at a sequence of times and:

  * write a ParaView time series  paraview_demo/skyrmion_dynamics/run.pvd
    (+ run_NNNN.vtk)  — press Play in ParaView to animate,
  * render a snapshot montage (m_z + in-plane arrows) and the trajectory /
    topological-charge / area curves.

    python examples/skyrmion_dynamics.py
"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "windows-msvc" / "python"))
import micromag as mm

OUT = ROOT / "paraview_demo" / "skyrmion_dynamics"; OUT.mkdir(parents=True, exist_ok=True)
INK = "#0b0b0b"

# ---- geometry & material (Co/Pt-like PMA + interfacial DMI) -----------------
nx, ny, nz = 120, 84, 1
dx = 3.5e-9
Lx, Ly = nx * dx, ny * dx
g = mm.StructuredGrid(nx, ny, nz, dx, dx, 1e-9)

mat = mm.Material()
mat.Ms         = 6.0e5
mat.A_exchange = 1.5e-11
mat.K_uniaxial = 0.6e6           # PMA, z easy axis (lower -> larger skyrmion)
mat.easy_axis  = mm.Vec3(0, 0, 1)
mat.alpha      = 0.20            # damping: stable skyrmion + visible skyrmion Hall

# ---- initial Néel skyrmion at centre ---------------------------------------
m = mm.neel_skyrmion(g, 32e-9, charge=1, pol=-1, cx=0.0, cy=0.0)

demag  = mm.DemagField(g)
exch   = mm.ExchangeField(mm.BoundaryCondition.Neumann)
aniso  = mm.UniaxialAnisotropyField()
dmi    = mm.InterfacialDMIField(3.0e-3)          # D = 3 mJ/m^2
zeeman = mm.ZeemanField(mm.Vec3(0, 0, 0))
heff = mm.EffectiveFieldSum()
for f in (demag, exch, aniso, dmi, zeeman):
    heff.add(f)

# ---- relax to a clean skyrmion ---------------------------------------------
print("relaxing skyrmion ...")
mm.run(mm.RK4Integrator(dt=5e-13), m, mat, heff, t_total=0.5e-9)
print(f"  Q after relax = {mm.topological_charge_Q(m):+.3f}")

# ---- SOT drive + snapshot capture ------------------------------------------
sot = mm.SpinOrbitTorque(J_c=3.5e11, theta_SH=0.25, d_fm=1e-9,
                         sigma=mm.Vec3(0, 1, 0), eta_DL=1.0, eta_FL=0.0)
stt = mm.SpinTorqueSum(); stt.add(sot)

frames, times, cx_nm, cy_nm, Qs, area = [], [], [], [], [], []


def snap(t, mv):
    arr = np.array(mm.to_numpy(mv), copy=True)          # (nz,ny,nx,3)
    mz = arr[0, :, :, 2]
    iy, ix = np.unravel_index(np.argmax(mz), mz.shape)   # core = max m_z (core up)
    core = (mz > 0.0)                                     # skyrmion core region (minority up)
    frames.append(arr)
    times.append(t * 1e9)
    cx_nm.append((ix + 0.5) * dx * 1e9); cy_nm.append((iy + 0.5) * dx * 1e9)
    Qs.append(mm.topological_charge_Q(mv))
    area.append(core.sum() * (dx * 1e9) ** 2)            # nm^2 (deformation proxy)


snap(0.0, m)                                             # relaxed state = frame 0
print("SOT dynamics ...")
mm.run(mm.RK4Integrator(dt=2e-13), m, mat, heff, t_total=0.9e-9,
       stt=stt, callback=snap, callback_dt=0.1e-9)   # dense capture of the deformation
print(f"  captured {len(frames)} snapshots; Q {Qs[0]:+.2f} -> {Qs[-1]:+.2f}")

# ---- ParaView time series ---------------------------------------------------
mm.save_paraview_series(frames, str(OUT / "run"), spacing=dx, dt=0.3e-9)
print(f"  wrote {OUT / 'run.pvd'}  (+ {len(frames)} .vtk frames)")

# ---- snapshot montage -------------------------------------------------------
n = len(frames); ncol = 4; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow * ny / nx + 1))
for k in range(nrow * ncol):
    ax = axes.flat[k]
    if k < n:
        arr = frames[k]; mz = arr[0, :, :, 2]; mx = arr[0, :, :, 0]; my = arr[0, :, :, 1]
        ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                  extent=[0, nx, 0, ny], aspect="equal")
        st = max(1, nx // 26)
        ys, xs = np.mgrid[0:ny:st, 0:nx:st]
        ax.quiver(xs + .5, ys + .5, mx[::st, ::st], my[::st, ::st],
                  color=INK, scale=30, width=0.005, pivot="mid")
        ax.set_title(f"t = {times[k]:.1f} ns   Q = {Qs[k]:+.2f}\narea = {area[k]:.0f} nm²",
                     fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if k >= n: ax.axis("off")
fig.suptitle("SOT-driven Néel skyrmion — motion, skyrmion-Hall deflection & deformation "
             "($m_z$ colour, in-plane arrows)", fontsize=13, y=1.0)
fig.tight_layout()
mont = OUT / "snapshots_montage.png"; fig.savefig(mont, dpi=135, bbox_inches="tight", facecolor="white")
print("  wrote", mont)

# ---- trajectory / Q / area curves ------------------------------------------
fig2, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(cx_nm, cy_nm, "-o", ms=3); ax[0].set_xlabel("x core (nm)"); ax[0].set_ylabel("y core (nm)")
ax[0].set_title("core trajectory (skyrmion Hall)"); ax[0].grid(alpha=.3)
ax[1].plot(times, Qs, "-o", ms=3); ax[1].axhline(-1, ls="--", color="gray")
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("Q"); ax[1].set_title("topological charge"); ax[1].grid(alpha=.3)
ax[2].plot(times, area, "-o", ms=3); ax[2].set_xlabel("t (ns)"); ax[2].set_ylabel("core area (nm²)")
ax[2].set_title("area (deformation / breathing)"); ax[2].grid(alpha=.3)
fig2.tight_layout()
traj = OUT / "trajectory.png"; fig2.savefig(traj, dpi=140, bbox_inches="tight", facecolor="white")
print("  wrote", traj)
print(f"\nParaView: open {OUT / 'run.pvd'} and press Play.")
