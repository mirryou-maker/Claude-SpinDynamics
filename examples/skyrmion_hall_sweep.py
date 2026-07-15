"""Fig 2 — quantitative skyrmion-Hall: velocity vs current, Hall angle vs damping,
with the Thiele-model overlay. GPU.

  (a) |v| vs J at fixed alpha  -> linear mobility
  (b) Hall angle vs alpha at fixed J, points + Thiele: tan(theta_H) = C / alpha

    python examples/skyrmion_hall_sweep.py
-> benchmarks/skyrmion_hall/hall_sweep.png
"""
import os, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
assert mm.cuda_available()
OUT = ROOT / "benchmarks" / "skyrmion_hall"; OUT.mkdir(parents=True, exist_ok=True)
INK, BLUE, RED = "#0b0b0b", "#2a78d6", "#e34948"

nx, ny, dx = 300, 180, 3.0e-9
Lx, Ly = nx*dx, ny*dx
g = mm.StructuredGrid(nx, ny, 1, dx, dx, 1e-9)
mat = mm.Material(); mat.Ms=5.8e5; mat.A_exchange=1.5e-11
mat.K_uniaxial=0.8e6; mat.easy_axis=mm.Vec3(0,0,1)

demag = mm.DemagFieldGPU(g)
fields = mm.FieldSumGPU()
fields.add(mm.ExchangeFieldGPU(g)); fields.add(mm.UniaxialAnisotropyFieldGPU(g))
fields.add(mm.InterfacialDMIFieldGPU(g, 3.0e-3))

# relax one clean skyrmion at centre -> reused for every run
m0 = mm.neel_skyrmion(g, 28e-9, charge=1, pol=-1, cx=0.0, cy=0.0)
mat.alpha = 0.30
mini = mm.MinimizeGPU(g); o = mm.MinimizeGPUOptions(); o.threshold=1e-5; o.max_steps=60000
mini.upload(m0); mini.run(mat, demag, fields, o); mini.download(m0)
print(f"relaxed Q={mm.topological_charge_Q(m0):+.3f}", flush=True)


def core(m):
    mz = np.array(mm.to_numpy(m), copy=True)[0][:, :, 2]
    c = mz < -0.3
    if not c.sum(): return np.nan, np.nan
    yy, xx = np.where(c); return xx.mean()*dx*1e9, yy.mean()*dx*1e9


def drive(J, alpha, t_meas=(0.5e-9, 1.5e-9), dt=2e-13):
    """Return (v [m/s], hall_angle_deg) from displacement between the two times."""
    mat.alpha = alpha
    sot = mm.SpinOrbitTorqueGPU(g, J, 0.30, 1e-9, mm.Vec3(0,1,0), 1.0, 0.0)
    tq = mm.SpinTorqueSumGPU(); tq.add(sot)
    integ = mm.RK4IntegratorGPU(g, dt); m = mm.VectorField3D(g)
    mm.from_numpy(m, mm.to_numpy(m0)); integ.upload(m)
    k1, k2 = int(round(t_meas[0]/dt)), int(round(t_meas[1]/dt))
    p1 = p2 = None
    for k in range(1, k2+1):
        integ.step(mat, demag, fields, tq)
        if k == k1: integ.download(m); p1 = core(m)
    integ.download(m); p2 = core(m)
    dxs = (p2[0]-p1[0])*1e-9; dys = (p2[1]-p1[1])*1e-9
    v = np.hypot(dxs, dys) / (t_meas[1]-t_meas[0])
    ang = np.degrees(np.arctan2(dys, dxs))
    return v, ang


# ---- (a) velocity vs J  (alpha = 0.30) --------------------------------------
Js = np.array([0.75, 1.5, 2.25, 3.0, 3.75]) * 1e11
vJ = []
for J in Js:
    v, a = drive(J, 0.30); vJ.append(v)
    print(f"  J={J:.2e}  v={v:.1f} m/s  angle={a:+.0f}", flush=True)
vJ = np.array(vJ)

# ---- (b) Hall angle vs alpha  (J = 2.0e11) ----------------------------------
alphas = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
thA = []
for al in alphas:
    v, a = drive(2.0e11, al); thA.append(abs(a))
    print(f"  alpha={al:.2f}  hall={a:+.0f}  v={v:.1f}", flush=True)
thA = np.array(thA)

# save raw sweep data (so the figure can be regenerated without re-running)
np.savetxt(OUT / "hall_sweep_data.txt",
           np.column_stack([np.r_[Js, np.full(len(alphas), np.nan)],
                            np.r_[vJ, np.full(len(alphas), np.nan)],
                            np.r_[np.full(len(Js), np.nan), alphas],
                            np.r_[np.full(len(Js), np.nan), thA]]),
           header="J v_ofJ  alpha thH_ofAlpha")

# ---- plot -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
# (a) v vs J  (linear mobility)
cf = np.polyfit(Js, vJ, 1)
ax[0].plot(Js/1e11, vJ, "o", ms=8, color=BLUE)
xx = np.linspace(0, Js.max()*1.05, 50)
ax[0].plot(xx/1e11, np.polyval(cf, xx), "-", color=BLUE, lw=1.6,
           label=f"linear fit: {cf[0]*1e11:.1f} m/s per 10¹¹ A/m²")
ax[0].set_xlabel("current density J  (×10¹¹ A/m²)", fontsize=11)
ax[0].set_ylabel("skyrmion speed |v|  (m/s)", fontsize=11)
ax[0].set_title("(a)  velocity vs current  (α = 0.30)", fontsize=12, fontweight="bold")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3); ax[0].set_xlim(0, None); ax[0].set_ylim(0, None)
# (b) hall angle vs alpha + Thiele  tan(theta)=C/alpha  (valid points only)
ok = np.isfinite(thA)
al_v, th_v = alphas[ok], thA[ok]
C = np.nanmedian(np.tan(np.radians(th_v)) * al_v)       # fit C from tan(th)*alpha
al_fine = np.linspace(al_v.min()*0.85, al_v.max()*1.05, 100)
th_th = np.degrees(np.arctan(C / al_fine))
ax[1].plot(al_v, th_v, "s", ms=8, color=RED, label="simulation")
ax[1].plot(al_fine, th_th, "-", color=INK, lw=1.6,
           label=r"Thiele  $\tan\theta_{sk}=\mathcal{G}/(\alpha\mathcal{D})$" +
                 f"\n" + r"$\mathcal{G}/\mathcal{D}=$" + f"{C:.2f}")
ax[1].set_xlabel("Gilbert damping α", fontsize=11)
ax[1].set_ylabel("skyrmion-Hall angle  θ$_{sk}$  (deg)", fontsize=11)
ax[1].set_title("(b)  Hall angle vs damping  (J = 2×10¹¹)", fontsize=12, fontweight="bold")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.suptitle("Skyrmion-Hall transport — Claude-SD (Thiele-model consistent)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "hall_sweep.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", OUT / "hall_sweep.png")
