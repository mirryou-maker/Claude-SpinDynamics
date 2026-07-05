"""Compare SOT-driven skyrmion dynamics: Claude-SD vs mumax3, same parameters.

Runs the Claude-SD dynamics (same as examples/skyrmion_dynamics.py) and loads the
mumax3 frames produced by sk_dyn.mx3 (mumax3 -f sk_dyn.mx3 ; mumax3-convert -numpy
sk_dyn.out/*.ovf). Builds a two-row snapshot montage at matched times and overlays
the core trajectory / |Q| / area.
"""
import sys, glob, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "build" / "windows-msvc" / "python"))
import micromag as mm

nx, ny, dx = 120, 84, 3.5e-9
targets = np.arange(0.0, 0.85, 0.1)          # ns columns

# ---------------- Claude-SD run (same params) --------------------------------
g = mm.StructuredGrid(nx, ny, 1, dx, dx, 1e-9)
mat = mm.Material(); mat.Ms = 6e5; mat.A_exchange = 1.5e-11
mat.K_uniaxial = 0.6e6; mat.easy_axis = mm.Vec3(0, 0, 1); mat.alpha = 0.20
m = mm.neel_skyrmion(g, 32e-9, charge=1, pol=-1)
heff = mm.EffectiveFieldSum()
for f in (mm.DemagField(g), mm.ExchangeField(mm.BoundaryCondition.Neumann),
          mm.UniaxialAnisotropyField(), mm.InterfacialDMIField(3.0e-3),
          mm.ZeemanField(mm.Vec3(0, 0, 0))):
    heff.add(f)
print("CS: relax..."); mm.run(mm.RK4Integrator(dt=5e-13), m, mat, heff, t_total=0.5e-9)
sot = mm.SpinOrbitTorque(J_c=3.5e11, theta_SH=0.25, d_fm=1e-9,
                         sigma=mm.Vec3(0, 1, 0), eta_DL=1.0, eta_FL=0.0)
stt = mm.SpinTorqueSum(); stt.add(sot)
cs = {"t": [], "m": [], "Q": []}
def cb(t, mv):
    cs["t"].append(t * 1e9); cs["m"].append(np.array(mm.to_numpy(mv), copy=True)[0])
    cs["Q"].append(mm.topological_charge_Q(mv))
cb(0.0, m)
print("CS: SOT dynamics...")
mm.run(mm.RK4Integrator(dt=2e-13), m, mat, heff, t_total=0.9e-9, stt=stt, callback=cb, callback_dt=0.1e-9)

# ---------------- mumax3 frames ---------------------------------------------
mfs = sorted(glob.glob(str(HERE / "sk_dyn.out" / "*.npy")))
mx = {"t": [], "m": [], "Q": []}
for i, f in enumerate(mfs):
    a = np.load(f)                     # (3,1,ny,nx)
    m2 = np.moveaxis(a[:, 0], 0, -1)   # (ny,nx,3)
    mx["t"].append(i * 0.1); mx["m"].append(m2)
    dmx = np.gradient(m2, axis=1); dmy = np.gradient(m2, axis=0)
    mx["Q"].append((m2 * np.cross(dmx, dmy)).sum() / (4 * np.pi))

def nearest(tlist, mlist, qlist, tt):
    j = int(np.argmin([abs(x - tt) for x in tlist])); return mlist[j], qlist[j]

def panel(ax, mm2, q, tt, tag):
    mz = mm2[:, :, 2]; a = mm2[:, :, 0]; b = mm2[:, :, 1]
    ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
              extent=[0, nx, 0, ny], aspect="equal")
    st = max(1, nx // 24); ys, xs = np.mgrid[0:ny:st, 0:nx:st]
    ax.quiver(xs + .5, ys + .5, a[::st, ::st], b[::st, ::st], color="#0b0b0b",
              scale=30, width=0.006, pivot="mid")
    ax.set_title(f"{tag}  t={tt:.1f}ns\nQ={q:+.2f}", fontsize=8.5, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

ncol = len(targets)
fig, axes = plt.subplots(2, ncol, figsize=(2.5 * ncol, 5.6))
for c, tt in enumerate(targets):
    mm2, q = nearest(cs["t"], cs["m"], cs["Q"], tt); panel(axes[0, c], mm2, q, tt, "Claude-SD")
    mm2, q = nearest(mx["t"], mx["m"], mx["Q"], tt); panel(axes[1, c], mm2, q, tt, "mumax3")
fig.suptitle("SOT-driven skyrmion — Claude-SD (top) vs mumax3 (bottom), identical parameters "
             "($m_z$ colour, in-plane arrows)", fontsize=13, y=1.01)
fig.tight_layout()
out = HERE / "compare_montage.png"; fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
print("wrote", out)

# ---------------- trajectory / Q / area curves ------------------------------
def track(frames):
    xs, ys, ar = [], [], []
    for mm2 in frames:
        mz = mm2[:, :, 2]; bg = np.sign(np.median(mz)); core = np.sign(mz) != bg
        if core.sum() == 0: xs.append(np.nan); ys.append(np.nan); ar.append(0); continue
        yy, xx = np.where(core); xs.append(xx.mean() * dx * 1e9); ys.append(yy.mean() * dx * 1e9)
        ar.append(core.sum() * (dx * 1e9) ** 2)
    return np.array(xs), np.array(ys), np.array(ar)
csx, csy, csa = track(cs["m"]); mxx, mxy, mxa = track(mx["m"])
fig2, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(csx, csy, "-o", ms=3, label="Claude-SD"); ax[0].plot(mxx, mxy, "-s", ms=3, label="mumax3")
ax[0].set_xlabel("x core (nm)"); ax[0].set_ylabel("y core (nm)"); ax[0].set_title("core trajectory")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(cs["t"], np.abs(cs["Q"]), "-o", ms=3, label="Claude-SD")
ax[1].plot(mx["t"], np.abs(mx["Q"]), "-s", ms=3, label="mumax3")
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("|Q|"); ax[1].set_title("topological charge |Q|")
ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].plot(cs["t"], csa, "-o", ms=3, label="Claude-SD"); ax[2].plot(mx["t"], mxa, "-s", ms=3, label="mumax3")
ax[2].set_xlabel("t (ns)"); ax[2].set_ylabel("core area (nm²)"); ax[2].set_title("area (deformation)")
ax[2].legend(); ax[2].grid(alpha=.3)
fig2.tight_layout()
out2 = HERE / "compare_curves.png"; fig2.savefig(out2, dpi=140, bbox_inches="tight", facecolor="white")
print("wrote", out2)
print(f"\nCS  Q: {cs['Q'][0]:+.2f} -> {cs['Q'][-1]:+.2f}   mumax3 Q: {mx['Q'][0]:+.2f} -> {mx['Q'][-1]:+.2f}")
print("(opposite Q sign = opposite DMI/skyrmion chirality convention -> mirrored skyrmion-Hall direction)")
