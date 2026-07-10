"""µMAG SP#5 — vortex-core gyration spiral.

Reads core_<tag>.txt (t, xc, yc from run_sp5_claude_sd.py) and plots the
vortex-core trajectory: a spiral converging to the steady gyrating state under
the in-plane Zhang-Li current.

    python benchmarks/sp5/plot_gyration.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

HERE = pathlib.Path(__file__).parent
INK, BLUE, RED = "#0b0b0b", "#2a78d6", "#e34948"

d = np.loadtxt(HERE / "core_double.txt")            # t, xc_nm, yc_nm
t, xc, yc = d[:, 0] * 1e9, d[:, 1], d[:, 2]

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])

# ---- (a) spiral trajectory, coloured by time -------------------------------
axS = fig.add_subplot(gs[:, 0])
norm = plt.Normalize(t.min(), t.max()); cmap = cm.viridis
pts = np.array([xc, yc]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.0)
lc.set_array(t[:-1]); axS.add_collection(lc)
axS.plot(xc[0], yc[0], "o", color="k", ms=7, zorder=5, label="start (t=0)")
axS.plot(xc[-1], yc[-1], "*", color=RED, ms=15, zorder=5, label="steady (t=8 ns)")
axS.set_xlim(xc.min()-1, xc.max()+1); axS.set_ylim(yc.min()-1, yc.max()+1)
axS.set_aspect("equal", "box")
axS.set_xlabel("vortex-core x (nm)", fontsize=11)
axS.set_ylabel("vortex-core y (nm)", fontsize=11)
axS.set_title("(a)  SP#5 vortex-core gyration spiral", fontsize=12, fontweight="bold")
axS.legend(fontsize=9, loc="upper right"); axS.grid(alpha=.3)
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axS, shrink=0.8, pad=0.02)
cb.set_label("time (ns)", fontsize=10)

# ---- (b) x(t), y(t) ---------------------------------------------------------
axX = fig.add_subplot(gs[0, 1]); axX.plot(t, xc, color=BLUE, lw=1.4)
axX.set_ylabel("x$_c$ (nm)", fontsize=10); axX.grid(alpha=.3)
axX.set_title("(b)  core position vs time", fontsize=11, fontweight="bold")
axX.tick_params(labelbottom=False)
axY = fig.add_subplot(gs[1, 1]); axY.plot(t, yc, color=RED, lw=1.4)
axY.set_ylabel("y$_c$ (nm)", fontsize=10); axY.set_xlabel("time (ns)", fontsize=10); axY.grid(alpha=.3)

fig.suptitle("µMAG Standard Problem 5 — Zhang-Li current-driven vortex-core gyration "
             "(100×100×10 nm Permalloy)", fontsize=12.5, y=1.02)
fig.tight_layout()
out = HERE / "sp5_gyration.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", out, f"  ({len(t)} points, steady core = ({xc[-1]:+.2f}, {yc[-1]:+.2f}) nm)")
