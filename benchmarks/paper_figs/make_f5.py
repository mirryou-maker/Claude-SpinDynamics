"""F5 — solver capability matrix (standalone; throughput lives in F3)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

OUT = pathlib.Path(__file__).parent
INK, SEC, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
CAP_NONE, CAP_PART, CAP_FULL = "#eef2f7", "#9ec5f4", "#2a78d6"   # single-hue ordinal
plt.rcParams.update({"text.color": INK, "axes.labelcolor": SEC,
                     "xtick.color": SEC, "ytick.color": SEC, "font.size": 10})

solvers = ["Claude-SD", "mumax3", "MuMax-CO", "mumax+", "OOMMF"]
feats = ["float64", "float32", "cuFFT", "VkFFT", "native\nSOT", "DMI",
         "RKKY", "magneto-\nelastic", "per-cell", "auto-\nintegr.", "Python", "mx3"]
M = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],          # Claude-SD
    [0,1,1,0,0.5,1,0,0,0.5,0,0,1],      # mumax3
    [0,1,1,0,0.5,1,0,0,0.5,0,0,1],      # MuMax-CO
    [0,1,1,0,0.5,1,0.5,1,0.5,0,1,0],    # mumax+
    [1,0,0,0,0,0.5,0,0,0.5,0,0.5,0],    # OOMMF
], dtype=float)

fig, ax = plt.subplots(figsize=(10.5, 4.0))
cmap = ListedColormap([CAP_NONE, CAP_PART, CAP_FULL])
norm = BoundaryNorm([-0.25, 0.25, 0.75, 1.25], cmap.N)
ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(solvers), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.tick_params(which="minor", length=0)
ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(solvers)))
ax.set_yticklabels(solvers, fontsize=10.5, fontweight="bold")
for i in range(len(solvers)):
    for j in range(len(feats)):
        v = M[i, j]
        if v == 1:
            ax.text(j, i, "✓", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        elif v == 0.5:
            ax.text(j, i, "◐", ha="center", va="center", color=SEC, fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("Solver capability matrix", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
ax.legend(handles=[Patch(fc=CAP_FULL, label="full  ✓"),
                   Patch(fc=CAP_PART, label="partial  ◐"),
                   Patch(fc=CAP_NONE, ec=GRID, label="none")],
          loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=9.5)

fig.tight_layout()
fig.savefig(OUT / "fig_f5_landscape.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote fig_f5_landscape.png (capability matrix, standalone)")
