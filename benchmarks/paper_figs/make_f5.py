"""F5 — solver capability matrix + throughput landscape."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent.parent
data = json.loads((ROOT / "benchmarks/results/all_solvers.json").read_text())
OUT = pathlib.Path(__file__).parent

fig, (axM, axP) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})

# ---- (a) capability matrix --------------------------------------------------
solvers = ["Claude-SD", "mumax3", "MuMax-CO", "mumax+", "OOMMF"]
feats = ["float64", "float32", "cuFFT", "VkFFT", "native SOT", "DMI",
         "RKKY", "magneto-\nelastic", "per-cell", "auto-\nintegrator", "Python", "mx3"]
# 1 = yes, 0.5 = partial/emulated, 0 = no
M = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],          # Claude-SD
    [0,1,1,0,0.5,1,0,0,0.5,0,0,1],      # mumax3 (SOT via Slonczewski)
    [0,1,1,0,0.5,1,0,0,0.5,0,0,1],      # MuMax-CO
    [0,1,1,0,0.5,1,0.5,1,0.5,0,1,0],    # mumax+
    [1,0,0,0,0,0.5,0,0,0.5,0,0.5,0],    # OOMMF (CPU double)
], dtype=float)
im = axM.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
axM.set_xticks(range(len(feats))); axM.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
axM.set_yticks(range(len(solvers))); axM.set_yticklabels(solvers, fontsize=9)
for i in range(len(solvers)):
    for j in range(len(feats)):
        v = M[i, j]; s = "Y" if v == 1 else ("~" if v == 0.5 else "")
        axM.text(j, i, s, ha="center", va="center", fontsize=8)
axM.set_title("(a) Solver capability matrix")

# ---- (b) throughput landscape (ms/eval vs cells) ---------------------------
EVALS = {"RK4": 4, "RK45-DP": 6, "Heun": 2, "relax": 1}
def lab(r):
    return f"CS {r['build']}" if r["solver"] == "claude-sd" else r["solver"]
tp = [r for r in data if r["metric"] == "throughput" and r.get("ms_step")]
series = {}
for r in tp:
    series.setdefault(lab(r), []).append((r["cells"], r["ms_step"]/EVALS.get(r["integrator"], 1)))
markers = "osd^v><p*"
for i, (k, pts) in enumerate(sorted(series.items())):
    pts = sorted(pts); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    axP.loglog(xs, ys, marker=markers[i % len(markers)], label=k, ms=5)
axP.axvspan(1e5, 3e5, color="gray", alpha=0.12)
axP.text(1.4e5, axP.get_ylim()[0]*2, "crossover", fontsize=8, color="#666")
axP.set_xlabel("cell count"); axP.set_ylabel("ms / field-eval")
axP.set_title("(b) throughput landscape")
axP.legend(fontsize=7, ncol=2); axP.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("F5 — Capability & performance landscape", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_f5_landscape.png", dpi=150, bbox_inches="tight")
print("wrote fig_f5_landscape.png")
