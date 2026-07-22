"""F3 — four-solver throughput: crossover curve (3a) + per-scenario bars (3b)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent.parent
data = json.loads((ROOT / "benchmarks/results/all_solvers.json").read_text())
OUT = pathlib.Path(__file__).parent

INK, SEC, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {
    "CS cuFFT_f32": "#2a78d6", "CS VkFFT_f32": "#1baf7a", "mumax3": "#eb6834",
    "MuMax-CO": "#e34948", "mumax+": "#4a3aa7", "CS cuFFT_f64": "#9ec5f4",
}
plt.rcParams.update({"axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": SEC,
                     "xtick.color": SEC, "ytick.color": SEC, "font.size": 10})

# ============ Fig 3a — crossover curve =======================================
EVALS = {"RK4": 4, "RK45-DP": 6, "Heun": 2, "relax": 1}
DISPLAY = {"mumax-co": "MuMax-CO", "mumax3": "mumax3", "mumax+": "mumax+"}
def lab(r): return f"CS {r['build']}" if r["solver"] == "claude-sd" else DISPLAY.get(r["solver"], r["solver"])
series = {}
for r in data:
    if r["metric"] != "throughput" or not r.get("ms_step"): continue
    k = lab(r)
    if k not in SERIES: continue
    series.setdefault(k, []).append((r["cells"], r["ms_step"]/EVALS.get(r["integrator"], 1)))

fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.axvspan(1e5, 5e5, color="#f0efec", zorder=0)
ax.text(3.5e3, 0.030, "◀  Claude-SD f32 fastest", fontsize=9.5, color="#2a78d6",
        ha="left", va="bottom", fontweight="bold")
ax.text(1.4e6, 0.030, "mumax family fastest  ▶", fontsize=9.5, color="#e34948",
        ha="right", va="bottom", fontweight="bold")
ax.text(2.2e5, 30, "crossover\n~0.1–0.5 M cells", fontsize=8.5, color=MUTED,
        ha="center", style="italic")

order = ["CS cuFFT_f32", "MuMax-CO", "mumax3", "CS VkFFT_f32", "mumax+", "CS cuFFT_f64"]
for k in order:
    if k not in series: continue
    pts = sorted(series[k]); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    prim, dashed = (k == "CS cuFFT_f32"), (k == "CS cuFFT_f64")
    ax.loglog(xs, ys, marker="o", ms=7 if prim else 5, lw=2.8 if prim else 1.8,
              ls="--" if dashed else "-", color=SERIES[k], label=k,
              zorder=6 if prim else 3, mec="white", mew=0.9)

# speed-up callout at the small-grid end
ax.annotate("5.3× vs mumax3\n14× vs mumax+", xy=(1e4, 0.042), xytext=(2.3e4, 0.0075),
            fontsize=8.5, color=SEC, ha="left",
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

ax.set_xlabel("cell count", fontsize=11)
ax.set_ylabel("ms / field-evaluation", fontsize=11)
ax.set_title("(a)  Four-solver throughput crossover  (float32, lower = faster)",
             loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)
ax.grid(True, which="major", ls="-", lw=0.6, color=GRID, zorder=1)
ax.grid(True, which="minor", ls=":", lw=0.4, color=GRID, alpha=0.6, zorder=1)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_f3_crossover.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote fig_f3_crossover.png")

# ============ Fig 3b — per-scenario grouped bars (ms/eval, Table 2) ==========
scen = ["SP#4 2-D\n10 K", "pow-2 3-D\n65 K", "medium 3-D\n540 K", "large 3-D\n2.5 M"]
bars = {  # canonical Table 2 values (from all_solvers.json)
    "CS cuFFT_f32": [0.042, 0.155, 2.610, 12.72],
    "CS VkFFT_f32": [0.157, 0.295, 2.777, 13.16],
    "mumax3":       [0.223, 0.305, 1.708, 11.28],
    "MuMax-CO":     [0.211, 0.290, 1.663, 11.13],
    "mumax+":       [0.595, 0.739, 3.302, 18.87],
}
order2 = ["CS cuFFT_f32", "CS VkFFT_f32", "mumax3", "MuMax-CO", "mumax+"]
fig2, ax2 = plt.subplots(figsize=(8.6, 5.0))
x = np.arange(len(scen)); w = 0.16
for i, k in enumerate(order2):
    ax2.bar(x + (i - 2)*w, bars[k], w, label=k, color=SERIES[k],
            edgecolor="white", linewidth=0.6, zorder=3)
# mark the fastest per scenario
allvals = np.array([bars[k] for k in order2])
for j in range(len(scen)):
    imin = allvals[:, j].argmin()
    ax2.text(x[j] + (imin - 2)*w, allvals[imin, j]*0.78, "★", ha="center", va="top",
             color="white", fontsize=9)
ax2.set_yscale("log")
ax2.set_xticks(x); ax2.set_xticklabels(scen, fontsize=9.5)
ax2.set_ylabel("ms / field-evaluation  (log)", fontsize=11)
ax2.set_title("(b)  Per-scenario throughput  (★ = fastest)", loc="left",
              fontsize=12.5, fontweight="bold", color=INK, pad=10)
ax2.grid(True, axis="y", which="major", ls="-", lw=0.6, color=GRID, zorder=0)
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
fig2.tight_layout()
fig2.savefig(OUT / "fig_f3_bars.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote fig_f3_bars.png")
