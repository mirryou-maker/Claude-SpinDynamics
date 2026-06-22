"""Draft paper figures for the cross-solver benchmark PLAN.

These use ALREADY-MEASURED numbers from existing reports (benchmarks/fair_comparison
compare.py ms/eval ratios and BENCH_REPORT_2026-06-21.md) purely to demonstrate the
figure templates (F1, F2). They are illustrative drafts, NOT new measurements — the
final figures will be regenerated from benchmarks/results/all_solvers.json after the run.

Run: py -3.13 benchmarks/plan_figs/make_draft_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# F1 — Scenario speedup (CS-f32 best build vs mumax3), ms/field-eval.
# Source: benchmarks/fair_comparison compare.py embedded summary (RTX 5060 Ti).
# Positive = CS faster; <1 = mumax3 faster. 2D vs 3D colour-coded.
# ---------------------------------------------------------------------------
scenarios = [
    ("S2 SP#4 2D 10K",        38.5, "2D"),
    ("S7 DW-motion 2D 8K",    33.7, "2D"),
    ("S4 SP#4-adaptive 2D",   30.4, "2D"),
    ("S8 precession 2D 10K",   8.8, "2D"),
    ("S1 pow2 3D 65K",         6.3, "3D"),
    ("S6 medium-adaptive 540K",1/2.4, "3D"),
    ("S3 medium 3D 540K",     1/3.2, "3D"),
    ("S5 large 3D 2.5M",      1/3.2, "3D"),
]
labels = [s[0] for s in scenarios]
ratios = np.array([s[1] for s in scenarios])
dims   = [s[2] for s in scenarios]
colors = ["#1f77b4" if d == "2D" else "#d62728" for d in dims]

order = np.argsort(ratios)
labels = [labels[i] for i in order]
ratios = ratios[order]
colors = [colors[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 4.5))
y = np.arange(len(labels))
ax.barh(y, ratios, color=colors, edgecolor="black", linewidth=0.5)
ax.axvline(1.0, color="black", ls="--", lw=1)
ax.set_xscale("log")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Speedup  (CS-f32 ms/field-eval relative to mumax3-f32);  >1 = Claude-SD faster")
ax.set_title("F1 (draft) — Per-scenario speedup, Claude-SD f32 vs mumax3\nRTX 5060 Ti · matched RK4/DOPRI5 · ms-per-eval normalized")
for yi, r in zip(y, ratios):
    ax.text(r * (1.15 if r >= 1 else 0.85), yi, f"{r:.1f}x" if r >= 1 else f"{1/r:.1f}x slower",
            va="center", ha="left" if r >= 1 else "right", fontsize=7)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#1f77b4", label="2D thin film"),
                   Patch(color="#d62728", label="3D bulk")], loc="lower right", fontsize=8)
ax.set_xlim(0.2, 80)
fig.tight_layout()
f1 = os.path.join(OUT, "fig_scenario_speedup.png")
fig.savefig(f1, dpi=150); plt.close(fig)
print("wrote", f1)

# ---------------------------------------------------------------------------
# F2 — Throughput vs cell count crossover (illustrative).
# Source: BENCH_REPORT_2026-06-21 RK4 ms/step (cuFFT_f32, cuFFT_f64, VkFFT_f32)
# and mumax3 f32 Heun reference, normalized to ms/field-eval (RK4=4, Heun=2).
# ---------------------------------------------------------------------------
cells = np.array([10_000, 200_000, 2_500_000], dtype=float)
# ms/step from report → ms/eval
cs_f32   = np.array([0.156, 2.721, 52.30]) / 4.0
cs_f64   = np.array([0.621, 21.08, 275.6]) / 4.0
vkfft_f32= np.array([0.745, 2.591, 53.90]) / 4.0
mumax3   = np.array([0.482, 1.289, 18.46]) / 2.0   # Heun = 2 eval/step

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.loglog(cells, cs_f32,    "o-", color="#1f77b4", label="Claude-SD cuFFT f32")
ax.loglog(cells, vkfft_f32, "s-", color="#2ca02c", label="Claude-SD VkFFT f32")
ax.loglog(cells, cs_f64,    "^-", color="#9467bd", label="Claude-SD cuFFT f64")
ax.loglog(cells, mumax3,    "d-", color="#d62728", label="mumax3 / MuMax-CO f32")
ax.axvspan(1e5, 2e5, color="gray", alpha=0.15, label="crossover ~0.1-0.2M")
ax.set_xlabel("Cell count")
ax.set_ylabel("ms per field-eval")
ax.set_title("F2 (draft) — Throughput vs problem size\nRTX 5060 Ti · ms/eval normalized · illustrative from existing data")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
f2 = os.path.join(OUT, "fig_throughput_crossover.png")
fig.savefig(f2, dpi=150); plt.close(fig)
print("wrote", f2)
