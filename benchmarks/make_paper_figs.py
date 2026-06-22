"""Additional paper figures (F5-F7) from measured data in all_solvers.json + NB43.

F6 — f32 vs f64 Tensor-Core speedup (within Claude-SD, cuFFT) per scenario.
F7 — T>0 SOT-driven canting <mz>(J): CS builds (identical) vs mumax+.
F5 — integrator decision tree (recommend_integrator rules) schematic.

(F3 SP#4 <m>(t) trajectories omitted: NB41 did not persist the per-step mx log.)
Run: py -3.13 benchmarks/make_paper_figs.py
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
OUT = HERE / "results"
data = json.loads((OUT / "all_solvers.json").read_text())


def sid(r): return r["scenario"].split("_")[0]


# ---- F6: f32 vs f64 speedup (cuFFT, Claude-SD) ----------------------------
tp = [r for r in data if r["metric"] == "throughput" and r["solver"] == "claude-sd"]
f64 = {sid(r): r["ms_step"] for r in tp if r["build"] == "cuFFT_f64"}
f32 = {sid(r): r["ms_step"] for r in tp if r["build"] == "cuFFT_f32"}
cells = {sid(r): r["cells"] for r in tp}
scen = sorted(set(f64) & set(f32), key=lambda s: cells[s])
spd = [f64[s] / f32[s] for s in scen]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(range(len(scen)), spd, color="#9467bd", edgecolor="black")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(range(len(scen)))
ax.set_xticklabels([f"{s}\n{cells[s]//1000}K" for s in scen])
ax.set_ylabel("cuFFT  f64 / f32  step-time ratio")
ax.set_title("F6 — Blackwell Tensor-Core f32 speedup (Claude-SD, cuFFT)")
for b, v in zip(bars, spd):
    ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.1f}x", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_f6_f32_f64.png", dpi=150); plt.close(fig)
print("wrote fig_f6_f32_f64.png")

# ---- F7: T>0 SOT canting <mz>(J) ------------------------------------------
J = np.array([1, 2, 4, 6])
cs_mz = np.array([0.99, 0.97, -0.18, -0.11])    # cuFFT_f64/f32/VkFFT_f32 identical
mp_mz = np.array([0.20, 0.20, 0.20, 0.20])      # mumax+ (Slonczewski-SOT, J-indep)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(J, cs_mz, "o-", color="#1f77b4", lw=2, ms=7,
        label="Claude-SD (cuFFT f64 = f32 = VkFFT)")
ax.plot(J, mp_mz, "s--", color="#d62728", lw=2, ms=7,
        label="mumax+ (Slonczewski-SOT emul.)")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel(r"SOT current $J$ ($\times10^{12}$ A/m$^2$)")
ax.set_ylabel(r"thermal-averaged $\langle m_z\rangle$  (T=300 K)")
ax.set_title("F7 — SOT-driven canting at finite T (NB43, macrospin)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(3.5, 0.6, "CS: all builds\nbit-identical", fontsize=8, color="#1f77b4")
fig.tight_layout(); fig.savefig(OUT / "fig_f7_sot_canting.png", dpi=150); plt.close(fig)
print("wrote fig_f7_sot_canting.png")

# ---- F5: integrator decision tree (schematic) -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")
def box(x, y, t, c="#e8e8e8"):
    ax.text(x, y, t, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc=c, ec="black"))
box(0.5, 0.92, "scenario (mat, T_K, goal, alpha, dt)", "#cfe8ff")
box(0.5, 0.74, "T_K > 0 ?")
ax.annotate("", (0.5,0.80),(0.5,0.88), arrowprops=dict(arrowstyle="->"))
box(0.16, 0.56, "Heun\n(mumax SetSolver 2)\nSLLG mandatory", "#d6f5d6")
ax.annotate("yes", (0.16,0.62),(0.45,0.72), arrowprops=dict(arrowstyle="->"), fontsize=8)
box(0.62, 0.56, "goal = relax ?")
ax.annotate("no", (0.62,0.62),(0.55,0.72), arrowprops=dict(arrowstyle="->"), fontsize=8)
box(0.40, 0.38, "Heun\n(2x cheaper,\npath-independent)", "#d6f5d6")
ax.annotate("yes", (0.40,0.44),(0.58,0.54), arrowprops=dict(arrowstyle="->"), fontsize=8)
box(0.80, 0.38, "alpha >= 0.3 ?")
ax.annotate("no", (0.80,0.44),(0.66,0.54), arrowprops=dict(arrowstyle="->"), fontsize=8)
box(0.66, 0.18, "Heun\n(overdamped)", "#d6f5d6")
ax.annotate("yes", (0.66,0.24),(0.78,0.34), arrowprops=dict(arrowstyle="->"), fontsize=8)
box(0.93, 0.18, "alpha < 0.05 ?\n-> RK45-DP (5)\nelse phase-err\nthreshold", "#fff0c2")
ax.annotate("no", (0.93,0.26),(0.84,0.34), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax.set_title("F5 — recommend_integrator() decision tree  (mumax3 SetSolver in ())", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig_f5_decision_tree.png", dpi=150); plt.close(fig)
print("wrote fig_f5_decision_tree.png")
