"""F1 — AI-agent development pipeline + Claude-SD architecture (schematic)."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = pathlib.Path(__file__).parent
fig, (axT, axB) = plt.subplots(2, 1, figsize=(10, 8),
                               gridspec_kw={"height_ratios": [1.1, 1]})

# ---- Top: development loop ------------------------------------------------
axT.set_xlim(0, 10); axT.set_ylim(0, 4); axT.axis("off")
axT.set_title("(a) AI-agent development & verification loop", fontsize=12, loc="left")


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec="black", lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.4, color="#333"))


box(axT, 0.2, 1.5, 1.9, 1.0, "CLAUDE.md\nspec & guardrails\n(layers, SI, tests)", "#cfe8ff")
box(axT, 2.6, 1.5, 1.8, 1.0, "Claude Code\nagent", "#ffe0b3")
box(axT, 4.9, 2.6, 1.9, 0.9, "CUDA kernel +\nC++ class", "#e8e8e8")
box(axT, 4.9, 1.45, 1.9, 0.9, "pybind11\nbinding", "#e8e8e8")
box(axT, 4.9, 0.3, 1.9, 0.9, "Catch2 test\n(GPU vs CPU)", "#d6f5d6")
box(axT, 7.3, 1.5, 2.4, 1.0, "4-build CI verify\n(cuFFT/VkFFT x f32/f64)\n345 tests", "#f5d6d6")
arrow(axT, 2.1, 2.0, 2.6, 2.0)
arrow(axT, 4.4, 2.0, 4.9, 3.0)
arrow(axT, 4.4, 2.0, 4.9, 1.9)
arrow(axT, 4.4, 2.0, 4.9, 0.75)
arrow(axT, 6.8, 3.0, 7.3, 2.2)
arrow(axT, 6.8, 1.9, 7.3, 2.0)
arrow(axT, 6.8, 0.75, 7.3, 1.8)
# feedback loop
axT.add_patch(FancyArrowPatch((8.5, 1.5), (3.5, 1.3), arrowstyle="-|>",
              mutation_scale=14, lw=1.2, color="#999",
              connectionstyle="arc3,rad=0.35", ls="--"))
axT.text(6.0, 0.55, "fail -> diagnose -> fix (e.g. DMI stream race)", fontsize=8,
         color="#777", style="italic")

# ---- Bottom: architecture --------------------------------------------------
axB.set_xlim(0, 10); axB.set_ylim(0, 4); axB.axis("off")
axB.set_title("(b) Claude-SD architecture & capabilities", fontsize=12, loc="left")
layers = ["types / grid / field / material",
          "effective fields: Demag(cuFFT/VkFFT), Exchange, Anisotropy,\nDMI, RKKY, Magnetoelastic, Surface  (CPU & GPU, per-cell)",
          "spin torques: Slonczewski STT, SOT, Zhang-Li",
          "integrators: RK4, RK45-DP, Heun (SLLG), Relax/Minimize",
          "Python API  +  mx3 runner  +  auto_integrator()"]
ys = [3.1, 2.25, 1.55, 0.85, 0.15]
hs = [0.55, 0.6, 0.45, 0.45, 0.45]
fcs = ["#eeeeee", "#cfe8ff", "#ffe0b3", "#d6f5d6", "#e0d6f5"]
for y, h, t, fc in zip(ys, hs, layers, fcs):
    box(axB, 0.3, y, 6.4, h, t, fc)
# capability badges
badges = ["f32 + f64", "cuFFT + VkFFT", "native SOT/DMI", "uMAG SP#1-5", "GPLv3, Python+C++"]
for i, bdg in enumerate(badges):
    box(axB, 7.0, 3.2 - i*0.62, 2.7, 0.5, bdg, "#fff6cc")

fig.tight_layout()
fig.savefig(OUT / "fig_f1_pipeline.png", dpi=150, bbox_inches="tight")
print("wrote fig_f1_pipeline.png")
