"""F4 — a GPU stream-race in DMI-coupled relaxation, exposed and fixed by
multi-build/cross-code validation.

(a) BEFORE fix: relaxed skyrmion Q vs D/Dc — run-to-run scatter (mean+/-std,
    min-max band) for each CS build; even f64 scatters across topological sectors.
(b) AFTER fix: the same study — Q is now deterministic (std=0) per build.
(c) Run-to-run std, before vs after, at each D/Dc — the headline collapse to 0.
mumax3 (deterministic, both relax() and minimize()) overlaid as the reference.

Reads p1_sensitivity_BEFORE.json (buggy) + p1_sensitivity.json (fixed)
+ p1_mumax3_relax.json / p1_mumax3_minimize.json.
Run: py -3.13 benchmarks/sensitivity/make_f4.py
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
before = json.loads((HERE / "p1_sensitivity_BEFORE.json").read_text())
after = json.loads((HERE / "p1_sensitivity.json").read_text())
mx3 = None
p = HERE / "p1_mumax3_relax.json"
if p.exists():
    mx3 = json.loads(p.read_text())

DD = before["DD"]; D_list = before["D_list"]
builds = list(before["results"].keys())
colors = {"cuFFT_f64": "#1f77b4", "cuFFT_f32": "#ff7f0e", "VkFFT_f32": "#2ca02c"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))


def panel(ax, data, title):
    for b in builds:
        if b not in data["results"]:
            continue
        means, stds, los, his = [], [], [], []
        for D in D_list:
            Qs = np.array(data["results"][b][f"{D:.4e}"]["Q"])
            means.append(Qs.mean()); stds.append(Qs.std())
            los.append(Qs.min()); his.append(Qs.max())
        ax.errorbar(DD, means, yerr=stds, marker="o", capsize=3,
                    color=colors.get(b), label=b, lw=1.8)
        ax.fill_between(DD, los, his, color=colors.get(b), alpha=0.12)
    if mx3:
        mq = [np.mean(mx3["Q"][f"{D:.4e}"]) for D in mx3["D_list"]]
        ax.plot(mx3["DD"], mq, "k*--", ms=11, label="mumax3 relax()")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel(r"$D/D_c$"); ax.set_ylabel("relaxed skyrmion charge $Q$")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(-1.7, 1.7)


panel(axes[0], before, "(a) BEFORE fix: $Q$ scatters run-to-run\n(stream race on $d\\!H_\\mathrm{out}$, even f64)")
panel(axes[1], after, "(b) AFTER fix: $Q$ deterministic\n(DMI set_stream override)")

# (c) std before vs after
ax = axes[2]
x = np.arange(len(DD)); w = 0.35
sb = [np.mean([np.std(before["results"][b][f"{D:.4e}"]["Q"]) for b in builds if b in before["results"]]) for D in D_list]
sa = [np.mean([np.std(after["results"][b][f"{D:.4e}"]["Q"]) for b in builds if b in after["results"]]) for D in D_list]
ax.bar(x - w/2, sb, w, color="#d62728", label="before fix")
ax.bar(x + w/2, sa, w, color="#1f77b4", label="after fix")
ax.set_xticks(x); ax.set_xticklabels([f"{d:.2f}" for d in DD])
ax.set_xlabel(r"$D/D_c$"); ax.set_ylabel("run-to-run std of $Q$ (build-avg)")
ax.set_title("(c) determinism restored")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("F4 — Cross-validation exposes & fixes a GPU stream race in DMI-coupled relaxation "
             "(Co/Pt skyrmion, RTX 5060 Ti)", y=1.03, fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "fig_f4_race_fix.png", dpi=150, bbox_inches="tight")
print("wrote fig_f4_race_fix.png")
