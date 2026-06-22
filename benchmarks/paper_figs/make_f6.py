"""F6 — AI-agent development & verification metrics (from git history)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent.parent
d = json.loads((ROOT / "benchmarks/sensitivity/dev_metrics.json").read_text())
OUT = pathlib.Path(__file__).parent

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# (a) test-case growth over time (test-driven development)
ax = axes[0]
g = d["test_case_growth"]
dates = [x["date"][5:] for x in g]; tc = [x["test_cases"] for x in g]
ax.plot(range(len(g)), tc, "o-", color="#2ca02c", lw=2)
ax.set_xticks(range(len(g))); ax.set_xticklabels(dates, rotation=30, fontsize=8)
ax.set_ylabel("Catch2 TEST_CASE count")
ax.set_title("(a) test-driven growth\n7 → %d over %d days" % (d["current_test_cases"]["total_incl_gpu"], d["period"]["days"]))
ax.grid(alpha=0.3)

# (b) code composition
ax = axes[1]
loc = d["loc_by_category"]
labels = ["C++ src\n(.cpp)", "CUDA\n(.cu)", "headers\n(.hpp)", "py bind\n(.cpp)", "Python\n(.py)", "tests\n(.cpp)"]
vals = [loc["src_cpp"]["lines"], loc["src_cu"]["lines"], loc["include_hpp"]["lines"],
        loc["python_bindings_cpp"]["lines"], loc["python_micromag_py"]["lines"], loc["tests_cpp"]["lines"]]
cols = ["#1f77b4", "#17becf", "#9467bd", "#8c564b", "#e377c2", "#2ca02c"]
ax.bar(range(len(vals)), vals, color=cols)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("lines of code")
ax.set_title("(b) code composition\ntest:source ratio = %.2f" % d["totals"]["test_to_source_ratio"])
ax.grid(alpha=0.3, axis="y")

# (c) commits per week + churn
ax = axes[2]
cw = d["commits_per_week"]
weeks = list(cw.keys()); vals = list(cw.values())
ax.bar(range(len(weeks)), vals, color="#ff7f0e")
ax.set_xticks(range(len(weeks))); ax.set_xticklabels([w[5:] for w in weeks], rotation=30, fontsize=8)
ax.set_ylabel("commits / week")
ax.set_title("(c) %d commits, +%.0fk LOC churn" % (d["commits"], d["churn"]["insertions"]/1000))
ax.grid(alpha=0.3, axis="y")

fig.suptitle("F6 — Agent-driven development metrics (Claude-SD)", y=1.03, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_f6_dev_metrics.png", dpi=150, bbox_inches="tight")
print("wrote fig_f6_dev_metrics.png")
