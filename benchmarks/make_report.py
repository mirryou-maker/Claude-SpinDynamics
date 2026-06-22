"""Render paper tables + figures from benchmarks/results/all_solvers.json.

Produces (in benchmarks/results/):
  - throughput_table.md   : ms/step (±IQR) pivot, scenario × solver/build
  - fig_throughput.png     : ms/eval vs cells, one line per solver/build
  - fig_scenario_bars.png  : per-scenario ms/step bars grouped by solver/build

Run: py -3.13 benchmarks/make_report.py
"""
import json, pathlib, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
STORE = HERE / "results" / "all_solvers.json"
OUT = HERE / "results"

EVALS = {"RK4": 4, "RK45-DP": 6, "Heun": 2, "relax": 1}


def sid(rec):
    """Scenario id (S1/S2/...) — normalises CS 'S2_throughput' and mumax
    'S2_s2_nonpow2_rk4' to the same row key."""
    return rec["scenario"].split("_")[0]


def load():
    return json.loads(STORE.read_text()) if STORE.exists() else []


def col_label(rec):
    s = rec["solver"]
    if s == "claude-sd":
        return f"CS {rec['build']}"
    return s  # mumax3 / mumax-co / mumax+


def throughput_table(data):
    tp = [r for r in data if r["metric"] == "throughput" and r.get("ms_step")]
    if not tp:
        return "No throughput data.\n"
    scens = sorted({sid(r) for r in tp}, key=lambda s: next(r["cells"] for r in tp if sid(r) == s))
    cols = sorted({col_label(r) for r in tp})
    cells = {s: next(r["cells"] for r in tp if sid(r) == s) for s in scens}
    grid = {(sid(r), col_label(r)): r for r in tp}

    lines = ["# Throughput — ms/step (±IQR), RK4 f32/f64\n",
             "| Scenario | cells | " + " | ".join(cols) + " |",
             "|---|---|" + "|".join(["---"] * len(cols)) + "|"]
    for s in scens:
        row = [s, f"{cells[s]:,}"]
        for c in cols:
            r = grid.get((s, c))
            row.append(f"{r['ms_step']:.3f}±{r.get('ms_step_iqr',0):.3f}" if r else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def throughput_table_eval(data):
    """ms per FIELD-EVAL — the fair cross-solver metric (normalises RK4=4 evals
    vs mumax+ DOPRI5=6 evals vs Heun=2)."""
    tp = [r for r in data if r["metric"] == "throughput" and r.get("ms_step")]
    if not tp:
        return ""
    scens = sorted({sid(r) for r in tp}, key=lambda s: next(r["cells"] for r in tp if sid(r) == s))
    cols = sorted({col_label(r) for r in tp})
    cells = {s: next(r["cells"] for r in tp if sid(r) == s) for s in scens}
    integ = {col_label(r): r["integrator"] for r in tp}

    def msev(r):
        return r["ms_step"] / EVALS.get(r["integrator"], 1)
    grid = {(sid(r), col_label(r)): msev(r) for r in tp}

    hdr = [f"{c}<br>({integ.get(c,'?')})" for c in cols]
    lines = ["\n# Throughput — ms per field-eval (fair metric)\n",
             "| Scenario | cells | " + " | ".join(hdr) + " |",
             "|---|---|" + "|".join(["---"] * len(cols)) + "|"]
    for s in scens:
        row = [s, f"{cells[s]:,}"]
        for c in cols:
            v = grid.get((s, c))
            row.append(f"{v:.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    # winner per scenario
    lines.append("\n**Fastest per scenario (ms/eval):**")
    for s in scens:
        best = min(((c, grid[(s, c)]) for c in cols if (s, c) in grid), key=lambda kv: kv[1])
        lines.append(f"- {s} ({cells[s]:,} cells): **{best[0]}** = {best[1]:.3f} ms/eval")
    return "\n".join(lines) + "\n"


def accuracy_table(data):
    """Cross-solver accuracy observables (from the notebook re-run, latest build)."""
    acc = [r for r in data if r["metric"] == "accuracy" and r.get("value") is not None]
    if not acc:
        return ""
    scens = sorted({r["scenario"] for r in acc})
    cols = sorted({col_label(r) for r in acc})
    grid = {(r["scenario"], col_label(r)): r for r in acc}
    lines = ["\n# Accuracy — cross-solver observables (latest build)\n",
             "| Observable | " + " | ".join(cols) + " | µMAG/analytic ref |",
             "|---|" + "|".join(["---"] * (len(cols) + 1)) + "|"]
    refs = {"SP4_accuracy": "mx(1ns) = -0.9862", "SP1_Lc": "L_c ~ 116 nm",
            "SP3_Hsw": "H_sw ~ -20 mT", "FMR_freq": "f = 1.4006 GHz",
            "DW_v_2e12": "—"}
    for s in scens:
        row = [s]
        for c in cols:
            r = grid.get((s, c))
            row.append(f"{r['value']:.4g}" if r else "—")
        row.append(refs.get(s, "—"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def fig_throughput(data):
    tp = [r for r in data if r["metric"] == "throughput" and r.get("ms_step")]
    if not tp:
        return
    series = collections.defaultdict(list)
    for r in tp:
        evals = EVALS.get(r["integrator"], 1)
        series[col_label(r)].append((r["cells"], r["ms_step"] / evals))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    markers = "osd^v><p*"
    for i, (lab, pts) in enumerate(sorted(series.items())):
        pts = sorted(pts)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.loglog(xs, ys, marker=markers[i % len(markers)], label=lab)
    ax.set_xlabel("Cell count"); ax.set_ylabel("ms per field-eval")
    ax.set_title("Throughput vs problem size (RTX 5060 Ti)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_throughput.png", dpi=150); plt.close(fig)
    print("wrote", OUT / "fig_throughput.png")


def fig_scenario_bars(data):
    tp = [r for r in data if r["metric"] == "throughput" and r.get("ms_step")]
    if not tp:
        return
    scens = sorted({sid(r) for r in tp}, key=lambda s: next(r["cells"] for r in tp if sid(r) == s))
    cols = sorted({col_label(r) for r in tp})
    grid = {(sid(r), col_label(r)): r["ms_step"] for r in tp}
    x = np.arange(len(scens)); w = 0.8 / max(1, len(cols))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(cols):
        ys = [grid.get((s, c), np.nan) for s in scens]
        ax.bar(x + i * w, ys, w, label=c)
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - w/2); ax.set_xticklabels(scens)
    ax.set_ylabel("ms/step (log)")
    ax.set_title("Per-scenario step time by solver/build")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_scenario_bars.png", dpi=150); plt.close(fig)
    print("wrote", OUT / "fig_scenario_bars.png")


if __name__ == "__main__":
    data = load()
    print(f"loaded {len(data)} records")
    md = throughput_table(data) + "\n" + throughput_table_eval(data) + "\n" + accuracy_table(data)
    (OUT / "throughput_table.md").write_text(md, encoding="utf-8")
    print("wrote", OUT / "throughput_table.md")
    # Console may be cp949 (Korean Windows); print ASCII-safe preview only.
    try:
        sys_stdout_ok = True
        print("\n" + md)
    except UnicodeEncodeError:
        print("\n(table written to file; console encoding cannot show it)")
    fig_throughput(data)
    fig_scenario_bars(data)
