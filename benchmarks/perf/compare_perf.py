"""Assemble the GPU performance comparison: parse perf_*.txt, normalise to
ms per field-evaluation (RK4=4, Heun=2) for a fair core-throughput compare,
print a table and plot throughput vs grid size."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# (file, label, evals_per_step)
SRC = [
    ("perf_f64.txt",     "Claude-SD f64 (RK4)", 4),
    ("perf_f32.txt",     "Claude-SD f32 (RK4)", 4),
    ("perf_mumax3.txt",  "mumax3 f32 (Heun)",   2),
    ("perf_mumaxco.txt", "MuMax-CO f32 (Heun)", 2),
]


def _read(path):
    raw = open(os.path.join(HERE, path), "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="ignore")


def parse(path):
    """Return {cells: ms_step} from a perf table; drop non-physical rows."""
    out = {}
    for line in _read(path).splitlines():
        p = line.split()
        if len(p) < 4 or not p[1].isdigit():
            continue
        cells = int(p[1]); ms_step = float(p[2])
        if ms_step > 0.05:                       # drop unmeasurable (launch-bound) rows
            out[cells] = ms_step
    return out


data = {label: (parse(f), ev) for f, label, ev in SRC}

# table — ms/eval at each grid
allcells = sorted({c for d, _ in data.values() for c in d})
print(f"{'cells':>10} " + " ".join(f"{lbl.split(' (')[0]:>20}" for lbl in data))
print(f"{'':>10} " + " ".join(f"{'ms/eval':>20}" for _ in data))
for c in allcells:
    row = f"{c:>10} "
    for lbl, (d, ev) in data.items():
        row += f"{(d[c]/ev):>20.3f}" if c in d else f"{'--':>20}"
    print(row)

# reference: ms/eval rel to mumax3 at the largest common grid
big = max(allcells)
ref = data["mumax3 f32 (Heun)"][0].get(big, None)
if ref:
    refpe = ref / 2
    print(f"\nAt {big} cells, ms per field-eval relative to mumax3:")
    for lbl, (d, ev) in data.items():
        if big in d:
            print(f"  {lbl:<24} {d[big]/ev/refpe:6.2f}x mumax3 "
                  f"({'faster' if d[big]/ev < refpe else 'slower'})")

# plot throughput (Mcell field-evals / s) vs cells
fig, ax = plt.subplots(figsize=(7, 5))
for lbl, (d, ev) in data.items():
    cs = sorted(d)
    thr = [c * ev / (d[c] * 1e-3) / 1e6 for c in cs]   # Mcell-evals/s
    ax.plot(cs, thr, "o-", label=lbl, lw=1.6)
ax.set_xscale("log"); ax.set_xlabel("cells")
ax.set_ylabel("throughput  (M cell-evals / s)")
ax.set_title("GPU micromagnetics throughput (demag-dominated step)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(HERE, "perf_throughput.png")
fig.savefig(out, dpi=110)
print(f"\nplot -> {out}")
