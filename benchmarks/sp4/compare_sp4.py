"""Assemble the SP#4 5-way comparison: load each config's trajectory, tabulate
<m>(1ns) / t_switch / RMS-vs-OOMMF, and plot <mx,my,mz>(t)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from parse import load_mumax_table, load_oommf_odt, resample  # noqa: E402

HERE = os.path.dirname(__file__)

CONFIGS = [
    ("mumax3",      os.path.join(HERE, "mumax.out", "table.txt"),       "table"),
    ("MuMax-CO",    os.path.join(HERE, "mumaxCO.out", "table.txt"),     "table"),
    ("OOMMF",       os.path.join(HERE, "sp4_oommf.odt"),                "odt"),
    ("ours-double", os.path.join(HERE, "ours_double.out", "m.txt"),     "table"),
    ("ours-f32",    os.path.join(HERE, "ours_f32.out", "m.txt"),        "table"),
]

trajs = {}
for name, path, kind in CONFIGS:
    if not os.path.exists(path):
        print(f"  (skip {name}: {path} missing)")
        continue
    trajs[name] = load_oommf_odt(path, stage=1) if kind == "odt" else load_mumax_table(path)

tg = np.linspace(0, 1e-9, 200)
ref = trajs.get("OOMMF")

print(f"\n{'config':<13}{'mx(1ns)':>9}{'my(1ns)':>9}{'mz(1ns)':>9}"
      f"{'t_sw(ps)':>9}{'mx_RMS':>9}{'my_RMS':>9}")
for name, _, _ in CONFIGS:
    tr = trajs.get(name)
    if tr is None:
        continue
    mxg = resample(tr[0], tr[1], tg)
    myg = resample(tr[0], tr[2], tg)
    mzg = resample(tr[0], tr[3], tg)
    cross = np.where(np.diff(np.sign(mxg)))[0]
    tsw = tg[cross[0]] * 1e12 if len(cross) else float("nan")
    if ref is not None:
        rmx = np.sqrt(np.mean((mxg - resample(ref[0], ref[1], tg)) ** 2))
        rmy = np.sqrt(np.mean((myg - resample(ref[0], ref[2], tg)) ** 2))
    else:
        rmx = rmy = float("nan")
    print(f"{name:<13}{mxg[-1]:>9.4f}{myg[-1]:>9.4f}{mzg[-1]:>9.4f}"
          f"{tsw:>9.1f}{rmx:>9.4f}{rmy:>9.4f}")

# plot
fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for name, _, _ in CONFIGS:
    tr = trajs.get(name)
    if tr is None:
        continue
    for k, comp in enumerate((1, 2, 3)):
        ax[k].plot(tr[0] * 1e9, tr[comp], label=name, lw=1.2)
for k, lbl in enumerate(("<mx>", "<my>", "<mz>")):
    ax[k].set_ylabel(lbl)
    ax[k].grid(alpha=0.3)
ax[0].legend(fontsize=8, ncol=5, loc="upper right")
ax[2].set_xlabel("t (ns)")
ax[0].set_title("muMAG Standard Problem 4 (field A, 250x64x1) - 5-way")
fig.tight_layout()
out = os.path.join(HERE, "sp4_trajectories.png")
fig.savefig(out, dpi=110)
print(f"\nplot -> {out}")
