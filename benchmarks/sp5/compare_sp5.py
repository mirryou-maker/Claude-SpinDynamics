"""SP#5 vortex-core trajectory comparison: Claude-SD (f64/f32) vs mumax3."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_sd(p):
    d = [l.split() for l in open(p) if not l.startswith("#") and l.strip()]
    a = np.array([[float(x) for x in r] for r in d])
    return a[:, 0] * 1e9, a[:, 1], a[:, 2]            # t(ns), x(nm), y(nm)


def load_mm(p):
    a = np.array([[float(x) for x in l.split()] for l in open(p)
                  if not l.startswith("#") and l.strip()])
    return a[:, 0] * 1e9, a[:, 4] * 1e9, a[:, 5] * 1e9


series = []
for tag, f in [("Claude-SD f64", "core_double.txt"), ("Claude-SD f32", "core_float32.txt")]:
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        series.append((tag, *load_sd(p)))
mmp = os.path.join(HERE, "sp5.out", "table.txt")
if os.path.exists(mmp):
    series.append(("mumax3", *load_mm(mmp)))

print(f"{'config':<16}{'core(8ns) x,y nm':>22}{'orbit centre x,y':>22}")
for tag, t, x, y in series:
    i = np.argmin(abs(t - 8.0)); k = t >= 6
    print(f"{tag:<16}{f'({x[i]:+.2f}, {y[i]:+.2f})':>22}"
          f"{f'({x[k].mean():+.2f}, {y[k].mean():+.2f})':>22}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for tag, t, x, y in series:
    ax[0].plot(x, y, lw=1.2, label=tag)
    ax[1].plot(t, x, lw=1.2, label=tag + " x")
    ax[1].plot(t, y, lw=1.2, ls="--", label=tag + " y")
ax[0].set_xlabel("core x (nm)"); ax[0].set_ylabel("core y (nm)")
ax[0].set_title("SP#5 vortex-core trajectory"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("core position (nm)")
ax[1].set_title("core x(t), y(t)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
out = os.path.join(HERE, "sp5_core.png")
fig.savefig(out, dpi=110)
print("plot ->", out)
