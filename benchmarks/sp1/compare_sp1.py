"""SP#1 long-axis hysteresis: Claude-SD (f64/f32) vs mumax3."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_sd(p):
    H, mx = [], []
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        r = l.split("\t"); H.append(float(r[0])); mx.append(float(r[1]))
    return np.array(H), np.array(mx)


def load_mm(p):
    hdr = open(p).readline().strip("# \n").split("\t")
    imx = next(i for i, h in enumerate(hdr) if h.startswith("mx"))
    ibx = next(i for i, h in enumerate(hdr) if "B_extx" in h)
    d = np.array([[float(x) for x in l.split("\t")] for l in open(p)
                  if not l.startswith("#") and l.strip()])
    return d[:, ibx] * 1e3, d[:, imx]


def metrics(H, mx):
    n = len(H) // 2
    Hd, md = H[:n], mx[:n]               # descending branch
    hc = float("nan")
    for i in range(len(Hd) - 1):
        if (md[i] > 0) != (md[i + 1] > 0):
            hc = Hd[i] + (Hd[i + 1] - Hd[i]) * (0 - md[i]) / (md[i + 1] - md[i]); break
    rem = float(np.interp(0, Hd[::-1], md[::-1]))
    return hc, rem


series = []
for tag, f in [("Claude-SD f64", "loop_double.txt"), ("Claude-SD f32", "loop_float32.txt")]:
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        series.append((tag, *load_sd(p)))
mmp = os.path.join(HERE, "sp1.out", "table.txt")
if os.path.exists(mmp):
    series.append(("mumax3", *load_mm(mmp)))

print(f"{'config':<16}{'Hc (mT)':>10}{'remanence <mx>':>16}")
for tag, H, mx in series:
    hc, rem = metrics(H, mx)
    print(f"{tag:<16}{hc:>10.2f}{rem:>16.4f}")

fig, ax = plt.subplots(figsize=(6.5, 5))
for tag, H, mx in series:
    ax.plot(H, mx, "o-", ms=3, lw=1.2, label=tag)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("B_ext along long axis (mT)"); ax.set_ylabel("<mx>")
ax.set_title("muMAG SP#1 long-axis hysteresis (100x50x1, 20 nm cells)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
out = os.path.join(HERE, "sp1_loop.png")
fig.savefig(out, dpi=110)
print("plot ->", out)
