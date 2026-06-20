"""Parsers for mumax3 (table.txt), our runner (table), and OOMMF (.odt),
returning a uniform (t, mx, my, mz) trajectory as numpy arrays."""
import re
import numpy as np


def load_mumax_table(path):
    """mumax3 / our-runner table: tab-separated, '# t mx my mz [...]'."""
    t, mx, my, mz = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            t.append(float(c[0])); mx.append(float(c[1]))
            my.append(float(c[2])); mz.append(float(c[3]))
    return (np.array(t), np.array(mx), np.array(my), np.array(mz))


def load_oommf_odt(path, stage=None):
    """OOMMF .odt: locate Oxs_*Driver::mx/my/mz and Simulation time columns by
    header, optionally filter to one Stage, and re-zero time to that stage."""
    cols = None
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("# Columns:"):
                # column titles are brace-delimited or bare words
                cols = re.findall(r"\{[^}]*\}|\S+", line[len("# Columns:"):])
                cols = [c.strip("{} ") for c in cols]
            elif not line.startswith("#") and line.strip():
                rows.append([float(x) for x in line.split()])
    if cols is None:
        raise ValueError("no Columns header in " + path)
    data = np.array(rows)

    def idx(substr):
        for i, c in enumerate(cols):
            if c.endswith(substr):
                return i
        raise KeyError(substr)

    imx, imy, imz = idx("::mx"), idx("::my"), idx("::mz")
    it = idx("Simulation time")
    istage = next((i for i, c in enumerate(cols) if c.endswith("::Stage")
                   and "iteration" not in c), None)
    if stage is not None and istage is not None:
        sel = data[:, istage] == stage
        data = data[sel]
    t = data[:, it]
    t = t - t[0]            # re-zero to start of the (selected) trajectory
    return (t, data[:, imx], data[:, imy], data[:, imz])


def resample(t, y, tgrid):
    return np.interp(tgrid, t, y)


def metrics(traj, ref, n=200, t_end=1e-9):
    """Per-component RMS/max error of traj vs ref on a common time grid,
    plus mx(t_end) and t_switch (first ref mx sign change)."""
    tg = np.linspace(0, t_end, n)
    out = {}
    rmx = resample(ref[0], ref[1], tg)
    for name, tr in [("traj", traj), ("ref", ref)]:
        pass
    txx = {}
    for lbl, comp in (("mx", 1), ("my", 2), ("mz", 3)):
        a = resample(traj[0], traj[comp], tg)
        b = resample(ref[0], ref[comp], tg)
        txx[lbl + "_end"] = float(a[-1])
        txx[lbl + "_rms"] = float(np.sqrt(np.mean((a - b) ** 2)))
        txx[lbl + "_max"] = float(np.max(np.abs(a - b)))
    # t_switch from this trajectory's mx crossing zero
    mxg = resample(traj[0], traj[1], tg)
    cross = np.where(np.diff(np.sign(mxg)))[0]
    txx["t_switch"] = float(tg[cross[0]]) if len(cross) else float("nan")
    return txx


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    tr = load_oommf_odt(p, stage=1) if p.endswith(".odt") else load_mumax_table(p)
    print(f"{p}: rows={len(tr[0])}  mx(end)={tr[1][-1]:+.4f} my={tr[2][-1]:+.4f} mz={tr[3][-1]:+.4f}")
