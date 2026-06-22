"""SP#2 cross-solver: run mumax3 over the d/lex sweep (same grids as CS) and
compare the remanent <mx>/Ms to the Claude-SD curve. Overlays both on fig_sp2.

Run: py -3.13 benchmarks/sp2/sp2_crosssolver.py
"""
import os, math, subprocess, shutil, pathlib, time
import numpy as np

HERE = pathlib.Path(__file__).parent
EXE = r"D:/Mumax3/mumax3.exe"
MU0 = 4e-7 * math.pi
Ms, A = 8e5, 13e-12
LEX = math.sqrt(2 * A / (MU0 * Ms * Ms))
CPL, MAXCELLS = 1.5, 40000

TPL = """SetGridSize({nx}, {ny}, {nz})
SetCellSize({dx:.6e}, {dy:.6e}, {dz:.6e})
Msat = 8e5
Aex  = 13e-12
alpha = 0.5
m = uniform(1, 1, 1)
Bsat := 0.5 * mu0 * 8e5
u    := 1.0 / sqrt(3.0)
NPTS := 21
for i := 0; i < NPTS; i++ {{
    amp := Bsat - 2.0*Bsat*i/(NPTS-1)
    B_ext = vector(amp*u, amp*u, amp*u)
    relax()
    tablesave()
}}
"""


def make_grid(dl):
    d = dl * LEX; L = 5*d; t = 0.1*d; dx = LEX/CPL
    nx, ny, nz = max(2, round(L/dx)), max(2, round(d/dx)), max(1, round(t/dx))
    while nx*ny*nz > MAXCELLS:
        nx = max(2, nx//2); ny = max(2, ny//2); nz = max(1, nz//2)
    return nx, ny, nz, L/nx, d/ny, t/nz


def run_mumax3(dl):
    nx, ny, nz, dx, dy, dz = make_grid(dl)
    mx3 = HERE / f"_sp2x_{dl}.mx3"
    out = HERE / f"_sp2x_{dl}.out"
    mx3.write_text(TPL.format(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz))
    if out.exists(): shutil.rmtree(out, ignore_errors=True)
    t0 = time.perf_counter()
    p = subprocess.run([EXE, "-f", "-o", str(out), str(mx3)],
                       capture_output=True, text=True, timeout=600)
    wall = time.perf_counter() - t0
    tbl = out / "table.txt"
    mx = my = mz = float("nan")
    if tbl.exists():
        rows = [l.split() for l in tbl.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        if len(rows) >= 11:           # H=0 is the 11th relax point (i=10)
            r = rows[10]
            mx, my, mz = float(r[1]), float(r[2]), float(r[3])
    mx3.unlink(missing_ok=True); shutil.rmtree(out, ignore_errors=True)
    return (nx, ny, nz), mx, my, mz, wall


if __name__ == "__main__":
    cs = np.loadtxt(HERE / "sp2_remanence_double.csv", delimiter=",", skiprows=1)
    cs_d = {round(r[0], 2): r[1] for r in cs}   # d/lex -> CS <mx>/Ms
    sweep = [2.0, 5.0, 10.0, 20.0]
    print(f"SP#2 cross-solver remanence <mx>/Ms (lex={LEX*1e9:.2f} nm)")
    print(f"{'d/lex':>6}{'grid':>12}{'CS_mx':>9}{'mumax3_mx':>11}{'wall_s':>8}")
    rows = []
    for dl in sweep:
        g, mx, my, mz, wall = run_mumax3(dl)
        csmx = cs_d.get(round(dl, 2), float("nan"))
        rows.append((dl, csmx, mx))
        print(f"{dl:>6.1f}{f'{g[0]}x{g[1]}x{g[2]}':>12}{csmx:>9.3f}{mx:>11.3f}{wall:>8.1f}", flush=True)

    # Overlay figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.semilogx(cs[:, 0], cs[:, 1], "o-", color="#1f77b4", label="Claude-SD f64")
    d_m = [r[0] for r in rows]; m_m = [r[2] for r in rows]
    ax.semilogx(d_m, m_m, "s", color="#d62728", ms=9, label="mumax3 relax()")
    ax.set_xlabel(r"$d/\ell_{ex}$"); ax.set_ylabel(r"remanent $\langle m_x\rangle/M_s$")
    ax.set_title("SP#2 remanence: Claude-SD vs mumax3 (same grids, relax protocol)")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(HERE / "fig_sp2_crosssolver.png", dpi=150)
    print("wrote fig_sp2_crosssolver.png")
