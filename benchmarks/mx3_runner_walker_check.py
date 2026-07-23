# mx3-runner convention check: run the Zhang-Li Walker-breakdown strip as an
# UNMODIFIED mumax3 script through Claude-SD's .mx3 runner and compare DW
# velocities against native mumax3 (benchmarks/results/walker_mx3_native.json).
# With the runner's mumax3 semantics (Zhang-Li thiaville_u, Dbulk sign map)
# the two codes must agree point by point.
#
# Run: py -3 benchmarks/mx3_runner_walker_check.py
import sys, json, pathlib, tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from micromag_locate import add_micromag_to_path
add_micromag_to_path()
from micromag import mx3

MX3_TEMPLATE = """SetGridSize(200, 10, 1)
SetCellSize(5e-9, 5e-9, 5e-9)
Msat = 860e3
Aex = 13e-12
alpha = 0.05
Xi = 0.5
Pol = 0.5
m = twodomain(-1,0,0, 0,1,0, 1,0,0)
relax()
J = vector({J:.4e}, 0, 0)
tableautosave(1e-11)
run(1e-9)
"""

L = 200 * 5e-9
J_arr = np.array([0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.8]) * 1e12

ref_path = ROOT / "benchmarks" / "results" / "walker_mx3_native.json"
ref = json.loads(ref_path.read_text()) if ref_path.exists() else None

print("J(e12)   v_runner(m/s)   v_mumax3(m/s)   ratio")
rows = {}
tmp = pathlib.Path(tempfile.mkdtemp(prefix="mx3walker_"))
for J in J_arr:
    scr = tmp / f"wk_{J:.0e}.mx3"
    scr.write_text(MX3_TEMPLATE.format(J=J))
    out = tmp / f"wk_{J:.0e}_out"
    eng = mx3.run_mx3(str(scr), outdir=str(out))
    tbl = np.loadtxt(eng.table_path)
    t, mx_avg = tbl[:, 0], tbl[:, 1]
    sel = t > 0.5e-9
    # DW position from <mx>: x_w = L (1 - <mx>) / 2 -> v = -(L/2) d<mx>/dt
    v = -(L / 2) * np.polyfit(t[sel], mx_avg[sel], 1)[0]
    rows[f"{J:.3e}"] = v
    if ref:
        vref = ref[f"{J:.3e}"]
        print(f"{J/1e12:5.2f}   {v:12.1f}   {vref:12.1f}   {v/vref:7.3f}")
    else:
        print(f"{J/1e12:5.2f}   {v:12.1f}")

out_json = ROOT / "benchmarks" / "results" / "walker_mx3_runner.json"
out_json.write_text(json.dumps(rows, indent=1))
print(f"wrote {out_json}")
if ref:
    ratios = [rows[k] / ref[k] for k in rows if k in ref and abs(ref[k]) > 1]
    print(f"max |ratio-1| over sub-Walker points = "
          f"{max(abs(r - 1) for r in ratios):.3f}")
