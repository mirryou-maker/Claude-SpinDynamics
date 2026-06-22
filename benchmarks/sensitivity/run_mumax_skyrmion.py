"""P1 cross-code replication: run mumax3 skyrmion relax at the sensitive D points
and report Q (ext_topologicalcharge), N repeats each. Compares against the
Claude-SD run-to-run distribution to show the metastable state is ill-determined
ACROSS codes (not a CS-specific artifact).

Run: py -3.13 benchmarks/sensitivity/run_mumax_skyrmion.py
"""
import os, math, subprocess, shutil, pathlib, json

HERE = pathlib.Path(__file__).parent
EXE = r"D:/Mumax3/mumax3.exe"
A, K = 15e-12, 0.8e6
DC = 4.0 * math.sqrt(A * K) / math.pi
DD = [0.68, 0.79, 0.91]          # the sensitive band
D_LIST = [round(r * DC, 6) for r in DD]
N = int(os.environ.get("MX_N", "6"))

TPL = (HERE / "skyrmion_Q.mx3").read_text(encoding="utf-8")


def run_one(D, tag):
    src = TPL.replace("DIND_PLACEHOLDER", f"{D:.6e}")
    p = HERE / f"_skx_{tag}.mx3"; out = HERE / f"_skx_{tag}.out"
    p.write_text(src, encoding="utf-8")
    if out.exists(): shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([EXE, "-f", "-o", str(out), str(p)], capture_output=True, text=True, timeout=300)
    Q = float("nan")
    tbl = out / "table.txt"
    if tbl.exists():
        lines = tbl.read_text().splitlines()
        hdr = next((l for l in lines if l.startswith("#")), "")
        cols = hdr.lstrip("# ").split("\t")
        qi = next((i for i, c in enumerate(cols) if "topologicalcharge" in c.lower()), None)
        data = [l.split() for l in lines if l.strip() and not l.startswith("#")]
        if data and qi is not None and qi < len(data[-1]):
            Q = float(data[-1][qi])
    p.unlink(missing_ok=True); shutil.rmtree(out, ignore_errors=True)
    return Q


if __name__ == "__main__":
    print(f"mumax3 skyrmion Q cross-check  Dc={DC*1e3:.2f} mJ/m^2  N={N} reps")
    print(f"{'D/Dc':>6}{'D(mJ)':>7}   Q values (run-to-run)")
    res = {}
    for dd, D in zip(DD, D_LIST):
        Qs = [run_one(D, f"{dd}_{i}") for i in range(N)]
        res[f"{D:.4e}"] = Qs
        qstr = " ".join(f"{q:+.3f}" for q in Qs)
        print(f"{dd:>6.2f}{D*1e3:>7.2f}   {qstr}", flush=True)
    (HERE / "p1_mumax3.json").write_text(json.dumps({"DD": DD, "D_list": D_LIST, "Q": res}, indent=2))
    print(f"\nwrote {HERE/'p1_mumax3.json'}")
