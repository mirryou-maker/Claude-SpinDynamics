"""Run every example notebook 01-40 against the CURRENT build and record
status, wall time, and the tail of stdout/stderr.

Usage:
  D:/anaconda3/python.exe notebooks/run_all_examples.py            # all 01-40
  D:/anaconda3/python.exe notebooks/run_all_examples.py 10 11 12   # subset by prefix

Writes run_all_summary.json + run_all_summary.md in the notebooks folder.
"""
import sys, os, pathlib, subprocess, time, json, re

try:                                     # this runner echoes child output (µ, em-dash);
    sys.stdout.reconfigure(encoding="utf-8")   # its own stdout is cp949 on Korean Windows
except Exception:
    pass

HERE = pathlib.Path(__file__).parent
PY = r"D:/anaconda3/python.exe"
PER_NB_TIMEOUT = 1500   # s
# Korean Windows console codec is cp949; notebooks print UTF-8 glyphs (µ, em-dash).
# Force UTF-8 on the child's stdout so prints never crash the run.
# MPLBACKEND=Agg: notebooks call plt.show(), which BLOCKS on an interactive
# backend in headless subprocess runs; Agg makes it a no-op (figures still save).
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE",
       "MPLBACKEND": "Agg"}

# every NN_*.py / NNx_*.py with NN in 01..40 (the runnable example notebooks;
# 41-50 are the separate mumax3 cross-comparison suite handled by
# rerun_all_nbs.py). Matches lettered companions too (e.g. 23b_, 26b_) --
# a plain "[0-3][0-9]_*.py" glob would both miss NB40 and skip lettered
# files, since the character after the two digits isn't "_" for those.
def discover():
    out = []
    for p in sorted(HERE.glob("*.py")):
        m = re.match(r"(\d+)[a-z]?_", p.name)
        if m and 1 <= int(m.group(1)) <= 40:
            out.append(p.name)
    return out


def run(nb):
    t0 = time.perf_counter()
    try:
        r = subprocess.run([PY, "-u", str(HERE / nb)],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=ENV,
                           timeout=PER_NB_TIMEOUT, cwd=str(HERE))
        ok = r.returncode == 0
        out = (r.stdout or "") + "\n===STDERR===\n" + (r.stderr or "")
    except subprocess.TimeoutExpired:
        ok, out = False, "TIMEOUT"
    return ok, time.perf_counter() - t0, out


if __name__ == "__main__":
    args = sys.argv[1:]
    nbs = [nb for nb in discover() if (not args or any(nb.startswith(a) for a in args))]
    summary = []
    print("=" * 72)
    print(f"Running {len(nbs)} example notebooks against the current build")
    print("=" * 72)
    for nb in nbs:
        print(f"\n>>> {nb}", flush=True)
        ok, wall, out = run(nb)
        tail = [l for l in out.splitlines() if l.strip()][-15:]
        for l in tail:
            print("    " + l, flush=True)
        print(f"    -> {'OK' if ok else 'FAILED'}  {wall:.1f}s", flush=True)
        summary.append({"nb": nb, "ok": ok, "wall_s": round(wall, 1), "tail": tail})
        (HERE / "run_all_summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# Example notebook run summary (current build)\n",
          "| Notebook | status | wall (s) |", "|---|---|---|"]
    for s in summary:
        md.append(f"| {s['nb']} | {'OK' if s['ok'] else 'FAILED'} | {s['wall_s']} |")
    n_ok = sum(1 for s in summary if s["ok"])
    md.append(f"\n**{n_ok}/{len(summary)} notebooks passed.**")
    (HERE / "run_all_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{n_ok}/{len(summary)} OK.  Wrote run_all_summary.{{json,md}}")
