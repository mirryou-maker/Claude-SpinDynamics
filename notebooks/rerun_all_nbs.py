"""Re-run ALL comparison notebooks (41-50) against the CURRENT (latest) build.

The notebooks load Claude-SD from build/windows-msvc-cuda*/python via bench_utils,
which now reflect the latest commits (Ku2, ExchangeFieldGPU geometry, etc.).
Re-running guarantees every reported number comes from one consistent build set.

Captures per-NB: exit status, wall time, and the tail of stdout. Writes
rerun_summary.json + rerun_summary.md.

Usage:  D:/anaconda3/python.exe notebooks/rerun_all_nbs.py
"""
import sys, pathlib, subprocess, time, json

HERE = pathlib.Path(__file__).parent
PY = r"D:/anaconda3/python.exe"

ALL_NBS = [
    "41_sp4_comparison.py",
    "42_stt_comparison.py",
    "43_sot_thermal_comparison.py",
    "44_dw_motion_comparison.py",
    "45_skyrmion_comparison.py",
    "46_sp1_comparison.py",
    "47_sp3_comparison.py",
    "48_fmr_comparison.py",
    "49_walker_breakdown_comparison.py",
    "50_sp3_llg_hysteresis.py",
]
# Optional CLI args: NB number prefixes to run (e.g. "46 47 48 49 50")
_args = sys.argv[1:]
NBS = [nb for nb in ALL_NBS if (not _args or any(nb.startswith(a) for a in _args))]
PER_NB_TIMEOUT = 1800   # s


def run(nb):
    t0 = time.perf_counter()
    try:
        # Force UTF-8 decoding (notebook output contains em-dash etc.; the
        # default locale codec cp949 chokes on UTF-8 bytes -> reader crash).
        r = subprocess.run([PY, "-u", str(HERE / nb)],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=PER_NB_TIMEOUT, cwd=str(HERE))
        ok = r.returncode == 0
        out = (r.stdout or "") + "\n" + (r.stderr or "")
    except subprocess.TimeoutExpired:
        ok, out = False, "TIMEOUT"
    wall = time.perf_counter() - t0
    return ok, wall, out


if __name__ == "__main__":
    summary = []
    print("=" * 72)
    print("Re-running comparison notebooks 41-50 against the LATEST build")
    print("=" * 72)
    for nb in NBS:
        if not (HERE / nb).exists():
            print(f"[SKIP] {nb} (missing)"); continue
        print(f"\n>>> {nb}", flush=True)
        ok, wall, out = run(nb)
        tail = [l for l in out.splitlines() if l.strip()][-12:]
        for l in tail:
            print("    " + l, flush=True)
        print(f"    -> {'OK' if ok else 'FAILED'}  {wall:.1f}s", flush=True)
        summary.append({"nb": nb, "ok": ok, "wall_s": round(wall, 1),
                        "tail": tail})
        (HERE / "rerun_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown summary
    md = ["# Notebook re-run summary (latest build)\n",
          "| Notebook | status | wall (s) |", "|---|---|---|"]
    for s in summary:
        md.append(f"| {s['nb']} | {'OK' if s['ok'] else 'FAILED'} | {s['wall_s']} |")
    n_ok = sum(1 for s in summary if s["ok"])
    md.append(f"\n**{n_ok}/{len(summary)} notebooks passed.**")
    (HERE / "rerun_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{n_ok}/{len(summary)} OK.  Wrote rerun_summary.{{json,md}}")
