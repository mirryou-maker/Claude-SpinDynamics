"""P1 driver: skyrmion-Q precision sensitivity across CS builds.

Runs the worker once per build (subprocess), aggregates Q mean/std + run-to-run
spread, writes p1_sensitivity.json. Dc = 4 sqrt(A K)/pi = 4.41 mJ/m^2 for the
Co/Pt disk (A=15p, K=0.8M).

Run: py -3.13 benchmarks/sensitivity/run_p1_sensitivity.py
"""
import os, sys, json, math, subprocess, pathlib, statistics

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
PY = r"D:/anaconda3/python.exe"
CUDA_BIN = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"

A, K = 15e-12, 0.8e6
DC = 4.0 * math.sqrt(A * K) / math.pi          # 4.41 mJ/m^2

BUILDS = {
    "cuFFT_f64": ROOT / "build/windows-msvc-cuda/python",
    "cuFFT_f32": ROOT / "build/windows-msvc-cuda-f32/python",
    "VkFFT_f32": ROOT / "build/windows-msvc-cuda-vkfft-f32/python",
}
DD = [0.50, 0.68, 0.79, 0.91, 1.00]            # D/Dc points
D_LIST = [round(r * DC, 6) for r in DD]
N = int(os.environ.get("P1_N", "12"))
MAX_STEPS = int(os.environ.get("P1_STEPS", "20000"))


def run_build(py_dir):
    cfg = json.dumps({"D_list": D_LIST, "N": N, "max_steps": MAX_STEPS})
    p = subprocess.run([PY, "-u", str(HERE / "_p1_worker.py"), str(py_dir), CUDA_BIN, cfg],
                       capture_output=True, text=True, timeout=7200)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    print(f"  worker failed:\n{p.stdout[-400:]}\n{p.stderr[-400:]}")
    return None


if __name__ == "__main__":
    print(f"P1 skyrmion-Q sensitivity  Dc={DC*1e3:.2f} mJ/m^2  N={N} reps/point")
    print(f"{'build':<11}{'D/Dc':>6}{'D(mJ)':>7}{'Q_mean':>9}{'Q_std':>8}{'Q_min':>8}{'Q_max':>8}")
    print("-" * 60)
    results = {}
    for build, py_dir in BUILDS.items():
        if not py_dir.exists():
            print(f"{build}: module missing, skip"); continue
        res = run_build(py_dir)
        if not res:
            continue
        results[build] = res
        for dd, D in zip(DD, D_LIST):
            Qs = res[f"{D:.4e}"]["Q"]
            mean = statistics.mean(Qs); std = statistics.pstdev(Qs)
            print(f"{build:<11}{dd:>6.2f}{D*1e3:>7.2f}{mean:>9.3f}{std:>8.3f}"
                  f"{min(Qs):>8.3f}{max(Qs):>8.3f}", flush=True)

    out = {"Dc": DC, "DD": DD, "D_list": D_LIST, "N": N, "max_steps": MAX_STEPS,
           "results": results}
    (HERE / "p1_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'p1_sensitivity.json'}")
