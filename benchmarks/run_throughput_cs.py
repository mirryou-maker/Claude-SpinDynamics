"""Claude-SD throughput driver — runs the worker once per build (subprocess),
collects ms/step, writes records to all_solvers.json.

Usage:  py -3.13 benchmarks/run_throughput_cs.py [S1 S2 S3 S5]
"""
import os, sys, json, subprocess, pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "results"))
import results_io as rio  # noqa: E402

PY = r"D:/anaconda3/python.exe"
CUDA_BIN = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"

BUILDS = {
    "cuFFT_f64":  ROOT / "build/windows-msvc-cuda/python",
    "cuFFT_f32":  ROOT / "build/windows-msvc-cuda-f32/python",
    "VkFFT_f64":  ROOT / "build/windows-msvc-cuda-vkfft/python",
    "VkFFT_f32":  ROOT / "build/windows-msvc-cuda-vkfft-f32/python",
}

# (id, nx, ny, nz, dim) — match the mumax s1/s2/s3/s5 grids
SCENARIOS = {
    "S1": (128, 128,  4, "3D"),
    "S2": (200,  50,  1, "2D"),
    "S3": (300, 300,  6, "3D"),
    "S5": (500, 500, 10, "3D"),
}


def run_build(build, py_dir, scen_list):
    arg = json.dumps([[s, *SCENARIOS[s][:3]] for s in scen_list])
    p = subprocess.run([PY, "-u", str(HERE / "_cs_tp_worker.py"),
                        str(py_dir), CUDA_BIN, arg],
                       capture_output=True, text=True, timeout=1800)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    print(f"  [{build}] worker failed:\n{p.stdout[-500:]}\n{p.stderr[-500:]}")
    return None


if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(SCENARIOS)
    print(f"{'build':<12}{'scenario':<8}{'cells':>10}{'ms/step':>10}{'ms/eval':>10}{'IQR':>9}")
    print("-" * 60)
    for build, py_dir in BUILDS.items():
        if not (py_dir).exists():
            print(f"{build:<12} (module dir missing, skipped)")
            continue
        res = run_build(build, py_dir, want)
        if res is None:
            continue
        for r in res:
            if not r.get("ok"):
                print(f"{build:<12}{r['sid']:<8}{r['cells']:>10}   FAILED {r.get('err','')[:40]}")
                continue
            dim = SCENARIOS[r["sid"]][3]
            rio.append(rio.make_record(
                f"{r['sid']}_throughput", "claude-sd", build, "RK4", dim, r["cells"],
                ms_step=r["ms_step"], metric="throughput", repeats=5,
                ms_step_iqr=r["iqr"], notes="CS RK4IntegratorGPU, in-process timing + sync"))
            print(f"{build:<12}{r['sid']:<8}{r['cells']:>10}{r['ms_step']:>10.3f}"
                  f"{r['ms_step']/4:>10.3f}{r['iqr']:>9.3f}", flush=True)
    print(f"\nappended to {rio.STORE}")
