"""Claude-SD GPU performance sweep: fixed-step RK4 ms/step, throughput and
VRAM across grid sizes.  Precision = whatever build is on PYTHONPATH.

Timing is measured around a download() (which syncs the stream), so the wall
time includes the full GPU work of each step — no async under-counting.

Usage:  py -3.13 perf_sweep.py [double|float32]
"""
import os, sys, math, time, subprocess

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
import micromag as mm  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "double"

# permalloy-like SP#4 material; cell 2 nm in-plane, 3 nm thick
MAT = mm.Material(); MAT.Ms = 800e3; MAT.A_exchange = 13e-12; MAT.alpha = 0.02
DX, DY, DZ = 2e-9, 2e-9, 3e-9
DT = 1e-14
N_WARM, N_TIME = 20, 100

GRIDS = [
    (128, 32, 1), (256, 64, 1), (512, 128, 1),
    (1024, 256, 1), (1024, 512, 1), (2048, 512, 1),
]


def vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL)
        return int(out.decode().splitlines()[0].strip())
    except Exception:
        return -1


def bench(nx, ny, nz):
    g = mm.StructuredGrid(nx, ny, nz, DX, DY, DZ)
    base_vram = vram_used_mb()
    demag = mm.DemagFieldGPU(g)
    exch = mm.ExchangeFieldGPU(g)
    zee = mm.ZeemanFieldGPU(g, mm.Vec3(1e3, 0, 0))
    fs = mm.FieldSumGPU(); fs.add(exch); fs.add(zee)
    m = mm.VectorField3D(g); m.set_uniform(mm.Vec3(1, 0.05, 0)); m.normalize()
    integ = mm.RK4IntegratorGPU(g, DT); integ.upload(m)
    for _ in range(N_WARM):
        integ.step(MAT, demag, fs)
    integ.download(m)                      # sync after warmup
    peak_vram = vram_used_mb()
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        integ.step(MAT, demag, fs)
    integ.download(m)                      # sync — ensures all steps complete
    dt_wall = time.perf_counter() - t0
    ms_step = dt_wall / N_TIME * 1e3
    cells = nx * ny * nz
    mcell_steps_s = cells * N_TIME / dt_wall / 1e6
    model_vram = (peak_vram - base_vram) if (peak_vram > 0 and base_vram > 0) else -1
    # sanity: final |m| ~ 1 (no NaN/divergence)
    a = mm.to_numpy(m).reshape(-1, 3)
    norm_ok = abs(float((a * a).sum(1).mean()) - 1.0) < 1e-3
    return cells, ms_step, mcell_steps_s, model_vram, norm_ok


if __name__ == "__main__":
    print(f"[Claude-SD {TAG}] fixed-step RK4 perf sweep (dt={DT:.0e}, {N_TIME} timed steps)")
    print(f"{'grid':>16}{'cells':>10}{'ms/step':>10}{'Mcell-st/s':>12}{'VRAM(MB)':>10}{'|m|ok':>7}")
    for nx, ny, nz in GRIDS:
        cells, mss, thr, vram, ok = bench(nx, ny, nz)
        print(f"{f'{nx}x{ny}x{nz}':>16}{cells:>10}{mss:>10.3f}{thr:>12.1f}"
              f"{vram:>10}{str(ok):>7}", flush=True)
