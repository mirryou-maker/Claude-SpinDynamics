"""µMAG Standard Problem 1 — Claude-SD (long-axis hysteresis loop).

2 µm x 1 µm x 20 nm permalloy, Ms=8e5, A=1.3e-11, Ku=500 J/m^3 with the easy
axis along the LONG (x) edge.  Sweep the applied field along x (with a tiny +y
component to break symmetry) from +Hmax to -Hmax and back, relaxing at each
step (continuation), and record <mx>(H).  Reports coercivity Hc and remanence.

Usage:  py -3.13 run_sp1_claude_sd.py [double|float32]
"""
import os, sys, math

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
import micromag as mm  # noqa: E402
import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else "double"
HERE = os.path.dirname(os.path.abspath(__file__))

NX, NY, NZ = 100, 50, 1
g = mm.StructuredGrid(NX, NY, NZ, 20e-9, 20e-9, 20e-9)
mu0 = 4e-7 * math.pi
mat = mm.Material()
mat.Ms = 8e5; mat.A_exchange = 1.3e-11; mat.K_uniaxial = 500.0
mat.easy_axis = mm.Vec3(1, 0, 0); mat.alpha = 1.0

demag = mm.DemagFieldGPU(g)
exch = mm.ExchangeFieldGPU(g)
ani = mm.UniaxialAnisotropyFieldGPU(g)

omega = 1.7595e11 * (2 * mat.A_exchange / mat.Ms) * 4.0 / (20e-9 ** 2)
SAFE_DT = 0.2 / omega


def relax_at(m, H_Am):
    zee = mm.ZeemanFieldGPU(g, mm.Vec3(H_Am[0], H_Am[1], H_Am[2]))
    fs = mm.FieldSumGPU(); fs.add(exch); fs.add(ani); fs.add(zee)
    rel = mm.RelaxGPU(g); o = mm.RelaxGPUOptions()
    o.dt = SAFE_DT; o.threshold = 50.0; o.max_steps = 200000
    rel.upload(m); rel.run(mat, demag, fs, o); rel.download(m)
    return mm.mean_magnetization(m)


def sweep(Hs_mT):
    m = mm.VectorField3D(g); m.set_uniform(mm.Vec3(1, 0.02, 0)); m.normalize()
    out = []
    for HmT in Hs_mT:
        Hx = HmT * 1e-3 / mu0                      # T -> A/m
        Hy = 0.5e-3 / mu0                          # small +y symmetry breaker
        mx, my, mz = relax_at(m, (Hx, Hy, 0.0))
        out.append((HmT, mx, my))
    return out


def coercivity(loop):
    # field where mx crosses 0 on the descending branch
    for (h1, m1, _), (h2, m2, _) in zip(loop, loop[1:]):
        if (m1 > 0) != (m2 > 0):
            return h1 + (h2 - h1) * (0 - m1) / (m2 - m1)
    return float("nan")


if __name__ == "__main__":
    print(f"[Claude-SD {TAG}] SP#1 long-axis loop  {NX}x{NY}x{NZ}, 20nm cells")
    down = list(np.arange(80, -80.001, -8.0))
    up = list(np.arange(-80, 80.001, 8.0))
    print(f"{'H(mT)':>7}{'<mx>':>9}{'<my>':>9}  branch", flush=True)
    loop = []
    for branch, Hs in [("down", down), ("up", up)]:
        res = sweep(Hs)
        for HmT, mx, my in res:
            loop.append((HmT, mx, my, branch))
            print(f"{HmT:>7.1f}{mx:>9.4f}{my:>9.4f}  {branch}", flush=True)
    with open(os.path.join(HERE, f"loop_{TAG}.txt"), "w") as fh:
        fh.write("# H_mT mx my branch\n")
        for HmT, mx, my, br in loop:
            fh.write(f"{HmT:.3f}\t{mx:.5f}\t{my:.5f}\t{br}\n")
    desc = [(h, mx, my) for h, mx, my, br in loop if br == "down"]
    rem = next((mx for h, mx, my in desc if abs(h) < 1e-6), float("nan"))
    print(f"\n  Hc(descending) = {coercivity(desc):.1f} mT,  remanence <mx>(H=0) = {rem:.4f}")
