"""µMAG Standard Problem 5 — Claude-SD.

100x100x10 nm permalloy, vortex initial state, in-plane spin-polarised current
(Zhang-Li STT) along +x.  Relax the vortex, switch on the current, integrate
and track the vortex-core position x_c(t), y_c(t) (mz^2-weighted centroid) to
its steady gyrating state.

Compared against mumax3 with the IDENTICAL J / Pol / Xi (run_sp5 mumax3).

Usage:  py -3.13 run_sp5_claude_sd.py [double|float32]
"""
import os, sys, math, time

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
import micromag as mm  # noqa: E402
import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else "double"
HERE = os.path.dirname(os.path.abspath(__file__))

NX, NY, NZ = 40, 40, 2          # 2.5 x 2.5 x 5 nm cells
LX, LY, LZ = 100e-9, 100e-9, 10e-9
DX, DY, DZ = LX / NX, LY / NY, LZ / NZ
g = mm.StructuredGrid(NX, NY, NZ, DX, DY, DZ)
mat = mm.Material(); mat.Ms = 8e5; mat.A_exchange = 13e-12; mat.alpha = 0.1
J = mm.Vec3(1e12, 0, 0); POL = 1.0; XI = 0.05          # current along +x

demag = mm.DemagFieldGPU(g)
exch = mm.ExchangeFieldGPU(g)
fs = mm.FieldSumGPU(); fs.add(exch)

# x,y cell-centre coordinates (m), centred on the sample
xs = (np.arange(NX) + 0.5) * DX - LX / 2
ys = (np.arange(NY) + 0.5) * DY - LY / 2


def vortex_ic():
    m = mm.VectorField3D(g)
    cx, cy = (NX - 1) / 2.0, (NY - 1) / 2.0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                dx, dy = i - cx, j - cy
                r = math.hypot(dx, dy)
                mz = math.exp(-(r / 2.0) ** 2)
                ip = math.sqrt(max(0.0, 1.0 - mz * mz))
                mx, my = (-dy / r * ip, dx / r * ip) if r > 1e-9 else (0.0, 0.0)
                m[i + NX * (j + NY * k)] = mm.Vec3(mx, my, mz)
    return m


def _parab(f, i, n):
    """Sub-cell peak via 3-point parabolic interpolation around index i."""
    if i <= 0 or i >= n - 1:
        return float(i)
    fm, f0, fp = f[i - 1], f[i], f[i + 1]
    d = fm - 2 * f0 + fp
    return i + (0.5 * (fm - fp) / d if d != 0 else 0.0)


def core_xy(m):
    # core = peak of |mz| (the vortex core points +-z), sub-cell interpolated.
    # Less biased than an mz^2 centroid, which the current-induced asymmetry
    # shifts along the current (x) direction.
    a = mm.to_numpy(m)                      # (NZ,NY,NX,3)
    absmz = np.abs(a[..., 2]).mean(axis=0)  # (NY,NX)
    jy, jx = np.unravel_index(absmz.argmax(), absmz.shape)
    xi = _parab(absmz[jy, :], jx, NX)
    yi = _parab(absmz[:, jx], jy, NY)
    return ((xi + 0.5) * DX - LX / 2) * 1e9, ((yi + 0.5) * DY - LY / 2) * 1e9


def relax_vortex():
    m = vortex_ic()
    rmat = mm.Material(); rmat.Ms = 8e5; rmat.A_exchange = 13e-12; rmat.alpha = 1.0
    rel = mm.RelaxGPU(g); o = mm.RelaxGPUOptions()
    omega = 1.7595e11 * (2 * mat.A_exchange / mat.Ms) * 4.0 / (DX * DX)
    o.dt = 0.2 / omega; o.threshold = 10.0; o.max_steps = 300000
    rel.upload(m); rel.run(rmat, demag, fs, o); rel.download(m)
    return m


if __name__ == "__main__":
    print(f"[Claude-SD {TAG}] SP#5  {NX}x{NY}x{NZ}, J={J.x:.0e} A/m^2, P={POL}, xi={XI}")
    m = relax_vortex()
    print(f"  relaxed vortex core = ({core_xy(m)[0]:+.2f}, {core_xy(m)[1]:+.2f}) nm  "
          f"<m>={tuple(round(float(x),3) for x in mm.mean_magnetization(m))}", flush=True)

    zl = mm.ZhangLiSTTGPU(g, J, POL, XI)   # keep a Python ref alive: add() stores a raw ptr
    torques = mm.SpinTorqueSumGPU()
    torques.add(zl)
    dt = 5e-14
    integ = mm.RK4IntegratorGPU(g, dt); integ.upload(m)
    t_end, sample = 8e-9, 20e-12
    n = int(round(t_end / dt)); rec = int(round(sample / dt))
    rows = []; t0 = time.time()
    for kk in range(n):
        integ.step(mat, demag, fs, torques)
        if kk % rec == 0:
            integ.download(m); xc, yc = core_xy(m)
            rows.append(((kk + 1) * dt, xc, yc))
    integ.download(m); xc, yc = core_xy(m)
    rows.append((t_end, xc, yc))
    with open(os.path.join(HERE, f"core_{TAG}.txt"), "w") as fh:
        fh.write("# t xc_nm yc_nm\n")
        for t, x, y in rows:
            fh.write(f"{t:.6e}\t{x:.4f}\t{y:.4f}\n")
    print(f"  core(8ns) = ({xc:+.2f}, {yc:+.2f}) nm   wall={time.time()-t0:.0f}s", flush=True)
