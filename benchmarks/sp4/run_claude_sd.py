"""Claude-SD SP#4 (field A) benchmark driver.

Relaxes the S-state, then runs field A for 1 ns with BOTH a fixed-step RK4
and the adaptive RK45 (DOPRI5) GPU integrator.  Writes mumax3-style tables
(# t mx my mz) and prints <m>(1ns) / t_switch / wall / step counts.

All GPU fields are created ONCE at module scope and reused — recreating /
destroying a DemagFieldGPU (cuFFT plans + streams) mid-run is both wasteful
and fragile.

Usage:  py -3.13 run_claude_sd.py [double|float32]
The precision is whatever build is on PYTHONPATH; the arg is only a label.
"""
import os, sys, math, time

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
import micromag as mm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "double"

# ---- problem (muMAG SP#4, field A, 250x64x1) -----------------------------
g = mm.StructuredGrid(250, 64, 1, 2e-9, 1.953125e-9, 3e-9)
mat = mm.Material(); mat.Ms = 800e3; mat.A_exchange = 13e-12; mat.alpha = 0.02
mu0 = 4e-7 * math.pi
H_A = mm.Vec3(-24.6e-3 / mu0, 4.3e-3 / mu0, 0.0)     # field A in A/m

dmin = min(2e-9, 1.953125e-9, 3e-9)
omega = 1.7595e11 * (2.0 * mat.A_exchange / mat.Ms) * 4.0 / (dmin * dmin)
SAFE_DT = 0.2 / omega                                # ~3.3e-14 s

# ---- shared fields (created once, live for the whole run) ----------------
demag = mm.DemagFieldGPU(g)
exch = mm.ExchangeFieldGPU(g)
zee0 = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 0))
zeeA = mm.ZeemanFieldGPU(g, H_A)
fs_relax = mm.FieldSumGPU(); fs_relax.add(exch); fs_relax.add(zee0)
fs_dyn = mm.FieldSumGPU(); fs_dyn.add(exch); fs_dyn.add(zeeA)


def save_table(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# t mx my mz\n")
        for t, mx, my, mz in rows:
            fh.write(f"{t:.6e}\t{mx:.6e}\t{my:.6e}\t{mz:.6e}\n")


def metrics(rows):
    tsw = next((t for t, mx, _, _ in rows if mx < 0), float("nan"))
    return rows[-1][1], rows[-1][2], rows[-1][3], tsw


def fresh_from(m_src):
    m = mm.VectorField3D(g)
    for i in range(g.size):
        m[i] = m_src[i]
    return m


def relax_sstate():
    m = mm.VectorField3D(g); m.set_uniform(mm.Vec3(1, 0.1, 0)); m.normalize()
    rmat = mm.Material(); rmat.Ms = 800e3; rmat.A_exchange = 13e-12; rmat.alpha = 0.5
    rel = mm.RelaxGPU(g); o = mm.RelaxGPUOptions()
    o.threshold = 0.5; o.dt = SAFE_DT; o.max_steps = 600000
    rel.upload(m); rel.run(rmat, demag, fs_relax, o); rel.download(m)
    return m


def run_fixed(m0, dt=2e-14, t_end=1e-9, sample=5e-12):
    integ = mm.RK4IntegratorGPU(g, dt); integ.upload(m0)
    n = int(round(t_end / dt)); rec = max(1, int(round(sample / dt)))
    rows = []; t0 = time.time()
    for k in range(n):
        integ.step(mat, demag, fs_dyn)
        if k % rec == 0:
            integ.download(m0); mx, my, mz = mm.mean_magnetization(m0)
            rows.append(((k + 1) * dt, mx, my, mz))
    integ.download(m0); mx, my, mz = mm.mean_magnetization(m0)
    rows.append((t_end, mx, my, mz))
    return rows, time.time() - t0, n


def run_adaptive(m0, t_end=1e-9, sample=5e-12, rtol=1e-5):
    o = mm.RK45GPUOptions()
    o.rtol = rtol; o.atol = 1e-6
    o.dt_init = SAFE_DT; o.dt_min = SAFE_DT * 1e-3; o.dt_max = SAFE_DT * 50
    integ = mm.RK45IntegratorGPU(g, o); integ.upload(m0)
    rows = []; t = 0.0; next_s = 0.0; t0 = time.time(); guard = 0
    while t < t_end and guard < 2_000_000:
        integ.step(mat, demag, fs_dyn)
        t += float(integ.dt_current); guard += 1
        if t >= next_s:
            integ.download(m0); mx, my, mz = mm.mean_magnetization(m0)
            rows.append((t, mx, my, mz)); next_s += sample
    integ.download(m0); mx, my, mz = mm.mean_magnetization(m0)
    rows.append((t, mx, my, mz))
    return rows, time.time() - t0, integ.n_accepted, integ.n_rejected


if __name__ == "__main__":
    print(f"[Claude-SD {TAG}] relaxing S-state (safe_dt={SAFE_DT:.2e}) ...", flush=True)
    s0 = relax_sstate()
    print("  S-state <m> =", tuple(round(float(x), 4) for x in mm.to_numpy(s0).reshape(-1, 3).mean(0)), flush=True)

    rows_f, wall_f, nsteps = run_fixed(fresh_from(s0))
    save_table(os.path.join(HERE, f"ours_{TAG}_fixed.out", "m.txt"), rows_f)
    mx, my, mz, tsw = metrics(rows_f)
    print(f"  RK4 fixed dt=2e-14: <m>(1ns)=({mx:+.4f} {my:+.4f} {mz:+.4f}) "
          f"t_switch={tsw*1e12:.0f}ps  {nsteps} steps  wall={wall_f:.0f}s", flush=True)

    rows_a, wall_a, na, nr = run_adaptive(fresh_from(s0))
    save_table(os.path.join(HERE, f"ours_{TAG}_adapt.out", "m.txt"), rows_a)
    mx, my, mz, tsw = metrics(rows_a)
    print(f"  RK45 adaptive MaxErr=1e-5: <m>(1ns)=({mx:+.4f} {my:+.4f} {mz:+.4f}) "
          f"t_switch={tsw*1e12:.0f}ps  {na} acc/{nr} rej  wall={wall_a:.0f}s", flush=True)
