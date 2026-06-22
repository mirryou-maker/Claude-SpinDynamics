"""Lightweight SP#2 cost diagnostic — find WHERE the time goes and WHY.

Measures, for a couple of small d/lex sizes:
  - wall time of remanence() (12-point field descent)
  - wall time of coercivity() (15-point field sweep)
  - per-field-point convergence steps (does each point hit the max_steps cap?)
A point hitting the cap = RK4 dynamic relaxation NOT converging -> the whole
budget is burned every point. That is the suspected root cause.
"""
import os, sys, time
import numpy as np

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
import micromag as mm  # noqa: E402

import importlib.util
spec = importlib.util.spec_from_file_location("sp2", os.path.join(os.path.dirname(__file__), "run_sp2_claude_sd.py"))
sp2 = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(sp2)

INV3 = sp2.INV_SQRT3
Ms = sp2.Ms


def instrumented_descent(d_over_lex, n_pts, h_lo, h_hi, max_steps, tol_deg=2.0):
    """Replicates the hysteresis descent but reports per-point step counts via
    a manual RK4 loop with convergence check, so we can see if points hit cap."""
    g = sp2.make_grid(d_over_lex)
    mat = sp2.make_mat(0.5)
    demag = mm.DemagFieldGPU(g); exch = mm.ExchangeFieldGPU(g)
    u = np.array([INV3, INV3, INV3])
    H_amp = np.linspace(h_hi, h_lo, n_pts) * Ms

    zee = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 0))
    fsum = mm.FieldSumGPU(); fsum.add(exch); fsum.add(zee)
    rk = mm.RK4IntegratorGPU(g, 5e-14)
    m0 = mm.VectorField3D(g); m0.set_uniform(mm.Vec3(INV3, INV3, INV3))
    rk.upload(m0)

    cells = g.nx * g.ny * g.nz
    steps_per_point = []
    capped = 0
    t0 = time.perf_counter()
    for amp in H_amp:
        zee.H_ext = mm.Vec3(amp*u[0], amp*u[1], amp*u[2])
        steps = 0
        while steps < max_steps:
            rk.step(mat, demag, fsum)
            steps += 200
            for _ in range(199):
                rk.step(mat, demag, fsum)
            if rk.max_angle_gpu() < tol_deg:
                break
        steps_per_point.append(steps)
        if steps >= max_steps:
            capped += 1
    wall = time.perf_counter() - t0
    return cells, wall, steps_per_point, capped


if __name__ == "__main__":
    print(f"SP#2 cost diagnostic  (lex={sp2.lex*1e9:.2f} nm, cells/lex={os.environ.get('SP2_CPL','1.5')})\n")
    for d in (2.0, 10.0):
        # remanence-like descent: +0.5Ms -> 0, 8 points, cap 30000
        cells, wall, steps, capped = instrumented_descent(d, 8, 0.0, 0.5, 30000)
        print(f"[remanence d/lex={d:>4}] cells={cells:>6}  wall={wall:6.2f}s  "
              f"steps/pt={[s for s in steps]}  capped={capped}/8")
    for d in (2.0, 10.0):
        # coercivity-like sweep: +0.5 -> -0.5 Ms, 15 points, cap 30000
        cells, wall, steps, capped = instrumented_descent(d, 15, -0.5, 0.5, 30000)
        print(f"[coercivity d/lex={d:>4}] cells={cells:>6}  wall={wall:6.2f}s  "
              f"capped={capped}/15  total_steps={sum(steps)}")
