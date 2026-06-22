"""µMAG Standard Problem 2 — Claude-SD.

Rectangular prism L : d : t = 5 : 1 : 0.1.  Material is parameterised only by the
exchange length lex = sqrt(2A/(µ0 Ms^2)); results depend on d/lex.

Deliverables (the µMAG SP#2 observables):
  1. Remanent magnetisation after saturating along the body diagonal [1,1,1]
     and removing the field:  <Mx>/Ms and <My>/Ms vs d/lex.
  2. Coercive field Hc along [1,1,1] (where the projection m·[1,1,1] crosses 0).

As d/lex -> 0 (single-domain limit) the accepted reference values are
  <Mx>/Ms ~ 0.97,  <My>/Ms ~ 0.13,  Hc/Ms ~ 0.057   (µMAG submissions).

Usage:  py -3.13 run_sp2_claude_sd.py [double|float32] [d_over_lex ...]
"""
import os, sys, math
import numpy as np

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
import micromag as mm  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "double"

# Material (values are arbitrary; only lex matters). Use permalloy-like.
Ms  = 8e5
A   = 13e-12
mu0 = 4e-7 * math.pi
lex = math.sqrt(2 * A / (mu0 * Ms * Ms))
INV_SQRT3 = 1.0 / math.sqrt(3.0)

# Cells per lex (resolution). Cap total cells so large d/lex stays tractable.
CELLS_PER_LEX = float(os.environ.get("SP2_CPL", "2.0"))
MAX_CELLS     = int(os.environ.get("SP2_MAXCELLS", "400000"))


def make_mat(alpha=1.0):
    m = mm.Material()
    m.Ms = Ms; m.A_exchange = A; m.K_uniaxial = 0.0
    m.alpha = alpha
    return m


def make_grid(d_over_lex):
    """Prism L=5d, d, t=0.1d, sized so dx ~ lex/CELLS_PER_LEX, capped at MAX_CELLS."""
    d = d_over_lex * lex
    L = 5.0 * d
    t = 0.1 * d
    dx = lex / CELLS_PER_LEX
    nx = max(2, round(L / dx))
    ny = max(2, round(d / dx))
    nz = max(1, round(t / dx))
    # Cap: coarsen uniformly if too many cells
    while nx * ny * nz > MAX_CELLS:
        nx = max(2, nx // 2); ny = max(2, ny // 2); nz = max(1, nz // 2)
    return mm.StructuredGrid(nx, ny, nz, L / nx, d / ny, t / nz)


def _relax_sweep(g, H_amp):
    """Quasistatic field sweep along [1,1,1] using RelaxGPU (damping-only energy
    minimisation, torque-based convergence) at each amplitude — the same
    protocol as mumax3 relax(). Continuation: each point starts from the
    previous equilibrium. Returns (mx, my, mz) arrays normalised to Ms."""
    mat = make_mat(1.0)
    demag = mm.DemagFieldGPU(g)
    exch  = mm.ExchangeFieldGPU(g)
    u = np.array([INV_SQRT3, INV_SQRT3, INV_SQRT3])

    zee  = mm.ZeemanFieldGPU(g, mm.Vec3(0, 0, 0))
    fsum = mm.FieldSumGPU(); fsum.add(exch); fsum.add(zee)
    rel  = mm.RelaxGPU(g); o = mm.RelaxGPUOptions()
    cell = min(g.dx, g.dy, g.dz)
    omega = 1.7595e11 * (2 * A / Ms) * 4.0 / (cell * cell)
    o.dt = 0.2 / omega
    o.threshold = 3e-4 * Ms      # torque-based; converges for non-uniform states
    # Small coherent particles relax near a low-torque saddle (slow); cap steps
    # by grid size — the small-d/lex answer (m -> long axis) needs few real iters.
    o.max_steps = 12000 if (g.nx * g.ny * g.nz) < 2000 else 30000

    m0 = mm.VectorField3D(g)
    m0.set_uniform(mm.Vec3(INV_SQRT3, INV_SQRT3, INV_SQRT3))
    rel.upload(m0)
    m_cpu = mm.VectorField3D(g)

    mx = np.zeros(len(H_amp)); my = np.zeros(len(H_amp)); mz = np.zeros(len(H_amp))
    for i, amp in enumerate(H_amp):
        zee.H_ext = mm.Vec3(amp*u[0], amp*u[1], amp*u[2])
        rel.run(mat, demag, fsum, o)       # continue from current equilibrium
        rel.download(m_cpu)
        mx[i], my[i], mz[i] = mm.mean_magnetization(m_cpu)
    return mx, my, mz


def remanence(d_over_lex, n_desc=10):
    """µMAG SP#2 remanence: descend the field along [1,1,1] from saturation to 0
    (RelaxGPU at each step); report the H=0 state."""
    g = make_grid(d_over_lex)
    H_amp = np.linspace(0.5, 0.0, n_desc) * Ms
    mx, my, mz = _relax_sweep(g, H_amp)
    return g, mx[-1], my[-1], mz[-1]


def coercivity(d_over_lex, n_pts=21):
    """Sweep H along [1,1,1] from +Hsat through 0 to -Hsat (RelaxGPU per point);
    coercive field Hc/Ms = where the projection m·[1,1,1] crosses zero."""
    g = make_grid(d_over_lex)
    u = np.array([INV_SQRT3, INV_SQRT3, INV_SQRT3])
    H_amp = np.linspace(0.5, -0.5, n_pts) * Ms
    mx, my, mz = _relax_sweep(g, H_amp)
    proj = mx * u[0] + my * u[1] + mz * u[2]
    Hc = None
    for i in range(len(H_amp) - 1):
        if proj[i] == 0 or (proj[i] > 0) != (proj[i + 1] > 0):
            f = (0 - proj[i]) / (proj[i + 1] - proj[i])
            Hc = H_amp[i] + f * (H_amp[i + 1] - H_amp[i])
            break
    return Hc / Ms if Hc is not None else float("nan")


if __name__ == "__main__":
    sweep = [float(x) for x in sys.argv[2:]] or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    print(f"[Claude-SD {TAG}] SP#2  lex={lex*1e9:.3f} nm  cells/lex={CELLS_PER_LEX}")
    print(f"{'d/lex':>7}{'grid':>14}{'<mx>/Ms':>10}{'<my>/Ms':>10}{'<mz>/Ms':>10}")
    rem = []
    for d in sweep:
        g, mx, my, mz = remanence(d)
        rem.append((d, mx, my, mz))
        print(f"{d:>7.2f}{f'{g.nx}x{g.ny}x{g.nz}':>14}{mx:>10.4f}{my:>10.4f}{mz:>10.4f}", flush=True)

    # Coercivity for a representative subset (expensive)
    print(f"\n{'d/lex':>7}{'Hc/Ms':>10}")
    for d in (sweep if len(sweep) <= 4 else [sweep[0], sweep[len(sweep)//2], sweep[-1]]):
        hc = coercivity(d)
        print(f"{d:>7.2f}{hc:>10.4f}", flush=True)

    # Save remanence table for the report
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sp2_remanence_{TAG}.csv")
    np.savetxt(out, np.array(rem), delimiter=",",
               header="d_over_lex,mx_over_Ms,my_over_Ms,mz_over_Ms", comments="")
    print(f"\nwrote {out}")
