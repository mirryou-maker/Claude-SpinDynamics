"""µMAG Standard Problem 3 — Claude-SD.

Cube of edge L (in units of lex = sqrt(2A/(µ0 Ms^2))); uniaxial anisotropy
Ku = 0.1 * Kd with Kd = 0.5 µ0 Ms^2, easy axis along z (a cube edge); no
applied field.  Relax a flower state and a vortex state at several L and find
the crossing L_c where E_vortex = E_flower (reference: L_c ~ 8.47 lex,
E/Kd/V ~ 0.303 at the crossing).

Energies are reported in units of Kd*V (= Km*V), the µMAG convention.

Usage:  py -3.13 run_sp3_claude_sd.py [double|float32]
"""
import os, sys, math

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64")
import micromag as mm  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "double"

Ms = 8e5
A = 13e-12
mu0 = 4e-7 * math.pi
Kd = 0.5 * mu0 * Ms * Ms
Ku = 0.1 * Kd
lex = math.sqrt(2 * A / (mu0 * Ms * Ms))
N = int(os.environ.get("SP3_N", "16"))     # cells per edge


def make_mat(alpha):
    m = mm.Material()
    m.Ms = Ms; m.A_exchange = A; m.K_uniaxial = Ku
    m.easy_axis = mm.Vec3(0, 0, 1); m.alpha = alpha
    return m


def relax(g, demag, exch, ani, m0, mat):
    fs = mm.FieldSumGPU(); fs.add(exch); fs.add(ani)
    rel = mm.RelaxGPU(g); o = mm.RelaxGPUOptions()
    # cell-size-aware stable step
    cell = g.dx
    omega = 1.7595e11 * (2 * A / Ms) * 4.0 / (cell * cell)
    o.dt = 0.2 / omega; o.threshold = 1e-3 * Ms / 100; o.max_steps = 400000
    rel.upload(m0); rel.run(mat, demag, fs, o); rel.download(m0)
    return m0


def energy_norm(g, demag, exch, ani, m, mat):
    V = g.size * g.dx * g.dy * g.dz
    E = demag.energy(m, mat) + exch.energy(m, mat) + ani.energy(m, mat)
    return E / (Kd * V)


def flower_ic(g):
    m = mm.VectorField3D(g); m.set_uniform(mm.Vec3(0.05, 0.05, 1.0)); m.normalize()
    return m


def vortex_ic(g):
    # proper vortex: in-plane curl around z with a small Gaussian core (mz->1
    # only within ~rc cells of the centre axis, ~0 elsewhere).
    m = mm.VectorField3D(g)
    c = (N - 1) / 2.0
    rc = 1.5
    for k in range(N):
        for j in range(N):
            for i in range(N):
                dx, dy = i - c, j - c
                r = math.hypot(dx, dy)
                mz = math.exp(-(r / rc) ** 2)
                ip = math.sqrt(max(0.0, 1.0 - mz * mz))
                if r > 1e-9:
                    mx, my = -dy / r * ip, dx / r * ip
                else:
                    mx, my = 0.0, 0.0
                m[i + N * (j + N * k)] = mm.Vec3(mx, my, mz)
    return m


def relax_state(L_in_lex, m_components):
    """Relax a state (given as an (N,N,N,3) numpy array) at edge L; return
    (E/Kd/V, <mz>, relaxed (N,N,N,3) components) so it can be CONTINUED to the
    next L — carrying the metastable branch into its unstable region, which a
    fresh IC + damped relax cannot reach."""
    import numpy as np
    L = L_in_lex * lex
    cell = L / N
    g = mm.StructuredGrid(N, N, N, cell, cell, cell)
    demag = mm.DemagFieldGPU(g); exch = mm.ExchangeFieldGPU(g); ani = mm.UniaxialAnisotropyFieldGPU(g)
    relmat = make_mat(1.0)
    m = mm.VectorField3D(g)
    flat = m_components.reshape(-1, 3)
    for idx in range(N * N * N):
        m[idx] = mm.Vec3(float(flat[idx, 0]), float(flat[idx, 1]), float(flat[idx, 2]))
    m.normalize()
    m = relax(g, demag, exch, ani, m, relmat)
    E = energy_norm(g, demag, exch, ani, m, relmat)
    arr = mm.to_numpy(m)
    return E, mm.mean_magnetization(m)[2], arr


def ic_array(kind):
    import numpy as np
    g = mm.StructuredGrid(N, N, N, 1e-9, 1e-9, 1e-9)
    return mm.to_numpy(flower_ic(g) if kind == "flower" else vortex_ic(g))


if __name__ == "__main__":
    print(f"[Claude-SD {TAG}] SP#3  lex={lex*1e9:.3f} nm, Kd={Kd:.4e}, Ku/Kd=0.1, N={N}^3")
    print("  (continuation: flower carried UP from small L, vortex carried DOWN from large L)")
    print(f"{'L/lex':>7}{'E_flower':>11}{'E_vortex':>11}{'dE=v-f':>11}{'<mz>_fl':>9}{'<mz>_vx':>9}")
    Ls = sorted(float(x) for x in (sys.argv[2:] or ["8.0", "8.3", "8.5", "8.7", "9.0", "9.5"]))

    # flower branch: ascending L, continued
    flower = {}; st = ic_array("flower")
    for L in Ls:
        Ef, mzf, st = relax_state(L, st); flower[L] = (Ef, mzf)
    # vortex branch: descending L, continued from a stable large-L vortex
    vortex = {}; st = ic_array("vortex")
    st = relax_state(max(Ls) + 3.0, st)[2]          # seed: relax vortex well above range
    for L in reversed(Ls):
        Ev, mzv, st = relax_state(L, st); vortex[L] = (Ev, mzv)

    rows = []
    for L in Ls:
        Ef, mzf = flower[L]; Ev, mzv = vortex[L]
        rows.append((L, Ef, Ev))
        print(f"{L:>7.2f}{Ef:>11.4f}{Ev:>11.4f}{Ev-Ef:>11.4f}{mzf:>9.3f}{mzv:>9.3f}", flush=True)

    cr = None
    for (L1, f1, v1), (L2, f2, v2) in zip(rows, rows[1:]):
        d1, d2 = v1 - f1, v2 - f2
        if d1 == 0 or (d1 < 0) != (d2 < 0):
            cr = L1 + (L2 - L1) * (0 - d1) / (d2 - d1)
            break
    print(f"\n  L_c (E_vortex = E_flower) = {cr:.3f} lex   (reference 8.47)" if cr
          else "\n  no crossing in this L range")
