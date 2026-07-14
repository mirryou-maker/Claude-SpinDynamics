"""
Notebook 49: Walker Breakdown (Zhang-Li STT, Flat Strip) -- Build Comparison
Permalloy flat strip 200×10×1 cells, dx=5nm (1µm × 50nm × 5nm).
xi=0.5, alpha=0.05 → dramatic Walker breakdown effect (xi/alpha = 10).

Sub-Walker (J < J_W): v = (xi/alpha) × u = 10u  (10× enhancement)
Above-Walker (J > J_W): v → (xi/sqrt(α²+ξ²)) × u ≈ 0.995u  (collapses to ~u)

Walker threshold: J_W ≈ (2eMsα²Δ)/(ℏPξ) (flat strip approximation)

Runs across CS cuFFT_f64 / cuFFT_f32 / mumax3 / mumax+.
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_WALKER = pathlib.Path(__file__).parent / "mx3" / "walker_breakdown.mx3"

Ms = 860e3; A = 13e-12; K = 0.0
alpha = 0.05; xi = 0.5; P = 0.5
NX, NY, NZ = 200, 10, 1
DX = 5e-9
DT  = 5e-14
T_SIM = 5e-10   # 0.5 ns per J value
N_STEPS = int(T_SIM / DT)  # 10,000

# J sweep: spans both sub-Walker and above-Walker regimes
J_VALS = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]) * 1e12

MU0 = bu.MU0
mu_B = 9.274e-24; e_ch = 1.6022e-19

# Theoretical quantities
l_ex = np.sqrt(2*A / (MU0 * Ms**2))
dw_width = np.pi * l_ex   # Bloch DW width
v_sub_ratio = xi / alpha           # 10 (sub-Walker enhancement)
v_abv_ratio = xi / np.sqrt(alpha**2 + xi**2)  # ~0.995

print("=" * 70)
print("Notebook 49: Walker Breakdown (Zhang-Li, Flat Strip) -- Build Comparison")
print(f"  Py strip {NX*DX*1e9:.0f}nm x {NY*DX*1e9:.0f}nm x {NZ*DX*1e9:.0f}nm ({NX}x{NY}x{NZ} cells)")
print(f"  xi={xi}, alpha={alpha}, xi/alpha={xi/alpha:.0f}")
print(f"  l_ex = {l_ex*1e9:.2f}nm, DW width = pi*l_ex = {dw_width*1e9:.1f}nm")
print(f"  Sub-Walker: v/u = {v_sub_ratio:.1f}  |  Above-Walker: v/u = {v_abv_ratio:.3f}")
print(f"  J sweep: {J_VALS[0]/1e12:.1f} - {J_VALS[-1]/1e12:.1f} x 10^1^2 A/m^2 ({len(J_VALS)} points)")
print("=" * 70)


def _init_neel_dw(mm, g, NX, NY, NZ, DX):
    """Néel DW: mx=-tanh, my=sech, mz=0 (core along +y for flat strip)."""
    m0 = mm.VectorField3D(g)
    mid = NX // 2
    arr = np.zeros((NZ, NY, NX, 3))
    for ix in range(NX):
        x = (ix - mid) * DX / dw_width
        arr[:, :, ix, 0] = -np.tanh(x)    # mx: domain wall
        arr[:, :, ix, 1] =  1.0 / np.cosh(x)  # my: Néel core
    norms = np.linalg.norm(arr, axis=-1, keepdims=True).clip(1e-10)
    arr /= norms
    mm.from_numpy(m0, arr)
    return m0


def _dw_pos(mm, m, NX, NY, NZ, DX):
    arr = mm.to_numpy(m)
    mx_col = arr[:, :, :, 0].mean(axis=(0, 1))  # mean over nz,ny → (nx,)
    grad = np.gradient(mx_col)
    return float(np.argmin(grad)) * DX


def run_cs(build_label):
    mm  = bu.load_mm(build_label)
    g   = mm.StructuredGrid(NX, NY, NZ, DX, DX, DX)
    mat = mm.Material()
    mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = K; mat.alpha = alpha

    demag  = mm.DemagFieldGPU(g)
    exch   = mm.ExchangeFieldGPU(g)
    fields = mm.FieldSumGPU(); fields.add(exch)
    zl     = mm.ZhangLiSTTGPU(g, mm.Vec3(1e12, 0, 0), P, xi)
    torques = mm.SpinTorqueSumGPU(); torques.add(zl)

    m0 = _init_neel_dw(mm, g, NX, NY, NZ, DX)
    t0_all = time.perf_counter(); v_list = []

    for J in J_VALS:
        zl.J = mm.Vec3(float(J), 0, 0)
        heun = mm.HeunIntegratorGPU(g, DT, 0)
        heun.upload(m0)
        tmp = mm.VectorField3D(g); heun.download(tmp)
        pos0 = _dw_pos(mm, tmp, NX, NY, NZ, DX)
        for _ in range(N_STEPS):
            heun.step(mat, demag, fields, 0.0, torques)
        tmp2 = mm.VectorField3D(g); heun.download(tmp2)
        pos1 = _dw_pos(mm, tmp2, NX, NY, NZ, DX)
        v_list.append(float((pos1 - pos0) / T_SIM))

    wall_ms = (time.perf_counter() - t0_all) * 1e3
    return {"build": build_label, "wall_ms": wall_ms,
            "J_vals": J_VALS.tolist(), "v_list": v_list,
            "ms_per_ns": wall_ms / (T_SIM * 1e9 * len(J_VALS))}


results = []
for bl in ["cuFFT_f64", "cuFFT_f32", "VkFFT_f32"]:
    if not bu.BUILDS[bl].exists():
        print(f"[SKIP] {bl}"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs(bl); results.append(r)
        print(f"  Wall: {r['wall_ms']:.0f} ms ({r['ms_per_ns']:.0f} ms/ns per J)")
        for J, v in zip(J_VALS, r['v_list']):
            regime = "sub-W" if abs(v) > 3 * abs(J/1e12 * 8.5e-2) else "abv-W"
            print(f"  J={J/1e12:.1f}e12  v={v:.0f} m/s  [{regime}]")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n--- mumax3 ---")
mx3r = bu.run_mumax3(MX3_WALKER, timeout_s=300) if MX3_WALKER.exists() else {"ok": False, "error": "mx3 not found"}
if mx3r["ok"]:
    print(f"  Wall: {mx3r['wall_ms']:.0f} ms")
else:
    print(f"  {mx3r.get('error', 'failed')}")

print("\n--- mumax+ (Zhang-Li, Neel DW, flat strip) ---")
def walker_mumaxplus(mxp):
    import time, numpy as np
    world = mxp.World(cellsize=(DX, DX, DX))
    mag   = mxp.Ferromagnet(world, mxp.Grid((NX, NY, NZ)))
    mag.msat  = Ms; mag.aex = A; mag.alpha = alpha
    mag.enable_zhang_li_torque    = True
    mag.enable_slonczewski_torque = False
    mag.xi  = xi
    mag.pol = P

    # Néel DW init (component-first: 3, nz, ny, nx)
    mid = NX // 2
    m_arr = np.zeros((3, NZ, NY, NX))
    for ix in range(NX):
        x = (ix - mid) * DX / dw_width
        m_arr[0, :, :, ix] = -np.tanh(x)
        m_arr[1, :, :, ix] =  1.0 / np.cosh(x)
    norms = np.linalg.norm(m_arr, axis=0, keepdims=True).clip(1e-10)
    m_arr /= norms

    t0_all = time.perf_counter(); v_list = []
    for J in J_VALS:
        mag.jcur = (float(J), 0, 0)
        mag.magnetization = m_arr
        world.timesolver.adaptive_timestep = False
        world.timesolver.timestep = DT
        world.timesolver.run(0)   # reset time (no-op, 0 sec)
        # Initial DW position
        m_np0 = mag.magnetization.eval()  # (3, nz, ny, nx)
        mx0 = m_np0[0, :, :, :].mean(axis=(0, 1))  # (nx,)
        pos0 = float(np.argmin(np.gradient(mx0))) * DX
        world.timesolver.run(T_SIM)
        m_np1 = mag.magnetization.eval()
        mx1 = m_np1[0, :, :, :].mean(axis=(0, 1))
        pos1 = float(np.argmin(np.gradient(mx1))) * DX
        v_list.append((pos1 - pos0) / T_SIM)
    wall_ms = (time.perf_counter() - t0_all) * 1e3
    return {"wall_ms": wall_ms, "v_list": [float(v) for v in v_list],
            "ms_per_ns": wall_ms / (T_SIM * 1e9 * len(J_VALS))}

mxpr = bu.run_mumaxplus(walker_mumaxplus, timeout_s=600)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms ({mxpr['ms_per_ns']:.0f} ms/ns per J)")
    for J, v in zip(J_VALS, mxpr["v_list"]):
        print(f"  J={J/1e12:.1f}e12  v={v:.0f} m/s")
else:
    print(f"  {mxpr.get('error', 'failed')}")

# Theoretical drift velocity u(J) = P*μ_B*J/(e*Ms)
u_vals = P * mu_B * J_VALS / (e_ch * Ms)
v_sub_theory = v_sub_ratio * u_vals   # sub-Walker
v_abv_theory = v_abv_ratio * u_vals   # above-Walker

print("\n" + "=" * 70)
print("SUMMARY -- Walker Breakdown (Py strip 1umx50nm, xi=0.5, alpha=0.05)")
print(f"  Drift velocity u = P*mu_B*J/(e*Ms):")
for J, u in zip(J_VALS, u_vals):
    print(f"    J={J/1e12:.1f}e12  u={u:.1f} m/s  v_sub={v_sub_ratio*u:.1f}  v_abv={v_abv_ratio*u:.1f}")

header = ["J (e12 A/m²)"] + [f"{J/1e12:.1f}" for J in J_VALS]
print("\n  " + "  ".join(f"{v:<10}" for v in header))
print("  " + "  ".join(f"{'Theory(sub)':<10}") +
      "  ".join(f"{v:.0f} m/s   " for v in v_sub_theory))
for r in results:
    row = [r['build']] + [f"{v:.0f}" for v in r['v_list']]
    print("  " + "  ".join(f"{v:<10}" for v in row))
if mx3r.get("ok"):
    print(f"  mumax3 wall: {mx3r['wall_ms']:.0f} ms")
if mxpr.get("ok"):
    row = ["mumax+"] + [f"{v:.0f}" for v in mxpr["v_list"]]
    print("  " + "  ".join(f"{v:<10}" for v in row))

out = {"scenario": "Walker Breakdown Zhang-Li", "J_vals": J_VALS.tolist(),
       "v_sub_theory": v_sub_theory.tolist(), "v_abv_theory": v_abv_theory.tolist(),
       "cs": results, "mumax3_wall_ms": mx3r.get("wall_ms"),
       "mumaxplus": mxpr}
(pathlib.Path(__file__).parent / "49_results.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    J_arr = J_VALS / 1e12

    # Theory bands
    ax.plot(J_arr, np.abs(v_sub_theory), 'k--', lw=1.5, alpha=0.5, label="Theory: sub-Walker (xi/alpha × u)")
    ax.plot(J_arr, np.abs(v_abv_theory), 'k:', lw=1.5, alpha=0.5, label="Theory: above-Walker (xi/sqrt(a²+xi²) × u)")

    colors = {"cuFFT_f64": "C0", "cuFFT_f32": "C1", "VkFFT_f32": "C2", "mumax+": "C3"}
    for r in results:
        c = colors.get(r['build'], 'C0')
        ax.plot(J_arr, np.abs(r['v_list']), 'o-', ms=6, lw=2, color=c, label=f"CS {r['build']}")

    if mxpr.get("ok"):
        ax.plot(J_arr, np.abs(mxpr['v_list']), '^--', ms=7, lw=2, color="C3", label="mumax+")

    ax.set_xlabel("J (×10¹² A/m²)")
    ax.set_ylabel("|v_DW| (m/s)")
    ax.set_title(f"Walker Breakdown: DW velocity vs J\n"
                 f"Py strip {NX*DX*1e9:.0f}nm × {NY*DX*1e9:.0f}nm, xi={xi}, alpha={alpha}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / "49_walker_comparison.png"), dpi=120)
    print("\nPlot: 49_walker_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
