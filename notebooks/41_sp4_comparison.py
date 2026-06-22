"""
Notebook 41: SP#4 Field A - Build Comparison
Runs muMAG Standard Problem #4 (200x50x1 Permalloy, Field A) across:
  - CS cuFFT f64 / cuFFT f32 / VkFFT f32  (RK45 adaptive DOPRI5)
  - mumax3 (DormandPrince, MaxErr=1e-4)
Outputs timing table + <mx>(t) comparison plot.
"""
import sys, os, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_SP4 = pathlib.Path(__file__).parent / "mx3" / "sp4_field_a.mx3"
MU0     = bu.MU0
Hx_SI   = -24.6e-3 / MU0
Hy_SI   =   4.3e-3 / MU0
MX_REF  = -0.9862
RTOL, ATOL = 1e-4, 1e-6
T_END   = 1e-9

print("=" * 70)
print("Notebook 41: SP#4 Field A -- Build Comparison")
print("  Permalloy 500x125x3 nm (200x50x1 cells, 2.5x2.5x3 nm)")
print(f"  H_ext = ({Hx_SI*MU0*1e3:.1f}, {Hy_SI*MU0*1e3:.1f}, 0) mT")
print(f"  muMAG consensus: <mx>(1ns) = {MX_REF}")
print("=" * 70)


def run_cs(build_label):
    mm  = bu.load_mm(build_label)
    g   = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3e-9)
    mat = mm.Material()
    mat.Ms = 860e3; mat.A_exchange = 13e-12; mat.K_uniaxial = 0.0; mat.alpha = 0.02
    m0  = bu.uniform_m0(mm, g, 1.0, 0.1, 0.01)
    demag  = mm.DemagFieldGPU(g)
    exch   = mm.ExchangeFieldGPU(g)
    zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(Hx_SI, Hy_SI, 0.0))
    fields = mm.FieldSumGPU(); fields.add(exch); fields.add(zeeman)
    opts = mm.RK45GPUOptions()
    opts.rtol = RTOL; opts.atol = ATOL
    rk45 = mm.RK45IntegratorGPU(g, opts)
    rk45.upload(m0)
    t_sim = 0.0; n = 0
    t0 = time.perf_counter()
    while t_sim < T_END:
        t_sim += rk45.step(mat, demag, fields)
        n += 1
    tmp = mm.VectorField3D(g); rk45.download(tmp)
    wall_ms = (time.perf_counter() - t0) * 1e3
    arr = mm.to_numpy(tmp)
    mx_fin = float(arr[..., 0].mean())
    return {"build": build_label, "wall_ms": wall_ms, "n_steps": n,
            "mx_1ns": mx_fin, "err_pct": abs(mx_fin - MX_REF) / abs(MX_REF) * 100,
            "t_switch_ps": None,
            "t_log": [], "mx_log": [],
            "ms_per_ns": wall_ms / (T_END * 1e9)}


results = []
for bl in bu.BUILDS:
    if not bu.BUILDS[bl].exists():
        print(f"[SKIP] {bl}: build not found"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs(bl)
        results.append(r)
        sw = f"{r['t_switch_ps']:.0f} ps" if r['t_switch_ps'] else "NO SWITCH"
        print(f"  Steps: {r['n_steps']}  Wall: {r['wall_ms']:.0f} ms  ms/ns: {r['ms_per_ns']:.1f}")
        print(f"  <mx>(1ns) = {r['mx_1ns']:.4f}  err={r['err_pct']:.2f}%  t_sw={sw}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n--- mumax3 (DormandPrince, MaxErr=1e-4) ---")
mx3r = bu.run_mumax3(MX3_SP4, timeout_s=300)
mx3_wall = mx3r.get("wall_ms")
mx3_mx_fin = None
if mx3r["ok"] and mx3_wall:
    outdir = MX3_SP4.parent / (MX3_SP4.stem + ".out")
    tbl = bu.parse_mumax3_table(outdir)
    if tbl is not None and len(tbl) > 0:
        mx3_mx_fin = float(tbl[-1, 1])
    print(f"  Wall: {mx3_wall:.0f} ms")
    if mx3_mx_fin is not None:
        print(f"  <mx>(1ns) = {mx3_mx_fin:.4f}  err={abs(mx3_mx_fin-MX_REF)/abs(MX_REF)*100:.2f}%")
else:
    print(f"  {mx3r.get('error', 'failed')}")

print("\n--- mumax+ (RK45 adaptive, max_err=1e-4) ---")
def sp4_mumaxplus(mxp):
    import time
    world  = mxp.World(cellsize=(2.5e-9, 2.5e-9, 3e-9))
    magnet = mxp.Ferromagnet(world, mxp.Grid((200, 50, 1)))
    magnet.msat  = 860e3
    magnet.aex   = 13e-12
    magnet.alpha = 0.02
    magnet.bias_magnetic_field = (-24.6e-3, 4.3e-3, 0.0)  # Tesla
    magnet.magnetization = (1.0, 0.1, 0.01)
    world.timesolver.adaptive_timestep = True
    world.timesolver.max_error = 1e-4
    t0 = time.perf_counter()
    world.timesolver.run(T_END)
    wall_ms = (time.perf_counter() - t0) * 1e3
    avg = magnet.magnetization.average()
    mx_fin = float(avg[0])
    return {"wall_ms": wall_ms, "mx_1ns": mx_fin,
            "err_pct": abs(mx_fin - MX_REF) / abs(MX_REF) * 100}

mxpr = bu.run_mumaxplus(sp4_mumaxplus, timeout_s=300)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms")
    print(f"  <mx>(1ns) = {mxpr['mx_1ns']:.4f}  err={mxpr['err_pct']:.2f}%")
else:
    print(f"  {mxpr.get('error','failed')}")

print("\n" + "=" * 70)
print("SUMMARY -- SP#4 Field A (1 ns adaptive DOPRI5)")
print("=" * 70)
rows = [["muMAG ref", "-", "-", f"{MX_REF:.4f}", "0.00", "~175"]]
for r in results:
    sw = f"{r['t_switch_ps']:.0f}" if r['t_switch_ps'] else "N/A"
    rows.append([r['build'], f"{r['wall_ms']:.0f}", f"{r['n_steps']}",
                 f"{r['mx_1ns']:.4f}", f"{r['err_pct']:.2f}", sw])
if mx3_wall:
    mxstr = f"{mx3_mx_fin:.4f}" if mx3_mx_fin else "n/a"
    rows.append(["mumax3", f"{mx3_wall:.0f}", "n/a", mxstr, "-", "~175"])
if mxpr["ok"]:
    rows.append(["mumax+", f"{mxpr['wall_ms']:.0f}", "adaptive",
                 f"{mxpr['mx_1ns']:.4f}", f"{mxpr['err_pct']:.2f}", "~175"])
bu.print_table(rows, ["Build", "wall_ms", "steps", "<mx>(1ns)", "err%", "t_sw(ps)"])

out = {"scenario": "SP#4 Field A 1ns", "cs": results,
       "mumax3_wall_ms": mx3_wall, "mumax3_mx_fin": mx3_mx_fin,
       "mumaxplus": mxpr}
(pathlib.Path(__file__).parent / "41_results.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["C0", "C1", "C2"]
    for r, c in zip(results, colors):
        ax.plot([t * 1e12 for t in r['t_log']], r['mx_log'],
                label=f"CS {r['build']} ({r['n_steps']} steps, {r['wall_ms']:.0f} ms)",
                lw=2, color=c)
    ax.axhline(MX_REF, color='k', ls=':', lw=1.5, label=f"muMAG ref {MX_REF}")
    ax.axvline(175, color='gray', ls='--', lw=1, alpha=0.7, label="t_sw=175 ps")
    ax.set_xlabel("Time (ps)"); ax.set_ylabel("<mx>"); ax.set_ylim(-1.1, 1.1)
    ax.set_title("SP#4 Field A: <mx>(t) -- CS build comparison")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / "41_sp4_comparison.png"), dpi=120)
    print("\nPlot saved: 41_sp4_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
