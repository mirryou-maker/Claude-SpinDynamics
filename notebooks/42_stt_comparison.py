"""
Notebook 42: STT Slonczewski Switching -- Build Comparison
Pt/Co macrospin (1x1x1, 3 nm), CPP-STT, sweep J, measure J_c and t_sw.
Runs across CS cuFFT f64 / f32 / VkFFT f32 + mumax3.
"""
import sys, os, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_STT = pathlib.Path(__file__).parent / "mx3" / "stt_switching.mx3"
mu0 = bu.MU0; mu_B = 9.274e-24; e_ch = 1.6022e-19; hbar = 1.0546e-34
Ms=580e3; A=15e-12; K=0.5e6; alpha=0.02; d=3e-9; P=0.5
mu0_Heff    = 2*K/Ms - mu0*Ms
J_c_theory  = 2*e_ch*alpha*Ms*d*mu0_Heff/(hbar*P)
J_VALS      = np.linspace(0.1e12, 1.5e12, 10)
DT = 5e-14; N_MAX = int(2e-9/DT); CHECK = 500

print("=" * 70)
print("Notebook 42: STT Slonczewski -- Build Comparison")
print(f"  Pt/Co 3 nm, Ms={Ms/1e3:.0f} kA/m, K={K/1e6:.1f} MJ/m3, alpha={alpha}, P={P}")
print(f"  Theory J_c = {J_c_theory/1e12:.3f} e12 A/m2")
print("=" * 70)


def run_cs(build_label):
    mm  = bu.load_mm(build_label)
    g   = mm.StructuredGrid(1, 1, 1, d, d, d)
    mat = mm.Material()
    mat.Ms=Ms; mat.A_exchange=A; mat.K_uniaxial=K
    mat.easy_axis=mm.Vec3(0,0,1); mat.alpha=alpha
    demag   = mm.DemagFieldGPU(g)
    aniso   = mm.UniaxialAnisotropyFieldGPU(g)
    stt     = mm.SlonczewskiSTTGPU(g, 1e12, P, d, mm.Vec3(0,0,1), 0.0)
    fields  = mm.FieldSumGPU(); fields.add(aniso)
    torques = mm.SpinTorqueSumGPU(); torques.add(stt)
    a0 = np.zeros((1,1,1,3)); a0[0,0,0] = [np.sin(np.deg2rad(5)),0,np.cos(np.deg2rad(5))]
    m0 = mm.VectorField3D(g); mm.from_numpy(m0, a0)
    t0_all = time.perf_counter()
    t_sw_list, mz_list = [], []
    for J in J_VALS:
        stt.J = J
        # HeunIntegratorGPU at T=0 (deterministic, avoids CUDA graph capture issues)
        heun = mm.HeunIntegratorGPU(g, DT, 0)
        heun.upload(m0)
        t_sw = None
        for step in range(0, N_MAX, CHECK):
            for _ in range(CHECK):
                heun.step(mat, demag, fields, 0.0, torques)
            tmp = mm.VectorField3D(g); heun.download(tmp)
            mz = float(mm.to_numpy(tmp)[0,0,0,2])
            if mz < -0.5 and t_sw is None:
                t_sw = (step + CHECK) * DT; break
        tmp = mm.VectorField3D(g); heun.download(tmp)
        mz_list.append(float(mm.to_numpy(tmp)[0,0,0,2]))
        t_sw_list.append(t_sw)
    wall_ms = (time.perf_counter()-t0_all)*1e3
    sw_idx = next((i for i,t in enumerate(t_sw_list) if t is not None), None)
    J_c_sim = float(J_VALS[sw_idx]) if sw_idx is not None else None
    return {"build": build_label, "wall_ms": wall_ms, "J_c_sim": J_c_sim,
            "J_c_theory": float(J_c_theory),
            "t_sw_list": [float(t*1e12) if t else None for t in t_sw_list],
            "mz_list": [float(v) for v in mz_list],
            "J_vals": J_VALS.tolist()}


results = []
for bl in bu.BUILDS:
    if not bu.BUILDS[bl].exists(): print(f"[SKIP] {bl}"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs(bl); results.append(r)
        Jc = r['J_c_sim']
        print(f"  Wall: {r['wall_ms']:.0f} ms")
        if Jc: print(f"  J_c_sim = {Jc/1e12:.3f} e12 A/m2")
        else:   print(f"  No switching in 2 ns")
        print(f"  J_c_theory = {J_c_theory/1e12:.3f} e12 A/m2")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n--- mumax3 ---")
mx3r = bu.run_mumax3(MX3_STT, 120)
print(f"  Wall: {mx3r.get('wall_ms'):.0f} ms" if mx3r['ok'] else f"  {mx3r.get('error','failed')}")

print("\n--- mumax+ (Slonczewski STT, Heun fixed dt) ---")
def stt_mumaxplus(mxp):
    import time, numpy as np
    # mumax+ sign convention: jcur_z < 0 drives +z -> -z switching
    t0_all = time.perf_counter()
    t_sw_list, mz_list = [], []
    for J in J_VALS:
        world = mxp.World(cellsize=(d, d, d))
        mag   = mxp.Ferromagnet(world, mxp.Grid((1, 1, 1)))
        mag.msat=Ms; mag.aex=A; mag.alpha=alpha
        mag.ku1=K; mag.anisU=(0, 0, 1)
        mag.enable_slonczewski_torque = True
        mag.enable_zhang_li_torque    = False
        mag.Lambda = 1.0
        mag.jcur = (0, 0, -float(J))   # negative = destabilise +z
        mag.pol  = P
        mag.fixed_layer = (0, 0, 1)
        mag.free_layer_thickness = d
        mag.magnetization = (float(np.sin(np.deg2rad(5))), 0.0, float(np.cos(np.deg2rad(5))))
        world.timesolver.adaptive_timestep = False
        world.timesolver.timestep = DT
        t_sw = None
        for step in range(0, N_MAX, CHECK):
            world.timesolver.run(CHECK * DT)
            mz = mag.magnetization.average()[2]
            if mz < -0.5 and t_sw is None:
                t_sw = (step + CHECK) * DT; break
        mz_list.append(float(mag.magnetization.average()[2]))
        t_sw_list.append(t_sw)
    wall_ms = (time.perf_counter() - t0_all) * 1e3
    sw_idx = next((i for i, t in enumerate(t_sw_list) if t is not None), None)
    J_c_sim = float(J_VALS[sw_idx]) if sw_idx is not None else None
    return {"wall_ms": wall_ms, "J_c_sim": J_c_sim,
            "t_sw_list": [float(t*1e12) if t else None for t in t_sw_list],
            "mz_list": [float(v) for v in mz_list]}

mxpr = bu.run_mumaxplus(stt_mumaxplus, timeout_s=300)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms")
    Jc = mxpr["J_c_sim"]
    print(f"  J_c_sim = {Jc/1e12:.3f} e12 A/m2" if Jc else "  No switching")
else:
    print(f"  {mxpr.get('error','failed')}")

print("\n" + "="*70)
print("SUMMARY -- STT Slonczewski Switching (Pt/Co macrospin)")
rows = [["Theory", "-", f"{J_c_theory/1e12:.3f}", "-"]]
for r in results:
    Jc = r['J_c_sim']
    rows.append([r['build'], f"{r['wall_ms']:.0f}",
                 f"{Jc/1e12:.3f}" if Jc else "N/A", "-"])
if mx3r['ok'] and mx3r.get('wall_ms'):
    rows.append(["mumax3", f"{mx3r['wall_ms']:.0f}", "-", "-"])
if mxpr["ok"]:
    Jc = mxpr["J_c_sim"]
    rows.append(["mumax+", f"{mxpr['wall_ms']:.0f}",
                 f"{Jc/1e12:.3f}" if Jc else "N/A", "neg jcur_z"])
bu.print_table(rows, ["Build", "wall_ms", "J_c (1e12 A/m2)", "note"])

out = {"scenario": "STT Slonczewski Macrospin", "cs": results,
       "mumax3_wall_ms": mx3r.get("wall_ms"), "mumaxplus": mxpr}
(pathlib.Path(__file__).parent/"42_results.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["C0", "C1", "C2"]
    for r, c in zip(results, colors):
        J_arr = np.array(r['J_vals'])/1e12
        t_arr = [t if t else np.nan for t in r['t_sw_list']]
        mask  = ~np.isnan(t_arr)
        if mask.any():
            ax.semilogy(J_arr[mask], np.array(t_arr)[mask], 'o-', color=c,
                        label=r['build'], lw=2, ms=5)
    ax.axvline(J_c_theory/1e12, color='k', ls='--', lw=1.5,
               label=f"Theory Jc={J_c_theory/1e12:.3f}")
    ax.set_xlabel("J (1e12 A/m2)"); ax.set_ylabel("t_sw (ps, log)")
    ax.set_title("STT Slonczewski: t_sw vs J (Pt/Co 3nm PMA)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent/"42_stt_comparison.png"), dpi=120)
    print("\nPlot: 42_stt_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
