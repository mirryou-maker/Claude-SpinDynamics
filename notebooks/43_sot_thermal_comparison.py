"""
Notebook 43: SOT-Driven Magnetization at Finite T -- Build Comparison
PMA macrospin (1x1x1, 3nm), Heun SLLG at T=300K.

Observable: SOT-driven canting <mz>(J) and switching probability P_sw(J).

DESIGN NOTE (reconfigured): the original Co/Pt parameters (K=0.5 MJ/m^3,
H_k=1.72 T) put the SOT switching threshold at an unphysically high current
(J_c >> 2e13 A/m^2), so P_sw was identically 0 -- an uninformative benchmark.
Reconfigured here with:
  * sigma = (0,1,0)  (spin polarisation PERP to the in-plane assist; the
    standard SOT-switching geometry. The old sigma=(1,0,0) drove m only to
    the equator along +-x.)
  * in-plane assist field H_x = 40 mT along the current (breaks +z/-z symmetry)
  * softer PMA K = 0.2 MJ/m^3 (H_k = 0.69 T) so the SOT response is in range
  * t_sim = 2 ns

The ROBUST, reproducible cross-solver observable is the thermal-averaged
canting <mz>(J): the SOT progressively tilts m from +z through the equator as
J rises (e.g. +0.99 -> +0.97 -> -0.18 over J = 1..6e12). Clean deterministic
P_sw additionally requires fine pulse/assist tuning and is reported as
secondary. (The headline finite-T throughput comparison is the Heun<->Heun
benchmark in benchmarks/RESULTS_2026.md.)
Runs across CS cuFFT f64 / cuFFT f32 / VkFFT f32 + mumax3 + mumax+.
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_SOT = pathlib.Path(__file__).parent / "mx3" / "sot_thermal.mx3"
Ms=580e3; A=15e-12; K=0.2e6; alpha=0.05; d=3e-9; P=0.35   # softer PMA (H_k=0.69T)
SIGMA = (0, 1, 0)                   # spin pol. perp to in-plane assist (std SOT geometry)
T_KELVIN=300; T_SIM=2e-9
N_TRIALS=10
J_VALS=np.array([1.0,2.0,4.0,6.0])*1e12
DT=5e-14
# In-plane assist field along the spin-polarisation axis (x). Without it, pure
# damping-like SOT with sigma=(1,0,0) drives a PMA magnet to the equator (+-x),
# not deterministically to -z, so P_sw stays 0. H_x breaks the symmetry so the
# DL torque selects -z above threshold -> a real P_sw(J) sigmoid.
MU0 = 4e-7*np.pi
B_ASSIST = 40e-3                     # 40 mT in-plane assist (along x)
H_ASSIST = B_ASSIST/MU0             # A/m

print("="*70)
print("Notebook 43: SOT Thermal Switching -- Build Comparison")
print(f"  Pt/Co 3nm, T={T_KELVIN}K, {N_TRIALS} trials/J, t_sim={T_SIM*1e9:.1f}ns")
print("="*70)


def run_cs(build_label):
    mm  = bu.load_mm(build_label)
    g   = mm.StructuredGrid(1, 1, 1, d, d, d)
    mat = mm.Material()
    mat.Ms=Ms; mat.A_exchange=A; mat.K_uniaxial=K
    mat.easy_axis=mm.Vec3(0,0,1); mat.alpha=alpha
    demag   = mm.DemagFieldGPU(g)
    aniso   = mm.UniaxialAnisotropyFieldGPU(g)
    sot     = mm.SpinOrbitTorqueGPU(g, 1e12, P, d, mm.Vec3(*SIGMA))
    zee     = mm.ZeemanFieldGPU(g, mm.Vec3(H_ASSIST, 0, 0))   # in-plane assist
    fields  = mm.FieldSumGPU(); fields.add(aniso); fields.add(zee)
    torques = mm.SpinTorqueSumGPU(); torques.add(sot)
    a0 = np.zeros((1,1,1,3)); a0[0,0,0]=[0.01,0.01,0.9999]
    a0[0,0,0] /= np.linalg.norm(a0[0,0,0])
    m0 = mm.VectorField3D(g); mm.from_numpy(m0, a0)
    N_STEPS = int(T_SIM/DT)
    psw = []; mz_avg = []; t0_all = time.perf_counter()
    for J in J_VALS:
        sot.J_c = J
        sw_count = 0; mzs = []
        for seed in range(N_TRIALS):
            heun = mm.HeunIntegratorGPU(g, DT, seed)
            heun.upload(m0)
            for _ in range(N_STEPS):
                heun.step(mat, demag, fields, float(T_KELVIN), torques)
            tmp = mm.VectorField3D(g); heun.download(tmp)
            mz = float(mm.to_numpy(tmp)[0,0,0,2]); mzs.append(mz)
            if mz < -0.5: sw_count += 1
        psw.append(sw_count/N_TRIALS); mz_avg.append(float(np.mean(mzs)))
    wall_ms = (time.perf_counter()-t0_all)*1e3
    return {"build":build_label,"wall_ms":wall_ms,
            "psw":[float(p) for p in psw],"mz_avg":mz_avg,"J_vals":J_VALS.tolist()}


results = []
for bl in bu.BUILDS:
    if not bu.BUILDS[bl].exists(): print(f"[SKIP] {bl}"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs(bl); results.append(r)
        print(f"  Wall: {r['wall_ms']:.0f} ms")
        for J,p,mz in zip(J_VALS,r['psw'],r['mz_avg']):
            print(f"  J={J/1e12:.1f}e12  <mz>={mz:+.2f}  P_sw={p:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n--- mumax3 ---")
mx3r = bu.run_mumax3(MX3_SOT, 300)
print(f"  Wall: {mx3r.get('wall_ms'):.0f} ms" if mx3r['ok'] else f"  {mx3r.get('error','failed')}")

print("\n--- mumax+ (SOT via Slonczewski, T=300K Heun) ---")
def sot_thermal_mumaxplus(mxp):
    import time, numpy as np
    # SOT via Slonczewski: jcur=(J_HM,0,0) in HM, fixed_layer=(0,1,0)=sigma_y, pol=theta_SH
    # Negative jcur_x -> spin pol in -y -> DL torque drives +z->-z at appropriate sigma
    # We replicate CS SpinOrbitTorqueGPU: sigma=(1,0,0) → for SOT, fixed_layer=(0,1,0)
    # Use jcur_x > 0 with fixed_layer=(0,-1,0) or jcur_x < 0 with fixed_layer=(0,1,0)
    t0_all = time.perf_counter()
    psw = []; mz_avg = []
    for J in J_VALS:
        sw_count = 0; mzs = []
        for seed in range(N_TRIALS):
            world = mxp.World(cellsize=(d, d, d))
            mag   = mxp.Ferromagnet(world, mxp.Grid((1, 1, 1)))
            mag.msat=Ms; mag.aex=A; mag.alpha=alpha
            mag.ku1=K; mag.anisU=(0, 0, 1)
            mag.enable_slonczewski_torque = True
            mag.enable_zhang_li_torque    = False
            mag.jcur        = (-float(J), 0, 0)  # in-plane HM current (-x)
            mag.pol         = P                   # theta_SH = 0.35
            mag.fixed_layer = SIGMA               # spin pol perp to assist (std SOT geometry)
            mag.free_layer_thickness = d
            mag.bias_magnetic_field = (B_ASSIST, 0, 0)   # in-plane assist (40 mT)
            mag.temperature  = float(T_KELVIN)
            mag.thermal_seed = seed
            mag.magnetization = (0.01, 0.01, 0.9999)
            world.timesolver.adaptive_timestep = False
            world.timesolver.timestep = DT
            world.timesolver.run(T_SIM)
            mz = mag.magnetization.average()[2]; mzs.append(mz)
            if mz < -0.5:
                sw_count += 1
        psw.append(sw_count / N_TRIALS); mz_avg.append(float(np.mean(mzs)))
    wall_ms = (time.perf_counter() - t0_all) * 1e3
    return {"wall_ms": wall_ms, "psw": psw, "mz_avg": mz_avg}

mxpr = bu.run_mumaxplus(sot_thermal_mumaxplus, timeout_s=600)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms")
    for J, p in zip(J_VALS, mxpr["psw"]):
        print(f"  J={J/1e12:.1f}e12  P_sw={p:.2f}")
else:
    print(f"  {mxpr.get('error','failed')}")

print("\n"+"="*70)
print("SUMMARY -- SOT-driven <mz>(J) at T=300K (primary), P_sw (secondary)")
header = ["J (1e12)"] + [f"{J/1e12:.1f}" for J in J_VALS]
print("<mz>(J):")
print("  ".join(f"{v:<10}" for v in header))
for r in results:
    row = [r['build']] + [f"{mz:+.2f}" for mz in r['mz_avg']]
    print("  ".join(f"{v:<10}" for v in row))
if mxpr["ok"] and mxpr.get("mz_avg"):
    row = ["mumax+"] + [f"{mz:+.2f}" for mz in mxpr["mz_avg"]]
    print("  ".join(f"{v:<10}" for v in row))
print("\nP_sw(J):")
print("  ".join(f"{v:<10}" for v in header))
for r in results:
    row = [r['build']] + [f"{p:.2f}" for p in r['psw']]
    print("  ".join(f"{v:<10}" for v in row))
if mxpr["ok"]:
    row = ["mumax+"] + [f"{p:.2f}" for p in mxpr["psw"]]
    print("  ".join(f"{v:<10}" for v in row))
if mx3r['ok'] and mx3r.get('wall_ms'):
    print(f"mumax3 wall: {mx3r['wall_ms']:.0f} ms")

out = {"scenario":"SOT Thermal Switching","cs":results,
       "mumax3_wall_ms":mx3r.get("wall_ms"), "mumaxplus": mxpr}
(pathlib.Path(__file__).parent/"43_results.json").write_text(
    json.dumps(out,indent=2,default=str),encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for r,c in zip(results,["C0","C1","C2"]):
        ax.plot(np.array(r['J_vals'])/1e12,r['psw'],'o-',color=c,label=r['build'],lw=2,ms=6)
    ax.set_xlabel("J_SOT (1e12 A/m2)"); ax.set_ylabel("P_sw")
    ax.set_title(f"SOT Thermal Switching (T={T_KELVIN}K, {N_TRIALS} trials)")
    ax.set_ylim(-0.05,1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent/"43_sot_thermal_comparison.png"),dpi=120)
    print("\nPlot: 43_sot_thermal_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
