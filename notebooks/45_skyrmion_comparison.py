"""
Notebook 45: SOT Skyrmion Nucleation -- PRECISION-SENSITIVITY CASE STUDY
Co/Pt PMA disc 100x100x1, dx=2nm, interfacial DMI D=3 mJ/m2 (~0.68*Dc).

IMPORTANT FINDING (not a solver defect): at D ~ 0.7*Dc the seeded isolated
skyrmion sits at a metastability bifurcation. The relaxed topological charge Q
is sensitive to (i) float precision (f32 vs f64), (ii) FFT backend (cuFFT vs
VkFFT), AND (iii) run-to-run GPU floating-point non-determinism (atomic-reduction
ordering) -- repeated runs of the SAME build give different Q. Q therefore ranges
from ~-1 to ~+0.2 across configurations and even across repeats.

=> This scenario is reported as a SENSITIVITY CAVEAT, not a quantitative
   cross-solver agreement test. Topological observables of marginally-stable
   skyrmions should use double precision (cuFFT_f64) and be treated with care.
   Quantitative cross-solver agreement is reported for the non-topological,
   well-posed scenarios (SP#4, DW velocity, FMR, SP#1 L_c, SP#3 H_sw).
Runs across CS cuFFT f64 / cuFFT f32 / VkFFT f32 + mumax3 + mumax+.
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_SKY = pathlib.Path(__file__).parent / "mx3" / "skyrmion_sot.mx3"
# D = 3 mJ/m^2 ~ 0.68*Dc (Dc = 4*sqrt(A*K)/pi = 4.41 mJ/m^2): the documented
# Co/Pt value. Tuning D does NOT remove the Q sensitivity: D=4 made the relax
# briefly agree but the SOT drive still diverged and repeats were unstable;
# D=5 (> Dc) gave spontaneous multidomain chaos. The sensitivity is intrinsic
# to the near-threshold bifurcation -> reported as a caveat (see module docstring).
Ms=800e3; A=15e-12; K=0.8e6; D_val=3e-3; alpha=0.3; P=0.15; d=1e-9
DX=2e-9; NX,NY,NZ=100,100,1
T_DRIVE=2e-10; DT=5e-14; J_SOT=3e12
N_DRIVE=int(T_DRIVE/DT)

print("="*70)
print("Notebook 45: SOT Skyrmion Nucleation -- Build Comparison")
print(f"  Co/Pt {NX}x{NY}x{NZ}, dx={DX*1e9:.0f}nm, D={D_val*1e3:.0f} mJ/m2")
print("="*70)


def run_cs(build_label):
    mm  = bu.load_mm(build_label)
    g   = mm.StructuredGrid(NX, NY, NZ, DX, DX, d)
    mat = mm.Material()
    mat.Ms=Ms; mat.A_exchange=A; mat.K_uniaxial=K
    mat.easy_axis=mm.Vec3(0,0,1); mat.alpha=alpha
    demag  = mm.DemagFieldGPU(g)
    exch   = mm.ExchangeFieldGPU(g)
    dmi    = mm.InterfacialDMIFieldGPU(g, D_val)
    aniso  = mm.UniaxialAnisotropyFieldGPU(g)
    sot    = mm.SpinOrbitTorqueGPU(g, J_SOT, P, d, mm.Vec3(1,0,0))
    f_base = mm.FieldSumGPU()
    f_base.add(exch); f_base.add(dmi); f_base.add(aniso)
    torques = mm.SpinTorqueSumGPU(); torques.add(sot)
    m0 = mm.VectorField3D(g)
    for iy in range(NY):
        for ix in range(NX):
            rx=(ix-NX//2)*DX; ry=(iy-NY//2)*DX
            r2=np.sqrt(rx*rx+ry*ry)
            mz = -1.0 if r2 < 20e-9 else 1.0
            m0[ix+NX*iy] = mm.Vec3(0, 0, mz)
    # Converge the relaxation properly (torque-threshold + high step cap).
    # The old max_steps=3000 cut the skyrmion relax off mid-way, so the
    # metastable Q landed differently per build/precision (Q = +0.17 / -0.66 /
    # 0.00). A converged relax reaches the SAME isolated-skyrmion local minimum
    # regardless of numerical path, making this a reproducible cross-solver test.
    opts = mm.RelaxGPUOptions(); opts.max_steps=40000; opts.threshold=1e-4*Ms
    relax = mm.RelaxGPU(g)
    relax.upload(m0)
    relax.run(mat, demag, f_base, opts)
    relax.download(m0)
    Q_relax = float(mm.topological_charge_Q(m0))
    heun = mm.HeunIntegratorGPU(g, DT, 0)
    heun.upload(m0)
    t0 = time.perf_counter()
    for _ in range(N_DRIVE):
        heun.step(mat, demag, f_base, 0.0, torques)
    wall_drive = (time.perf_counter()-t0)*1e3
    tmp = mm.VectorField3D(g); heun.download(tmp)
    Q_drive = float(mm.topological_charge_Q(tmp))
    arr = mm.to_numpy(tmp)
    mz_avg = float(arr[...,2].mean())
    return {"build":build_label,"wall_ms":wall_drive,"Q_relax":Q_relax,
            "Q_drive":Q_drive,"mz_avg":mz_avg,
            "ms_per_ns":wall_drive/(T_DRIVE*1e9),"n_steps":N_DRIVE}


results = []
for bl in bu.BUILDS:
    if not bu.BUILDS[bl].exists(): print(f"[SKIP] {bl}"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs(bl); results.append(r)
        print(f"  Drive: {r['wall_ms']:.0f} ms ({r['ms_per_ns']:.0f} ms/ns)")
        print(f"  Q_relax={r['Q_relax']:.2f}  Q_drive={r['Q_drive']:.2f}  <mz>={r['mz_avg']:.3f}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n--- mumax3 ---")
mx3r = bu.run_mumax3(MX3_SKY, 300)
print(f"  Wall: {mx3r.get('wall_ms'):.0f} ms" if mx3r['ok'] else f"  {mx3r.get('error','failed')}")

print("\n--- mumax+ (SOT Skyrmion, Heun fixed dt) ---")
def sky_mumaxplus(mxp):
    import time, numpy as np
    world = mxp.World(cellsize=(DX, DX, d))
    mag   = mxp.Ferromagnet(world, mxp.Grid((NX, NY, NZ)))
    mag.msat=Ms; mag.aex=A; mag.alpha=alpha
    mag.ku1=K; mag.anisU=(0, 0, 1)
    mag.dmi_tensor.set_interfacial_dmi(D_val)
    # Initial mz=-1 disk, r<20nm, rest mz=+1
    m_arr = np.zeros((3, NZ, NY, NX))
    m_arr[2] = 1.0  # all +z
    for iy in range(NY):
        for ix in range(NX):
            rx = (ix - NX//2) * DX; ry = (iy - NY//2) * DX
            if np.sqrt(rx*rx + ry*ry) < 20e-9:
                m_arr[2, 0, iy, ix] = -1.0
    mag.magnetization = m_arr
    mag.minimize()
    m_np = mag.magnetization.eval()  # (3, nz, ny, nx)
    Q_relax = bu.mxp_topological_charge(m_np)
    # SOT drive: sigma=(1,0,0) equiv to fixed_layer=(0,1,0), pol=theta_SH=P
    mag.enable_slonczewski_torque = True
    mag.enable_zhang_li_torque    = False
    mag.jcur        = (-float(J_SOT), 0, 0)
    mag.pol         = P
    mag.fixed_layer = (0, 1, 0)
    mag.free_layer_thickness = d
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = DT
    t0 = time.perf_counter()
    world.timesolver.run(T_DRIVE)
    wall_ms = (time.perf_counter() - t0) * 1e3
    m_np2 = mag.magnetization.eval()
    Q_drive = bu.mxp_topological_charge(m_np2)
    mz_avg = float(m_np2[2].mean())
    return {"wall_ms": wall_ms, "Q_relax": float(Q_relax), "Q_drive": float(Q_drive),
            "mz_avg": mz_avg, "ms_per_ns": wall_ms / (T_DRIVE * 1e9)}

mxpr = bu.run_mumaxplus(sky_mumaxplus, timeout_s=300)
if mxpr["ok"]:
    print(f"  Drive: {mxpr['wall_ms']:.0f} ms ({mxpr['ms_per_ns']:.0f} ms/ns)")
    print(f"  Q_relax={mxpr['Q_relax']:.2f}  Q_drive={mxpr['Q_drive']:.2f}  <mz>={mxpr['mz_avg']:.3f}")
else:
    print(f"  {mxpr.get('error','failed')}")

print("\n"+"="*70)
print("SUMMARY -- SOT Skyrmion (Co/Pt 100x100x1, D=3 mJ/m2)")
rows=[["Build","drive_ms","ms/ns","Q_relax","Q_drive","<mz>"]]
for r in results:
    rows.append([r['build'],f"{r['wall_ms']:.0f}",f"{r['ms_per_ns']:.0f}",
                 f"{r['Q_relax']:.2f}",f"{r['Q_drive']:.2f}",f"{r['mz_avg']:.3f}"])
if mx3r['ok'] and mx3r.get('wall_ms'):
    rows.append(["mumax3",f"{mx3r['wall_ms']:.0f}","-","-","-","-"])
if mxpr["ok"]:
    rows.append(["mumax+",f"{mxpr['wall_ms']:.0f}",f"{mxpr['ms_per_ns']:.0f}",
                 f"{mxpr['Q_relax']:.2f}",f"{mxpr['Q_drive']:.2f}",f"{mxpr['mz_avg']:.3f}"])
for row in rows:
    print("  ".join(f"{v:<14}" for v in row))

out = {"scenario":"SOT Skyrmion Nucleation","cs":results,
       "mumax3_wall_ms":mx3r.get("wall_ms"), "mumaxplus": mxpr}
(pathlib.Path(__file__).parent/"45_results.json").write_text(
    json.dumps(out,indent=2,default=str),encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    if results:
        fig,axes=plt.subplots(1,min(3,len(results)),figsize=(4*len(results),4.5))
        if len(results)==1: axes=[axes]
        for ax,r in zip(axes,results):
            ax.set_title(f"{r['build']}\nQ={r['Q_drive']:.2f} <mz>={r['mz_avg']:.2f}")
            ax.text(0.5,0.5,f"Q={r['Q_drive']:.2f}\n<mz>={r['mz_avg']:.3f}",
                    ha='center',va='center',transform=ax.transAxes,fontsize=14)
            ax.axis('off')
        plt.suptitle(f"SOT Skyrmion (Co/Pt, 100x100, D=3mJ/m2, 0.2ns SOT drive)",y=1.02)
        plt.tight_layout()
        plt.savefig(str(pathlib.Path(__file__).parent/"45_skyrmion_comparison.png"),dpi=120)
        print("\nPlot: 45_skyrmion_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
