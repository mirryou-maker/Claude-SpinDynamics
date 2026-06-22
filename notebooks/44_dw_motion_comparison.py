"""
Notebook 44: Zhang-Li DW Motion -- Build Comparison
Permalloy strip 400x20x1, dx=4nm (1.6um x 80nm).
Sweeps J, measures v_DW vs J. Shows Walker breakdown.
Runs across CS cuFFT f64 / cuFFT f32 / VkFFT f32 + mumax3.
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_DW = pathlib.Path(__file__).parent / "mx3" / "dw_motion.mx3"
Ms=860e3; A=13e-12; K=0.0; alpha=0.01; P=0.5; xi=0.05
DX=4e-9; NX,NY,NZ=400,20,1
DT=5e-14; T_SIM=5e-10
N_STEPS=int(T_SIM/DT)
J_VALS=np.array([0.5,1.0,2.0,4.0,8.0])*1e12
mu_B=9.274e-24; e_ch=1.6022e-19

print("="*70)
print("Notebook 44: Zhang-Li DW Motion -- Build Comparison")
print(f"  Py strip {NX}x{NY}x{NZ}, dx={DX*1e9:.0f} nm")
print("="*70)


def _init_dw(mm, g):
    m0 = mm.VectorField3D(g)
    mid = NX//2
    for idx in range(NX*NY*NZ):
        x = idx % NX
        if x < mid-5:
            m0[idx] = mm.Vec3(1,0,0)
        elif x > mid+5:
            m0[idx] = mm.Vec3(-1,0,0)
        else:
            theta = np.pi*(x-mid+5)/10.0
            m0[idx] = mm.Vec3(float(np.cos(theta)), float(np.sin(theta)), 0.0)
    return m0


def _dw_pos(arr):
    mx = arr[0,:,:,0]
    grad_x = np.gradient(mx.mean(0))
    return float(np.argmin(grad_x)) * DX


def run_cs(build_label):
    mm  = bu.load_mm(build_label)
    g   = mm.StructuredGrid(NX, NY, NZ, DX, DX, DX)
    mat = mm.Material()
    mat.Ms=Ms; mat.A_exchange=A; mat.K_uniaxial=K; mat.alpha=alpha
    demag   = mm.DemagFieldGPU(g)
    exch    = mm.ExchangeFieldGPU(g)
    fields  = mm.FieldSumGPU(); fields.add(exch)
    zl      = mm.ZhangLiSTTGPU(g, mm.Vec3(1e12, 0, 0), P, xi)
    torques = mm.SpinTorqueSumGPU(); torques.add(zl)
    m0 = _init_dw(mm, g)
    t0_all = time.perf_counter(); v_list = []
    for J in J_VALS:
        zl.J = mm.Vec3(float(J), 0, 0)
        heun = mm.HeunIntegratorGPU(g, DT, 0)
        heun.upload(m0)
        tmp = mm.VectorField3D(g); heun.download(tmp)
        pos0 = _dw_pos(mm.to_numpy(tmp))
        for _ in range(N_STEPS):
            heun.step(mat, demag, fields, 0.0, torques)
        tmp = mm.VectorField3D(g); heun.download(tmp)
        pos1 = _dw_pos(mm.to_numpy(tmp))
        v_list.append(float((pos1-pos0)/T_SIM))
    wall_ms = (time.perf_counter()-t0_all)*1e3
    return {"build":build_label,"wall_ms":wall_ms,
            "J_vals":J_VALS.tolist(),"v_list":v_list,
            "ms_per_ns":wall_ms/(T_SIM*1e9)}


results = []
for bl in bu.BUILDS:
    if not bu.BUILDS[bl].exists(): print(f"[SKIP] {bl}"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs(bl); results.append(r)
        print(f"  Wall: {r['wall_ms']:.0f} ms ({r['ms_per_ns']:.1f} ms/ns)")
        for J,v in zip(J_VALS,r['v_list']):
            print(f"  J={J/1e12:.1f}e12  v={v:.0f} m/s")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n--- mumax3 ---")
mx3r = bu.run_mumax3(MX3_DW, 300)
print(f"  Wall: {mx3r.get('wall_ms'):.0f} ms" if mx3r['ok'] else f"  {mx3r.get('error','failed')}")

print("\n--- mumax+ (Zhang-Li CIP, Heun fixed dt) ---")
def dw_mumaxplus(mxp):
    import time, numpy as np
    world = mxp.World(cellsize=(DX, DX, DX))
    mag   = mxp.Ferromagnet(world, mxp.Grid((NX, NY, NZ)))
    mag.msat=Ms; mag.aex=A; mag.alpha=alpha
    mag.enable_zhang_li_torque    = True
    mag.enable_slonczewski_torque = False
    mag.xi  = xi
    mag.pol = P
    # Build DW initial state (same as CS _init_dw)
    mid = NX // 2
    m_arr = np.zeros((3, NZ, NY, NX))   # (3, nz, ny, nx)
    for iy in range(NY):
        for ix in range(NX):
            if ix < mid - 5:
                m_arr[0, 0, iy, ix] = 1.0
            elif ix > mid + 5:
                m_arr[0, 0, iy, ix] = -1.0
            else:
                theta = np.pi * (ix - mid + 5) / 10.0
                m_arr[0, 0, iy, ix] = float(np.cos(theta))
                m_arr[1, 0, iy, ix] = float(np.sin(theta))
    m_arr /= np.linalg.norm(m_arr, axis=0, keepdims=True).clip(1e-30)
    t0_all = time.perf_counter()
    v_list = []
    for J in J_VALS:
        mag.jcur = (float(J), 0, 0)
        mag.magnetization = m_arr
        world.timesolver.adaptive_timestep = False
        world.timesolver.timestep = DT
        world.timesolver.run(0)  # reset time to 0 for each J
        # get initial DW position
        m_np0 = mag.magnetization.eval()   # (3, nz, ny, nx)
        mx0 = m_np0[0, 0, :, :].mean(0)   # (nx,)
        grad0 = np.gradient(mx0); pos0 = float(np.argmin(grad0)) * DX
        world.timesolver.run(T_SIM)
        m_np1 = mag.magnetization.eval()
        mx1 = m_np1[0, 0, :, :].mean(0)
        grad1 = np.gradient(mx1); pos1 = float(np.argmin(grad1)) * DX
        v_list.append((pos1 - pos0) / T_SIM)
    wall_ms = (time.perf_counter() - t0_all) * 1e3
    return {"wall_ms": wall_ms, "v_list": [float(v) for v in v_list],
            "ms_per_ns": wall_ms / (T_SIM * len(J_VALS) * 1e9)}

mxpr = bu.run_mumaxplus(dw_mumaxplus, timeout_s=300)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms ({mxpr['ms_per_ns']:.1f} ms/ns per run)")
    for J, v in zip(J_VALS, mxpr["v_list"]):
        print(f"  J={J/1e12:.1f}e12  v={v:.0f} m/s")
else:
    print(f"  {mxpr.get('error','failed')}")

print("\n"+"="*70)
print("SUMMARY -- DW Velocity vs J (Zhang-Li, Py 400x20x1)")
header = ["J (e12 A/m2)"] + [f"{J/1e12:.1f}" for J in J_VALS]
print("  ".join(f"{v:<14}" for v in header))
for r in results:
    row = [r['build']] + [f"{v:.0f} m/s" for v in r['v_list']]
    print("  ".join(f"{v:<14}" for v in row))
if mx3r['ok'] and mx3r.get('wall_ms'):
    print(f"mumax3 wall: {mx3r['wall_ms']:.0f} ms")
if mxpr["ok"]:
    row = ["mumax+"] + [f"{v:.0f} m/s" for v in mxpr["v_list"]]
    print("  ".join(f"{v:<14}" for v in row))

out = {"scenario":"Zhang-Li DW Motion","cs":results,
       "mumax3_wall_ms":mx3r.get("wall_ms"), "mumaxplus": mxpr}
(pathlib.Path(__file__).parent/"44_results.json").write_text(
    json.dumps(out,indent=2,default=str),encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for r,c in zip(results,["C0","C1","C2"]):
        ax.plot(np.array(r['J_vals'])/1e12,np.abs(r['v_list']),'o-',
                color=c,label=r['build'],lw=2,ms=6)
    ax.set_xlabel("J (1e12 A/m2)"); ax.set_ylabel("|v_DW| (m/s)")
    ax.set_title(f"Zhang-Li DW velocity vs J (Py {NX}x{NY}x{NZ}, dx={DX*1e9:.0f}nm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent/"44_dw_motion_comparison.png"),dpi=120)
    print("\nPlot: 44_dw_motion_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
