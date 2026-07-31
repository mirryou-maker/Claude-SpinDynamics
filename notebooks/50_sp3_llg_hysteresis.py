"""
Notebook 50: µMAG SP#3 Hysteresis — LLG Protocol vs Energy Minimization
Compares two relaxation strategies to understand the H_sw discrepancy:
  NB47 (minimize): CS -13.8 mT, mumax3 -13.3 mT, mumax+ -6.0 mT
  NB50 (LLG):      CS ?, mumax3 ?, mumax+ ?  → expect ~-20 mT (µMAG ref)

LLG protocol: RK4IntegratorGPU + run_until_converged_gpu (tol=0.5°)
Minimize protocol (NB47): RelaxGPU gradient descent
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_SP3_LLG = pathlib.Path(__file__).parent / "mx3" / "sp3_llg.mx3"

MU0 = bu.MU0
Ms  = 860e3; A = 13e-12; alpha = 0.5
NX, NY, NZ = 100, 100, 2
DX = 10e-9

# Same field sweep as NB47 (10 mT steps, 31 points)
H_MV = np.arange(150, -160, -10, dtype=float)[:31]   # 150, 140, ..., -150 mT
H_AM = H_MV * 1e-3 / MU0

DT   = 5e-13   # 500 fs — stable for alpha=0.5 Permalloy
TOL  = 0.5     # degrees: convergence threshold
MAX_STEPS = 12000  # 12000 × 500 fs = 6 ns max per field step

print("=" * 70)
print("Notebook 50: SP#3 Hysteresis -- LLG Protocol")
print(f"  Grid: {NX}x{NY}x{NZ}, dx={DX*1e9:.0f} nm")
print(f"  alpha={alpha}, dt={DT*1e15:.0f} fs, tol={TOL}deg, max={MAX_STEPS*DT*1e9:.1f} ns/step")
print(f"  Field: {H_MV[0]:.0f} -> {H_MV[-1]:.0f} mT ({len(H_MV)} points)")
print("=" * 70)


def _find_hsw(H_mT, mx_arr):
    for i in range(len(mx_arr) - 1):
        if mx_arr[i] >= 0 > mx_arr[i+1]:
            frac = mx_arr[i] / (mx_arr[i] - mx_arr[i+1])
            return H_mT[i] + frac * (H_mT[i+1] - H_mT[i])
    return None


def run_cs_sp3_llg(build_label):
    mm = bu.load_mm(build_label)
    g  = mm.StructuredGrid(NX, NY, NZ, DX, DX, DX)
    mat = mm.Material()
    mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = 0.0
    mat.easy_axis = mm.Vec3(1, 0, 0); mat.alpha = alpha

    demag = mm.DemagFieldGPU(g)
    exch  = mm.ExchangeFieldGPU(g)

    # Initial saturated state (+x)
    m0   = mm.VectorField3D(g)
    arr0 = np.zeros((NZ, NX, NY, 3))
    arr0[..., 0] = 1.0; arr0[..., 1] = 0.02
    arr0 /= np.linalg.norm(arr0, axis=-1, keepdims=True)
    mm.from_numpy(m0, arr0)

    # Single RK4 integrator reused across all field steps
    rk4 = mm.RK4IntegratorGPU(g, DT)
    rk4.upload(m0)   # initial upload — required before run_until_converged_gpu

    # Saturate at +150 mT before starting timing
    ze0   = mm.ZeemanFieldGPU(g, mm.Vec3(H_AM[0], 0, 0))
    fsum0 = mm.FieldSumGPU(); fsum0.add(exch); fsum0.add(ze0)
    rk4.invalidate_graph()
    mm.run_until_converged_gpu(rk4, mat, demag, fsum0, m0,
                               tol_deg=TOL, max_steps=MAX_STEPS)
    # GPU state now has saturated +x; m0 (CPU) not yet updated — OK, loop handles it

    mx_list, mag_list, steps_list = [], [], []
    t0 = time.perf_counter()
    for H in H_AM:
        ze   = mm.ZeemanFieldGPU(g, mm.Vec3(H, 0, 0))
        fsum = mm.FieldSumGPU(); fsum.add(exch); fsum.add(ze)
        rk4.invalidate_graph()
        mm.run_until_converged_gpu(rk4, mat, demag, fsum, m0,
                                   tol_deg=TOL, max_steps=MAX_STEPS)
        # Download GPU state → m0 so that to_numpy reads correct data
        # and next iteration inherits the current GPU state (no re-upload needed)
        rk4.download(m0)
        arr = mm.to_numpy(m0)
        mx_avg = float(arr[..., 0].mean())
        m_avg  = float(np.linalg.norm(arr.mean(axis=(0, 1, 2))))
        mx_list.append(mx_avg); mag_list.append(m_avg)
    wall_ms = (time.perf_counter() - t0) * 1e3

    H_sw = _find_hsw(H_MV.tolist(), mx_list)
    return {"build": build_label, "wall_ms": wall_ms,
            "H_mT": H_MV.tolist(), "mx": mx_list, "mag": mag_list,
            "H_sw_mT": float(H_sw) if H_sw else None}


# --- CS ---
results = []
for bl in ["cuFFT_f64"]:
    if not bu.BUILDS[bl].exists():
        print(f"[SKIP] {bl}"); continue
    print(f"\n--- CS {bl} (LLG, run_until_converged_gpu) ---")
    try:
        r = run_cs_sp3_llg(bl); results.append(r)
        hsw = r["H_sw_mT"]
        print(f"  Wall: {r['wall_ms']:.0f} ms")
        print(f"  H_sw = {hsw:.1f} mT" if hsw else "  H_sw: not found")
        print(f"  Final <mx> = {r['mx'][-1]:.4f}")
        # Show a few critical field points around switching
        for i, (H, mx) in enumerate(zip(H_MV, r["mx"])):
            if -40 <= H <= 20:
                print(f"    H={H:+.0f} mT  <mx>={mx:+.4f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {e}")

# --- mumax3 (relax protocol) ---
print("\n--- mumax3 (LLG relax()) ---")
mx3r = bu.run_mumax3(MX3_SP3_LLG, timeout_s=600) if MX3_SP3_LLG.exists() \
       else {"ok": False, "error": "mx3 not found"}
mx3_Hsw = None
if mx3r["ok"]:
    print(f"  Wall: {mx3r['wall_ms']:.0f} ms")
    outdir = MX3_SP3_LLG.parent / (MX3_SP3_LLG.stem + ".out")
    tbl = bu.parse_mumax3_table(outdir)
    if tbl is not None and len(tbl) > 1:
        mx_col = tbl[:, 1]
        B_mT   = [150.0 - k * 10.0 for k in range(len(mx_col))]
        mx3_Hsw = _find_hsw(B_mT, mx_col.tolist())
        print(f"  H_sw = {mx3_Hsw:.1f} mT" if mx3_Hsw else "  H_sw: not found")
    else:
        print("  table parse failed")
else:
    print(f"  {mx3r.get('error', 'failed')}")

# --- mumax+ (LLG timesolver) ---
print("\n--- mumax+ (LLG timesolver, 2 ns/step) ---")
T_RUN_PER_STEP = 2e-9   # 2 ns LLG relaxation per field step

def sp3_mumaxplus_llg(mxp):
    import time as tmod, numpy as np
    world  = mxp.World(cellsize=(DX, DX, DX))
    mag    = mxp.Ferromagnet(world, mxp.Grid((NX, NY, NZ)))
    mag.msat = Ms; mag.aex = A; mag.alpha = alpha
    mag.magnetization = (1.0, 0.02, 0.0)
    # Saturate at +150 mT
    mag.bias_magnetic_field = (H_AM[0] * MU0, 0, 0)
    world.timesolver.run(T_RUN_PER_STEP)
    mx_list, mag_list = [], []
    t0 = tmod.perf_counter()
    for H in H_AM:
        mag.bias_magnetic_field = (H * MU0, 0, 0)
        world.timesolver.run(T_RUN_PER_STEP)
        avg = mag.magnetization.average()
        mx_list.append(float(avg[0]))
        mag_list.append(float(np.linalg.norm(avg)))
    wall_ms = (tmod.perf_counter() - t0) * 1e3
    H_sw = None
    for i in range(len(mx_list) - 1):
        if mx_list[i] >= 0 > mx_list[i+1]:
            frac = mx_list[i] / (mx_list[i] - mx_list[i+1])
            H_sw = float(H_MV[i] + frac * (H_MV[i+1] - H_MV[i]))
            break
    return {"wall_ms": wall_ms, "mx": mx_list, "mag": mag_list,
            "H_sw_mT": H_sw}

mxpr = bu.run_mumaxplus(sp3_mumaxplus_llg, timeout_s=1800)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms")
    hsw = mxpr.get("H_sw_mT")
    print(f"  H_sw = {hsw:.1f} mT" if hsw else "  H_sw: not found")
    print(f"  Final <mx> = {mxpr['mx'][-1]:.4f}")
else:
    print(f"  {mxpr.get('error', 'failed')}")

# --- SUMMARY ---
print("\n" + "=" * 70)
print("SUMMARY -- SP#3 Hysteresis Protocol Comparison")
print(f"  uMAG reference H_sw ~ -20 mT (LLG integration protocol)")
print(f"  NB47 (minimize): CS -13.8 mT, mumax3 -13.3 mT, mumax+ -6.0 mT")
print("  NB50 (LLG relax):")
rows = []
for r in results:
    hsw = r["H_sw_mT"]
    rows.append([f"CS {r['build']}", f"{r['wall_ms']:.0f}",
                 f"{hsw:.1f} mT" if hsw else "n/a", "LLG run_until_converged"])
if mx3r.get("ok"):
    rows.append(["mumax3", f"{mx3r['wall_ms']:.0f}",
                 f"{mx3_Hsw:.1f} mT" if mx3_Hsw else "n/a", "LLG relax()"])
if mxpr.get("ok"):
    hsw = mxpr.get("H_sw_mT")
    rows.append(["mumax+", f"{mxpr['wall_ms']:.0f}",
                 f"{hsw:.1f} mT" if hsw else "n/a", "LLG timesolver 2ns/step"])
bu.print_table(rows, ["Build", "wall_ms", "H_sw (LLG)", "Method"])

out = {"scenario": "SP#3 Hysteresis LLG", "H_mT": H_MV.tolist(),
       "cs": results, "mumax3": mx3r, "mumaxplus": mxpr,
       "nb47_minimize_ref": {"cs_Hsw": -13.8, "mu3_Hsw": -13.3, "mup_Hsw": -6.0},
       "mumax_ref_Hsw": -20.0}
(pathlib.Path(__file__).parent / "50_results.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"cuFFT_f64": "C0", "mumax3": "C1", "mumax+": "C3"}

    for r in results:
        c = colors.get(r["build"], "C0")
        ax.plot(r["H_mT"], r["mx"], 'o-', ms=3, lw=1.5, color=c,
                label=f"CS {r['build']} LLG")
        if r["H_sw_mT"]:
            ax.axvline(r["H_sw_mT"], color=c, ls='--', lw=1,
                       label=f"H_sw(CS)={r['H_sw_mT']:.1f}mT")

    if mx3r.get("ok") and mx3_Hsw:
        outdir = MX3_SP3_LLG.parent / (MX3_SP3_LLG.stem + ".out")
        tbl = bu.parse_mumax3_table(outdir)
        if tbl is not None:
            B_mT = [150.0 - k*10.0 for k in range(len(tbl))]
            ax.plot(B_mT, tbl[:, 1], 's-', ms=4, lw=1.5, color="C1",
                    label="mumax3 relax()")
            ax.axvline(mx3_Hsw, color="C1", ls='--', lw=1,
                       label=f"H_sw(mu3)={mx3_Hsw:.1f}mT")

    if mxpr.get("ok") and mxpr.get("mx"):
        ax.plot(H_MV, mxpr["mx"], '^-', ms=4, lw=2, color="C3",
                label="mumax+ timesolver")
        if mxpr.get("H_sw_mT"):
            ax.axvline(mxpr["H_sw_mT"], color="C3", ls='--', lw=1,
                       label=f"H_sw(mu+)={mxpr['H_sw_mT']:.1f}mT")

    ax.axvline(-20, color='k', ls=':', lw=1.5, label="µMAG ref -20 mT")
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel("H (mT)"); ax.set_ylabel("<mx>")
    ax.set_title("SP#3: LLG protocol vs minimize — H_sw comparison")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / "50_sp3_llg_hysteresis.png"), dpi=120)
    print("\nPlot: 50_sp3_llg_hysteresis.png")
except Exception as e:
    print(f"Plot error: {e}")
