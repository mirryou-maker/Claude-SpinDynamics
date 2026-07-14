"""
Notebook 47: µMAG SP#3 Hysteresis -- Build Comparison
1 µm × 1 µm × 20 nm Permalloy element (100×100×2, 10nm cells).
Field sweep +150 → −150 mT, find nucleation H_nuc and switching H_sw.
Runs across CS cuFFT_f64 / mumax3 / mumax+.

Reference: µMAG SP#3, H_sw ≈ −20 mT (qualitative, 10nm cells)
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_SP3 = pathlib.Path(__file__).parent / "mx3" / "sp3_hysteresis.mx3"

MU0 = bu.MU0
Ms  = 860e3; A = 13e-12; alpha = 0.5
NX, NY, NZ = 100, 100, 2
DX = 10e-9  # 10nm cells

# Field sweep: +150 → −150 mT (29 points, ~10 mT spacing)
H_MV = np.concatenate([
    np.linspace(150, 10, 15),
    np.linspace(0, -150, 16),
])  # mT
H_AM = H_MV * 1e-3 / MU0   # A/m

T_RELAX = 2e-9  # relax time per field step

print("=" * 70)
print("Notebook 47: uMAG SP#3 Hysteresis -- Build Comparison")
print(f"  Permalloy {NX*DX*1e9:.0f}nm x {NY*DX*1e9:.0f}nm x {NZ*DX*1e9:.0f}nm ({NX}x{NY}x{NZ} cells)")
print(f"  Field sweep: {H_MV[0]:.0f} -> {H_MV[-1]:.0f} mT ({len(H_MV)} steps)")
print("=" * 70)


def _find_hsw(H_mT, mx_arr):
    """Find H_sw as field where <mx> crosses 0 (on descending branch)."""
    for i in range(len(mx_arr) - 1):
        if mx_arr[i] >= 0 > mx_arr[i+1]:
            frac = mx_arr[i] / (mx_arr[i] - mx_arr[i+1])
            return H_mT[i] + frac * (H_mT[i+1] - H_mT[i])
    return None


def run_cs_sp3(build_label):
    mm = bu.load_mm(build_label)
    g  = mm.StructuredGrid(NX, NY, NZ, DX, DX, DX)
    mat = mm.Material()
    mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = 0.0
    mat.easy_axis = mm.Vec3(1, 0, 0); mat.alpha = alpha

    demag = mm.DemagFieldGPU(g)
    exch  = mm.ExchangeFieldGPU(g)

    mx_list, mag_list = [], []
    opts = mm.RelaxGPUOptions(); opts.max_steps = 20000

    # Saturate at +150 mT
    zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(H_AM[0], 0, 0))
    fields = mm.FieldSumGPU(); fields.add(exch); fields.add(zeeman)
    m0 = mm.VectorField3D(g)
    arr0 = np.zeros((NZ, NX, NY, 3)); arr0[..., 0] = 1.0; arr0[..., 1] = 0.02
    norms = np.linalg.norm(arr0, axis=-1, keepdims=True); arr0 /= norms
    mm.from_numpy(m0, arr0)
    relax = mm.RelaxGPU(g); relax.upload(m0)
    relax.run(mat, demag, fields, opts); relax.download(m0)

    t0 = time.perf_counter()
    for H in H_AM:
        zeeman2 = mm.ZeemanFieldGPU(g, mm.Vec3(H, 0, 0))
        f2 = mm.FieldSumGPU(); f2.add(exch); f2.add(zeeman2)
        rel2 = mm.RelaxGPU(g); rel2.upload(m0)
        rel2.run(mat, demag, f2, opts); rel2.download(m0)
        arr = mm.to_numpy(m0)
        mx_avg = float(arr[..., 0].mean())
        m_avg  = float(np.linalg.norm(arr.mean(axis=(0, 1, 2))))
        mx_list.append(mx_avg); mag_list.append(m_avg)
    wall_ms = (time.perf_counter() - t0) * 1e3

    H_sw = _find_hsw(H_MV.tolist(), mx_list)
    return {"build": build_label, "wall_ms": wall_ms,
            "H_mT": H_MV.tolist(), "mx": mx_list, "mag": mag_list,
            "H_sw_mT": float(H_sw) if H_sw else None}


results = []
for bl in ["cuFFT_f64"]:
    if not bu.BUILDS[bl].exists():
        print(f"[SKIP] {bl}"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs_sp3(bl); results.append(r)
        print(f"  Wall: {r['wall_ms']:.0f} ms")
        hsw = r['H_sw_mT']
        print(f"  H_sw = {hsw:.1f} mT" if hsw else "  H_sw: not found")
        print(f"  Final <mx> = {r['mx'][-1]:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n--- mumax3 ---")
mx3r = bu.run_mumax3(MX3_SP3, timeout_s=300) if MX3_SP3.exists() else {"ok": False, "error": "mx3 not found"}
mx3_Hsw = None
if mx3r["ok"]:
    print(f"  Wall: {mx3r['wall_ms']:.0f} ms")
    outdir = MX3_SP3.parent / (MX3_SP3.stem + ".out")
    tbl = bu.parse_mumax3_table(outdir)
    if tbl is not None and len(tbl) > 1:
        # Default table: t(0), mx(1), my(2), mz(3)
        # Row k → B = 150 - k*10 mT  (row 0=+150, row 30=-150)
        mx_col = tbl[:, 1]
        B_mT   = [150.0 - k * 10.0 for k in range(len(mx_col))]
        mx3_Hsw = _find_hsw(B_mT, mx_col.tolist())
        print(f"  H_sw ~ {mx3_Hsw:.1f} mT" if mx3_Hsw else "  H_sw: not found in table")
    else:
        print("  table parse failed")
else:
    print(f"  {mx3r.get('error', 'failed')}")

print("\n--- mumax+ (field sweep + minimize) ---")
def sp3_mumaxplus(mxp):
    import time, numpy as np
    world  = mxp.World(cellsize=(DX, DX, DX))
    mag    = mxp.Ferromagnet(world, mxp.Grid((NX, NY, NZ)))
    mag.msat = Ms; mag.aex = A; mag.alpha = alpha
    mag.magnetization = (1.0, 0.02, 0.0)
    # Saturate at +150mT
    mag.bias_magnetic_field = (H_AM[0] * MU0, 0, 0)  # Tesla
    mag.minimize()
    mx_list, mag_list = [], []
    t0 = time.perf_counter()
    for H in H_AM:
        mag.bias_magnetic_field = (H * MU0, 0, 0)  # Tesla
        mag.minimize()
        avg = mag.magnetization.average()
        mx_list.append(float(avg[0]))
        mag_list.append(float(np.linalg.norm(avg)))
    wall_ms = (time.perf_counter() - t0) * 1e3
    H_sw = _find_hsw(H_MV.tolist(), mx_list)
    return {"wall_ms": wall_ms, "mx": mx_list, "mag": mag_list,
            "H_sw_mT": float(H_sw) if H_sw else None}

mxpr = bu.run_mumaxplus(sp3_mumaxplus, timeout_s=600)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms")
    hsw = mxpr.get("H_sw_mT")
    print(f"  H_sw = {hsw:.1f} mT" if hsw else "  H_sw: not found")
    print(f"  Final <mx> = {mxpr['mx'][-1]:.4f}")
else:
    print(f"  {mxpr.get('error', 'failed')}")

print("\n" + "=" * 70)
print("SUMMARY -- SP#3 Hysteresis (1um x 1um x 20nm Permalloy, 10nm cells)")
print(f"  uMAG reference H_sw ~ -20 mT (coarse-grid estimate)")
rows = []
for r in results:
    hsw = r['H_sw_mT']
    rows.append([r['build'], f"{r['wall_ms']:.0f}",
                 f"{hsw:.1f} mT" if hsw else "n/a"])
if mx3r.get("ok"):
    rows.append(["mumax3", f"{mx3r['wall_ms']:.0f}",
                 f"{mx3_Hsw:.1f} mT" if mx3_Hsw else "n/a"])
if mxpr.get("ok"):
    hsw = mxpr.get("H_sw_mT")
    rows.append(["mumax+", f"{mxpr['wall_ms']:.0f}",
                 f"{hsw:.1f} mT" if hsw else "n/a"])
bu.print_table(rows, ["Build", "wall_ms", "H_sw"])

out = {"scenario": "SP#3 Hysteresis", "H_mT": H_MV.tolist(),
       "cs": results, "mumax3_wall_ms": mx3r.get("wall_ms"),
       "mumaxplus": mxpr}
(pathlib.Path(__file__).parent / "47_results.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    color_map = {"cuFFT_f64": "C0", "mumax+": "C3"}

    for r in results:
        c = color_map.get(r['build'], "C0")
        axes[0].plot(r['H_mT'], r['mx'], 'o-', ms=3, lw=1.5, color=c,
                     label=f"CS {r['build']} ⟨mx⟩")
        axes[1].plot(r['H_mT'], r['mag'], 's-', ms=3, lw=1.5, color=c,
                     label=f"CS {r['build']} |⟨m⟩|", alpha=0.8)
        if r['H_sw_mT']:
            axes[0].axvline(r['H_sw_mT'], color=c, ls='--', lw=1,
                            label=f"H_sw={r['H_sw_mT']:.1f}mT")

    if mxpr.get("ok") and mxpr.get("mx"):
        axes[0].plot(H_MV, mxpr['mx'], '^-', ms=4, lw=2, color="C3",
                     label="mumax+ ⟨mx⟩")
        axes[1].plot(H_MV, mxpr['mag'], 'v-', ms=4, lw=2, color="C3",
                     label="mumax+ |⟨m⟩|", alpha=0.8)
        if mxpr.get('H_sw_mT'):
            axes[0].axvline(mxpr['H_sw_mT'], color="C3", ls='--', lw=1,
                            label=f"H_sw(mxp+)={mxpr['H_sw_mT']:.1f}mT")

    axes[0].axhline(0, color='k', lw=0.6, ls=':')
    axes[0].set_xlabel("H (mT)"); axes[0].set_ylabel("⟨mx⟩")
    axes[0].set_title("SP#3: Hysteresis loop ⟨mx⟩ vs H")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("H (mT)"); axes[1].set_ylabel("|⟨m⟩|")
    axes[1].set_title("SP#3: |⟨m⟩| vs H (multi-domain order parameter)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    plt.suptitle(f"µMAG SP#3 — {NX*DX*1e9:.0f}nm × {NY*DX*1e9:.0f}nm × {NZ*DX*1e9:.0f}nm Permalloy ({DX*1e9:.0f}nm cells)", y=1.01)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / "47_sp3_comparison.png"), dpi=120)
    print("\nPlot: 47_sp3_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
