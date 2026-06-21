"""
Notebook 46: µMAG SP#1 Phase Diagram -- Build Comparison
Permalloy squares L×L×10nm, 5nm cells.
Relax from S-state init and vortex init at each L.
Finds critical size L_c where E_vortex = E_sstate.
Runs across CS cuFFT_f64 / mumax3 / mumax+.

Reference: µMAG SP#1, L_c ≈ 110-120 nm (10nm thick, 5nm cells)
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_SP1 = pathlib.Path(__file__).parent / "mx3" / "sp1_phase.mx3"

# L sweep: Permalloy squares L×L×10nm, 5nm cells
L_NM_LIST = [80, 100, 110, 120, 130, 140, 160, 180, 200]
D = 5e-9   # cell size
T_NM = 10.0  # thickness nm
NZ = 2     # 10 nm / 5 nm cells

MU0    = bu.MU0
Ms     = 860e3
A      = 13e-12
alpha  = 0.5   # high damping for fast relaxation
T_RELAX = 3e-9

print("=" * 70)
print("Notebook 46: µMAG SP#1 Phase Diagram -- Build Comparison")
print(f"  Permalloy L×L×{T_NM:.0f}nm squares, d={D*1e9:.0f}nm cells")
print(f"  L values: {L_NM_LIST} nm")
print("=" * 70)


def run_cs_sp1(build_label):
    mm = bu.load_mm(build_label)
    mat = mm.Material()
    mat.Ms = Ms; mat.A_exchange = A; mat.K_uniaxial = 0.0
    mat.easy_axis = mm.Vec3(1, 0, 0); mat.alpha = alpha

    E_sstate, E_vortex = [], []
    t0 = time.perf_counter()

    for L_nm in L_NM_LIST:
        N = max(2, int(round(L_nm * 1e-9 / D)))
        g = mm.StructuredGrid(N, N, NZ, D, D, D)

        demag = mm.DemagFieldGPU(g)
        exch  = mm.ExchangeFieldGPU(g)
        fields = mm.FieldSumGPU()
        fields.add(exch)

        opts = mm.RelaxGPUOptions()
        opts.max_steps = 5000

        # S-state init: uniform +x with slight tilt
        m0 = mm.VectorField3D(g)
        arr = np.zeros((NZ, N, N, 3))
        arr[..., 0] = 1.0; arr[..., 1] = 0.05; arr[..., 2] = 0.01
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        arr /= norms
        mm.from_numpy(m0, arr)
        relax = mm.RelaxGPU(g)
        relax.upload(m0)
        relax.run(mat, demag, fields, opts)
        relax.download(m0)
        E_s = demag.energy(m0, mat) + exch.energy(m0, mat)
        E_sstate.append(E_s)

        # Vortex init: winding around center
        cx = N * D / 2; cy = N * D / 2
        arr_v = np.zeros((NZ, N, N, 3))
        for iy in range(N):
            for ix in range(N):
                rx = ix * D - cx; ry = iy * D - cy
                r = np.sqrt(rx*rx + ry*ry)
                if r < 1e-10: r = 1e-10
                arr_v[:, iy, ix, 0] = -ry / r
                arr_v[:, iy, ix, 1] =  rx / r
                # vortex core: small mz at center
                core_r = 3 * D
                arr_v[:, iy, ix, 2] = np.exp(-r**2 / (2 * core_r**2))
        norms_v = np.linalg.norm(arr_v, axis=-1, keepdims=True).clip(1e-10)
        arr_v /= norms_v
        mm.from_numpy(m0, arr_v)
        relax2 = mm.RelaxGPU(g)
        relax2.upload(m0)
        relax2.run(mat, demag, fields, opts)
        relax2.download(m0)
        E_v = demag.energy(m0, mat) + exch.energy(m0, mat)
        E_vortex.append(E_v)

    wall_ms = (time.perf_counter() - t0) * 1e3

    # Find L_c by linear interpolation of dE = E_v - E_s
    dE = np.array(E_vortex) - np.array(E_sstate)
    Lc = None
    for i in range(len(dE) - 1):
        if dE[i] >= 0 > dE[i+1] or dE[i] <= 0 < dE[i+1]:
            frac = -dE[i] / (dE[i+1] - dE[i])
            Lc = L_NM_LIST[i] + frac * (L_NM_LIST[i+1] - L_NM_LIST[i])
            break

    return {"build": build_label, "wall_ms": wall_ms,
            "L_nm": L_NM_LIST,
            "E_sstate_aJ": [e * 1e18 for e in E_sstate],
            "E_vortex_aJ": [e * 1e18 for e in E_vortex],
            "dE_pct": (dE / np.abs(np.array(E_sstate)) * 100).tolist(),
            "Lc_nm": float(Lc) if Lc else None}


results = []
for bl in ["cuFFT_f64"]:
    if not bu.BUILDS[bl].exists():
        print(f"[SKIP] {bl}: build not found"); continue
    print(f"\n--- {bl} ---")
    try:
        r = run_cs_sp1(bl); results.append(r)
        print(f"  Wall: {r['wall_ms']:.0f} ms")
        print(f"  L_c ≈ {r['Lc_nm']:.1f} nm" if r['Lc_nm'] else "  L_c: not found in sweep")
        for L, Es, Ev, dE in zip(r['L_nm'], r['E_sstate_aJ'], r['E_vortex_aJ'], r['dE_pct']):
            print(f"  L={L:3d}nm  E_s={Es:.3f}aJ  E_v={Ev:.3f}aJ  dE={dE:+.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")

# --- mumax3 ---
print("\n--- mumax3 ---")
mx3r = bu.run_mumax3(MX3_SP1, timeout_s=300) if MX3_SP1.exists() else {"ok": False, "error": "mx3 not found"}
if mx3r["ok"]:
    print(f"  Wall: {mx3r['wall_ms']:.0f} ms")
else:
    print(f"  {mx3r.get('error', 'failed')}")

# --- mumax+ ---
print("\n--- mumax+ (RelaxGPU at each L, minimize) ---")
def sp1_mumaxplus(mxp):
    import time, numpy as np
    E_sstate, E_vortex = [], []
    t0 = time.perf_counter()
    for L_nm in L_NM_LIST:
        N = max(2, int(round(L_nm * 1e-9 / D)))
        for init_type in ("sstate", "vortex"):
            world  = mxp.World(cellsize=(D, D, D))
            mag    = mxp.Ferromagnet(world, mxp.Grid((N, N, NZ)))
            mag.msat = Ms; mag.aex = A; mag.alpha = alpha
            if init_type == "sstate":
                mag.magnetization = (1.0, 0.05, 0.01)
            else:
                # Vortex init: component-first (3, nz, ny, nx)
                arr = np.zeros((3, NZ, N, N))
                cx = N * D / 2; cy = N * D / 2
                for iy in range(N):
                    for ix in range(N):
                        rx = ix * D - cx; ry = iy * D - cy
                        r = max(np.sqrt(rx*rx + ry*ry), 1e-10)
                        arr[0, :, iy, ix] = -ry / r
                        arr[1, :, iy, ix] =  rx / r
                        arr[2, :, iy, ix] = np.exp(-r**2 / (2*(3*D)**2))
                norms = np.linalg.norm(arr, axis=0, keepdims=True).clip(1e-10)
                arr /= norms
                mag.magnetization = arr
            mag.minimize()
            E = mag.total_energy()
            if init_type == "sstate":
                E_sstate.append(float(E))
            else:
                E_vortex.append(float(E))
    wall_ms = (time.perf_counter() - t0) * 1e3
    dE = np.array(E_vortex) - np.array(E_sstate)
    Lc = None
    for i in range(len(dE) - 1):
        if dE[i] >= 0 > dE[i+1] or dE[i] <= 0 < dE[i+1]:
            frac = -dE[i] / (dE[i+1] - dE[i])
            Lc = L_NM_LIST[i] + frac * (L_NM_LIST[i+1] - L_NM_LIST[i])
            break
    return {"wall_ms": wall_ms,
            "E_sstate_aJ": [e * 1e18 for e in E_sstate],
            "E_vortex_aJ": [e * 1e18 for e in E_vortex],
            "Lc_nm": float(Lc) if Lc else None}

mxpr = bu.run_mumaxplus(sp1_mumaxplus, timeout_s=600)
if mxpr["ok"]:
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms")
    print(f"  L_c ≈ {mxpr['Lc_nm']:.1f} nm" if mxpr.get('Lc_nm') else "  L_c: not found")
else:
    print(f"  {mxpr.get('error', 'failed')}")

print("\n" + "=" * 70)
print("SUMMARY -- SP#1 Phase Diagram (Permalloy squares, 5nm cells)")
print("=" * 70)
ref_lc = 116.0  # nm, from sp1_thickness.cpp at t=10nm
print(f"  Reference L_c (t=10nm, 5nm cells): ~{ref_lc:.0f} nm")
for r in results:
    lc = r['Lc_nm']
    lc_str = f"{lc:.1f} nm" if lc else "not found"
    print(f"  CS {r['build']:12s}: L_c = {lc_str}  wall={r['wall_ms']:.0f} ms")
if mx3r.get("ok"):
    print(f"  mumax3          : wall={mx3r['wall_ms']:.0f} ms  (Lc from table TBD)")
if mxpr.get("ok"):
    lc = mxpr.get('Lc_nm')
    lc_str = f"{lc:.1f} nm" if lc else "not found"
    print(f"  mumax+          : L_c = {lc_str}  wall={mxpr['wall_ms']:.0f} ms")

out = {"scenario": "SP#1 Phase Diagram", "L_nm": L_NM_LIST,
       "cs": results, "mumax3_wall_ms": mx3r.get("wall_ms"),
       "mumaxplus": mxpr}
(pathlib.Path(__file__).parent / "46_results.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"cuFFT_f64": "C0", "mumax+": "C3"}
    L_arr = np.array(L_NM_LIST)

    for r in results:
        Es = np.array(r['E_sstate_aJ']); Ev = np.array(r['E_vortex_aJ'])
        axes[0].plot(L_arr, Es, 'o-', color=colors.get(r['build'], 'C0'),
                     label=f"CS {r['build']} S-state", lw=1.5)
        axes[0].plot(L_arr, Ev, 's--', color=colors.get(r['build'], 'C0'),
                     label=f"CS {r['build']} Vortex", lw=1.5, alpha=0.7)
        if r['Lc_nm']:
            axes[0].axvline(r['Lc_nm'], color=colors.get(r['build'], 'C0'),
                            ls=':', lw=1, alpha=0.7)

    if mxpr.get("ok") and mxpr.get("E_sstate_aJ"):
        Es_mxp = np.array(mxpr['E_sstate_aJ'])
        Ev_mxp = np.array(mxpr['E_vortex_aJ'])
        axes[0].plot(L_arr, Es_mxp, '^-', color="C3", label="mumax+ S-state", lw=1.5)
        axes[0].plot(L_arr, Ev_mxp, 'v--', color="C3", label="mumax+ Vortex", lw=1.5, alpha=0.7)
        if mxpr.get('Lc_nm'):
            axes[0].axvline(mxpr['Lc_nm'], color="C3", ls=':', lw=1, alpha=0.7)

    axes[0].set_xlabel("L (nm)"); axes[0].set_ylabel("Energy (aJ)")
    axes[0].set_title("SP#1: Energy vs element size")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for r in results:
        dE = np.array(r['dE_pct'])
        axes[1].plot(L_arr, dE, 'o-', color=colors.get(r['build'], 'C0'),
                     label=f"CS {r['build']}", lw=2)
        if r['Lc_nm']:
            axes[1].axvline(r['Lc_nm'], color=colors.get(r['build'], 'C0'),
                            ls='--', lw=1, label=f"L_c={r['Lc_nm']:.0f}nm")

    if mxpr.get("ok") and mxpr.get("E_sstate_aJ"):
        Es_mxp = np.array(mxpr['E_sstate_aJ'])
        Ev_mxp = np.array(mxpr['E_vortex_aJ'])
        dE_mxp = (Ev_mxp - Es_mxp) / np.abs(Es_mxp) * 100
        axes[1].plot(L_arr, dE_mxp, '^-', color="C3", label="mumax+", lw=2)
        if mxpr.get('Lc_nm'):
            axes[1].axvline(mxpr['Lc_nm'], color="C3", ls='--', lw=1,
                            label=f"L_c(mxp+)={mxpr['Lc_nm']:.0f}nm")

    axes[1].axhline(0, color='k', lw=0.8, ls=':')
    axes[1].set_xlabel("L (nm)"); axes[1].set_ylabel("(E_vortex - E_sstate) / E_sstate (%)")
    axes[1].set_title("SP#1: Vortex vs S-state energy difference")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle(f"µMAG SP#1 — Permalloy L×L×10nm, 5nm cells", y=1.01)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / "46_sp1_comparison.png"), dpi=120)
    print("\nPlot: 46_sp1_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
