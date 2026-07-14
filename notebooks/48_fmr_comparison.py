"""
Notebook 48: FMR Spectrum -- Build Comparison
Macrospin free precession: 1×1×1, B_bias=50mT along z, alpha=0.005.
FFT of <mx>(t) → Kittel resonance frequency.
Runs across CS CPU RK4 / mumax3 / mumax+.

Kittel (no demag, no anisotropy):
  f_FMR = γ₀/(2π) × B_bias = 1.76×10¹¹/(2π) × 0.05 = 1.400 GHz

Physics note: 5° initial tilt (m ≈ ẑ) → free precession around B_bias.
  With α=0.005, coherence length >> 5ns → sharp FFT peak.
"""
import sys, pathlib, time, json
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_utils as bu
sys.stdout.reconfigure(encoding="utf-8")

MX3_FMR = pathlib.Path(__file__).parent / "mx3" / "fmr_macrospin.mx3"

MU0    = bu.MU0
gamma0 = 1.76e11   # rad/(T·s)

B_BIAS   = 50e-3     # T
H_BIAS   = B_BIAS / MU0
F_KITTEL = gamma0 / (2 * np.pi) * B_BIAS   # Hz

DT    = 1e-12    # 1 ps
T_TOT = 5e-9     # 5 ns
N_STEP = int(T_TOT / DT)  # 5000

# Initial tilt: 5° off z-axis
M0_X = np.sin(np.deg2rad(5.0))   # 0.08716
M0_Z = np.cos(np.deg2rad(5.0))   # 0.99619

print("=" * 70)
print("Notebook 48: FMR Spectrum -- Build Comparison")
print(f"  Macrospin 1x1x1, B_bias={B_BIAS*1e3:.0f}mT (z), alpha=0.005")
print(f"  Kittel f_FMR = {F_KITTEL/1e9:.4f} GHz")
print(f"  T={T_TOT*1e9:.0f}ns, dt={DT*1e12:.0f}ps, df={1/(T_TOT*1e9):.3f}GHz")
print("=" * 70)


def _fft_peak(t_arr, mx_arr):
    """Return peak frequency in GHz near the Kittel frequency."""
    mx = np.array(mx_arr)
    N  = len(mx)
    dt = (t_arr[-1] - t_arr[0]) / (N - 1) if N > 1 else DT
    freqs = np.fft.rfftfreq(N, d=dt)
    psd   = np.abs(np.fft.rfft(mx - mx.mean()))**2
    mask  = (freqs > F_KITTEL * 0.5) & (freqs < F_KITTEL * 2.0)
    if not mask.any():
        return None, freqs / 1e9, psd
    idx_peak = np.argmax(psd[mask])
    f_peak = freqs[mask][idx_peak]
    return f_peak / 1e9, freqs / 1e9, psd


# --- CS CPU RK4 ---
print("\n--- CS CPU (RK4, per-step constant B_bias) ---")
results = []
cpu_build = pathlib.Path(__file__).parent.parent / "build" / "windows-msvc" / "python"
if cpu_build.exists():
    sys.path.insert(0, str(cpu_build))
    try:
        import micromag as mm_cpu
        g   = mm_cpu.StructuredGrid(1, 1, 1, 5e-9, 5e-9, 5e-9)
        mat = mm_cpu.Material.permalloy(); mat.alpha = 0.005
        m   = mm_cpu.VectorField3D(g)
        arr0 = np.zeros((1, 1, 1, 3))
        arr0[0, 0, 0] = [M0_X, 0, M0_Z]
        mm_cpu.from_numpy(m, arr0)

        zeeman = mm_cpu.ZeemanField(mm_cpu.Vec3(0, 0, H_BIAS))
        heff   = mm_cpu.EffectiveFieldSum(); heff.add(zeeman)
        integ  = mm_cpu.RK4Integrator(DT)

        t_arr, mx_arr = [], []
        t_sim = 0.0
        t0 = time.perf_counter()
        for _ in range(N_STEP):
            integ.step(m, mat, heff)
            t_sim += DT
            avg = mm_cpu.mean_magnetization(m)
            mx_arr.append(avg[0]); t_arr.append(t_sim)
        wall_ms = (time.perf_counter() - t0) * 1e3

        f_peak, freqs, psd = _fft_peak(t_arr, mx_arr)
        err = abs(f_peak - F_KITTEL/1e9) / (F_KITTEL/1e9) * 100 if f_peak else None
        print(f"  Wall: {wall_ms:.0f} ms")
        print(f"  f_peak = {f_peak:.4f} GHz  err={err:.2f}%" if f_peak else "  f_peak: not found")
        results.append({"build": "CPU_RK4", "wall_ms": wall_ms,
                         "t_ns": [t*1e9 for t in t_arr[::5]],
                         "mx": mx_arr[::5], "f_peak_GHz": f_peak,
                         "freqs_GHz": freqs[:500].tolist(),
                         "psd": psd[:500].tolist()})
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print("  [SKIP] CPU build not found")

# --- mumax3 ---
print("\n--- mumax3 (free precession, 1ps output) ---")
mx3r = bu.run_mumax3(MX3_FMR, timeout_s=120) if MX3_FMR.exists() else {"ok": False, "error": "mx3 not found"}
mx3_fpeak = None
if mx3r["ok"]:
    print(f"  Wall: {mx3r['wall_ms']:.0f} ms")
    outdir = MX3_FMR.parent / (MX3_FMR.stem + ".out")
    tbl = bu.parse_mumax3_table(outdir)
    if tbl is not None and len(tbl) > 10:
        # table columns: [t, m.x, m.y, m.z]
        t_col  = tbl[:, 0].tolist()
        mx_col = tbl[:, 1].tolist()
        mx3_fpeak, _, _ = _fft_peak(t_col, mx_col)
        err = abs(mx3_fpeak - F_KITTEL/1e9) / (F_KITTEL/1e9) * 100 if mx3_fpeak else None
        print(f"  f_peak = {mx3_fpeak:.4f} GHz  err={err:.2f}%" if mx3_fpeak else "  f_peak: not found")
else:
    print(f"  {mx3r.get('error', 'failed')}")

# --- mumax+ ---
print("\n--- mumax+ (free precession, batch run + <mx> sampling) ---")
def fmr_mumaxplus(mxp):
    import time, numpy as np
    BATCH  = 50        # run 50 steps per timesolver call (50ps)
    N_BATCHES = N_STEP // BATCH

    world = mxp.World(cellsize=(5e-9, 5e-9, 5e-9))
    mag   = mxp.Ferromagnet(world, mxp.Grid((1, 1, 1)))
    mag.msat  = 860e3
    mag.aex   = 13e-12
    mag.alpha = 0.005
    mag.bias_magnetic_field = (0.0, 0.0, B_BIAS)   # Tesla
    mag.magnetization = (float(M0_X), 0.0, float(M0_Z))

    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = DT

    t_arr, mx_arr = [], []
    t_sim = 0.0
    t0 = time.perf_counter()
    for _ in range(N_BATCHES):
        world.timesolver.run(BATCH * DT)
        t_sim += BATCH * DT
        avg = mag.magnetization.average()
        mx_arr.append(float(avg[0])); t_arr.append(t_sim)
    wall_ms = (time.perf_counter() - t0) * 1e3

    f_peak, freqs, psd = _fft_peak(t_arr, mx_arr)
    return {"wall_ms": wall_ms,
            "t_ns": [t*1e9 for t in t_arr],
            "mx": mx_arr, "f_peak_GHz": f_peak,
            "freqs_GHz": freqs[:500].tolist(),
            "psd": psd[:500].tolist()}

BATCH = 50   # steps per timesolver call (50ps resolution)
mxpr = bu.run_mumaxplus(fmr_mumaxplus, timeout_s=300)
if mxpr["ok"]:
    fp  = mxpr.get("f_peak_GHz")
    err = abs(fp - F_KITTEL/1e9) / (F_KITTEL/1e9) * 100 if fp else None
    print(f"  Wall: {mxpr['wall_ms']:.0f} ms")
    print(f"  f_peak = {fp:.4f} GHz  err={err:.2f}%" if fp else "  f_peak: not found")
    print(f"  Note: mx sampled every {BATCH*DT*1e12:.0f}ps -> f_max~{1/(BATCH*DT*1e9):.1f}GHz (coarser than CS)")
else:
    print(f"  {mxpr.get('error', 'failed')}")

print("\n" + "=" * 70)
print("SUMMARY -- FMR Macrospin (B_bias=50mT, alpha=0.005)")
print(f"  Kittel f_FMR = {F_KITTEL/1e9:.4f} GHz")
rows = []
for r in results:
    fp  = r.get("f_peak_GHz")
    err = abs(fp - F_KITTEL/1e9) / (F_KITTEL/1e9) * 100 if fp else None
    rows.append([r['build'], f"{r['wall_ms']:.0f}",
                 f"{fp:.4f}" if fp else "n/a",
                 f"{err:.2f}%" if err is not None else "n/a"])
if mx3r.get("ok"):
    fp  = mx3_fpeak
    err = abs(fp - F_KITTEL/1e9) / (F_KITTEL/1e9) * 100 if fp else None
    rows.append(["mumax3", f"{mx3r['wall_ms']:.0f}",
                 f"{fp:.4f}" if fp else "n/a",
                 f"{err:.2f}%" if err is not None else "n/a"])
if mxpr.get("ok"):
    fp  = mxpr.get("f_peak_GHz")
    err = abs(fp - F_KITTEL/1e9) / (F_KITTEL/1e9) * 100 if fp else None
    rows.append(["mumax+ (50ps batch)", f"{mxpr['wall_ms']:.0f}",
                 f"{fp:.4f}" if fp else "n/a",
                 f"{err:.2f}%" if err is not None else "n/a"])
bu.print_table(rows, ["Build", "wall_ms", "f_peak (GHz)", "err vs Kittel"])

BATCH = 50
out = {"scenario": "FMR Macrospin", "f_kittel_GHz": F_KITTEL/1e9,
       "B_bias_mT": B_BIAS*1e3, "alpha": 0.005,
       "cs": results, "mumax3_wall_ms": mx3r.get("wall_ms"),
       "mumax3_fpeak_GHz": mx3_fpeak, "mumaxplus": mxpr}
(pathlib.Path(__file__).parent / "48_results.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    c_map = {"CPU_RK4": "C0", "mumax+": "C3"}

    for r in results:
        c = c_map.get(r['build'], 'C0')
        axes[0].plot(r['t_ns'], r['mx'], lw=1, color=c, label=r['build'])
        f = np.array(r['freqs_GHz']); p = np.array(r['psd'])
        mask = f < 5
        if p[mask].max() > 0:
            axes[1].semilogy(f[mask], p[mask], lw=1.5, color=c, label=r['build'])

    if mxpr.get("ok") and mxpr.get("t_ns"):
        axes[0].plot(mxpr['t_ns'], mxpr['mx'], 'C3--', lw=2, label="mumax+ (50ps)")
        f_m = np.array(mxpr['freqs_GHz']); p_m = np.array(mxpr['psd'])
        mask_m = f_m < 5
        if p_m[mask_m].max() > 0:
            axes[1].semilogy(f_m[mask_m], p_m[mask_m], 'C3--', lw=2, label="mumax+")

    axes[0].set_xlabel("Time (ns)"); axes[0].set_ylabel("⟨mx⟩")
    axes[0].set_title("FMR: Free precession mx(t)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].axvline(F_KITTEL/1e9, color='k', ls=':', lw=2,
                    label=f"Kittel = {F_KITTEL/1e9:.3f} GHz")
    axes[1].set_xlabel("Frequency (GHz)"); axes[1].set_ylabel("PSD (log)")
    axes[1].set_title("FMR: Power spectral density of ⟨mx⟩")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 5)

    plt.suptitle(f"FMR Macrospin: B_bias={B_BIAS*1e3:.0f}mT → Kittel={F_KITTEL/1e9:.3f}GHz", y=1.01)
    plt.tight_layout()
    plt.savefig(str(pathlib.Path(__file__).parent / "48_fmr_comparison.png"), dpi=120)
    print("\nPlot: 48_fmr_comparison.png")
except Exception as e:
    print(f"Plot error: {e}")
