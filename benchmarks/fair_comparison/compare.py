"""Compare mumax3 vs Claude-SD fair-comparison results.

Usage:
    py -3.13 compare.py
"""
import json, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")

HERE    = pathlib.Path(__file__).parent
MX_FILE = HERE / "fair_comparison_mumax.json"
CS_FILE = HERE / "fair_comparison_cs.json"

# (display_label, nx, ny, nz, type, t_ns_mumax, t_ns_cs, mx_key, cs_key, note)
ALL_SCENARIOS = [
    ("S1  pow2 128×128×4",
     128, 128,  4, "fixed",    None, None,
     "S1 pow2 128×128×4",        "S1 pow2 128×128×4",
     "mumax3 — smaller padded Z (7 vs 8), CUDA Graphs advantage"),
    ("S2  nonpow2 200×50×1",
     200,  50,  1, "fixed",    None, None,
     "S2 nonpow2 200×50×1",      "S2 nonpow2 200×50×1",
     "CS — identical FFT (400×100×1); mumax3 fixed overhead dominates at 10K cells"),
    ("S3  nonpow2 300×300×6",
     300, 300,  6, "fixed",    None, None,
     "S3 nonpow2 300×300×6",     "S3 nonpow2 300×300×6",
     "mumax3 — identical FFT (600×600×12); mumax3 cuFFT+CUDA Graphs 3.2× faster"),
    ("S4  SP#4 switch  1ns  (adaptive)",
     200,  50,  1, "adaptive", 1.0, 1.0,
     "S4 SP#4 switch 1ns",       "S4 SP#4 switch 1ns",
     "CS — 831 accepted steps vs ~14700 mumax3 Heun steps; 30× faster"),
    ("S5  large 500×500×10",
     500, 500, 10, "fixed",    None, None,
     "S5 large 500×500×10",      "S5 large 500×500×10",
     "mumax3 — identical FFT (1000×1000×20); mumax3 cuFFT+CUDA Graphs 3.2× faster"),
    ("S6  medium 300×300×6 (adaptive)*",
     300, 300,  6, "adaptive", 0.3, 0.03,
     "S6 Medium switch 0.3ns",   "S6 Medium switch 0.03ns",
     "mumax3 — DOPRI5 per-step FFT at 600-size; mumax3 cuFFT 2.4× faster (*CS ran 0.03ns)"),
    ("S7  DW motion 400×20×1 2ns (adaptive)",
     400,  20,  1, "adaptive", 2.0, 2.0,
     "S7 DW motion 2ns",         "S7 DW motion 2ns",
     "CS — Walker oscillation, 8K cells; mumax3 overhead at small grid; CS 33× faster"),
    ("S8  precession 200×50×1 1ns α=5e-3 (adaptive)",
     200,  50,  1, "adaptive", 1.0, 1.0,
     "S8 Precession 1ns α=5e-3", "S8 Precession 1ns α=5e-3",
     "CS — uniform precession, 10K cells; CS 8.8× faster"),
]

FFT_NOTE = {
    "S1 pow2 128×128×4":        "256×256×7 (mumax3 2N-1 for Z≤5) vs 256×256×8 (CS 2N)",
    "S2 nonpow2 200×50×1":      "400×100×1 IDENTICAL (2N for all dims)",
    "S3 nonpow2 300×300×6":     "600×600×12 IDENTICAL",
    "S4 SP#4 switch 1ns":       "400×100×1 IDENTICAL",
    "S5 large 500×500×10":      "1000×1000×20 IDENTICAL",
    "S6 Medium switch 0.3ns":   "600×600×12 IDENTICAL",
    "S7 DW motion 2ns":         "800×40×1 IDENTICAL",
    "S8 Precession 1ns α=5e-3": "400×100×1 IDENTICAL",
}

CS_BUILDS = ["cuFFT_f64", "cuFFT_f32", "VkFFT_f32"]


def _fmt(v, fmt):
    return (fmt % v) if v is not None else "n/a"


def _ratio(mx_val, cs_val):
    if not mx_val or not cs_val:
        return "—"
    r = mx_val / cs_val
    return f"CS {r:.1f}×" if r >= 1 else f"mumax3 {1/r:.1f}×"


def _best_cs(cs_dict, key, field):
    vals = []
    for b in CS_BUILDS:
        v = (cs_dict.get(b) or {}).get(key, {}).get(field)
        if isinstance(v, float):
            vals.append(v)
    return min(vals) if vals else None


def load_json(path):
    if not path.exists():
        print(f"[WARN] {path.name} not found"); return {}
    return json.loads(path.read_text())


if __name__ == "__main__":
    mx = load_json(MX_FILE)
    cs = load_json(CS_FILE)

    W = 110
    print("\n" + "=" * W)
    print("  FAIR COMPARISON: Claude-SD (CS) vs mumax3")
    print("  Fixed RK4 :  same dt=5e-14 s, 4 eval/step - metric: ms/step, ms/eval")
    print("  Adaptive  :  DOPRI5 FSAL, rtol=MaxErr=1e-4 - metric: ms/ns_sim, step count")
    print("  CS builds  : cuFFT f64 | cuFFT f32 | VkFFT f32")
    print("=" * W)

    for (disp, nx, ny, nz, stype, t_mx, t_cs, mx_key, cs_key, winner) in ALL_SCENARIOS:
        cells = nx * ny * nz
        tag   = "FIXED RK4" if stype == "fixed" else f"ADAPTIVE DOPRI5  (mumax3:{t_mx}ns / CS:{t_cs}ns)"
        print(f"\n{'─'*W}")
        print(f"  {disp}  ({cells:,} cells)  [{tag}]")
        print(f"  FFT:  {FFT_NOTE.get(mx_key, '—')}")
        print(f"  ►  {winner}")
        print()

        mx_r = mx.get(mx_key) or {}
        cs_builds_data = [(b, (cs.get(b) or {}).get(cs_key) or {}) for b in CS_BUILDS]

        if stype == "fixed":
            # ms/step row
            mx_ms = mx_r.get("ms_step")
            cs_ms = [d.get("ms_step") for _, d in cs_builds_data]
            best  = min((v for v in cs_ms if v), default=None)
            print(f"  {'ms/step':>30}   {_fmt(mx_ms,'%8.3f')}   " +
                  "   ".join(_fmt(v, "%8.3f") for v in cs_ms) +
                  f"   {_ratio(mx_ms, best)}")
            # ms/eval row
            mx_ev = mx_r.get("ms_eval")
            cs_ev = [d.get("ms_eval") for _, d in cs_builds_data]
            print(f"  {'ms/eval':>30}   {_fmt(mx_ev,'%8.3f')}   " +
                  "   ".join(_fmt(v, "%8.3f") for v in cs_ev))

        else:  # adaptive
            # ms/ns row
            mx_mns = mx_r.get("ms_per_ns")
            cs_mns = [d.get("ms_per_ns") for _, d in cs_builds_data]
            best   = min((v for v in cs_mns if v), default=None)
            print(f"  {'ms/ns_sim':>30}   {_fmt(mx_mns,'%9.1f')}   " +
                  "   ".join(_fmt(v, "%9.1f") for v in cs_mns) +
                  f"   {_ratio(mx_mns, best)}")
            # wall_ms row
            mx_wall = mx_r.get("wall_ms")
            cs_wall = [d.get("wall_ms") for _, d in cs_builds_data]
            print(f"  {f'wall ms (mumax3:{t_mx}ns / CS:{t_cs}ns)':>30}   {_fmt(mx_wall,'%9.1f')}   " +
                  "   ".join(_fmt(v, "%9.1f") for v in cs_wall))
            # step count
            mx_n = mx_r.get("n_steps")
            cs_n = [d.get("n_steps") for _, d in cs_builds_data]
            mx_n_str = str(mx_n) if mx_n else "n/a"
            print(f"  {'accepted steps':>30}   {mx_n_str:>9}   " +
                  "   ".join(f"{str(v):>9}" if v else "      n/a" for v in cs_n))
            # ms/eval
            cs_ev = [d.get("ms_eval") for _, d in cs_builds_data]
            print(f"  {'ms/eval (DOPRI5 ~5 evals)':>30}        n/a   " +
                  "   ".join(_fmt(v, "%9.3f") for v in cs_ev))

    print(f"\n{'─'*W}")
    print("""
SUMMARY TABLE (best CS build vs mumax3)
─────────────────────────────────────────────────────────────────────────────
Scenario              Cells  Type     mumax3 ms/ev  CS best ms/ev  CS vs mumax3
S1 128×128×4          65K    fixed        6.037          0.955      CS  6.3×
S2 200×50×1           10K    fixed        6.113          0.159      CS 38.5×
S3 300×300×6         540K    fixed        5.364         17.332      mumax3  3.2×
S4 SP#4 1ns           10K    adaptive    29400 ms/ns    968 ms/ns   CS 30.4×
S5 500×500×10        2.5M    fixed       24.283         78.250      mumax3  3.2×
S6 medium 300×300×6  540K    adaptive  105352 ms/ns  252783 ms/ns   mumax3  2.4× *
S7 DW 400×20×1        8K     adaptive   42679 ms/ns   1266 ms/ns   CS 33.7×
S8 precession 10K     10K    adaptive   10392 ms/ns   1175 ms/ns   CS  8.8×
─────────────────────────────────────────────────────────────────────────────
*S6: mumax3 ran 0.3ns, CS ran 0.03ns — ms/ns compared directly (quasi-stationary).
 For S1/S2/S3/S5: mumax3 uses RK4 (4 eval/step), CS uses RK4 (4 eval/step).
 For S4/S6/S7/S8: mumax3 uses DOPRI5 setsolver(5), CS uses RK45 FSAL.

CROSSOVER: ~100K–200K cells.  Below: CS faster; Above: mumax3 faster.
ROOT CAUSE: mumax3 cuFFT + CUDA Graphs at large FFT sizes vs CS sequential launches.
""")
