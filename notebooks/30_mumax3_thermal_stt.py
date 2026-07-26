"""
mumax3 cross-check of NB30 (thermal STT switching P_sw vs J).

Same Pt/Co PMA macrospin as notebook 30 run on the independent, validated
mumax3 code, to resolve whether Claude-SD's FDT-corrected thermal (mu0^2 sigma)
places the thermally-assisted switching transition at the right J/Jc0.

Procedure:
  1) T=0 sweep -> mumax3 deterministic threshold Jc0_mx.
  2) T=300 K ensemble (N seeds) -> P_sw vs J/Jc0_mx (final-state criterion).
Writes 30_mumax3_cache.json {jf: [n_sw, n_tr]} and prints the curve.
"""
import os, subprocess, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MX3 = os.environ.get("MX3_EXE",
    r"C:/Users/CHUN-Y~1/AppData/Local/Temp/claude/d--Claude-Code-R-Claude-SpinDynamics/f0c2b5e7-7db0-4deb-a0c6-be72efec5cfb/scratchpad/mumax312/mumax3.exe")
WORK = HERE / "_mx3_nb30"; WORK.mkdir(exist_ok=True)

# NB30 material
Ms, K, alpha, P, dx = 580e3, 0.5e6, 0.02, 0.5, 10e-9
t_max = 2.0e-9

def read_final_mz(outdir):
    ovf = sorted(outdir.glob("m*.ovf"))
    if not ovf:
        return None
    with open(ovf[-1], "rb") as f:
        while True:
            line = f.readline()
            if not line or b"Begin: Data" in line:
                break
        np.fromfile(f, dtype="<f4", count=1)
        d = np.fromfile(f, dtype="<f4", count=3)   # 1 cell, 3 comps
    return float(d[2])

def run_mx3(Jz, T, seed, tag, tilt=0.0):
    scr = WORK / f"{tag}.mx3"
    scr.write_text(f"""SetGridSize(1, 1, 1)
SetCellSize({dx:.3e}, {dx:.3e}, {dx:.3e})
Msat  = {Ms}
Aex   = 0
Ku1   = {K}
anisU = vector(0, 0, 1)
alpha = {alpha}
Pol   = {P}
Lambda = 1
EpsilonPrime = 0
FixedLayer = vector(0, 0, 1)
m = uniform({tilt}, 0, 1)
Temp = {T}
RandSeed({seed})
J = vector(0, 0, {Jz:.6e})
Run({t_max:.3e})
save(m)
""")
    out = WORK / f"{tag}.out"
    subprocess.run([MX3, "-o", str(out), str(scr)], capture_output=True)
    return read_final_mz(out)

# --- 1) T=0 deterministic threshold Jc0_mx (negative J; small tilt seeds STT) ---
print("Finding mumax3 T=0 threshold Jc0_mx ...")
Jc0_mx = None
for Jz in np.arange(-1.0e12, -3.01e12, -0.1e12):    # negative J switches +z->-z
    mz = run_mx3(Jz, 0.0, 1, "t0", tilt=0.15)
    if mz is not None and mz < -0.5:
        Jc0_mx = abs(Jz); break
print(f"  Jc0_mx = {Jc0_mx/1e12:.3f} e12 A/m2" if Jc0_mx else "  no switch up to 6e12")

# --- 2) T=300 ensemble P_sw vs J/Jc0 ---
CACHE = HERE / "30_mumax3_cache.json"
res = {k: tuple(v) for k, v in json.loads(CACHE.read_text()).items()} if CACHE.exists() else {}
res["Jc0_mx"] = (Jc0_mx, 1)
J_frac = np.round(np.arange(0.30, 0.851, 0.05), 4)
N = 24
for jf in J_frac:
    key = f"{jf:.4f}"
    nsw, ntr = res.get(key, (0, 0))
    while ntr < N:
        mz = run_mx3(-jf * Jc0_mx, 300.0, 1000 + ntr + int(jf * 1e4), f"e_{key}_{ntr}")
        if mz is not None and mz < -0.5:
            nsw += 1
        ntr += 1
    res[key] = (nsw, ntr)
    CACHE.write_text(json.dumps({k: list(v) for k, v in res.items()}))
    print(f"  J/Jc0={jf:.2f}  P_sw={nsw/ntr:.2f} (N={ntr})", flush=True)
print("mumax3 sweep done ->", CACHE.name)
