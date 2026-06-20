"""mumax3 reference for µMAG SP#3 (cube flower/vortex energy crossing).

For each L (in lex) writes a script that minimises a flower and a vortex state
and prints E_total for each, then normalises by Kd*V and finds the crossing.
Uses mumax3's built-in Vortex() seed + minimize() (Barzilai-Borwein) — the
standard recipe.

Usage:  py -3.13 mumax_sp3.py <path-to-mumax3.exe> [L1 L2 ...]
"""
import os, sys, subprocess, math, re

EXE = sys.argv[1] if len(sys.argv) > 1 else r"D:/Mumax3/mumax3.exe"
HERE = os.path.dirname(os.path.abspath(__file__))

Ms = 8e5; A = 13e-12; mu0 = 4e-7 * math.pi
Kd = 0.5 * mu0 * Ms * Ms; Ku = 0.1 * Kd
lex = math.sqrt(2 * A / (mu0 * Ms * Ms))
N = 28


def script(L_nm):
    return f"""SetGridSize({N}, {N}, {N})
SetCellSize({L_nm/N}e-9, {L_nm/N}e-9, {L_nm/N}e-9)
Msat = {Ms}
Aex  = {A}
Ku1  = {Ku}
anisU = vector(0, 0, 1)
m = uniform(0.05, 0.05, 1)
minimize()
print("EFLOWER", E_total)
m = Vortex(1, 1)
minimize()
print("EVORTEX", E_total)
"""


def run_L(L_lex):
    L_nm = L_lex * lex * 1e9
    path = os.path.join(HERE, "_mxsp3.mx3")
    with open(path, "w") as fh:
        fh.write(script(L_nm))
    p = subprocess.run([EXE, "-f", "-o", os.path.join(HERE, "_mxsp3.out"), path],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    ef = re.search(r"EFLOWER\s+([-\d.eE+]+)", out)
    ev = re.search(r"EVORTEX\s+([-\d.eE+]+)", out)
    V = (L_lex * lex) ** 3
    if not (ef and ev):
        return None
    return float(ef.group(1)) / (Kd * V), float(ev.group(1)) / (Kd * V)


if __name__ == "__main__":
    Ls = sorted(float(x) for x in (sys.argv[2:] or ["8.0", "8.3", "8.5", "8.7", "9.0"]))
    print(f"[mumax3] SP#3  lex={lex*1e9:.3f} nm, N={N}^3, Vortex()+minimize()")
    print(f"{'L/lex':>7}{'E_flower':>11}{'E_vortex':>11}{'dE=v-f':>11}")
    rows = []
    for L in Ls:
        r = run_L(L)
        if r is None:
            print(f"{L:>7.2f}   FAILED"); continue
        ef, ev = r; rows.append((L, ef, ev))
        print(f"{L:>7.2f}{ef:>11.4f}{ev:>11.4f}{ev-ef:>11.4f}", flush=True)
    cr = None
    for (L1, f1, v1), (L2, f2, v2) in zip(rows, rows[1:]):
        d1, d2 = v1 - f1, v2 - f2
        if (d1 < 0) != (d2 < 0):
            cr = L1 + (L2 - L1) * (0 - d1) / (d2 - d1); break
    print(f"\n  mumax3 L_c = {cr:.3f} lex   (reference 8.47)" if cr else "\n  no crossing")
