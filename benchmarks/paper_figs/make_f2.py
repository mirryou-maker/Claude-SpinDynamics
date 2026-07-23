"""F2 — µMAG standard-problem validation panel (SP#1-5), Claude-SD."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent.parent
NB = ROOT / "notebooks"
BM = ROOT / "benchmarks"
OUT = pathlib.Path(__file__).parent

fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 6)

# (a) SP#1 — L_c energy crossing
ax = fig.add_subplot(gs[0, 0:2])
d = json.loads((NB / "46_results.json").read_text())
c = d["cs"][0]
L = np.array(c["L_nm"]); Es = np.array(c["E_sstate_aJ"]); Ev = np.array(c["E_vortex_aJ"])
ax.plot(L, Es, "o-", label="S-state"); ax.plot(L, Ev, "s-", label="vortex")
# crossing
sgn = np.sign(Es - Ev); idx = np.where(np.diff(sgn) != 0)[0]
if len(idx):
    i = idx[0]; Lc = L[i] + (L[i+1]-L[i])*(Es-Ev)[i]/((Es-Ev)[i]-(Es-Ev)[i+1])
    ax.axvline(Lc, color="gray", ls="--"); ax.text(Lc, ax.get_ylim()[1]*0.95, f"$L_c$={Lc:.0f}nm", fontsize=8)
ax.set_xlabel(r"$L$ (nm)"); ax.set_ylabel(r"$E$ (aJ)"); ax.set_title("(a) SP#1 phase: $L_c$"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (b) SP#2 — remanence vs d/lex
ax = fig.add_subplot(gs[0, 2:4])
sp2 = np.loadtxt(BM / "sp2" / "sp2_remanence_double.csv", delimiter=",", skiprows=1)
ax.semilogx(sp2[:,0], sp2[:,1], "o-", label=r"$\langle m_x\rangle$")
ax.semilogx(sp2[:,0], sp2[:,2], "s-", label=r"$\langle m_y\rangle$")
ax.set_xlabel(r"$d/\ell_{ex}$"); ax.set_ylabel(r"remanent $\langle m\rangle/M_s$")
ax.set_title("(b) SP#2 remanence"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

# (c) SP#3 — hysteresis
ax = fig.add_subplot(gs[0, 4:6])
d3 = json.loads((NB / "47_results.json").read_text())
c3 = d3["cs"][0]
H = np.array(c3["H_mT"]); mx = np.array(c3["mx"])
ax.plot(H, mx, "o-", ms=3)
ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel(r"$\mu_0H$ (mT)"); ax.set_ylabel(r"$\langle m_x\rangle$")
ax.set_title("(c) SP#3 hysteresis"); ax.grid(alpha=0.3)

# (d) SP#4 — <m>(t)
ax = fig.add_subplot(gs[1, 0:3])
t4 = np.loadtxt(BM / "sp4_trajectory_cs.csv", delimiter=",", skiprows=1)
ax.plot(t4[:,0], t4[:,1], label=r"$\langle m_x\rangle$")
ax.plot(t4[:,0], t4[:,2], label=r"$\langle m_y\rangle$")
ax.plot(t4[:,0], t4[:,3], label=r"$\langle m_z\rangle$")
ax.axhline(-0.9862, color="k", ls=":", lw=1, label="µMAG ref $m_x(1ns)$")
ax.set_xlabel(r"$t$ (ns)"); ax.set_ylabel(r"$\langle m\rangle$")
ax.set_title(r"(d) SP#4 Field A: $\langle m_x\rangle(1ns)$=%.3f (ref −0.986)" % t4[-1,1])
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# (e) SP#5 — vortex-core trajectory
ax = fig.add_subplot(gs[1, 3:6])
sp5 = np.loadtxt(BM / "sp5" / "core_double.txt")
ax.plot(sp5[:,1], sp5[:,2], "-", lw=1)
ax.plot(sp5[0,1], sp5[0,2], "go", label="start"); ax.plot(sp5[-1,1], sp5[-1,2], "rs", label=r"$t=8$ ns")
ax.set_xlabel(r"$x_c$ (nm)"); ax.set_ylabel(r"$y_c$ (nm)")
ax.set_title("(e) SP#5 vortex-core gyration (Zhang-Li STT)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal", "datalim")

fig.suptitle("F2 — µMAG standard-problem validation (Claude-SD, f64)", y=1.01, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_f2_umag_validation.png", dpi=150, bbox_inches="tight")
print("wrote fig_f2_umag_validation.png")
