"""SP#2 figure — remanence components + coercivity vs d/lex (Claude-SD f64)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
csv = os.path.join(HERE, "sp2_remanence_double.csv")
d = np.loadtxt(csv, delimiter=",", skiprows=1)
dlex, mx, my, mz = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

# Coercivity (from the run log; 3 representative points)
hc_d  = np.array([0.5, 5.0, 30.0])
hc_v  = np.array([-0.4619, -0.0587, -0.0226])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

ax1.semilogx(dlex, mx, "o-", label=r"$\langle m_x\rangle/M_s$ (long axis)")
ax1.semilogx(dlex, my, "s-", label=r"$\langle m_y\rangle/M_s$")
ax1.semilogx(dlex, mz, "^-", label=r"$\langle m_z\rangle/M_s$ (thin axis)")
ax1.set_xlabel(r"$d/\ell_{ex}$"); ax1.set_ylabel(r"remanent $\langle m\rangle/M_s$")
ax1.set_title("SP#2 remanence after [111] saturation (Claude-SD f64)")
ax1.grid(True, which="both", ls=":", alpha=0.4); ax1.legend(fontsize=9)

ax2.semilogx(hc_d, -hc_v, "d-", color="#d62728")
ax2.set_xlabel(r"$d/\ell_{ex}$"); ax2.set_ylabel(r"coercive field $|H_c|/M_s$")
ax2.set_title("SP#2 coercivity along [111]")
ax2.grid(True, which="both", ls=":", alpha=0.4)

fig.tight_layout()
out = os.path.join(HERE, "fig_sp2.png")
fig.savefig(out, dpi=150)
print("wrote", out)
