"""
Three-way comparison of thermally-assisted STT switching P_sw(J/Jc0):
  1) Claude-SD (FDT-correct mu0^2 sigma)  -- 30_partB_cache.json  (Jc0=2.5e12)
  2) mumax3 (independent reference)        -- 30_mumax3_cache.json (Jc0=Jc0_mx)
  3) Neel-Brown analytic  P_sw = 1 - exp(-t*f0*exp(-Delta*(1-J/Jc0)^2))

Each simulation is normalised by ITS OWN T=0 deterministic threshold Jc0, so the
comparison isolates the thermally-assisted transition location (the sigma test),
not the STT-strength convention (which differs between codes).
"""
import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
kB, mu0, g0 = 1.38065e-23, 4e-7*np.pi, 1.76e11
Ms, K, alpha, dx, T, t_max = 580e3, 0.5e6, 0.02, 10e-9, 300.0, 2.0e-9
V = dx**3
Delta = K * V / (kB * T)                                  # 120.7 (cube: no demag anis.)

# --- Neel-Brown analytic (Brown high-barrier attempt frequency) ---
Hk = 2*K/(mu0*Ms)
f0 = (alpha*g0*mu0*Hk/(1+alpha**2)) * np.sqrt(Delta/np.pi) / (2*np.pi)
jf = np.linspace(0.30, 1.0, 200)
Delta_eff = Delta * (1 - jf)**2
P_theory = 1 - np.exp(-t_max * f0 * np.exp(-Delta_eff))

# --- Claude-SD (mu0^2) ---
csd = json.loads((HERE/"30_partB_cache.json").read_text())
Jcsd = np.array(sorted(float(k) for k in csd))
Pcsd = np.array([csd[f"{j:.6f}"][0]/csd[f"{j:.6f}"][1] if f"{j:.6f}" in csd
                 else csd[[k for k in csd if abs(float(k)-j)<1e-6][0]][0]/
                      csd[[k for k in csd if abs(float(k)-j)<1e-6][0]][1] for j in Jcsd])
Ncsd = np.array([csd[[k for k in csd if abs(float(k)-j)<1e-6][0]][1] for j in Jcsd])

# --- mumax3 ---
mxp = HERE/"30_mumax3_cache.json"
have_mx = mxp.exists()
if have_mx:
    mx = json.loads(mxp.read_text())
    Jc0_mx = mx.get("Jc0_mx", [None])[0]
    keys = sorted((k for k in mx if k != "Jc0_mx"), key=float)
    Jmx = np.array([float(k) for k in keys])
    Pmx = np.array([mx[k][0]/mx[k][1] for k in keys])
    Nmx = np.array([mx[k][1] for k in keys])

fig, ax = plt.subplots(figsize=(7.0, 5.0))
ax.plot(jf, P_theory, "k--", lw=1.8,
        label=fr"N\'eel-Brown ($\Delta$={Delta:.0f}, $f_0$={f0:.1e} Hz)")
ax.errorbar(Jcsd, Pcsd, yerr=np.sqrt(Pcsd*(1-Pcsd)/np.maximum(Ncsd,1)),
            fmt="o-", color="#1f77b4", ms=5, capsize=2,
            label=r"Claude-SD ($\mu_0^2$ $\sigma$, FDT-correct)")
if have_mx:
    ax.errorbar(Jmx, Pmx, yerr=np.sqrt(Pmx*(1-Pmx)/np.maximum(Nmx,1)),
                fmt="s-", color="#d62728", ms=5, capsize=2,
                label=fr"mumax3 ($J_{{c0}}$={Jc0_mx/1e12:.2f}e12)")
ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
ax.set_xlabel(r"$J / J_{c0}$"); ax.set_ylabel(r"$P_\mathrm{sw}$ (2 ns, 300 K)")
ax.set_title("Thermally-assisted STT switching: Claude-SD vs mumax3 vs theory")
ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=9)
ax.grid(alpha=0.3)
out = HERE/"30_compare_nb30_mumax3_theory.png"
fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out.name)

def transition(J, P):
    i = np.argmin(np.abs(P - 0.5)); return J[i]
print(f"Transition J/Jc0 @ P=0.5:  theory={jf[np.argmin(np.abs(P_theory-0.5))]:.2f}"
      f"  Claude-SD={transition(Jcsd,Pcsd):.2f}"
      + (f"  mumax3={transition(Jmx,Pmx):.2f}" if have_mx else ""))
