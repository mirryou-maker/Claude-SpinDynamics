# Cross-Solver Benchmark Plan — Claude-SD vs mumax3 / mumax+ / MuMax-CO

Refined strategy for a publishable performance comparison. Status: **plan (not yet executed)**.
Hardware of record: **RTX 5060 Ti (Blackwell GB206, 8 GB, sm_120), CUDA 13.2, Intel Core Ultra 7 265KF, Win 11 / MSVC**.

---

## 0. Reviewer's verdict on the original 8-point strategy

The 8 points are sound. The main corrections are: (i) **don't rebuild — unify** the two existing
harnesses (NB41–50 and `benchmarks/REPORT.md` C1–C5) into one 4-solver pipeline; (ii) make the
**float32/float64 precision asymmetry the explicit backbone** of every fair comparison; (iii) treat
**MuMax-CO as "optimized mumax3" (bit-identical, faster)**, not a separate accuracy target; (iv) for
T>0, only **Heun↔Heun** is valid and results must be **ensemble-averaged**; (v) add **statistical rigor**
(≥3 repeats, locked GPU clocks, median+IQR). Each point is addressed in detail below.

---

## 1. What already exists (leverage, do not duplicate)

| Asset | Solvers covered | Scenarios | Reuse |
|-------|-----------------|-----------|-------|
| `notebooks/NB41–50` + `bench_utils.py` | CS (f64/f32/VkFFT_f32), mumax3, mumax+ | SP#4, STT, SOT-thermal, DW, skyrmion, SP#1, SP#3, FMR, Walker | **Primary**: extend with MuMax-CO column |
| `benchmarks/fair_comparison/` (s1–s8) | CS (3 builds), mumax3 | pow2/non-pow2 2D/3D throughput + 4 adaptive | **Primary** throughput harness (two-run subtraction, ms/eval) |
| `benchmarks/perf/` | CS, mumax3, MuMax-CO (via `label`) | grid sweep 16K–2.5M | **Primary** scaling/crossover curve |
| `benchmarks/REPORT.md` (C1–C5) | CS f64/f32, mumax3, MuMax-CO, OOMMF | SP#4/#3/#5/#1 + perf | **Merge** its MuMax-CO/OOMMF data into the main report; reconcile older f32 numbers |

**Gaps to close:** SP#2 absent (optional add); SP#5 only under `benchmarks/sp5/` (promote to NB); MuMax-CO
present only in the older REPORT.md (add as 4th column to NB41–50); two hardware/era contexts to reconcile.

---

## 2. The fairness framework (the backbone)

### 2.1 Precision — the central caveat
mumax3, mumax+, MuMax-CO are **all float32 only**. Claude-SD has both. Therefore split every claim:

- **Speed claim** → compare **CS-f32** against the f32-only tools (apples-to-apples FLOPs).
- **Accuracy claim** → use **CS-f64 as the high-precision reference**; f32 tools plateau at ~1e-6–1e-7
  relative error. Report each tool's deviation from (a) the µMAG community reference and (b) CS-f64.
- Always state the build in every table cell. Never compare CS-f64 wall-time against f32 tools as if equal.

### 2.2 Integrator equivalence map
| Claude-SD | mumax3 / MuMax-CO `SetSolver` | mumax+ | Use for |
|-----------|------------------------------|--------|---------|
| `RK4IntegratorGPU` (fixed) | `4` RK4 (fixed) | RK4 fixed | **Throughput** (deterministic, identical work/step) |
| `RK45IntegratorGPU` DOPRI5/FSAL | `5` RK45 Dormand–Prince (default) | RK45-DP (default) | **Adaptive dynamics** (SP#4, precession) |
| `HeunIntegratorGPU` (Stratonovich) | `2` Heun | Heun | **T>0 SLLG** + fixed-step T=0 |

Matched controls: **same cell size, same field terms, same tolerance** (adaptive: `MaxErr = rtol = 1e-4`,
tighten to `1e-5` for the accuracy table), same `t_end`, same initial state, same applied field schedule.
Normalize wall time to **ms/step AND ms/field-eval** (RK4=4, Heun=2, DOPRI5=6 evals) so eval-count
asymmetry never contaminates the comparison.

### 2.3 Timing methodology (already in fair_comparison — keep + harden)
- **Two-run subtraction** (t(N₂)−t(N₁))/(N₂−N₁) to cancel one-time setup/JIT/plan costs.
- **`cudaDeviceSynchronize()` before stopping the clock** (the DeviceSync bug already cost a 3× error at f32).
- **Warm-up** run discarded; then **≥3 measured repeats → report median + IQR** (new requirement).
- **Lock GPU clocks** `nvidia-smi -lgc <clk>` + `-lmc` and run **exclusive** (no concurrent GPU jobs — the
  earlier NB46/47 concurrent run inflated both times). Record clocks + driver in the report.
- mumax+ via its Python API (time the solver loop, not import); mumax3/MuMax-CO via subprocess on `.mx3`
  with `NSTEPS` placeholder; parse `table.txt`.

### 2.4 MuMax-CO special handling
Bit-identical to mumax3, CUDA-Graph path **auto-disables** when: Temp≠0, cells>800k, steps<20, or custom
field terms. So: **MuMax-CO = mumax3 at T>0 and at >0.8M cells** — show it as the "fast mumax3" point on
small/medium T=0 scenarios only, and state explicitly where the graph path is active.

---

## 3. Scenario matrix (2D/3D × who-should-win × T)

Designed so each tool's sweet spot appears. "Expected winner" is a hypothesis to be tested, grounded in the
measured ~100–200K-cell crossover (below it CS-f32+CUDA-Graph wins on launch overhead; above it the
comparison is cuFFT-bound and mumax3/MuMax-CO/VkFFT win).

| # | Scenario | Dim | Cells | Integrator (matched) | T | Hypothesis: favors |
|---|----------|-----|-------|----------------------|---|--------------------|
| A1 | SP#4 field-A reversal | 2D | 10K | DOPRI5 adaptive | 0 | **CS-f32** (small, launch-bound) |
| A2 | Thin-film throughput (non-pow2) | 2D | 10K | RK4 fixed | 0 | **CS-f32** |
| A3 | Low-damping precession (α=0.005) | 2D | 10K | DOPRI5 | 0 | **CS-f32** |
| A4 | Zhang-Li DW / Walker | 2D | 2–8K | Heun / DOPRI5 | 0 | **CS-f32** |
| B1 | Medium film | 3D | 0.2–0.5M | RK4 fixed | 0 | ~crossover (report both) |
| B2 | Large film | 3D | 2.5M | RK4 fixed | 0 | **mumax3 / MuMax-CO / VkFFT_f32** (FFT-bound) |
| B3 | Large adaptive | 3D | 0.5M | DOPRI5 | 0 | **mumax3 / MuMax-CO** |
| C1 | SOT thermal switching (ensemble) | 2D | 10K or macrospin | **Heun↔Heun** | **300 K** | report mean±std; MuMax-CO=mumax3 |
| C2 | Thermal equilibrium ⟨m²⟩ | 3D | 0.1M | Heun↔Heun | **>0** | accuracy (FDT) check, not speed |
| D1 | SP#1 L_c phase / SP#3 H_sw | quasi-static | 20–80K | Relax/Minimize ↔ minimize() | 0 | accuracy parity + relax speed |
| D2 | SP#5 Zhang-Li vortex core | 2D | ~16K | DOPRI5 | 0 | accuracy vs ext_corepos |

cuFFT vs VkFFT within CS: include **both f32 builds** on B1/B2/B3 to show cuFFT_f32 wins small (CUDA-Graph)
while VkFFT_f32 wins large 3D — this is a CS-internal figure, a selling point of the multi-backend design.

---

## 4. T=0 primary, T>0 secondary (point 5)

- **T=0 is the headline** (deterministic, reproducible, exact integrator match). All A/B/D scenarios.
- **T>0 (C scenarios):** only Heun↔Heun(solver 2) is physically valid (Stratonovich SLLG). Requirements:
  - **Ensemble of N≥20 trials**, different seeds; report **P_switch / ⟨m⟩ mean ± std**, never a single run.
  - Verify the **fluctuation–dissipation** sanity check (⟨m²⟩ vs analytic) before timing.
  - Note MuMax-CO's graph path is off at T>0 ⇒ equals mumax3; mumax+ thermal via its own RNG.
  - Fairness caveat: RNG algorithms differ (cuRAND vs mumax3 xorwow) — compare **statistics, not trajectories**.

---

## 5. Auto-integrator selection in the NBs (point 6)

`recommend_integrator(mat, T_K, goal, dt, B_eff_T, t_end)` already exists (rules: T>0→Heun; relax→Heun;
α≥0.3→Heun; α<0.05→RK45; else phase-error threshold; ε=ω³dt²t/6). Refinement:

1. Add a thin wrapper `auto_setup(scenario)` in `bench_utils.py` that calls `recommend_integrator`, **logs the
   chosen integrator + reason into the results JSON**, and instantiates it — so each NB is self-configuring.
2. Emit a **per-scenario "auto-pick" table** (scenario → recommended integrator → reason → matched mumax solver)
   as a paper table, and a **decision-tree figure** (α / T / goal → integrator).
3. Validate the recommender: for 2–3 scenarios run *all* integrators and show the recommended one is on the
   speed/accuracy Pareto front (this turns point 6 into a defensible result, not just a convenience feature).

---

## 6. Paper deliverables

### 6.1 Tables (templates)
- **T1 — Solver capability matrix**: precision (f32/f64), integrators, demag backend (cuFFT/VkFFT), native
  SOT/DMI/RKKY/magnetoelastic/AFM, scripting (mx3/Python/C++), license. (CS / mumax3 / mumax+ / MuMax-CO / OOMMF)
- **T2 — Throughput** (ms/step + ms/eval) per scenario × {CS-f32 cuFFT, CS-f32 VkFFT, CS-f64, mumax3, MuMax-CO,
  mumax+}, median±IQR, with FFT pad sizes annotated to prove identical FFT work.
- **T3 — Accuracy** vs µMAG reference and vs CS-f64: SP#4 ⟨mₓ⟩(1ns) & t_switch, SP#3 H_sw, SP#1 L_c, SP#5 core
  trajectory RMS.
- **T4 — Auto-integrator** picks (point 6).
- **T5 — T>0** P_switch mean±std (Heun↔Heun).

### 6.2 Figures (python/matplotlib)
- **F1 — Scenario speedup bars** (CS-f32 / mumax3, log scale, 2D vs 3D color, 1.0 line). *Draft generated now from
  existing data: `benchmarks/plan_figs/fig_scenario_speedup.png`.*
- **F2 — Throughput-vs-cells crossover curve** (ms/eval vs cell count; CS-f32 cuFFT, CS-f32 VkFFT, mumax3, MuMax-CO)
  with the ~100–200K crossover region shaded.
- **F3 — SP#4 ⟨m⟩(t) trajectories** overlaid (all solvers) + inset error vs CS-f64.
- **F4 — Accuracy–vs–wall-time Pareto** (each solver/integrator a point; SP#4).
- **F5 — Integrator decision tree** (point 6).
- **F6 — f32 vs f64** Blackwell Tensor-Core speedup (CS-internal) + precision cost (skyrmion Q).
- **F7 — T>0** P_switch(J) with error bars, CS-Heun vs mumax3-Heun.

### 6.3 One unified results artifact
All harnesses write to a single `benchmarks/results/all_solvers.json` (schema: scenario, solver, build,
integrator, cells, dim, ms_step, ms_eval, wall, observable, error_vs_ref, error_vs_csf64, repeats, clocks).
One `make_report.py` renders T2–T5 + F1–F7 from it → reproducible paper assets.

---

## 7. Pros/cons synthesis to write up (point 7)

Axes to populate from the data:
- **Speed × size×dim:** CS-f32 wins small/2D (launch-overhead regime, CUDA Graphs); mumax3/MuMax-CO/VkFFT win
  large/3D (cuFFT-bound). State the crossover (~100–200K cells) quantitatively.
- **Precision:** CS uniquely offers **f64** (reference-grade) and a 4–8× f32 Tensor-Core speedup on Blackwell;
  the mumax family is f32-only.
- **Features:** CS native SOT/DMI/RKKY/magnetoelastic/surface-anisotropy + per-cell + auto-integrator; mumax+
  extensibility (AFM, elastodynamics, Python); mumax3/MuMax-CO mature & validated; MuMax-CO = launch-overhead win.
- **Usability/reproducibility:** CS Python+C++; mumax+ Python; mumax3/MuMax-CO `.mx3`.
- **"Use X when…":** small 2D dynamics / need f64 / custom torques → CS; huge 3D production f32 → mumax3/MuMax-CO;
  novel physics (AFM/elastic) in Python → mumax+.

---

## 8. References (BibTeX-ready list)

1. µMAG Standard Problems — NIST/CTCMS µMAG, https://www.ctcms.nist.gov/~rdm/mumag.org.html
2. A. Vansteenkiste et al., "The design and verification of MuMax3," *AIP Advances* **4**, 107133 (2014).
3. L. Moreels, J. Lateur, I. De Gusem et al., "mumax+: extensible GPU-accelerated micromagnetics and beyond,"
   *npj Comput. Mater.* (2025/26).
4. M. J. Donahue, D. G. Porter, "OOMMF User's Guide," NISTIR 6376 (1999).
5. M. J. Donahue, "A variational approach to exchange energy …" / Newell, Dunlop, Williams, "A generalization of
   the demagnetizing tensor for nonuniform magnetization," *J. Geophys. Res.* **98**, B6, 9551 (1993).
6. J. R. Dormand, P. J. Prince, "A family of embedded Runge–Kutta formulae," *J. Comput. Appl. Math.* **6**, 19 (1980).
7. J. L. García-Palacios, F. J. Lázaro, "Langevin-dynamics study of … magnetic nanoparticles," *Phys. Rev. B* **58**,
   14937 (1998). (Stochastic LLG / Heun-Stratonovich.)
8. W. F. Brown, "Thermal fluctuations of a single-domain particle," *Phys. Rev.* **130**, 1677 (1963). (FDT.)
9. J. C. Slonczewski, "Current-driven excitation of magnetic multilayers," *JMMM* **159**, L1 (1996); L. Berger,
   *Phys. Rev. B* **54**, 9353 (1996). (STT.)
10. S. Zhang, Z. Li, "Roles of nonequilibrium conduction electrons … spin-transfer torques," *PRL* **93**, 127204 (2004).
11. A. V. Khvalkovskiy et al. / A. Thiele, "Steady-state motion of magnetic domains," *PRL* **30**, 230 (1973). (DW/Walker.)
12. D. Tolmachev, "VkFFT — Vulkan/CUDA FFT library," *IEEE Access* (2023).
13. L. Landau, E. Lifshitz (1935); T. L. Gilbert, *IEEE Trans. Magn.* **40**, 3443 (2004). (LLG.)
14. NVIDIA, "CUDA C++ Programming Guide — CUDA Graphs"; NVIDIA Blackwell architecture whitepaper (2024/25).
15. C.-Y. You, "Optimization of MuMax3 by Using Claude Code: A CUDA-Graph-Based Case Study," (MuMax-CO manuscript).

---

## 9. Recommended execution order (once approved)

1. **Add MuMax-CO runner** to `bench_utils.py` / `run_mumax_bench.py` (`label="mumaxco"`, exe under
   `MuMax-CO/mumax3-src/mumax3.exe`); add 4th column to NB41–50.
2. **Harden timing**: lock clocks, ≥3 repeats, median+IQR, exclusive GPU; write unified `all_solvers.json`.
3. **Run T=0 throughput** (A/B scenarios, all builds + 3 mumax tools) → T2, F1, F2, F6.
4. **Run T=0 accuracy** (SP#4/#3/#1/#5 at rtol=1e-5) → T3, F3, F4.
5. **Run auto-integrator** wrapper + Pareto validation → T4, F5.
6. **Run T>0 ensembles** (C scenarios, Heun↔Heun, N≥20) → T5, F7.
7. **`make_report.py`** renders all tables/figures; write Discussion (§7) + assemble references.

### Open decisions for the user
- **OOMMF inclusion?** It's CPU/double and already in REPORT.md but crashes on some configs — include as a
  CPU-double accuracy anchor only, or drop? (Recommend: keep for SP#4 accuracy anchor, exclude from speed plots.)
- **SP#2 add?** Currently unimplemented in CS. (Recommend: optional, low priority.)
- **Scale of run:** quick (A1/A2/B2/C1 representative, ~30 min) vs full matrix (all scenarios × all solvers × 3
  repeats, multi-hour). 
