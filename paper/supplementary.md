# Supplementary Information

**Building and validating a GPU micromagnetic simulator with an AI coding agent: cross-implementation
testing exposes a concurrency defect**

Chun-Yeol You

---

## S1. Benchmark methodology

All timings were taken on an NVIDIA RTX 5060 Ti (Blackwell GB206, 8 GB, sm_120), CUDA 13.2, Intel Core Ultra
7 265KF, Windows 11 / MSVC, with the GPU otherwise idle.

**Two-run subtraction.** For a fixed-step run of N steps the wall time is t(N) = t_setup + N·t_step, where
t_setup (CUDA context, FFT plan, kernel JIT) is one-time. We therefore report
t_step = [t(N₂) − t(N₁)]/(N₂ − N₁), which cancels t_setup. A `cudaDeviceSynchronize()` precedes every stop of
the clock; omitting it caused the host to measure dispatch overhead instead of GPU time for the fast float32
builds (a 3× error we encountered and corrected).

**Size-tiered step counts.** The two-run difference must contain several seconds of genuine compute or it is
swamped by ≈1 s of run-to-run start-up jitter (we initially observed an inter-quartile range larger than the
measured value with 200/600-step runs). Step counts were tiered by cell count so (N₂−N₁)·t_step ≳ 3 s.

**Statistics.** Five measured repeats per configuration; we report the median and the inter-quartile range.
After hardening, all reported IQRs are ≤4 % of the value.

**Fair metric.** Solvers use different-order integrators: RK4 evaluates the effective field four times per
step, Dormand–Prince six, Heun two. The implementation-independent unit of work is therefore the
field-evaluation, and we report milliseconds per field-evaluation (ms/eval) as the primary comparison.
mumax⁺ has no fixed-step RK4; we time its Dormand–Prince with a fixed time step (6 evals/step). mumax3 and
MuMax-CO are driven from `.mx3` scripts via subprocess; mumax⁺ through its Python `TimeSolver`; OOMMF via
`boxsi`.

**Reproduction.** `py -3.13 benchmarks/run_throughput_cs.py`,
`benchmarks/run_throughput_mumax.py`, `benchmarks/run_throughput_mumaxplus.py` populate a single
`benchmarks/results/all_solvers.json`; `benchmarks/make_report.py` then renders all throughput and accuracy
tables and figures.

---

## S2. Standard-problem details

| Problem | Geometry / parameters | Protocol | Observable |
|---|---|---|---|
| SP#1 | L×L×t Permalloy squares, 5 nm cells, t = 10 nm | relax S-state and vortex; energy crossing | L_c |
| SP#2 | prism L:d:t = 5:1:0.1, field along [1,1,1] | quasistatic relax sweep of d/ℓ_ex | remanent ⟨m⟩/M_s, coercivity |
| SP#3 | cube of edge L (in ℓ_ex), K_u = 0.1 K_d | relax flower vs vortex; energy crossing | L_c; and minimize hysteresis H_sw |
| SP#4 | 500×125×3 nm, 2.5 nm cells, field A | relax S-state, apply field A, 1 ns | ⟨m⟩(t), ⟨mₓ⟩(1 ns), t_switch |
| SP#5 | 100×100×10 nm Permalloy, vortex + Zhang–Li current | relax vortex, switch on current | vortex-core trajectory |

**SP#2 (new in Claude-SD).** The remanent ⟨mₓ⟩/M_s after [1,1,1] saturation, quasistatically relaxed with the
GPU damped-LLG relaxer, against mumax3 relax() on identical grids:

| d/ℓ_ex | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| Claude-SD | 0.994 | 1.000 | 0.998 | 0.970 |
| mumax3 | 1.000 | 1.000 | 0.999 | 0.970 |

The remanence dips at large d/ℓ_ex as flux-closure sets in; both codes track it identically (≤0.006).

**SP#4 accuracy table (extended).**

| Solver | build | ⟨mₓ⟩(1 ns) | error vs µMAG (−0.9862) |
|---|---|---|---|
| Claude-SD | cuFFT_f64 / f32 / VkFFT_f32 | −0.9795 | 0.67 % |
| mumax⁺ | f32 | −0.9802 | 0.61 % |
| mumax3 | f32 (DP45) | −0.9686 | 1.78 % |
| OOMMF | f64 (CPU, RKF54) | −0.9843 | 0.19 % |

**SP#3 protocol dependence.** Energy-minimization methods (Claude-SD relax, mumax3 minimize) give
H_sw ≈ −13 to −14 mT; the µMAG reference −20 mT is a dynamic switching field from time-domain LLG, a
different protocol. mumax⁺'s FIRE minimizer reaches −6.0 mT, a different metastable branch. The spread is
protocol-driven and reproduced consistently within each protocol.

---

## S3. Precision/race study (full data)

**Setup.** Co/Pt disk, 100×100×1 cells, 2 nm, Mₛ = 8×10⁵ A m⁻¹, A = 15 pJ m⁻¹, K = 0.8 MJ m⁻³, α = 0.3,
interfacial DMI with D_c = 4√(AK)/π = 4.41 mJ m⁻². A Néel-skyrmion seed (or mz = −1 disk) is relaxed and the
topological charge Q recorded.

**Before the fix (buggy).** Repeated identical relaxations of the same build gave different Q. Run-to-run
standard deviation of Q (N = 12 repeats per point):

| D/D_c | 0.50 | 0.68 | 0.79 | 0.91 | 1.00 |
|---|---|---|---|---|---|
| cuFFT_f64 σ(Q) | 0.087 | 0.261 | 0.434 | 0.479 | 0.242 |
| cuFFT_f32 σ(Q) | 0.453 | 0.520 | 0.420 | 0.342 | 0.651 |
| VkFFT_f32 σ(Q) | 0.102 | 0.352 | 0.340 | 0.345 | 0.760 |

Q scattered across [−1.5, +1.5], i.e. across topological sectors, even in double precision.

**Cross-code reference.** mumax3, both relax() (damped LLG) and minimize() (conjugate gradient), was
deterministic and robust, returning Q ≈ −0.92 (D/D_c = 0.68), −0.89 (0.79), −0.85 (0.91) with run-to-run
variation ≤0.002. This deterministic reference is what flagged Claude-SD's scatter as a defect.

**Diagnosis.** (i) Each field evaluated alone — demag (cuFFT), exchange, anisotropy, DMI — was bit-identical
across repeats. (ii) Fixed-step RK4 with all fields for 1000 steps was bit-identical. (iii) Field-content
bisection in the damped-LLG relaxer: {exchange}, {exchange, anisotropy} were deterministic;
{exchange, DMI} was not. (iv) Code audit: the DMI GPU classes carried a private CUDA stream but did not
override the compositor `set_stream` hook, so in single-stream mode (where the compositor omits inter-field
synchronization) the DMI field ran on its own stream, racing on the shared effective-field buffer.

**Fix and verification.** Adding the `set_stream` override (and ownership flag) to both DMI classes — matching
the pattern already present in the other GPU fields — eliminated the race. After the fix, σ(Q) = 0 at every
D/D_c for all three builds; the full GPU test suite (3906 assertions, 113 cases) remained green.

**Residual (post-fix) characterization.** The fixed-step damped-LLG relaxer does not reach the convergence
threshold for this stiff problem within 20 000 steps (it returns at the step cap); at 80 000 steps Q → −0.85,
approaching the mumax3 skyrmion. Energy comparison confirms the relaxer is mid-trajectory, not at a wrong
minimum. Recommendation: use the energy minimizer or a larger step budget for topological observables near
phase boundaries. This is reflected in `recommend_integrator()` guidance.

---

## S4. AI-agent development protocol

**Specification.** A root `CLAUDE.md` fixes: the build commands and four CUDA presets; the architectural
layering and the `IEffectiveField`/`ISpinTorque`/`IDemagGPU` interfaces; SI conventions (H in A m⁻¹,
H_demag = −N·M, energy E = −µ₀/2 Mₛ Σ m·H_demag dV, γ₀ = 1.76×10¹¹); the demag normalization; and the
test-tag map. The agent consults this before each change.

**Loop.** For a new feature the agent (i) reads the nearest existing field/integrator as a template,
(ii) writes the CUDA kernel, the host class behind the standard interface, the pybind11 binding, and a
Catch2 test that compares the GPU result to the CPU reference (≈10⁻⁶) or to an analytic limit, (iii) builds
and runs the relevant test tag, and (iv) verifies all four CUDA builds still pass. Large changes are made in
small steps. The same loop was used to add the SP#2 standard problem and to diagnose and fix the DMI race
described in S3.

**Guardrails.** SI conventions and interface composition are mandated; every change must carry a test; GPU
code must agree with the CPU path and keep all four builds green.

**Metrics (git history).** 155 commits over 31 days; +136,510 / −4,358 lines; 16.8k C++/CUDA + 5.3k Python +
9.3k test lines (test:source = 0.55); Catch2 cases 7 → 89 → 155 → 302 → 345; 25 effective-field
implementations; 22 applications; four CUDA build variants. (Fig. 6.)

---

## S5. Reproducibility

**Builds.**

```
cmake --preset windows-msvc-cuda          # f64, cuFFT
cmake --preset windows-msvc-cuda-f32      # f32, cuFFT
cmake --preset windows-msvc-cuda-vkfft    # f64, VkFFT
cmake --preset windows-msvc-cuda-vkfft-f32
cmake --build build/windows-msvc-cuda --config Release
```

**Tests.** `ctest --preset windows-msvc` (232 CPU); `unit_tests_gpu.exe` (113 GPU).

**Benchmarks.** `run_throughput_{cs,mumax,mumaxplus}.py` → `merge_nb_accuracy.py` → `make_report.py`;
sensitivity study `benchmarks/sensitivity/run_p1_sensitivity.py` (+ `run_mumax_skyrmion.py`, `make_f4.py`);
SP#2 `benchmarks/sp2/`; SP#4 trajectory `benchmarks/sp4_trajectory.py`.

**Results schema.** All measurements share one record schema in `benchmarks/results/all_solvers.json`
(scenario, solver, build, integrator, dim, cells, T_K, metric, ms_step, ms_eval, repeats, ms_step_iqr,
observable, value, error_vs_ref, error_vs_csf64, hw, notes).

**Environment.** NVIDIA RTX 5060 Ti, CUDA 13.2; mumax3 3.11.1 (`D:/Mumax3`); MuMax-CO (CUDA-Graph mumax3
fork); mumax⁺ 1.2.1 (Python); OOMMF 1.2.1.0.

---

## S6. Extended performance

**Throughput, ms per field-eval (median; full table in `benchmarks/results/throughput_table.md`).**

| Scenario | cells | dim | CS cuFFT_f32 | CS VkFFT_f32 | mumax3 | MuMax-CO | mumax⁺ |
|---|---|---|---|---|---|---|---|
| SP#4 2-D | 10 K | 2D | **0.042** | 0.157 | 0.223 | 0.211 | 0.595 |
| pow2 3-D | 65 K | 3D | **0.155** | 0.295 | 0.305 | 0.290 | 0.739 |
| medium 3-D | 540 K | 3D | 2.610 | 2.777 | 1.708 | **1.663** | 3.302 |
| large 3-D | 2.5 M | 3D | 12.72 | 13.16 | 11.28 | **11.13** | 18.87 |

**Claude-SD float32 scaling sweep (cuFFT, RK4, ms/step).** 45 k → 0.51, 90 k → 0.70, 180 k → 2.07,
360 k → 10.70, 720 k → 17.12, 1.5 M → 58.14, locating the crossover near 0.1–0.5 M cells.

**float32 vs float64 (Claude-SD, cuFFT, large 3-D).** float32 is 4–6× faster (Blackwell Tensor-Core FFT);
the residual error vs float64 is ≤10⁻⁶ for non-topological observables.

**Finite temperature (Heun ↔ Heun, T = 300 K, ms/step).** SP#4 grid: Claude-SD 0.262 vs mumax3 0.553
(2.1×); 0.2 M cells: 1.487 vs 1.326 (mumax3 1.12×).

**MuMax-CO dynamics.** Full SP#4 1-ns adaptive run: mumax3 5.08 s, MuMax-CO 2.24 s (2.3× via CUDA-Graph;
bit-identical physics).

---

## Supplementary figures

- **Fig. S1** Throughput-vs-cell-count crossover (draft and final). (`fig_throughput.png`)
- **Fig. S2** Precision/race study, before/after fix, all builds + mumax3 reference. (`fig_f4_race_fix.png`)
- **Fig. S3** SP#2 remanence and coercivity vs d/ℓ_ex; cross-solver overlay.
  (`benchmarks/sp2/fig_sp2.png`, `fig_sp2_crosssolver.png`)
