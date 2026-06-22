# Building and validating a GPU micromagnetic simulator with an AI coding agent: cross-implementation testing exposes a concurrency defect

Chun-Yeol You¹*

¹ Department of Physics and Chemistry, DGIST, Daegu, Republic of Korea
\* Correspondence: cyyou@dgist.ac.kr

---

## Abstract

Large language model coding agents can now write substantial scientific software, but whether they can
produce a *validated, performant* simulator — and how its correctness should be established — remains open.
We report Claude-SpinDynamics (Claude-SD), a C++20/CUDA micromagnetic simulator with a Python interface,
developed end-to-end with an AI coding agent under a specification-and-test discipline. Claude-SD is, to our
knowledge, the first GPU micromagnetic code to offer **both single and double precision** together with two
demag FFT backends (cuFFT and VkFFT), native spin–orbit/spin-transfer torques and Dzyaloshinskii–Moriya
interaction, and adaptive integrators. Across the µMAG standard problems it agrees with the established
codes mumax3, mumax⁺ and MuMax-CO (and an OOMMF double-precision anchor) to within their mutual spread, and
its float32 build is up to 5× faster than mumax3 per field evaluation on small/2-D problems, with a
throughput crossover near 0.1–0.5 M cells. Crucially, the simulator's native multi-precision/multi-backend
design enabled a cross-implementation comparison that exposed a GPU stream-synchronization race —
undetectable from any single build — which produced non-deterministic topological charge in
Dzyaloshinskii–Moriya-coupled relaxation; we diagnosed and fixed it. These results show that AI agents can
build research-grade simulation software and that built-in cross-validation is both feasible and essential.

---

## Introduction

Micromagnetic simulation underpins the design of magnetic memories, sensors, and spintronic logic by solving
the Landau–Lifshitz–Gilbert (LLG) equation on discretized geometries [1,2]. The community standardizes on a
small number of mature solvers — OOMMF [3], mumax3 [4], and more recently the extensible mumax⁺ [5] — whose
correctness is anchored to the NIST/µMAG standard problems [1]. Writing such a solver is a significant
software undertaking: tens of thousands of lines of numerically delicate CPU and GPU code, demag tensor
evaluation, FFT convolution, stiff time integration, and stochastic thermal dynamics, all of which must be
verified.

In parallel, AI coding agents have advanced to the point of authoring non-trivial software across many files
under human direction. Early demonstrations in computational science have used such agents to *optimize*
existing codes — for example, a CUDA-Graph acceleration of mumax3 achieved with an agent [6] — but the harder
question is whether an agent can build a *complete and validated* simulator from scratch, and what verification
methodology makes that trustworthy.

Here we address both questions with Claude-SD, a micromagnetic simulator written end-to-end with an AI coding
agent (Claude Code) governed by a project specification and a test-driven loop. We make three contributions.
First, we describe a reproducible agent-development workflow and quantify it. Second, we validate the
resulting code against the µMAG standard problems and against three independent solvers, and characterize its
performance, showing competitive — and on small/2-D problems superior — throughput, with the distinguishing
capability of double precision. Third, and most importantly for the methodological argument, we show that the
simulator's native multi-build (float32/float64 × cuFFT/VkFFT) and multi-solver design enabled a cross-
implementation comparison that *exposed a real concurrency defect* — a GPU stream race that randomized
topological charge near the skyrmion metastability boundary — which we then diagnosed and fixed. The defect
was invisible to any single build and to the standard problems; only cross-validation surfaced it. We argue
this is the crux of trustworthy AI-assisted scientific software: not that the agent writes correct code on
the first try, but that an aggressive, built-in cross-validation regime catches what slips through.

---

## Results

### Agent-driven development and verification

Claude-SD was developed under a root specification file (`CLAUDE.md`) that fixes the architectural layering
(`types → grid → field → material → effective_field → integrator`), the SI conventions
(H in A m⁻¹, H_demag = −N·M, γ₀ = 1.76 × 10¹¹ rad s⁻¹ T⁻¹), and a test-tag map. The agent reads this
specification, implements each feature as a CUDA kernel plus its CPU reference plus a Catch2 unit test, and
the change is accepted only after the test passes across all four CUDA build variants (cuFFT/VkFFT × f32/f64)
(Fig. 1). This loop is the operational core of the methodology: every GPU field is required to reproduce its
CPU twin to ≈10⁻⁶, and physics is checked against analytic limits or the µMAG references.

The development is quantified in Fig. 6. Over 31 days and 155 commits the project accrued ≈16.8k lines of
C++/CUDA and 5.3k lines of Python, with 9.3k lines of tests — a test-to-source ratio of 0.55. The Catch2 test
count grew in lockstep with features, from 7 to 345 cases (232 CPU + 113 GPU), reflecting the test-driven
discipline rather than tests bolted on at the end. The delivered surface comprises 25 effective-field
implementations (CPU and GPU), three time integrators plus two energy minimizers, and 22 validation/benchmark
applications.

### Validation against the µMAG standard problems and independent solvers

Claude-SD reproduces the µMAG standard problems SP#1–SP#5 and agrees with the independent solvers within
their mutual spread (Fig. 2, Table 1). For the dynamic switching problem SP#4 (field A), the magnetization
component ⟨mₓ⟩ at 1 ns is −0.981 (Claude-SD), −0.969 (mumax3), −0.980 (mumax⁺) and −0.984 (OOMMF, CPU
double), versus the µMAG reference −0.986 — all within 2 %, with the double-precision OOMMF anchor closest.
For SP#1, Claude-SD and mumax⁺ agree on the single-domain/vortex crossover length L_c = 99.7 nm. For SP#3,
the energy-minimization switching field is −13.8 mT (Claude-SD) and −13.3 mT (mumax3); mumax⁺'s FIRE
minimizer reaches a different metastable branch (−6.0 mT), illustrating that the residual spread is
protocol-, not implementation-, driven. The ferromagnetic resonance frequency (SP-style macrospin) is within
0.06 % of the Kittel value for all codes. For the newly implemented SP#2, Claude-SD and mumax3 agree on the
remanent ⟨mₓ⟩/M_s to ≤0.006 across the full d/ℓ_ex sweep. Domain-wall velocities (Zhang–Li) agree to within
the ≈10 % spread expected from finite-cell demag corrections.

### Performance and the precision/backend trade-off

We benchmarked throughput with a hardened protocol (two-run subtraction to cancel start-up cost,
size-tiered step counts, five-repeat medians; Methods) and report the cross-order-fair metric of
milliseconds per field-evaluation (Table 2, Fig. 5). Two regimes emerge, separated by a crossover near
0.1–0.5 M cells. Below it, Claude-SD's float32 build with cuFFT and CUDA-Graph replay is fastest: 0.042 ms
per eval at the SP#4 grid (10 K cells), 5.3× faster than mumax3 and 14× faster than mumax⁺, because at small
sizes kernel-launch overhead dominates and graph replay removes it. Above it, the comparison becomes
cuFFT-bound and the mumax family leads: at 2.5 M cells MuMax-CO is fastest (11.1 ms per eval) with Claude-SD
within 1.1×. Within Claude-SD, float32 is 4–6× faster than float64 at large 3-D sizes on the Blackwell GPU
(Tensor-Core FFT), and VkFFT overtakes cuFFT for large 3-D transforms while losing on small grids. The
finite-temperature picture mirrors the T = 0 one: with matched Heun stochastic-LLG integrators, Claude-SD is
2.1× faster than mumax3 at the SP#4 grid and ≈parity at 0.2 M cells. As an aside, the CUDA-Graph-optimized
MuMax-CO ran the full SP#4 1-ns adaptive simulation 2.3× faster than stock mumax3, consistent with [6].

The practical message of Table 2 and Fig. 5 is that no single code dominates: Claude-SD is preferable for
small/2-D dynamics, for any study requiring double-precision reference accuracy, and for native custom-torque
physics; the mumax family is preferable for large float32 production runs.

### Cross-validation exposes and fixes a GPU concurrency defect

The most consequential result is what the multi-implementation comparison revealed (Fig. 4). We relaxed a
seeded Néel skyrmion in a Co/Pt disk across the Dzyaloshinskii–Moriya boundary (D/D_c = 0.5–1.0, with
D_c = 4√(AK)/π) and recorded the relaxed topological charge Q. Initially, repeated identical runs of the
*same* Claude-SD build gave *different* Q — scattering across nearly the full range [−1.5, +1.5], with a
run-to-run standard deviation up to 0.48 even in double precision and up to 0.76 in float32 (Fig. 4a). In
contrast, mumax3's relax() and minimize() were deterministic and robust there, returning a clean skyrmion
Q ≈ −0.9 every time. A physically meaningful, deterministic micromagnetic relaxation must not depend on
run-to-run thread scheduling, so the scatter signalled a defect in Claude-SD.

We localized it by bisection. The individual field kernels (demag/cuFFT, exchange, anisotropy, DMI) were each
bit-identical run-to-run, and a fixed-step RK4 integration with the same field set was deterministic; only
the damped-LLG relaxation with DMI present scattered. The cause was a missing single-stream redirect: the
GPU Dzyaloshinskii–Moriya field classes declared a private CUDA stream but did not override the
compositor's `set_stream` hook. In single-stream mode the field compositor omits the inter-field
synchronization, trusting every field to run on the shared stream — but the DMI field continued to execute
on its own stream, accumulating into the shared effective-field buffer concurrently with the other fields, a
read–modify–write race. The tiny last-bit differences this produced were amplified, near the metastability
bifurcation, into different topological sectors. Adding the `set_stream` override (and the corresponding
ownership flag) eliminated the race: relaxation became bit-identical run-to-run, with topological charge
standard deviation collapsing to zero across all builds (Fig. 4b,c), while all 113 GPU unit tests continued
to pass.

Two points follow. First, this defect was invisible to every single-build run and to the standard problems —
which are not near a bifurcation — and only the cross-build/cross-solver comparison made it diagnosable, by
contrasting Claude-SD's scatter against mumax3's determinism. Second, after the fix a residual, *now
deterministic*, difference remains: Claude-SD's fixed-step damped-LLG relaxer (`RelaxGPU`) converges more
slowly than mumax3's adaptive relaxer for this stiff problem and, within a fixed step budget, lands on a
different metastable state; given a larger budget it approaches the same Q ≈ −0.85 skyrmion. This is a
convergence-rate limitation, not an error, and is documented as such.

---

## Discussion

Claude-SD demonstrates that an AI coding agent, directed by a specification and disciplined by tests, can
produce a micromagnetic simulator that is correct on the community benchmarks and competitive in performance,
while adding a capability the incumbent GPU codes lack — double precision — alongside a second FFT backend and
native spin torques. But the central lesson is methodological. The agent did *not* write defect-free code;
it wrote a subtle GPU concurrency bug that passed every unit test. What made the bug catchable was the
simulator's deliberately redundant design: multiple precisions, multiple FFT backends, and a benchmark
harness that pits the code against three independent solvers. Discrepancy — Claude-SD's non-determinism
against mumax3's determinism — was the diagnostic signal. We therefore argue that trustworthy AI-assisted
scientific software should treat cross-implementation validation not as an afterthought but as a first-class
design goal, exactly because the failure mode of agent-written code is plausible-looking code that is subtly
wrong.

The limitations are honestly bounded. Claude-SD's damped-LLG relaxer is slower-converging than adaptive
minimizers for stiff Dzyaloshinskii–Moriya textures; topological-charge studies near phase boundaries should
use the energy minimizer or a larger step budget. The performance advantage is confined to the small/2-D
regime; large float32 production remains the mumax family's domain. And the development metrics, while
quantitative, describe a single project; whether the workflow generalizes to other physics domains is an open
question we expect future work to address.

---

## Methods

**Physics and discretization.** Claude-SD integrates the Landau–Lifshitz–Gilbert equation
dm/dt = −γ′µ₀(m×H) − γ′αµ₀ m×(m×H) with γ′ = γ₀/(1+α²), on a structured finite-difference grid with
x-fastest linear indexing. Effective-field contributions (each adds to H): uniform/Spatial Zeeman, uniaxial
and cubic anisotropy, six-point Laplacian exchange (Neumann or periodic BC), open-BC demag by FFT convolution
with the Newell tensor [7] zero-padded to 2N, and periodic-BC demag by an image-sum kernel; plus bulk and
interfacial DMI, RKKY interlayer exchange, magnetoelastic and surface anisotropy. Spin torques (Slonczewski
STT, spin–orbit torque, Zhang–Li) are added per integrator stage. Finite-temperature dynamics use the
stochastic LLG with a Stratonovich Heun scheme and cuRAND-generated thermal fields [8,9].

**GPU implementation.** Effective fields and integrators run entirely on device with no per-step host
transfer; a field compositor accumulates contributions either on per-field streams (synchronized between
fields) or on a single shared stream (serialized by stream order). Demag uses cuFFT or, optionally, VkFFT
[10]. Fixed-step integrators capture the per-step kernel sequence as a CUDA Graph and replay it. Single and
double precision are selected at build time.

**Integrators.** RK4 (fixed step), RK45 Dormand–Prince (adaptive, FSAL) [11], and Heun (stochastic LLG), with
damping-only `RelaxGPU` and a Barzilai–Borwein `MinimizeGPU`. A helper `recommend_integrator()` selects among
them from the damping, temperature, goal, and an analytic phase-error estimate.

**Benchmark protocol.** Throughput was measured by two-run subtraction (t(N₂)−t(N₁))/(N₂−N₁) to cancel
one-time set-up, with size-tiered step counts so the timed difference is several seconds of compute, five
measured repeats reported as median with interquartile range, on an NVIDIA RTX 5060 Ti (Blackwell, CUDA 13.2)
with the GPU otherwise idle. Because solvers use different-order integrators (RK4 = 4, Dormand–Prince = 6,
Heun = 2 field evaluations per step), the fair metric is milliseconds per field-evaluation. mumax3 and
MuMax-CO were driven from `.mx3` scripts; mumax⁺ through its Python API; OOMMF via boxsi. Reference values
for SP#4 used the relaxed S-state followed by field application.

**Reproducibility.** The cross-solver benchmark is fully scripted; `make_report.py` regenerates all
performance and accuracy tables and figures from a single results file. Build presets, environment, and the
exact commands are given in Supplementary Information.

---

## Data and code availability

The source code, benchmark suite, and all data needed to regenerate the tables and figures are available at
https://github.com/mirryou-maker/Claude-SpinDynamics (GPLv3), archived on Zenodo (DOI to be assigned on
release). Citation metadata is provided in `CITATION.cff`.

---

## References

1. M. J. Donahue and the µMAG group. *µMAG Standard Problems*. NIST/CTCMS,
   https://www.ctcms.nist.gov/~rdm/mumag.org.html.
2. L. D. Landau, E. M. Lifshitz. On the theory of the dispersion of magnetic permeability in ferromagnetic
   bodies. *Phys. Z. Sowjetunion* **8**, 153 (1935); T. L. Gilbert. *IEEE Trans. Magn.* **40**, 3443 (2004).
3. M. J. Donahue, D. G. Porter. *OOMMF User's Guide*, NISTIR 6376 (1999).
4. A. Vansteenkiste et al. The design and verification of MuMax3. *AIP Adv.* **4**, 107133 (2014).
5. L. Moreels et al. mumax⁺: extensible GPU-accelerated micromagnetics and beyond. *npj Comput. Mater.*
   (2025/26).
6. C.-Y. You. Optimization of MuMax3 by using Claude Code: a CUDA-Graph-based case study in AI-assisted
   performance engineering (2026).
7. A. J. Newell, W. Williams, D. J. Dunlop. A generalization of the demagnetizing tensor for nonuniform
   magnetization. *J. Geophys. Res.* **98** (B6), 9551 (1993).
8. W. F. Brown Jr. Thermal fluctuations of a single-domain particle. *Phys. Rev.* **130**, 1677 (1963).
9. J. L. García-Palacios, F. J. Lázaro. Langevin-dynamics study of the dynamical properties of small magnetic
   particles. *Phys. Rev. B* **58**, 14937 (1998).
10. D. Tolmachev. VkFFT — a performant GPU FFT library. *IEEE Access* **11** (2023).
11. J. R. Dormand, P. J. Prince. A family of embedded Runge–Kutta formulae. *J. Comput. Appl. Math.* **6**,
    19 (1980).
12. J. C. Slonczewski. Current-driven excitation of magnetic multilayers. *J. Magn. Magn. Mater.* **159**, L1
    (1996).
13. S. Zhang, Z. Li. Roles of nonequilibrium conduction electrons on the magnetization dynamics of
    ferromagnets. *Phys. Rev. Lett.* **93**, 127204 (2004).
14. NVIDIA. *CUDA C++ Programming Guide — CUDA Graphs*; *NVIDIA Blackwell Architecture Whitepaper* (2024).
15. A. Fert, N. Reyren, V. Cros. Magnetic skyrmions: advances in physics and potential applications.
    *Nat. Rev. Mater.* **2**, 17031 (2017).

---

## Figures (main text)

- **Fig. 1** AI-agent development and verification loop, and Claude-SD architecture. (`fig_f1_pipeline.png`)
- **Fig. 2** µMAG standard-problem validation, SP#1–SP#5. (`fig_f2_umag_validation.png`)
- **Fig. 3** Cross-solver throughput and the precision/backend trade-off. (`fig_throughput.png`,
  `fig_scenario_bars.png`)
- **Fig. 4** Cross-validation exposes and fixes a GPU stream race: topological charge before (scatter) and
  after (deterministic) the fix. (`fig_f4_race_fix.png`)
- **Fig. 5** Solver capability matrix and throughput landscape. (`fig_f5_landscape.png`)
- **Fig. 6** Agent-driven development and verification metrics. (`fig_f6_dev_metrics.png`)
