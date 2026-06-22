# Paper Plan — Claude-SpinDynamics for *npj Computational Materials*

Working plan (pre-writing). Target: **npj Computational Materials** (Nature Portfolio, OA).
Framing **A** (AI-assisted development) per the venue strategy, sharpened so the AI workflow and a
genuine physics finding reinforce each other.

---

## 1. The thesis (one sentence)

> *An AI coding agent (Claude Code) was used to build, validate, and benchmark a research-grade,
> dual-precision GPU micromagnetic simulator (Claude-SD) from scratch; the resulting native
> multi-precision / multi-FFT-backend / multi-solver capability enabled a systematic cross-validation
> that reveals the float-precision and floating-point-nondeterminism sensitivity of topological
> observables near metastability boundaries — a general caveat for GPU micromagnetics.*

**Why this is npj-worthy (not "just another simulator"):**
1. **Methodological novelty** — a reproducible *AI-agent → architecture → CUDA kernels → tests →
   cross-solver benchmark* workflow, quantified (not anecdotal).
2. **Enabled discovery** — the AI workflow makes multi-implementation cross-validation cheap, which
   surfaces a *physics* result (topological-charge precision sensitivity) hard to see with one code.
3. **Rigorous validation** — µMAG SP#1–5 + head-to-head vs mumax3 / mumax+ / MuMax-CO (+ OOMMF anchor).

**Contribution ranking for the cover letter:** methodology ≈ physics finding > performance > the tool.

---

## 2. Article structure (npj CM format)

Abstract (~150 words) → Introduction → Results → Discussion → Methods → Data/Code availability → Refs.
Figures live in Results. Target ~6 main figures, ~3 main tables, rich Supplementary.

| Section | Content | Status of material |
|---|---|---|
| **Abstract** | thesis + 3 headline numbers (validation, crossover, Q-sensitivity) | write |
| **Introduction** | (i) micromagnetics + existing codes (OOMMF/mumax3/mumax+/Vampire); (ii) the rise of AI coding agents in computational science; (iii) gap: can an agent build a *validated, performant* simulator, and what does multi-implementation comparison reveal?; (iv) contributions | write |
| **Results 2.1 — AI-agent development workflow** | the pipeline (CLAUDE.md spec → agent loop → test-driven kernels → 4-build CI) + quantified metrics (LOC, test counts 232 CPU+113 GPU, # fields/integrators, dev iterations, human-vs-agent split). **Fig 1** | partial (need metrics mining) |
| **Results 2.2 — Validation** | µMAG SP#1–5 + cross-solver agreement; physics correctness. **Fig 2, Table 3** | mostly have |
| **Results 2.3 — Performance** | throughput crossover, f32/f64 Tensor-Core, cuFFT/VkFFT, ms/eval fairness. **Fig 3, Table 1–2** | have |
| **Results 2.4 — Enabled finding: precision sensitivity of topological observables** | systematic Q(D/Dc) sensitivity across precision/backend/run-to-run; contrast non-topological robustness. **Fig 4** | **need rigorous study** |
| **Discussion** | what the AI workflow does/doesn't do well; generalizability to other codes; implications of the Q caveat for skyrmion micromagnetics; limitations | write |
| **Methods** | LLG/SLLG, demag (Newell), integrators, GPU design, the agent workflow protocol + guardrails, benchmark methodology, hardware | have (CLAUDE.md + RESULTS_2026.md) |

---

## 3. Main-text figure plan (6)

| Fig | Title | Shows | Data source | Status |
|---|---|---|---|---|
| **F1** | AI-agent development pipeline + Claude-SD architecture | schematic: CLAUDE.md spec → agent → kernel/binding/test → 4-build verify loop; layer model; capability badges | schematic | **make** |
| **F2** | µMAG + cross-solver validation | 5-panel: SP#1 L_c, SP#2 remanence(d/lex), SP#3 hysteresis, SP#4 ⟨m⟩(t), SP#5 core; CS vs mumax3/mumax+/ref | NB46/47/41 + sp2 + benchmarks/sp5 | **assemble** (have most; SP#4 ⟨m⟩(t) needs trajectory log; SP#5 re-run) |
| **F3** | Cross-solver performance | (a) throughput ms/eval vs cells (crossover), (b) per-scenario bars, (c) f32/f64 speedup | all_solvers.json (fig_throughput, fig_scenario_bars, F6) | **have** |
| **F4** | Precision sensitivity of topological charge | Q vs D/Dc with per-build spread + run-to-run error bars; map of the sensitive region; non-topological robustness inset | **new study** | **need sim** |
| **F5** | Accuracy–performance & capability landscape | Pareto (accuracy err vs wall) + capability matrix heatmap (precision/backend/torques/scripting) | all_solvers.json + T1 | **assemble** |
| **F6** | Development & verification metrics | test-coverage growth over commits, code composition (CPU/GPU/Python), agent-iteration counts, validation funnel | git history + logs | **mine data** |

(F1, F6 can merge if the editor wants ≤5 figures. F5 can move to Supplementary if space-limited.)

---

## 4. Main-text tables (3)

- **T1 — Solver capability matrix** (precision, integrators, FFT, native SOT/DMI/RKKY/ME/AFM, scripting, license; CS / mumax3 / mumax+ / MuMax-CO / OOMMF). *Have.*
- **T2 — Throughput** (ms/step + ms/eval per scenario × solver/build, median±IQR). *Have.*
- **T3 — Accuracy vs µMAG** (SP#1 L_c, SP#2 remanence, SP#3 H_sw, SP#4 mx & t_switch, SP#5 core; per solver, error vs reference & vs CS-f64). *Have/partial — fill mumax3 + add OOMMF SP#4.*

---

## 5. Supplementary plan

| § | Content | Status |
|---|---|---|
| **S1 Benchmark methodology** | two-run subtraction, size-tiered steps, 5-repeat median+IQR, clock state, exclusivity; ms/eval normalization rationale | have (BENCHMARK_PLAN.md) |
| **S2 Per-standard-problem detail** | geometry, params, protocols, full result tables + convergence for SP#1–5 | have (NB + sp2) |
| **S3 Precision-sensitivity full study** | Q(D/Dc) tables, run-to-run histograms, per-build/backend breakdown, cross-solver (mumax3/mumax+) replication, the atomic-reduction-ordering mechanism | **need sim** |
| **S4 AI-development protocol** | the CLAUDE.md spec, representative prompts, the test-driven loop, guardrails, an example feature (SP#2 or a GPU field) as a worked case; human-intervention log | have (CLAUDE.md, git) + write |
| **S5 Reproducibility** | exact build presets, environment, commands, code+data DOI (Zenodo), all_solvers.json schema | have |
| **S6 Extended performance** | VkFFT vs cuFFT breakdown, demag FFT fraction, CUDA-Graph effect, large-grid scaling | have (BENCH_REPORT) |

---

## 6. Gap analysis — additional simulation / benchmarking needed

Prioritized; **the Q-sensitivity study is the critical new science.**

### P1 (critical) — Precision-sensitivity of topological charge (Fig 4 + S3)
Current data is ~4 ad-hoc runs. To be a defensible *finding*:
- **Sweep D/Dc** (e.g. 0.5–1.1 in ~8 steps) on the Co/Pt disk; at each D, relax a seeded skyrmion.
- For each (D, build ∈ {cuFFT_f64, cuFFT_f32, VkFFT_f32}) run **N≥20 repeats** → report mean Q + spread.
- **Quantify run-to-run nondeterminism** within one build (same seed, repeated) — the striking claim.
- **Replicate in mumax3 and mumax+** at matched D to show it's a *general* FP/metastability effect, not a CS bug.
- **Contrast**: show a non-topological observable (⟨mz⟩, DW velocity) is bit-reproducible under the same conditions.
- Define/illustrate the **"sensitive band"** in (D/Dc, precision) space.
- Effort: moderate GPU (skyrmion 10K cells is fast; ~8 D × 3 builds × 20 reps + mumax replication). ~1–2 GPU-hours.

### P2 (strengthens validation) — Complete the accuracy table
- Run **OOMMF** SP#4 as a CPU-double accuracy anchor (already in older REPORT.md C2 config; re-run clean).
- Parse **mumax3** physics observables (not just wall) for SP#1/#3/#4 → fill Table 3.
- Persist **SP#4 ⟨m⟩(t) trajectory** (NB41 currently drops the log) for Fig 2 panel + a trajectory-error inset.
- Effort: low–moderate.

### P3 (methodology rigor) — Development metrics (Fig 6 + S4)
- Mine **git history**: commits, LOC added per phase, test-count growth, files touched, CPU/GPU/Python split.
- Tally **fields/integrators/standard-problems** delivered; estimate human-vs-agent contribution from commit/transcript records.
- This is data mining (no GPU), but essential to make the methodology claim quantitative not anecdotal.

### P4 (optional polish)
- SP#5 Zhang–Li vortex-core clean re-run for Fig 2.
- Large-grid scaling curve (16K→2.5M) for S6.
- MuMax-CO timing column on the dynamics scenarios.

---

## 7. Reviewer concerns & mitigation

| Likely concern | Mitigation |
|---|---|
| "AI-built simulator = incremental / anecdotal" | Quantify the workflow (Fig 6, S4); frame as a *reproducible methodology*, and let the **physics finding** (Fig 4) carry novelty. |
| "Why another micromagnetic code?" | T1 differentiators: f64 capability (mumax family is f32-only), dual FFT backend, native SOT; plus the cross-validation it enables. |
| "Is the Q-sensitivity a bug?" | P1 shows it in **mumax3 and mumax+ too** + the FP-reduction-ordering mechanism → a general phenomenon, not a CS defect. |
| "Reproducibility" | Public repo + Zenodo DOI + `make_report.py` one-command regeneration; S5. |
| "Validation depth" | µMAG SP#1–5 + 3–4 independent solvers + f64 reference. |

---

## 8. Pre-submission checklist
- [ ] P1 precision-sensitivity study run + Fig 4 + S3
- [ ] P2 OOMMF anchor + mumax3 accuracy + SP#4 trajectory
- [ ] P3 development metrics (Fig 6, S4)
- [ ] F1 pipeline schematic, F2 validation panel, F5 capability landscape
- [ ] Make repo **public** + Zenodo DOI (code & data availability statements)
- [ ] Abstract + cover letter (lead with methodology + finding)
- [ ] Confirm npj CM current Aims & Scope / article type fit

---

*Assets in hand: `benchmarks/RESULTS_2026.md`, `benchmarks/results/all_solvers.json` (+ figures),
`benchmarks/sp2/`, NB41–50 (latest-build re-run), `CLAUDE.md` (Methods source). New work concentrated in
P1–P3 above.*
