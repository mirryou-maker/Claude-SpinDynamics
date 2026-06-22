# Claude-SD Development — Condensed Prompt Arc

A condensed selection of ~30 milestone prompts (of 281) that trace the development story,
in English. Numbers in [brackets] refer to [DEVELOPMENT_PROMPTS.md](DEVELOPMENT_PROMPTS.md).
Suitable as a paper figure/box or a one-page narrative of the agent-driven build.

> The full set is dominated by a recurring control loop — *"break it step by step,"
> "analyze the result and recommend the next task," "save / commit / document."* The
> milestones below are the moments where direction, not just continuation, was set.

---

### I. Core physics (2026-05-25 → 05-27)
- **[4]** Start Phase 1d STT — implement SOT (spin-orbit torque) at the same time.
- **[9]** First implement the #1-priority Demag Field, continue through priority #2, then review.
- **[10]** Continue — and if possible, proceed without asking me in the middle.

### II. The demag-bug ordeal under a token ceiling (05-27 → 06-02)
- **[38]** The demag bug still isn't fixed. Completely remove the existing demag code and rewrite it considering all errors seen — but since writing it at once keeps hitting the token max, plan it as multiple steps that never exceed the max token usage.
- **[40]** It keeps failing from token overflow. Analyze the root cause and propose a strategy that minimizes token consumption.
- **[45]** Apply only the step 1-b change (term-3 coefficient → 1/3) and report.
- **[48]** Compare against the newell_g reference implementation — don't write code, just compare and report.
- **[53]** Apply from A1 one step at a time; after each step, analyze and re-plan the fix strategy, then report.

### III. Integrators & finite temperature (06-02)
- **[57]** Implement the recommended RK45 step by step, review, then recommend next.
- **[58]** For finite-temperature you can't use the adaptive-timestep RK45 — you must use a fixed timestep. Add finite-T micromagnetics to the plan. *(domain knowledge steering the design)*

### IV. GPU build-out (06-03 → 06-06)
- **[73]** Implement Phase 3 (CUDA); break it step by step and present a concrete plan first.
- **[93]** Make a plan to implement the GPU full LLG step by step.
- **[94]** I'll develop it sequentially — in 40 min when usage resets, implement just G1 first. *(pacing around quota)*
- **[104]** Implement HeunIntegratorGPU.

### V. Validation, examples, project identity (06-04 → 06-07)
- **[88]** Run µMAG SP#1 and save the result.
- **[115]** Run the mumax.github.io examples, make notebooks for them, and save the results.
- **[137]** Rename the project to Claude-SpinDynamics and move the working folder. Request the needed permissions.
- **[138]** Don't delete the old folder when done — keep it.

### VI. mumax3 API coverage (06-19)
- **[148]** Defer Phase D; first organize which mumax3 API functions are implemented and which aren't.
- **[171]** Is multi-GPU support feasible? → **[172]** implement multi-GPU via approach A.

### VII. Performance & precision (06-19 → 06-21)
- **[180]** Fix priorities 1–5, then review again for improvements. Consider 3rd-party libraries.
- **[184]** Download the needed packages and implement P9, P12, P11, P10, P13, P14 in that order.
- **[211]** Provide a thorough VkFFT-vs-cuFFT comparison for non-power-of-two cell sizes; analyze the mumax→cuFFT data flow (layout, normalization, padding). *(the longest, most technical prompt)*
- **[220]** In which cases can Claude-SD show an advantage over mumax?

### VIII. The cross-solver benchmark campaign (06-20 → 06-21)
- **[196]** Plan a benchmark comparing mumax3, OOMMF, our program (float32), and our program (double) — I'll run it after reviewing the plan.
- **[199]** First investigate the solver discrepancy, find and fix the cause, then prepare the benchmark. Call our app "Claude-SD". *(the seed of the race-bug discovery)*
- **[221]** Write new scenarios — both CS-favorable and mumax-favorable — but evaluate fairly under identical conditions (same precision, step size, RK45).
- **[251]** Plan a benchmark vs mumax/mumax+/MuMax-CO with paper-ready tables/figures: fair same-integrator comparison, T=0 focus plus T>0, auto-integrator selection, and references. *(consolidated 8-point spec)*
- **[254]** Is the SP#2 problem a Claude-SD problem, or does it appear in mumax/mumax+/MuMax-CO too? *(cross-validation instinct)*

### IX. Release & paper (06-22)
- **[248]** I'm about to wrap up — one last time, check and report any improvements, fixes, or supplements.
- **[267]** Register the files on GitHub so users can download and use it.
- **[269]** I'd like to submit a paper about the Claude-SD development — recommend a journal.
- **[271]** I'll target npj Computational Materials; include figures/tables and Supplementary, plan extra simulations, and present the overall flow/TOC/strategy first.
- **[276]** Write the paper (abstract, body, references), Supplementary, and cover letter.

---

### The arc in one sentence
A physicist directed an AI agent through ~280 short, iterative instructions — pacing around token and quota limits, injecting domain knowledge at key forks (fixed-step SLLG, fair benchmarking, cross-solver validation) — to build, validate, optimize, and publish a GPU micromagnetic simulator, with the cross-validation he insisted on ([199], [254]) ultimately exposing the GPU stream-race bug that became the paper's central result.
