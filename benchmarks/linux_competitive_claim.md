# Does the competitive claim hold on native Linux? — scope & evidence

**For the paper.** This note states precisely what the Linux measurements let us
claim about Claude-SD *vs other solvers* (mumax3, mumax⁺, MuMax-CO), and what is
still required for a fully rigorous Linux competitive claim. The distinction
matters for scientific integrity: **portability of our solver** and **a
cross-solver competitive advantage** are two different claims with different
evidence requirements.

## 1. The thesis and how it was measured

The competitive thesis is: *Claude-SD (float32) is faster than or on par with
mumax3 (float32) at small-to-medium problem sizes, and competitive up to ~262 K
cells; mumax3's FFT pipeline pulls ahead only at very large grids (≥1 M).*

**Hardware/OS of record for that campaign** (`benchmarks/BENCHMARK_PLAN.md`,
`REPORT.md`): RTX 5060 Ti (Blackwell GB206, sm_120, 8 GB), CUDA 13.2, **Windows
11 / MSVC**. Competitors are **float32-only**; the small-grid advantage leans on
Claude-SD's CUDA-Graph launch-overhead reduction and (at f32) the Blackwell
Tensor-Core FFT.

## 2. What the Linux measurements DO establish

**Claude-SD is platform-portable.** On a native-Linux GPU host (AWS g6.xlarge,
NVIDIA L4), the Linux CUDA build reproduces the Windows CUDA build of Claude-SD:

| grid | Linux L4 | Windows baseline | ratio |
|---|--:|--:|--:|
| 2.5 M cells (compute-bound) | 292.4 ms/step | 290 ms/step | **1.01×** |

within ~1 % at the compute-bound size (`benchmarks/linux_cpu_parity.md`). CPU is
likewise on par (Linux single-thread actually faster). **Conclusion we can state
outright:** *Claude-SD has no Windows-specific performance dependency; its
absolute performance on Linux equals its Windows performance.*

## 3. What the Linux measurements DO NOT (yet) establish

The competitive *advantage* does **not** transfer to Linux automatically, for
three reasons — all of which the paper must acknowledge:

1. **Competitors were never measured on Linux.** mumax3 is Linux-native
   (Go + CUDA) and typically runs *faster* on Linux than Windows (no WDDM, lower
   kernel-launch latency). If mumax3 gains more from Linux than Claude-SD does,
   the margin changes.
2. **The winning regime is the most platform-sensitive quantity.** The advantage
   lives at **small grids (≤65 K cells)**, where **kernel-launch overhead**
   dominates — exactly the thing that differs most between Windows (WDDM) and
   Linux. A same-solver Win↔Linux parity (§2) does not settle a *cross-solver*
   comparison in this regime, because both solvers' launch overheads move.
3. **Precision/architecture gap.** The L4 parity validated the **f64** path on
   **Ada**. The competitive claim is an **f32 / Blackwell** story; the L4 has no
   Blackwell Tensor-Core FFT, so it neither reproduces nor refutes the f32
   magnitude that underpins the headline numbers.

## 3a. UPDATE — same-host Linux cross-solver run DONE (Ada / L4)

The gap in §3 has been closed for the **Ada** regime: Claude-SD (f32) and mumax3
(v3.12, f32) were run on the **same** Linux GPU host (AWS L4). Full data +
methodology: [`linux_crosssolver_results.md`](linux_crosssolver_results.md).

| cells | CS f32 / mumax3 | verdict |
|--:|:--:|---|
| 10 K | 0.42× | **CS 2.4× faster** |
| 65 K | 0.96× | **on par** |
| 540 K – 1 M | 1.6–1.7× | mumax3 leads (its FFT sweet spot) |
| 4.2 M | 1.10× | near parity |

**The small-grid competitive advantage is confirmed on native Linux** — and
crucially it is the launch-overhead-bound regime that the self-parity of §2 could
not settle. The crossover shape matches the Windows/Blackwell campaign. What still
requires **Blackwell-Linux** (RTX 50-series or `p6-b200`) is only the *headline
f32/Blackwell magnitude*, since L4 (Ada) lacks the Blackwell Tensor-Core FFT that
lifts Claude-SD's mid/large throughput on the campaign card.

## 4. Defensible wording for the paper (as of now)

- ✅ *"Claude-SD's GPU performance is platform-independent: the Linux CUDA build
  matches the Windows build within ~1 % at compute-bound sizes (2.5 M cells,
  NVIDIA L4)."* — fully supported.
- ✅ *"All competitive comparisons in §X were performed on Windows 11 (RTX 5060
  Ti, Blackwell); given Claude-SD's demonstrated platform-independence, its side
  of every comparison transfers to Linux unchanged."* — supported.
- ⚠️ *"Claude-SD outperforms mumax3 on Linux"* — **NOT yet supported.** Requires
  §5. State the competitive results as Windows-measured, and either (a) present
  the Linux cross-solver run once done, or (b) explicitly scope the competitive
  claim to the measured platform and cite portability separately.

## 5. What closes the gap — same-host Linux cross-solver run

To claim the competitive advantage *on Linux* rigorously, run **both** Claude-SD
and the competitors on the **same Linux GPU host**, reusing the existing harness:

- `benchmarks/run_throughput_cs.py` (CS f64/f32) — already cross-platform.
- `benchmarks/run_throughput_mumax.py` (mumax3) — mumax3 installs natively on
  Linux; time the solver loop with the same two-run subtraction.
- Optionally `run_throughput_mumaxplus.py` (mumax⁺, Python API).

Two tiers:

| host | what it proves | note |
|---|---|---|
| **L4 / Ada (available now)** | competitive relationship on Linux in the **Ada, f32+f64** regime | does *not* reproduce the Blackwell f32 magnitude |
| **Blackwell on Linux** (RTX 50-series workstation, or AWS `p6-b200`, `CUDA_ARCH=100`) | the **headline f32/Blackwell** claim on Linux, matching the campaign silicon | the only way to state the headline claim on Linux with identical arch |

**Recommendation:** the strongest, lowest-cost first step is a same-host L4 run
of CS vs mumax3 across the 16 K–1 M grid sweep. If the crossover shape matches
the Windows result (CS ahead at small grids, mumax3's FFT ahead at ~1 M), that is
strong Linux evidence for the Ada regime; the Blackwell-Linux run then only needs
to confirm the f32 magnitude. Until then, keep the competitive claim scoped to
the measured Windows/Blackwell platform and cite the portability result (§2)
separately — that is the honest, reviewer-proof framing.
