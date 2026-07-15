# Native-Linux cross-solver results — Claude-SD vs mumax3 (Ada / L4)

**Same-host** throughput comparison built to test whether the competitive thesis
holds on native Linux (see `linux_competitive_claim.md` for the scope argument).
This closes the "competitors never measured on Linux" gap for the **Ada** regime.

## Setup
- **Host:** AWS g6.xlarge — NVIDIA **L4** (Ada, sm_89, 23 GB), driver 580 / CUDA
  driver 13.0, Ubuntu 26.04.
- **Claude-SD:** `linux-gcc-cuda` (f64) and `linux-gcc-cuda-f32` (f32) builds,
  `RK4IntegratorGPU`, in-process timing with `download()` device sync.
- **mumax3:** official **v3.12** Linux binary (`cuda12.9`), `setsolver(4)` fixed
  `dt`, two-run subtraction to cancel process-startup + kernel-compile.
- **Matched physics:** Msat = 860 kA/m, Aex = 13 pJ/m, α = 0.02, demag + exchange,
  dt = 5e-14 s. Throughput is demag-FFT + exchange bound. 5 repeats, median.
- Harness: [`linux_crosssolver_bench.py`](linux_crosssolver_bench.py). Raw:
  `_crosssolver_linux_L4.json`, `_crosssolver_linux_L4_large.json`.

mumax3 and the other solvers are **float32-only**, so the fair competitive axis
is **CS f32 vs mumax3 f32**. (CS f64 is shown for context — a different accuracy
tier, not a like-for-like comparison.)

## Results — ms/step, ratio = CS_f32 / mumax3_f32 (< 1 ⇒ Claude-SD faster)

| grid | cells | dim | CS f64 | CS f32 | mumax3 f32 | **CS f32 / mumax3** |
|---|--:|:--:|--:|--:|--:|:--:|
| S2 |    10 K | 2D |   0.53 |  0.154 |  0.370 | **0.42× — CS 2.4× faster** |
| S1 |    65 K | 3D |   3.27 |  0.659 |  0.688 | **0.96× — on par** |
| S3 |   540 K | 3D |  64.4  | 13.56  |  8.33  | 1.63× |
| L1M |  1.05 M | 3D |   —    | 31.54  | 18.11  | 1.74× |
| S5 |   2.5 M | 3D | 299.1  | 80.27  | 69.46  | 1.16× |
| L4M | 4.19 M | 3D |   —    | 133.2  | 121.4  | **1.10× — near parity** |

## Realistic multi-field workload — DMI + PMA (skyrmion film)

Same sweep with `--physics dmi` (Pt/Co-like: demag + exchange + **interfacial
DMI + uniaxial PMA**, Ms = 580 kA/m, A = 15 pJ/m, K = 0.8 MJ/m³, D = 3 mJ/m²) —
the workload class where CS's per-field kernels and mumax3's fused pipeline
differ most, and the physics regime (skyrmions) where CS's f64/topology
capabilities matter. Raw: `_crosssolver_linux_L4_dmi.json`.

| grid | cells | CS f32 | mumax3 f32 | CS/mumax3 |
|---|--:|--:|--:|:--:|
| S2 |  10 K | 0.178 | 0.431 | **0.41× — CS 2.4× faster** |
| S1 |  65 K | 0.680 | 0.706 | **0.96× — on par** |
| S3 | 540 K | 13.97 | 8.32 | 1.68× |
| S5 | 2.5 M | 82.7 | 70.7 | 1.17× |

**The crossover curve is unchanged from the demag+exchange case** (0.42/0.96/
1.63/1.16 there) — the small-grid advantage is robust to adding DMI + anisotropy,
i.e. it is not an artifact of a minimal field set.

## Interpretation

Three regimes, all on the same Linux host:

1. **Small grids (≤ 65 K): Claude-SD f32 is faster or on par** — 2.4× faster at
   10 K, on par at 65 K. This is the launch-overhead-bound regime, the one that is
   *most* platform-sensitive (WDDM vs Linux) and therefore the one the earlier
   Windows↔Linux self-parity could **not** settle. It now has direct Linux data:
   **the small-grid competitive advantage holds on native Linux.**
2. **Mid grids (540 K – 1 M): mumax3 leads (~1.6–1.7×)** — its FFT pipeline's
   sweet spot, consistent with the Windows campaign's "mumax3 more efficient at
   large grids" finding.
3. **Very large grids (2.5 M – 4.2 M): Claude-SD narrows to within 10–16 %** —
   near parity again at 4.2 M (1.10×), as both become FFT-bandwidth bound.

The **qualitative crossover shape matches the Windows/Blackwell campaign**
(Claude-SD wins small, mumax3 wins mid/large), so the competitive conclusion is
platform-robust, not a Windows artefact.

## Scope / caveats (for the paper)

- **This is the Ada (L4) regime.** The Windows campaign's headline is
  **f32 on Blackwell** (RTX 5060 Ti), where Claude-SD additionally benefits from
  the Blackwell Tensor-Core FFT — an advantage **absent on L4**. So Claude-SD's
  mid/large position here is a *conservative lower bound*; on Blackwell-Linux it
  would improve (as it does on Blackwell-Windows). The **small-grid win does not
  depend on Blackwell** and is confirmed outright.
- To state the **headline f32/Blackwell** numbers on Linux with identical silicon,
  run this same harness on a Blackwell-Linux host (RTX 50-series workstation or
  AWS `p6-b200`, `CUDA_ARCH=100`). That is the one remaining measurement.
- mumax3 v3.12 (Linux) vs v3.11.1 (Windows campaign): **measured on the campaign
  GPU** (RTX 5060 Ti, Win11, same grids/methodology —
  `_mumax_version_delta_win_5060ti.json`): v3.12/v3.11.1 = 1.08× @ 10 K,
  1.00× @ 65 K, 1.01× @ 540 K. The ≤8 % small-grid delta (v3.12 slightly
  *slower*) is far below the 2.4× CS margin there — the version difference does
  not affect any conclusion.

## Bottom line

On native Linux (Ada), **Claude-SD f32 is faster than mumax3 at small grids
(2.4× at 10 K), on par at 65 K, and within ~10 % at the largest grids (4.2 M)**,
with mumax3 leading in the mid-range. The core competitive claim — *Claude-SD is
competitive with, and at small sizes faster than, mumax3* — is **confirmed on
native Linux**, not only on Windows.
