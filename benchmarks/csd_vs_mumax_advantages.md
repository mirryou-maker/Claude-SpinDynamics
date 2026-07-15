# Claude-SD vs mumax3 — comprehensive advantages (for the paper)

The competitive story is **not** "we are uniformly faster" (we are not — mumax3's
FFT pipeline leads in the mid-range). The defensible thesis is a **capability
superset at competitive performance, validated to reference accuracy, delivered
as an open, clean-room, AI-assisted implementation.** Below, claims are split
into *measured facts*, *capability advantages*, and — for integrity — *where
mumax3 leads*.

## A. Measured performance (facts, with data)

- **Faster at small grids.** CS f32 vs mumax3 f32, same host: 10 K cells → 0.42×
  (2.4× faster); 65 K → on par. Confirmed on **both** Windows/Blackwell (campaign)
  and native **Linux/Ada** (`linux_crosssolver_results.md`). This is the
  launch-overhead-bound regime, aided by CS's CUDA-Graph replay.
- **Competitive at the extremes, mumax3-led in the middle.** Near parity at the
  largest grids (4.2 M → 1.10× on L4); mumax3 leads 540 K–1 M (~1.6×).
- **Adaptive RK45 (DOPRI5/FSAL) GPU integrator**: ~3.7× fewer steps than fixed
  RK4 on SP#4, with per-step cost parity.
- **Platform-independent**: Linux CUDA build == Windows within ~1 % (2.5 M cells);
  CPU build on Linux is on par with / faster than Windows.

## B. Capability advantages (things mumax3 cannot do, or does not expose)

1. **Double precision.** mumax3, mumax⁺, and MuMax-CO are **float32-only**.
   Claude-SD offers a true **f64** path *and* an f32 path. f64 matters for
   topology-sensitive quantities (topological charge *Q*), energy-landscape
   accuracy, long-time integration, and bit-reproducible validation — a class of
   problems mumax3 fundamentally cannot address at full precision. **This is the
   single strongest differentiator.**
2. **CPU *and* GPU from one API.** The identical `IEffectiveField` / integrator
   interface runs on CPU (no GPU required) or GPU. mumax3 is GPU-only (mandatory
   NVIDIA card). CS is usable for development, teaching, CI, and small problems
   with zero GPU, and its GPU drop-ins are literal swaps.
3. **Python-native, composable.** CS is a `pybind11` library with a NumPy bridge:
   parameter sweeps (`parameter_sweep`, `multi_gpu_sweep`), programmatic geometry
   and materials, and integration into the scientific-Python ecosystem. mumax3 is
   a domain-specific `.mx3` script language + Go binary — powerful but a closed
   scripting surface. (CS can even *run* mumax3 `.mx3` scripts via its `mx3`
   runner, easing migration.)
4. **Per-cell material control.** `MaterialField3D` gives per-cell
   Ms/A/K/α/easy-axis; Voronoi/Poisson-disk grains; harmonic-mean exchange at
   grain boundaries; per-cell DMI, cubic, and surface anisotropy. Fine-grained,
   spatially-varying materials are first-class.
5. **Choice of numerics.** Fixed RK4, adaptive RK45 (DOPRI5/FSAL), and Stratonovich
   **Heun** for finite-T SLLG, plus a `recommend_integrator()` helper. Two demag
   backends (open + periodic BC) and **two FFT libraries** (cuFFT + VkFFT — VkFFT
   wins on non-power-of-two sizes).
6. **Broad, modern physics.** Bulk + interfacial DMI, RKKY interlayer exchange,
   cubic and 2nd-order-uniaxial (Ku2) anisotropy, magnetoelastic (B1/B2), surface
   anisotropy, Slonczewski + Zhang-Li STT, spin-Hall SOT, thermal SLLG, MFM
   imaging, OVF (mumax3-compatible) + VTK I/O.
7. **Deployment portability.** Runs on Windows and Linux (MSVC and GCC); ships
   self-contained binary packages with bundled CUDA runtime (no toolkit install),
   with a CPU fallback. mumax3 always needs an NVIDIA GPU + matching runtime.

## C. Scientific / methodological advantages

- **Validated to reference accuracy.** µMAG SP#1/#3/#4/#5, Bloch DW width, FMR
  (Kittel 0.06 %), Walker breakdown, skyrmion-Hall — all reproduced. Dynamic SP#4
  agrees with the **mumax3 ↔ OOMMF spread** (the two references disagree by more
  than CS deviates from either). Static energies match mumax3 to < 3e-4.
- **Cross-validation *found and fixed* real bugs.** Building an independent solver
  and cross-checking exposed a concurrency race (DMI/field-sum stream) that a
  single-codebase workflow would not surface — a methodological argument for
  reimplementation, not just a reimplementation.
- **Determinism / reproducibility.** The f64 path plus the race fix give
  bit-stable topological-charge results; f32-only tools cannot offer this
  reproducibility tier.
- **Open, auditable, clean-room.** A from-scratch C++20 + Python implementation
  whose internals are transparent and hackable, demonstrating that an
  AI-assisted solver can *match an established community tool* — itself a
  publishable result about the development methodology.

## D. Where mumax3 leads (state this honestly)

- **Mid/large-grid throughput**: mumax3's FFT pipeline is ~1.6× faster at
  540 K–1 M cells; CS closes to ~1.1× only at the extremes.
- **Maturity & ecosystem**: mumax3 is a decade-old, heavily-cited, community-
  hardened tool with a large user base and body of published results. CS is new
  and less battle-tested at scale.
- **Turn-key single-file workflow**: for a classic `.mx3` job on one GPU, mumax3
  is a mature, frictionless path.

## E. One-paragraph claim (paper-ready)

> Claude-SD matches established micromagnetic solvers to reference accuracy
> (µMAG SP#1/3/4/5, within the mumax3↔OOMMF spread) while providing a strict
> capability superset: a true double-precision path absent from mumax3 and its
> derivatives (all float32-only), a unified CPU/GPU Python-native API,
> first-class per-cell materials, a choice of fixed/adaptive/stochastic
> integrators and FFT backends, and a broad modern physics set (DMI, RKKY,
> magnetoelastic, SOT, SLLG). On throughput it is faster than mumax3 at small
> problem sizes (2.4× at 10⁴ cells, confirmed on both Windows/Blackwell and
> native Linux/Ada) and competitive at the largest (within ~10 % at 4×10⁶
> cells), with mumax3's mature FFT pipeline leading in the mid-range. The
> comparison holds on native Linux, not only Windows.
