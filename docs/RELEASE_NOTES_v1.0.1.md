# Claude-SpinDynamics v1.0.1 — Release Notes

Bug-fix release. **All v1.0.0 GPU-package users should upgrade** — v1.0.0's GPU
binaries ran only on Blackwell (sm_120) GPUs.

## Fixed: GPU package crashed on non-Blackwell GPUs

v1.0.0's GPU kernels were compiled for a single architecture (`sm_120`, the
development GPU) with no PTX fallback. On any other GPU — RTX 20/30/40, A100,
H100, L4/L40S — the first GPU call failed with *"no kernel image is available
for execution on the device"*, and the C++ benchmark apps died silently
(`0xC0000409`). Thanks to **JYCho** for the detailed report (binary-level
diagnosis included; archived as `claude-sd-gpu-arch-issue_JYCho.md`).

### What changed

- **Multi-architecture kernels** — v1.0.1 embeds real code for
  **sm_75 / 80 / 86 / 89 / 90 / 100 / 120** (Turing → Blackwell) plus
  `compute_120` PTX so future GPUs run via driver JIT. Build flag:
  `-DMICROMAG_CUDA_ARCHS=release` (toolkit-aware; CUDA ≥ 12.8 required for the
  Blackwell targets). Verified on RTX 5060 Ti (sm_120) and NVIDIA L4 (sm_89).
- **`cuda_available()` is now truthful** — it launches a cached probe kernel
  and returns `False` on a kernel/GPU mismatch instead of `True`-then-crash.
- **New `micromag.gpu_diagnostic()`** — one string with the device, compute
  capability, embedded kernel architectures, and the reason if the GPU is
  unusable. Please include it in bug reports.
- **No more silent crashes** — every GPU console app now prints
  `FATAL: <reason>` to stderr and exits non-zero instead of aborting.
- **`MANIFEST.md`** in the package root maps every bundled exe to CPU-only or
  GPU-dependent (folder location is not a reliable signal).
- **Release gate** — `scripts/check_fatbin_archs.py` (cuobjdump) verifies every
  shipped binary embeds the full architecture set; wired into the release
  checklist so a single-arch package cannot ship again.

## Also in this release

- Bundled examples now locate the module via `micromag_locate.py` (package
  root) — one shared resolver instead of 50+ inlined copies.
- Warning-clean tree enforced (`/WX`, `-Werror`); a dead-code NaN-divide in a
  demo app was found and removed in the process.
- Python binding test suite (20 pytest cases) runs in CI on Linux + Windows.
- FFTW plan handles are RAII; constructors validate arguments
  (`std::invalid_argument` on bad grid dims / `dt <= 0`).
- `MICROMAG_SYNC_DEBUG=1` env var: device-sync after every kernel launch for
  stream-race debugging.

## Supported GPUs (GPU package)

NVIDIA compute capability **≥ 7.5**: GTX 16 series, RTX 20/30/40/50, A100,
H100, L4/L40S, B200. Newer architectures run via embedded PTX.

## How to cite

See `CITATION.cff`.
