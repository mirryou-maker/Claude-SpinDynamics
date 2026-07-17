# Binary manifest — CPU vs GPU dependence

Ship this file at the root of every release package. QA rule: **do not infer
CPU/GPU dependence from the folder a binary sits in** — the GPU package's
per-variant `bin/` folders contain both kinds (see
claude-sd-gpu-arch-issue_JYCho.md for how that misled testing).

## CUDA-independent (run on any x64 machine, no GPU, no CUDA DLLs)

| binary | purpose |
|---|---|
| `hello_micromag.exe` | vortex init + VTK write demo |
| `field_demo.exe` | field API demo |
| `llg_demo.exe` | CPU LLG demo |
| `stt_sot_demo.exe` | CPU STT/SOT macrospin demo |
| `sp1.exe`, `sp1_phase.exe`, `sp1_thickness.exe` | µMAG SP#1 |
| `sp3.exe` | µMAG SP#3 hysteresis |
| `sp4.exe`, `sp4_rk45.exe`, `sp4_fieldB.exe` | µMAG SP#4 (CPU) |
| `sp4_thermal.exe`, `thermal_equilibrium.exe`, `thermal_neel_brown.exe` | CPU SLLG |
| `bloch_dw.exe` | Bloch DW width validation |
| `unit_tests.exe` | CPU test suite (232 cases) |

Running any of these does **not** exercise GPU code, even from a GPU-variant
folder.

## CUDA-dependent (need an NVIDIA GPU + the bundled `runtime-dll/`)

| binary | purpose | on kernel-arch mismatch |
|---|---|---|
| `sp4_gpu.exe` | SP#4 CPU-vs-GPU benchmark | `FATAL: CUDA ...` on stderr, exit 1 |
| `sp4_gpu_1ns.exe` | SP#4 1 ns GPU validation | same |
| `sp4_rk45_gpu.exe` | GPU RK45 adaptive benchmark | same |
| `sp4_full_gpu_bench.exe` | full-GPU LLG benchmark | same |
| `benchmark_large.exe` | large-grid GPU demag benchmark | same |
| `llg_large_bench.exe` | 2.5 M-cell scaling benchmark | same |
| `demag_profile.exe` | GPU demag profiler | same |
| `unit_tests_gpu.exe` | GPU test suite (116 cases) | Catch2 reports the exception |

Since v1.0.1 these print a `FATAL:` message and exit non-zero instead of the
old silent `0xC0000409` abort.

## Python module (`<variant>/python/_micromag*.pyd`)

- `micromag.cuda_available()` — True only when CUDA is compiled in **and** the
  installed GPU can execute this build's kernels (a probe kernel is launched
  once). On a mismatched GPU it returns **False** instead of crashing later.
- `micromag.gpu_diagnostic()` — prints device, compute capability, the kernel
  architectures embedded in this build, and the reason if unusable. **Ask bug
  reporters to include this string.**

## Supported GPU architectures (release builds)

Release packages are built with `-DMICROMAG_CUDA_ARCHS=release`:
real kernels for **sm_75 / 80 / 86 / 89 / 90 / 100 / 120**
(Turing, Ampere, Ada, Hopper, Blackwell) **+ compute_120 PTX** for
forward-compatibility with newer GPUs via driver JIT.
Verify before shipping: `python scripts/check_fatbin_archs.py <binary>`.
