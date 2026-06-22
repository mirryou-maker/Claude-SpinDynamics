# Claude-SpinDynamics — User Guide

A GPU-accelerated micromagnetics simulator (C++20 / CUDA core + Python API) for
Landau–Lifshitz–Gilbert dynamics, spin torques, and µMAG standard problems.

> **Contents**
> 1. [What is Claude-SD & why use it](#1-what-is-claude-sd--why-use-it)
> 2. [Installation](#2-installation)
> 3. [Beginner guide](#3-beginner-guide)
> 4. [Advanced guide](#4-advanced-guide)
> 5. [Extending Claude-SD with Claude Code](#5-extending-claude-sd-with-claude-code)
> 6. [Troubleshooting](#6-troubleshooting)

---

## 1. What is Claude-SD & why use it

Claude-SD solves the LLG equation on a structured finite-difference grid:

```
dm/dt = -γ'µ₀ (m×H) - γ'αµ₀ m×(m×H),   γ' = γ₀/(1+α²)
```

with spin-torque terms (STT/SOT/Zhang–Li) added per integrator stage. It ships a
C++/CUDA engine and a thin `micromag` Python module.

### Strengths (when Claude-SD is the right tool)

| Feature | Detail |
|---|---|
| **Double *and* single precision** | Unique among GPU micromagnetic codes — `f64` for reference-grade accuracy, `f32` for 4–6× Tensor-Core speed on Blackwell. (mumax3 / mumax+ / MuMax-CO are f32-only.) |
| **Two FFT backends** | cuFFT (best on small grids, CUDA-Graph replay) **and** VkFFT (faster on large 3D). Pick per problem. |
| **Fast on small / 2D problems** | At SP#4 (10 K cells) `f32` is ~5× faster than mumax3 per field-eval — CUDA-Graph eliminates kernel-launch overhead. Crossover with mumax3/MuMax-CO is ~0.1–0.5 M cells. |
| **Native spin torques & DMI** | Slonczewski STT, Spin–Orbit Torque, Zhang–Li, bulk & interfacial DMI, RKKY, magnetoelastic, surface anisotropy — all GPU, all composable. (mumax3 has no native SOT.) |
| **Per-cell materials & geometry** | `MaterialField3D` (per-cell Ms/A/K/α/axis), `GeomMask`, `RegionMap` + inter-region exchange — on CPU **and** GPU. |
| **Auto integrator selection** | `recommend_integrator()` picks RK4 / RK45-DP / Heun from α, T, goal, and a phase-error estimate. |
| **µMAG-validated** | SP#1 (L_c), SP#2 (remanence), SP#3 (H_sw), SP#4 (switching), SP#5 (Zhang–Li). Cross-checked against mumax3 / mumax+ / OOMMF. |
| **Python + C++** | Script in Python (NumPy bridge, mx3 runner) or link the C++ library directly. |

### When to prefer another code

- **Huge 3D f32 production runs (≳1 M cells):** mumax3 / MuMax-CO are cuFFT-bound and edge ahead there.
- **Antiferromagnets / coupled elastodynamics in Python:** mumax+ is purpose-built for that.

Full numbers: `benchmarks/RESULTS_2026.md`.

---

## 2. Installation

### 2.1 Prerequisites

- **OS / compiler:** Windows 11 + MSVC (the presets target `Visual Studio 18 2026`; for VS 2022 edit
  the `"generator"` in `CMakePresets.json` to `Visual Studio 17 2022`). Linux + GCC/Clang also works
  with an equivalent CMake invocation.
- **CMake ≥ 3.24**, **vcpkg** (dependency manager), **Python 3.13** (NumPy, Matplotlib).
- **GPU build only:** NVIDIA CUDA Toolkit 13.x (cuFFT, cuRAND). Optional VkFFT at `C:/vkfft`.
- Dependencies pulled by vcpkg: FFTW3, pybind11, Catch2 v3.

### 2.2 CPU build (start here — no GPU needed)

```powershell
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release
ctest --preset windows-msvc            # 232 tests should pass
```

The Python module lands in `build/windows-msvc/python/` (CUDA = False).

### 2.3 GPU build (CUDA required)

```powershell
cmake --preset windows-msvc-cuda
cmake --build build/windows-msvc-cuda --config Release
.\build\windows-msvc-cuda\bin\Release\unit_tests_gpu.exe   # 113 GPU tests
```

There are four CUDA presets — pick the precision/backend you need:

| Preset | Precision | FFT |
|---|---|---|
| `windows-msvc-cuda` | f64 | cuFFT |
| `windows-msvc-cuda-f32` | f32 | cuFFT |
| `windows-msvc-cuda-vkfft` | f64 | VkFFT |
| `windows-msvc-cuda-vkfft-f32` | f32 | VkFFT |

### 2.4 Using the Python module

```python
import os, sys
# GPU module needs the CUDA runtime DLLs on the search path FIRST:
os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
sys.path.insert(0, r".../build/windows-msvc-cuda/python")
import micromag as mm
print(mm.cuda_available())   # True for a GPU build
```

> Tip: if your system Python's NumPy is broken, use a clean interpreter (e.g. Anaconda).

---

## 3. Beginner guide

### 3.1 Your first script — make a magnetization, save it

```python
import micromag as mm

grid  = mm.StructuredGrid(nx=64, ny=64, nz=1, dx=4e-9, dy=4e-9, dz=4e-9)
m     = mm.VectorField3D(grid)
m.set_vortex(cx=grid.nx*grid.dx*0.5, cy=grid.ny*grid.dy*0.5, core_radius=12e-9)
mm.write_vtk_legacy("vortex.vtk", m, "m")     # open in ParaView
```

### 3.2 Your first dynamics — relax a thin film (GPU)

```python
import micromag as mm

g   = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3e-9)   # SP#4 geometry
mat = mm.Material.permalloy()                                # Ms, A, α presets

# initial state: tilted, then normalised
m0 = mm.VectorField3D(g); m0.set_uniform(mm.Vec3(1, 0.1, 0)); m0.normalize()

# effective fields (each is a GPU drop-in)
demag  = mm.DemagFieldGPU(g)
exch   = mm.ExchangeFieldGPU(g)
zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(-24.6e3, 4.3e3, 0))    # SP#4 field A

# fixed-step RK4 integrator, all on GPU, zero PCIe per step
rk4 = mm.RK4IntegratorGPU(g, dt=5e-14)
rk4.upload(m0)
for _ in range(2000):
    rk4.step(mat, demag, exch, zeeman)
rk4.download(m0)

mx, my, mz = mm.mean_magnetization(m0)
print(f"<mx>={mx:.4f}  <my>={my:.4f}  <mz>={mz:.4f}")
```

### 3.3 Let the library pick the integrator

```python
# damping is read from mat.alpha automatically
rec = mm.recommend_integrator(mat, T_K=0, goal="dynamics", dt=5e-13)
print(rec["integrator"], "—", rec["reason"])
```

- `T_K > 0` → **HeunIntegratorGPU** (stochastic LLG, mandatory at finite T)
- low damping, T = 0 → **RK45IntegratorGPU** (adaptive, fewest steps)
- relaxation / high damping → **HeunIntegratorGPU** (2× cheaper)

### 3.4 Visualize

```python
import numpy as np
arr = mm.to_numpy(m0)          # shape (nz, ny, nx, 3)
# ...matplotlib quiver / imshow of arr[0, :, :, 2] (the mz component)...
```

µMAG SP#4 and friends are pre-built apps: `sp4_gpu.exe`, `sp4_rk45_gpu.exe`, `bloch_dw.exe`.

---

## 4. Advanced guide

### 4.1 Composing many fields — `FieldSumGPU`

```python
exch  = mm.ExchangeFieldGPU(g)
dmi   = mm.InterfacialDMIFieldGPU(g, 3e-3)        # D = 3 mJ/m²
ani   = mm.UniaxialAnisotropyFieldGPU(g)
fields = mm.FieldSumGPU(); fields.add(exch); fields.add(dmi); fields.add(ani)

rk45 = mm.RK45IntegratorGPU(g)
rk45.upload(m0)
rk45.step(mat, demag, fields)          # adaptive DOPRI5, one call = one accepted step
```

All three GPU integrators (`RK4IntegratorGPU`, `RK45IntegratorGPU`, `HeunIntegratorGPU`)
and `RelaxGPU` accept a `FieldSumGPU`.

### 4.2 Spin torques (STT / SOT / Zhang–Li)

```python
sot     = mm.SpinOrbitTorqueGPU(g, 3e12, 0.35, 3e-9, mm.Vec3(0,1,0))  # J, P, thickness, σ̂
torques = mm.SpinTorqueSumGPU(); torques.add(sot)
heun    = mm.HeunIntegratorGPU(g, dt=5e-14, seed=0)
heun.step(mat, demag, fields, T_K=300.0, torques=torques)   # finite-T SLLG + SOT
```

### 4.3 Per-cell materials, geometry, regions

```python
matf = mm.MaterialField3D(g, mat)          # start uniform, then edit per cell
exch.set_material_field(matf)               # harmonic-mean A at grain boundaries
ani.set_material_field(matf)

mask_disk = mm.circle(g, ...)               # disk geometry (see mm.ellipse/rect/sphere/layer)
exch.set_mask(mask_disk)                     # 1 = magnet, 0 = vacuum; Neumann BC at the edge

rm = mm.RegionMap(g, 0); # ...assign region IDs...
exch.set_region_map(rm)
exch.set_inter_exchange(0, 1, A_IEC=5e-12)   # explicit coupling across a region pair
```

`voronoi_grains()`, `poisson_disk_grains()`, and the shape factories (`ellipse`, `rect`,
`sphere`, `layer`, …) build geometries and grain structures quickly.

### 4.4 Quasistatic: relaxation, hysteresis, energy minimization

```python
relax = mm.RelaxGPU(g)
opts  = mm.RelaxGPUOptions(); opts.threshold = 1e-4*mat.Ms; opts.max_steps = 40000
relax.upload(m0); relax.run(mat, demag, fields, opts); relax.download(m0)

# field sweep (e.g. SP#3 hysteresis or SP#2 remanence) — see benchmarks/sp2/
res = mm.gpu_hysteresis_loop(rk4, mat, demag, fields, zeeman, H_list, m_cpu)
```

> **Convergence tip:** for *non-uniform* equilibria use `RelaxGPU` (torque-based) — a
> neighbour-angle criterion never converges for flux-closure/vortex states.

### 4.5 Precision & FFT backend — choosing a build

- **Small / 2D, or you need a reference:** `cuda` (f64 cuFFT) or `cuda-f32` (fast, CUDA-Graph).
- **Large 3D production:** `cuda-vkfft` / `cuda-vkfft-f32` (VkFFT wins on big FFTs).
- `f32` gives 4–6× over `f64` on Blackwell but plateaus at ~1e-6 relative error — validate
  against an `f64` run for topology-sensitive observables (skyrmion charge *Q* is precision-sensitive).

### 4.6 Running mumax3 `.mx3` scripts

```python
import micromag.mx3 as mx3
mx3.run("myscript.mx3")        # CPU or GPU; a large mumax3 subset is supported
```

### 4.7 Benchmarking your own setup

The `benchmarks/` suite is reproducible: `run_throughput_cs.py`, `run_throughput_mumax.py`,
`run_throughput_mumaxplus.py` → `make_report.py`. Timing is hardened (size-tiered step counts,
5-repeat median + IQR). See `benchmarks/BENCHMARK_PLAN.md` for the methodology.

---

## 5. Extending Claude-SD with Claude Code

Claude-SD was itself **built and optimized with [Claude Code](https://claude.com/claude-code)**, and the
repository is structured so you can keep extending it the same way. (The sibling project *MuMax-CO* is a
published case study of optimizing mumax3 with Claude Code.)

### 5.1 Why the repo is "agent-friendly"

- **`CLAUDE.md`** at the root is the agent's playbook: build commands, the layer model
  (`types → grid → field → material → effective_field → integrator`), every field/integrator class, SI
  conventions, and the test-tag map. An agent reads this first and stays consistent with the codebase.
- **Tight test harness:** 232 CPU + 113 GPU Catch2 tests, tagged (`[demag]`, `[llg]`, `[gpu]`, …). New
  code is expected to come with a test; the agent can run `ctest` / `unit_tests_gpu.exe "[tag]"` to verify.
- **Mirrored CPU/GPU design:** every GPU field implements the same `IEffectiveField` interface as its CPU
  twin, so adding a feature has a clear template to copy.

### 5.2 A good extension workflow

1. **Describe the goal precisely** — e.g. *"Add a cubic-plus-uniaxial combined anisotropy GPU field, with a
   CPU reference and a `[anisotropy][gpu]` test comparing the two."*
2. Let Claude Code **read `CLAUDE.md` + the nearest existing field** (e.g. `field_kernels_gpu.cu`) and
   implement the new kernel, the host class, the pybind11 binding, and the test.
3. **Build & test:** `cmake --build build/windows-msvc-cuda --config Release` then run the new test tag.
4. **Validate physics** against a known limit or the CPU path (the codebase's habit: GPU-vs-CPU agreement
   to ~1e-6, and µMAG/analytic checks).

### 5.3 High-value things users commonly add this way

- **A new effective-field term** (e.g. a custom anisotropy, biquadratic exchange, dipolar tweak): copy an
  existing `*_gpu.cu` kernel + `IEffectiveField` class + binding + a GPU-vs-CPU test.
- **A new standard problem or validation app** (SP#2 was added exactly this way: `benchmarks/sp2/`).
- **A new integrator or a kernel optimization** (the fused local-field kernel and CUDA-Graph capture were
  added/iterated with the agent; *MuMax-CO* shows a 2–5× CUDA-Graph speedup achieved this way).
- **A Python convenience/analysis helper** in `python/micromag/__init__.py` (e.g. a new init state,
  topological diagnostic, or plotting routine) — these need no rebuild.
- **More benchmarks**: drop a scenario into the `benchmarks/` harness; it writes to the unified
  `all_solvers.json` and `make_report.py` renders the tables/figures.

### 5.4 Guardrails to ask the agent to honor

- Keep **SI conventions** (`H` in A/m, `H_demag = -N·M`, `γ₀ = 1.76e11`) — they're documented in `CLAUDE.md`.
- Add code behind the existing **interfaces** (`IEffectiveField`, `ISpinTorque`, `IDemagGPU`) so it composes.
- **Always add a test** and run the relevant tag; for GPU code, verify against the CPU path and confirm all
  four CUDA builds still pass.
- Respond in small steps for large changes (don't regenerate whole files unnecessarily).

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `import micromag` fails on GPU build | Call `os.add_dll_directory(...CUDA/bin/x64)` **before** importing. |
| CMake configure fails: generator not found | Edit `"generator"` in `CMakePresets.json` to your VS version (e.g. `Visual Studio 17 2022`). |
| `mm.cuda_available()` is `False` | You imported the CPU build's module; point `sys.path` at a `windows-msvc-cuda*` build. |
| Skyrmion charge `Q` differs run-to-run / by build | Near the metastability boundary `Q` is precision/FP-sensitive — use `f64` (cuFFT) as the reference; see `benchmarks/RESULTS_2026.md` §3a. |
| Hysteresis/relaxation never converges | Use `RelaxGPU` (torque threshold), not a neighbour-angle criterion, for non-uniform states. |
| mumax3 thermal run panics (`CURAND_LENGTH_NOT_MULTIPLE`) | mumax3 needs an even cell count for `Temp>0`; Claude-SD handles a single cell directly. |

---

*License: GPLv3. Issues & contributions welcome. Full benchmark methodology and results in
`benchmarks/RESULTS_2026.md`; build/architecture reference in `CLAUDE.md`.*
