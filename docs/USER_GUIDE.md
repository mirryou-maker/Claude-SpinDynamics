# Claude-SpinDynamics — User Guide & Reference Manual

A GPU-accelerated micromagnetics simulator (C++20 / CUDA core + Python API) for
Landau–Lifshitz–Gilbert dynamics, spin torques, DMI, and the µMAG standard problems.

**Version 1.0**  |  Toolchain: MSVC (VS 2022/2026), CUDA 13.x, Python 3.13  |  License: GPLv3

> **Contents**
>
> **Part A — Getting started**
> 1. [What is Claude-SD & why use it](#1-what-is-claude-sd--why-use-it)
> 2. [Installation](#2-installation)
> 3. [Beginner guide](#3-beginner-guide)
> 4. [Advanced guide](#4-advanced-guide)
> 5. [Using & extending Claude-SD with Claude Code](#5-using--extending-claude-sd-with-claude-code)
>
> **Part B — Reference**
> 6. [Physical model](#6-physical-model)
> 7. [Numerical methods](#7-numerical-methods)
> 8. [Python API reference](#8-python-api-reference)
> 9. [µMAG validation](#9-µmag-validation)
> 10. [Performance benchmarks](#10-performance-benchmarks)
> 11. [Example gallery](#11-example-gallery)
> 12. [Troubleshooting](#12-troubleshooting)
>
> **Appendix**
> - [A. mumax3 `.mx3` API coverage](#appendix-a--mumax3-mx3-api-coverage)

---
---

# Part A — Getting started

## 1. What is Claude-SD & why use it

Claude-SD solves the LLG equation on a structured finite-difference grid:

```
dm/dt = -γ'µ₀ (m×H) - γ'αµ₀ m×(m×H),   γ' = γ₀/(1+α²)
```

with spin-torque terms (STT / SOT / Zhang–Li) added per integrator stage. It ships a
C++/CUDA engine and a thin `micromag` Python module. Full theory is in [Part B](#6-physical-model).

### Strengths (when Claude-SD is the right tool)

| Feature | Detail |
|---|---|
| **Double *and* single precision** | Unique among GPU micromagnetic codes — `f64` for reference-grade accuracy, `f32` for 4–6× Tensor-Core speed on Blackwell. (mumax3 / mumax+ / MuMax-CO are f32-only.) |
| **Two FFT backends** | cuFFT (best on small grids, CUDA-Graph replay) **and** VkFFT (faster on large 3D). Pick per problem. |
| **Fast on small / 2D problems** | At SP#4 (10 K cells) `f32` is ~5× faster than mumax3 per field-eval — CUDA-Graph eliminates kernel-launch overhead. Crossover with mumax3/MuMax-CO is ~0.1–0.5 M cells. |
| **Native spin torques & DMI** | Slonczewski STT, Spin–Orbit Torque, Zhang–Li, bulk & interfacial DMI, RKKY, magnetoelastic, surface anisotropy — all GPU, all composable. (mumax3 has no native SOT.) |
| **Per-cell materials & geometry** | `MaterialField3D` (per-cell Ms/A/K/α/axis), `GeomMask`, `RegionMap` + inter-region exchange — on CPU **and** GPU. |
| **Auto integrator selection** | `recommend_integrator()` picks RK4 / RK45-DP / Heun from α, T, goal, and a phase-error estimate. |
| **µMAG-validated** | SP#1 (L_c), SP#2 (remanence), SP#3 (H_sw), SP#4 (switching), SP#5 (Zhang–Li). Cross-checked against mumax3 / mumax+ / MuMax-CO / OOMMF. |
| **Python + C++** | Script in Python (NumPy bridge, mx3 runner) or link the C++ library directly. |

### When to prefer another code

- **Huge 3D f32 production runs (≳1 M cells):** mumax3 / MuMax-CO are cuFFT-bound and edge ahead there.
- **Antiferromagnets / coupled elastodynamics in Python:** mumax+ is purpose-built for that.

Full numbers: `benchmarks/RESULTS_2026.md`. Feature comparison:

| Feature | Claude-SD | OOMMF | mumax3 | mumax+ |
|---|---|---|---|---|
| Language | C++20 / Python | C++ / Tcl | Go / CUDA | C++ / Python |
| GPU | Yes (CUDA) | No | Yes | Yes |
| Double precision (GPU) | **Yes** | — (CPU f64) | No | No |
| Two FFT backends (cuFFT+VkFFT) | **Yes** | No | No | No |
| Adaptive integrator (GPU) | Yes (DOPRI5) | No | Yes | Yes |
| Native SOT | **Yes** | No | No | Partial |
| Thermal (SLLG) on GPU | Yes | No | Yes | Yes |
| Python API | Yes | Partial | No (scripted) | Yes |

---

## 2. Installation

### 2.1 Prerequisites

- **OS / compiler:** Windows 11 + MSVC (presets target `Visual Studio 18 2026`; for VS 2022 edit the
  `"generator"` in `CMakePresets.json` to `Visual Studio 17 2022`). Linux + GCC/Clang also works with an
  equivalent CMake invocation.
- **CMake ≥ 3.24**, **vcpkg** at `C:/vcpkg` (triplet `x64-windows`), **Python 3.13** (NumPy, Matplotlib).
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

### 2.5 Choosing a build (CPU / GPU · precision · FFT backend)

Claude-SD ships one CPU build and four GPU variants. The build is fixed at **configure time** by a CMake
preset — precision and FFT backend are *not* run-time switches. Three independent axes decide it:

1. **Compute target** — CPU (FFTW) vs GPU (CUDA).
2. **Precision** — `float64` (default, reference-grade) vs `float32` (`MICROMAG_FLOAT32`, faster).
3. **FFT backend (GPU only)** — cuFFT (default) vs VkFFT (`MICROMAG_VKFFT`).

| Preset | Target | Precision | FFT | Needs |
|---|---|---|---|---|
| `windows-msvc` | **CPU** | f64 | FFTW | MSVC + vcpkg (no GPU) |
| `windows-msvc-debug` | CPU (Debug) | f64 | FFTW | same; for debugging |
| `windows-msvc-cuda` | **GPU** | f64 | cuFFT | CUDA 13.x + NVIDIA GPU |
| `windows-msvc-cuda-f32` | GPU | **f32** | cuFFT | — |
| `windows-msvc-cuda-vkfft` | GPU | f64 | **VkFFT** | VkFFT headers at `C:/vkfft` |
| `windows-msvc-cuda-vkfft-f32` | GPU | **f32** | **VkFFT** | VkFFT at `C:/vkfft` |

Build any of them with `cmake --preset <name>` then `cmake --build build/<name> --config Release`.
Executables land in `build/<name>/bin/Release/`, the Python module in `build/<name>/python/`. The four
CUDA builds use separate directories, so you can keep them side by side.

**Which one?**

| Situation | Preset | Why |
|---|---|---|
| No NVIDIA GPU / development / max portability | `windows-msvc` (CPU) | only FFTW + MSVC runtime; runs anywhere |
| GPU, need **reference accuracy** or topology-sensitive results (skyrmion charge *Q*, DMI near a phase boundary) | `windows-msvc-cuda` (f64 cuFFT) | double precision; the validation reference |
| GPU, **small / 2-D** problems or the fastest small runs | `windows-msvc-cuda-f32` (f32 cuFFT) | cuFFT + CUDA-Graph wins on small grids (~5× mumax3 at SP#4) |
| GPU, **large 3-D production** (≳0.1–0.5 M cells) | `windows-msvc-cuda-vkfft-f32` | VkFFT wins on big FFTs; f32 = 4–6× f64 on Blackwell |
| Large 3-D but need f64 accuracy | `windows-msvc-cuda-vkfft` | VkFFT for size, double precision |
| **Non-power-of-two** cell counts | either `*-vkfft*` | VkFFT handles non-pow2 better than cuFFT |
| Debugging a crash / asserts | `windows-msvc-debug` | unoptimized + checks |

Rule of thumb: **CPU** for portability/development/reference, **GPU** for production; **f64** for accuracy
(topology, near metastability), **f32** for speed at scale; **cuFFT** for small/power-of-two, **VkFFT** for
large-3-D / non-power-of-two.

#### Selecting the build from Python

Each build compiles its own `micromag` module into its own `build/<preset>/python/` directory. You "select"
a build by pointing `sys.path` at that directory (and, for GPU builds, adding the CUDA DLLs first).
Precision and FFT backend are baked in at compile time — there is no run-time flag.

```python
import os, sys

BUILD = "windows-msvc-cuda-f32"          # the build you compiled
ROOT  = r"D:/Claude-Code-R/Claude-SpinDynamics"

if "cuda" in BUILD:                       # GPU builds need the CUDA runtime DLLs FIRST
    os.add_dll_directory(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64")
sys.path.insert(0, rf"{ROOT}/build/{BUILD}/python")

import micromag as mm
print(mm.cuda_available())                # True for any *-cuda* build, False for the CPU build
```

- To switch builds, point `sys.path` at a different `build/<preset>/python/` in a **fresh interpreter**
  (a module already imported stays loaded for that process).
- `mm.cuda_available()` confirms CPU vs GPU; the precision/FFT backend are whatever you compiled — they
  follow the build directory you selected (there is no `mm`-level switch).

**Performance & accuracy** (measured; full numbers in `benchmarks/RESULTS_2026.md`):

- Crossover with mumax3 / MuMax-CO is ~0.1–0.5 M cells — below it CS `f32` (CUDA-Graph) is fastest; above it
  the mumax family edges ahead, with CS within ~1.1×.
- `f32` is 4–6× faster than `f64` at large 3-D sizes but its relative error plateaus near 1e-6 — validate
  topology-sensitive observables (skyrmion charge *Q*) against an `f64` run.
- All builds reproduce the µMAG standard problems; they differ in speed and last-bit precision, not physics.

**Caveats:** GPU builds need an NVIDIA GPU + CUDA 13.x and `os.add_dll_directory(...)` before import; VkFFT
builds expect headers at `C:/vkfft` (`MICROMAG_VKFFT_PATH`); presets target VS 2026 (edit the generator for
VS 2022); `windows-msvc-debug` is CPU-only and not a performance build.

---

## 3. Beginner guide

### 3.1 Your first script — make a magnetization, save it

```python
import micromag as mm

grid = mm.StructuredGrid(nx=64, ny=64, nz=1, dx=4e-9, dy=4e-9, dz=4e-9)
m    = mm.VectorField3D(grid)
m.set_vortex(cx=grid.nx*grid.dx*0.5, cy=grid.ny*grid.dy*0.5, core_radius=12e-9)
mm.write_vtk_legacy("vortex.vtk", m, "m")     # open in ParaView
```

### 3.2 Your first dynamics — relax a thin film (GPU)

```python
import micromag as mm

g   = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3e-9)   # SP#4 geometry
mat = mm.Material.permalloy()                                # Ms, A, α presets

m0 = mm.VectorField3D(g); m0.set_uniform(mm.Vec3(1, 0.1, 0)); m0.normalize()

demag  = mm.DemagFieldGPU(g)
exch   = mm.ExchangeFieldGPU(g)
zeeman = mm.ZeemanFieldGPU(g, mm.Vec3(-24.6e3, 4.3e3, 0))    # SP#4 field A

rk4 = mm.RK4IntegratorGPU(g, dt=5e-14)
rk4.upload(m0)
for _ in range(2000):
    rk4.step(mat, demag, exch, zeeman)        # all GPU, zero PCIe per step
rk4.download(m0)

mx, my, mz = mm.mean_magnetization(m0)
print(f"<mx>={mx:.4f}  <my>={my:.4f}  <mz>={mz:.4f}")
```

### 3.3 Let the library pick the integrator

```python
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
# matplotlib quiver / imshow of arr[0, :, :, 2] (the mz component)
```

**ParaView export.** `mm.save_paraview` writes a `.vtk` that ParaView opens directly, carrying the vector
`m` plus scalars `mx`/`my`/`mz`/`m_norm`/`q_topo` — so you can *Glyph* the spins, colour by `mz`, or
*Contour* `q_topo` to isolate a skyrmion. For a time series use `mm.save_paraview_series` (writes `.vtk`
frames + a `.pvd` collection ParaView animates):

```python
mm.save_paraview(m0, "state.vtk")                       # single state
mm.save_paraview_series(frames, "run", dt=5e-12)        # -> run.pvd (+ run_NNNN.vtk)
```

A runnable gallery is [`examples/paraview_gallery.py`](../examples/paraview_gallery.py) — Néel/Bloch
skyrmions, a Bloch domain wall, a two-domain wall, an SP#1 vortex, and a 3-D skyrmion tube. For each state it
writes the `.vtk` file **and** renders previews into `paraview_demo/`:

- **2-D previews** (`2d_<state>.png`) — `mz` colour + in-plane arrows; the domain-wall / two-domain views are
  cropped around the transition.
- **3-D renders** (`3d_<state>.png`), if [`pyvista`](https://pyvista.org) is installed (`pip install
  pyvista`) — glyph arrows coloured by `mz` with the `mz`=0 isosurface.
- **Thickness views for the tube** — `mz` on all 8 z-layers, the 8 slices stacked with the z-axis
  exaggerated, and the core `mz`=0 isosurface drawn as a column through the layers.

`pyvista` is an optional dependency: without it the `.vtk` export and 2-D previews still work (open the `.vtk`
in ParaView itself for full 3-D interaction — Glyph the `m` vector, colour by `mz`, Contour `q_topo`).

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

rm = mm.RegionMap(g, 0)  # ...assign region IDs...
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

> **Convergence tip:** for *non-uniform* equilibria use `RelaxGPU` (torque-based) — a neighbour-angle
> criterion never converges for flux-closure/vortex states. For stiff DMI skyrmion textures, prefer
> `MinimizeGPU` (BB/FIRE) or raise `max_steps`; the fixed-step damped-LLG relaxer converges slowly there.

### 4.5 Precision & FFT backend

See [§2.5 Choosing a build](#25-choosing-a-build-cpu--gpu--precision--fft-backend) for the full preset
matrix, selection guidance, and how to point Python at a given build. In short: `f32` is 4–6× faster than
`f64` on Blackwell but plateaus at ~1e-6 relative error — validate topology-sensitive observables (skyrmion
charge *Q*) against an `f64` run; pick **cuFFT** for small / power-of-two grids and **VkFFT** for large 3-D
or non-power-of-two sizes.

### 4.6 Running mumax3-syntax `.mx3` scripts (on the Claude-SD engine)

`micromag.mx3` is an **interpreter**: it parses a `.mx3` file written in the mumax3 scripting *language* and
executes it **entirely on the Claude-SD (micromag) engine** — GPU if available, else CPU. **It does not call
the mumax3 program** and mumax3 does not need to be installed; only the input syntax is mumax3-compatible, the
computation is 100 % Claude-SD. (Only a practical subset of the language is supported; unsupported statements
warn and are skipped.) This is distinct from the benchmark scripts under `benchmarks/` and notebooks 41–50,
which *do* invoke the real mumax3 executable for cross-validation.

Ready-made scripts live in [`examples/mx3/`](../examples/mx3/) (`sp4`, `disk`, `regions`, `relax_demo`,
`loop_test`, `bext_ac`); see `examples/mx3/README.md` for the supported commands.

```python
import micromag as mm
eng = mm.run_mx3("examples/mx3/sp4.mx3", outdir="out")   # CPU or GPU (auto-selected)
print(mm.mean_magnetization(eng.m))                       # final <m>
```

Or from the command line: `python -m micromag.mx3 examples/mx3/sp4.mx3 [out_dir]`. Supported subset:
grid/cell, `Msat`/`Aex`/`alpha`/`Ku1`/`Dind`, `m` init, regions, time-dependent `B_ext`,
`relax`/`minimize`/`run`/`steps`, `save`/`tableSave`, and Go-style `for`/`if`. A ready-to-run script is
[`examples/run_mx3_example.py`](../examples/run_mx3_example.py) (auto-selects GPU/CPU). For the complete
list of supported and **unsupported** mumax3 API, see
[Appendix A — mumax3 `.mx3` API coverage](#appendix-a--mumax3-mx3-api-coverage).

### 4.7 Benchmarking your own setup

The `benchmarks/` suite is reproducible: `run_throughput_cs.py`, `run_throughput_mumax.py`,
`run_throughput_mumaxplus.py` → `make_report.py`. Timing is hardened (size-tiered step counts,
5-repeat median + IQR). See `benchmarks/RESULTS_2026.md` for the methodology and full results.

---

## 5. Using & extending Claude-SD with Claude Code

Claude-SD was **built and optimized with [Claude Code](https://claude.com/claude-code)** — and we recommend
*using* it with Claude Code too. The repo ships a `CLAUDE.md` playbook (architecture, SI conventions,
build/test commands, the full API surface) that an agent reads first, so it stays consistent with the
codebase from the start. Typical things to ask:

- **Run & script simulations** — *"relax a 256 nm Permalloy disk and plot the vortex core"*, or *"sweep DMI
  from 2–5 mJ/m² and report the skyrmion charge Q"*: it writes the Python (correct build, fields, integrator),
  runs it, and shows the result.
- **Pick the right build & integrator** — it knows the preset matrix (§2.5) and `recommend_integrator()`.
- **Explain & debug** — *"how does the periodic demag kernel work?"*, or diagnose a run that won't converge.
- **Analyze results** — compute the topological charge, domain-wall width, FMR spectrum, or plot a sweep.
- **Extend the engine** — add a field / standard problem / kernel with a CPU reference and a test (§5.1–5.4).

You don't need Claude Code to use Claude-SD, but it is the fastest path from a physics question to a running,
validated simulation. (The sibling project *MuMax-CO* is a published case study of optimizing mumax3 with
Claude Code.)

### 5.1 Why the repo is "agent-friendly"

- **`CLAUDE.md`** at the root is the agent's playbook: build commands, the layer model
  (`types → grid → field → material → effective_field → integrator`), every field/integrator class, SI
  conventions, and the test-tag map. An agent reads this first and stays consistent with the codebase.
- **Tight test harness:** 232 CPU + 113 GPU Catch2 tests, tagged (`[demag]`, `[llg]`, `[gpu]`, …). New
  code is expected to come with a test; the agent runs `ctest` / `unit_tests_gpu.exe "[tag]"` to verify.
- **Mirrored CPU/GPU design:** every GPU field implements the same `IEffectiveField` interface as its CPU
  twin, so adding a feature has a clear template to copy.

### 5.2 A good extension workflow

1. **Describe the goal precisely** — e.g. *"Add a cubic-plus-uniaxial combined anisotropy GPU field, with a
   CPU reference and a `[anisotropy][gpu]` test comparing the two."*
2. Let Claude Code **read `CLAUDE.md` + the nearest existing field** (e.g. `field_kernels_gpu.cu`) and
   implement the new kernel, the host class, the pybind11 binding, and the test.
3. **Build & test:** `cmake --build build/windows-msvc-cuda --config Release` then run the new test tag.
4. **Validate physics** against a known limit or the CPU path (GPU-vs-CPU agreement to ~1e-6, and
   µMAG/analytic checks).

### 5.3 High-value things users commonly add this way

- **A new effective-field term** (custom anisotropy, biquadratic exchange, dipolar tweak): copy an
  existing `*_gpu.cu` kernel + `IEffectiveField` class + binding + a GPU-vs-CPU test.
- **A new standard problem or validation app** (SP#2 was added exactly this way: `benchmarks/sp2/`).
- **A new integrator or a kernel optimization** (the fused local-field kernel and CUDA-Graph capture were
  added/iterated with the agent; *MuMax-CO* shows a 2–5× CUDA-Graph speedup achieved this way).
- **A Python convenience/analysis helper** in `python/micromag/__init__.py` (a new init state, topological
  diagnostic, or plotting routine) — these need no rebuild.
- **More benchmarks**: drop a scenario into the `benchmarks/` harness; it writes the unified
  `all_solvers.json` and `make_report.py` renders the tables/figures.

### 5.4 Guardrails to ask the agent to honor

- Keep **SI conventions** (`H` in A/m, `H_demag = -N·M`, `γ₀ = 1.76e11`) — documented in `CLAUDE.md`.
- Add code behind the existing **interfaces** (`IEffectiveField`, `ISpinTorque`, `IDemagGPU`) so it composes.
- **Always add a test** and run the relevant tag; for GPU code, verify against the CPU path and confirm all
  four CUDA builds still pass.
- Respond in small steps for large changes (don't regenerate whole files unnecessarily).

---
---

# Part B — Reference

## 6. Physical model

### 6.1 Landau–Lifshitz–Gilbert equation

The magnetization dynamics obey the LLG equation in Landau–Lifshitz form:

```
dm/dt = -γ' µ₀ (m × H_eff) - γ' α µ₀ m × (m × H_eff)
```

where:
- **m** = M/Ms is the unit magnetization vector
- **H_eff** [A/m] is the effective field (sum of all contributions)
- **γ₀** = 1.7609×10¹¹ rad T⁻¹ s⁻¹ is the gyromagnetic ratio
- **γ'** = γ₀ / (1 + α²) is the reduced gyromagnetic ratio
- **α** is the dimensionless Gilbert damping parameter
- **μ₀** = 4π×10⁻⁷ T·m/A is the permeability of free space

The two terms are precession (first) and damping toward the effective field (second). Spin-torque terms
(§6.6–6.8) are added as extra `dm/dt` contributions at each integrator stage.

### 6.2 Effective field contributions

```
H_eff = H_exchange + H_demag + H_Zeeman + H_aniso + H_DMI + H_RKKY + H_ME + H_surf [+ H_thermal]
```

Each contribution implements `IEffectiveField` and **adds** to `H_eff` (never zeros it). All have GPU
drop-ins with the same interface, and support per-cell material parameters (§8.7).

#### 6.2.1 Exchange field

```
H_exchange = (2A / μ₀ Ms) ∇²m
```

Discretized as a 6-point Laplacian on the structured grid (x-fastest layout):

```
∇²m ≈ [m(i+1) + m(i-1) + m(j+1) + m(j-1) + m(k+1) + m(k-1) - 6 m(i,j,k)] / dx²
```

Boundary conditions: **Neumann** (∂m/∂n = 0 at surfaces, default) or **Periodic**. At grain boundaries
the per-cell exchange uses the harmonic mean of A. Parameter: exchange stiffness **A** [J/m].

#### 6.2.2 Demagnetization field (open BC)

```
H_demag = -N · M = -N · (Ms m)
```

**N** is the symmetric demag tensor from the Newell (1993) analytical formulas. The convolution is done in
frequency space (FFTW on CPU, cuFFT/VkFFT on GPU):

1. Zero-pad magnetization to 2N in each dimension.
2. Forward FFT → pointwise multiply with the kernel → inverse FFT.
3. Divide by the (padded) transform size (unnormalized FFT convention).

Zero-padding converts the FFT's circular convolution into the required linear (aperiodic) convolution.

#### 6.2.3 Demagnetization field (periodic BC)

For full 3D periodicity, the periodic kernel sums standard Newell tensors over ±n_rep image cells:

```
N^periodic(r) = Σ_{n ∈ Z³, |n| ≤ n_rep} N^Newell(r + n·L)
```

(default `n_rep = 2` → 125 images). No zero-padding (FFT size = grid size → 8× smaller FFT than open BC).
The k = 0 mode is set to zero (toroidal convention: uniform m gives no demag field).

#### 6.2.4 Zeeman field

```
H_Zeeman = H_ext   (applied field, A/m)
```

Uniform (`ZeemanField`) or spatially varying / time-dependent via per-cell fields.

#### 6.2.5 Uniaxial anisotropy

```
H_aniso = (2K₁/μ₀Ms)(m·û)û  [+ (4K₂/μ₀Ms)(m·û)³û]
```

**K₁, K₂** [J/m³] are the anisotropy constants, **û** the easy axis. Cubic, surface, and 2nd-order
anisotropies are also available (`CubicAnisotropyField`, `SurfaceAnisotropyField`).

#### 6.2.6 Dzyaloshinskii–Moriya interaction (DMI)

Interfacial and bulk DMI are provided (`InterfacialDMIFieldGPU`, `BulkDMIFieldGPU`), parameterized by
**D** [J/m²]. Interfacial DMI stabilizes Néel skyrmions for D below the critical
`D_c = 4√(AK)/π`. The GPU kernels are per-cell gather (race-free) and honour single-stream composition.

#### 6.2.7 RKKY, magnetoelastic, surface

- **RKKY** interlayer exchange couples spins across a spacer (region-pair coupling).
- **Magnetoelastic** anisotropy from a strain tensor.
- **Surface** (Néel) anisotropy at film interfaces.

### 6.3 Spin-transfer torque (Slonczewski)

Current-perpendicular-to-plane (CPP) STT:

```
τ_STT = a_J [m × (m × p̂)] + b_J [m × p̂]
a_J = γ₀ ħ J P / (2 e Ms d)   [rad/s]   (damping-like)
b_J = −β a_J                            (field-like)
```

- **p̂** fixed-layer polarization direction; **J** [A/m²] current density (J < 0 switches AP→P);
  **P** ∈ [0,1] spin polarization; **d** [m] free-layer thickness.

### 6.4 Spin-orbit torque (SOT)

```
τ_SOT = a_SOT [ η_DL m×(m×σ̂) + η_FL (m×σ̂) ]
a_SOT = γ₀ ħ |J_c| |θ_SH| / (2 e Ms d_FM)
```

**θ_SH** spin-Hall angle, **σ̂** = ẑ × Ĵ_c the spin-polarization direction set by the heavy-metal geometry.

### 6.5 Zhang–Li (in-plane current-driven) torque

Adiabatic + non-adiabatic torque from an in-plane spin-polarized current (used for SP#5 vortex-core
gyration): parameters current density **J**, polarization **P**, non-adiabaticity **ξ**.

### 6.6 Stochastic LLG (thermal fluctuations)

```
dm/dt = τ_LLG(m, H_eff + H_th)
H_th ~ N(0, σ),  σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V Δt))
```

The Heun (Stratonovich) scheme preserves the correct Boltzmann distribution at equilibrium. Noise scales
with temperature **T**, damping **α**, cell volume **V**, and time step **Δt**. Finite-T runs require a
fixed-step integrator (Heun) — the adaptive RK45 is not valid for the SDE.

---

## 7. Numerical methods

### 7.1 Fixed-step RK4 (CPU and GPU)

Classic 4-stage Runge–Kutta with fixed Δt, followed by per-cell renormalization |m| = 1:

```
k1 = f(m);  k2 = f(m + Δt/2 k1);  k3 = f(m + Δt/2 k2);  k4 = f(m + Δt k3)
m_new = m + (Δt/6)(k1 + 2k2 + 2k3 + k4)
```

Exchange-stability limit:  `Δt_max ≈ 2.83 / (γ₀ × (2A/Ms) × (π/dx)²)`.
For Permalloy at dx = 5 nm: Δt_max ≈ 1.25 ps.

### 7.2 Adaptive DOPRI5 / RK45 (CPU and GPU)

Dormand–Prince embedded RK pair (5th/4th order) with First-Same-As-Last (FSAL):

- 7 evaluations per trial step; 6 effective per accepted step (FSAL).
- Error norm: ‖e‖ = sqrt[(1/3N) Σ (eᵢ/scᵢ)²], scᵢ = atol + rtol·max(|mᵢ|,|m̃ᵢ|).
- Step control: h_new = h × clip(safety × (1/‖e‖)^0.2, fac_min, fac_max).
- GPU: error reduction by a block-parallel kernel; one device-to-host scalar copy per trial step
  (negligible vs the H_eff cost). Default tolerances rtol = 10⁻⁴, atol = 10⁻⁶.

### 7.3 Stratonovich Heun (SLLG)

Predictor–corrector for the Stratonovich SDE, using the **same** noise sample η in both evaluations:

```
Predictor:  m* = normalize(m + Δt f(m, H_eff + η))
Corrector:  m_new = normalize(m + (Δt/2)[f(m, H+η) + f(m*, H_eff(m*)+η)])
```

Noise is generated per step on GPU with cuRAND.

### 7.4 Energy minimizers

- **RelaxGPU** — damped-LLG to a torque threshold (good for flux-closure/vortex equilibria).
- **MinimizeGPU** — Barzilai–Borwein / FIRE energy minimization (faster for stiff DMI textures).

### 7.5 Exchange-stability time-step guideline

| dx (nm) | l_ex / dx | Δt_max (ps) | Recommended Δt (fs) |
|---------|----------|------------|---------------------|
| 10 | 0.57 | 5011 | 50 |
| 5  | 1.14 | 1253 | 50 |
| 3  | 1.90 | 451  | 18 |
| 2  | 2.84 | 200  | 8  |

(Permalloy: Ms = 800 kA/m, A = 13 pJ/m)

### 7.6 Choosing an integrator (`recommend_integrator`)

| Situation | Recommended | Why |
|---|---|---|
| Finite temperature (T > 0) | **Heun** | SLLG requires a fixed-step Stratonovich scheme |
| Low damping, T = 0, dynamics | **RK45-DP** | adaptive step → fewest evaluations |
| High damping / relaxation | **Heun** or **RelaxGPU** | cheap per step; no adaptive overhead |
| Stiff DMI relaxation | **MinimizeGPU** | damped-LLG converges slowly there |

`recommend_integrator(mat, T_K, goal, dt)` returns the choice and a one-line reason from these rules plus
an analytic phase-error estimate.

---

## 8. Python API reference

All examples use the modern module name: `import micromag as mm` (CPU and GPU builds both export
`micromag`). Symbols below exist on both unless marked GPU.

### 8.1 Grid and field

```python
grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)   # dx,dy,dz in meters
grid.nx, grid.ny, grid.nz       # cell counts
grid.dx, grid.dy, grid.dz       # cell sizes [m]
grid.size                        # total cell count

m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(mx, my, mz))
m.set_vortex(cx, cy, core_radius)    # in-plane vortex
m.normalize()                         # enforce |m| = 1 per cell

arr = mm.to_numpy(m)             # (nz, ny, nx, 3) float array
mm.from_numpy(m, arr)            # copy a numpy array into the field
mx, my, mz = mm.mean_magnetization(m)
mm.write_vtk_legacy("m.vtk", m, "m")
```

### 8.2 Material

```python
mat = mm.Material.permalloy()   # Ms=800 kA/m, A=13 pJ/m, K=0, α=0.02
mat = mm.Material.cobalt()      # Ms=1400 kA/m, A=28 pJ/m, K=520 kJ/m³
mat = mm.Material.iron()        # Ms=1710 kA/m, A=20 pJ/m, K=48 kJ/m³

mat.Ms          # saturation magnetization [A/m]
mat.A_exchange  # exchange stiffness [J/m]
mat.K_uniaxial  # uniaxial anisotropy constant [J/m³]
mat.easy_axis   # easy-axis unit vector (Vec3)
mat.alpha       # Gilbert damping
```

### 8.3 Effective fields (CPU)

```python
zeeman = mm.ZeemanField(mm.Vec3(Hx, Hy, Hz))            # H in A/m; .H_ext updatable
exch   = mm.ExchangeField()                             # Neumann BC (default)
exch   = mm.ExchangeField(mm.BoundaryCondition.Periodic)
demag  = mm.DemagField(grid)                            # open BC (FFTW; slow ctor)
demag_p= mm.DemagFieldPeriodic(grid, n_rep=2)           # periodic BC
aniso  = mm.UniaxialAnisotropyField()                   # uses mat.K_uniaxial / easy_axis
dmi    = mm.InterfacialDMIField(grid, 3e-3)             # D [J/m²]

heff = mm.EffectiveFieldSum()
heff.add(exch); heff.add(demag); heff.add(zeeman)
heff.total_energy(m, mat)        # total energy [J]
```

### 8.4 GPU field drop-ins

```python
demag  = mm.DemagFieldGPU(grid)                  # cuFFT / VkFFT (build-selected)
exch   = mm.ExchangeFieldGPU(grid)               # CUDA Laplacian
zeeman = mm.ZeemanFieldGPU(grid, mm.Vec3(Hx,Hy,Hz))
aniso  = mm.UniaxialAnisotropyFieldGPU(grid)
dmi    = mm.InterfacialDMIFieldGPU(grid, 3e-3)
cubic  = mm.CubicAnisotropyFieldGPU(grid, Kc1, Kc2)

fields = mm.FieldSumGPU()                          # compose for the integrators
fields.add(exch); fields.add(dmi); fields.add(aniso)
```

### 8.5 Integrators (CPU)

```python
integ = mm.RK4Integrator(dt=1e-13)
integ.step(m, mat, heff)                 # advance by dt
integ.step(m, mat, heff, stt_sum)        # with spin torque

opts = mm.RK45Options(); opts.rtol = 1e-4; opts.atol = 1e-6
opts.dt_init = 5e-14; opts.dt_max = 1e-11
integ = mm.RK45Integrator(opts)
dt = integ.step(m, mat, heff)            # returns the dt actually taken

thermal = mm.ThermalField(grid, T_K=300.0, dt=1e-13, seed=42)
integ = mm.HeunIntegrator(dt=1e-13)
integ.step(m, mat, heff, thermal)
```

### 8.6 GPU integrators

```python
# Fixed-step RK4 — zero PCIe per step
integ = mm.RK4IntegratorGPU(grid, dt=5e-14)
integ.upload(m)
for _ in range(N):
    integ.step(mat, demag, fields)
integ.download(m)

# Adaptive DOPRI5
opts = mm.RK45GPUOptions(); opts.rtol = 1e-4; opts.dt_init = 5e-14
integ = mm.RK45IntegratorGPU(grid, opts)
integ.upload(m)
t = 0.0
while t < t_end:
    t += integ.step(mat, demag, fields)
print(integ.n_accepted, integ.n_rejected)

# GPU SLLG (Stratonovich Heun + cuRAND)
integ = mm.HeunIntegratorGPU(grid, dt=1e-13, seed=42)
integ.upload(m); integ.step(mat, demag, fields, T_K=300.0)

# Quasistatic
relax = mm.RelaxGPU(grid)        # damped-LLG to torque threshold
mini  = mm.MinimizeGPU(grid)     # BB/FIRE energy minimization
```

### 8.7 Per-cell materials, geometry, regions

```python
matf = mm.MaterialField3D(grid, mat)       # per-cell Ms/A/K/α/axis
exch.set_material_field(matf)               # GPU & CPU
ani.set_material_field(matf)

mask = mm.circle(grid, ...)                 # ellipse/rect/sphere/layer also available
exch.set_mask(mask)                          # 1 = magnet, 0 = vacuum

rm = mm.RegionMap(grid, 0)
exch.set_region_map(rm)
exch.set_inter_exchange(0, 1, A_IEC=5e-12)

grains = mm.voronoi_grains(grid, n_grains=64, seed=0)   # also poisson_disk_grains
```

### 8.8 Spin torques

```python
stt = mm.SlonczewskiSTT(J=-3e12, P=0.5, d=4e-9, p=mm.Vec3(1,0,0))  # J<0: AP→P
sot = mm.SpinOrbitTorque(J_c=1e12, theta_SH=0.12, d_fm=3e-9, sigma=mm.Vec3(0,1,0))
zl  = mm.ZhangLiTorque(J=1e12, P=1.0, xi=0.05)

stt_sum = mm.SpinTorqueSum(); stt_sum.add(stt)        # CPU
torques = mm.SpinTorqueSumGPU(); torques.add(            # GPU
    mm.SpinOrbitTorqueGPU(grid, 3e12, 0.35, 3e-9, mm.Vec3(0,1,0)))
```

### 8.9 Utilities

```python
arr = mm.batch_to_numpy(frames)                 # (n_frames, nz, ny, nx, 3)
mm.save_animation(frames, "out.gif", component="z", fps=10)
lam, x0 = mm.bloch_dw_width(m, axis=0, comp=2)  # Lilley DW width λ = π/max|dm/dx|
res = mm.skyrmion_phase_diagram_gpu(D_vals, K_vals, grid, mat)
results = mm.parameter_sweep(fn, {"D": D_vals, "K": K_vals}, n_jobs=4)
import micromag.mx3 as mx3; mx3.run("script.mx3")   # mumax3 .mx3 subset
```

---

## 9. µMAG validation

Claude-SD reproduces the µMAG standard problems and is cross-checked against mumax3, mumax+, MuMax-CO, and
an OOMMF double-precision anchor. Full cross-solver numbers: `benchmarks/RESULTS_2026.md`.

### 9.1 Standard Problem #4 (dynamic switching)

**Geometry**: 500×125×3 nm Permalloy (200×50×1 cells, 2.5 nm). **Protocol**: saturate +x, apply Field A
H = (−24.6, 4.3, 0) kA/m at t = 0. **Observable**: ⟨mx⟩(t), t_switch.

| Method | ⟨mx⟩(1 ns) | t_switch | vs µMAG |
|--------|-----------|---------|---------|
| CPU RK45 | −0.982 | 175 ps | **0.4 %** |
| GPU RK45 (adaptive) | −0.982 | 175 ps | **0.4 %** |
| GPU RK4 (dt = 5×10⁻¹⁴ s) | −0.944 | ~125 ps | 4.2 %* |
| OOMMF (CPU f64, RKF54) | −0.984 | — | 0.19 % |
| mumax3 / mumax+ | −0.969 / −0.980 | 135 ps | 1.8 % / 0.6 % |
| µMAG reference | −0.9862 | 174–176 ps | — |

*Fixed-step RK4 oscillates near equilibrium; RK45 removes the artifact.

### 9.2 Standard Problem #1 (vortex/single-domain crossover)

**Geometry**: square Permalloy L×L×10 nm (5 nm cells, α = 0.5). **Observable**: critical length L_c.

| Quantity | Claude-SD | Literature |
|---|---|---|
| L_c (t = 10 nm) | ≈ 100–115 nm (cell-size dependent; = mumax+ at matched cells) | 110–120 nm |
| Vortex ground state (500 nm) | ✓ | ✓ |
| S-state ground state (200 nm) | ✓ | ✓ |

### 9.3 Standard Problem #2 (remanence & coercivity, new)

**Geometry**: prism L:d:t = 5:1:0.1, field along [1,1,1]. **Observable**: remanent ⟨mx⟩/Ms vs d/ℓ_ex.

| d/ℓ_ex | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| Claude-SD | 0.994 | 1.000 | 0.998 | 0.970 |
| mumax3 | 1.000 | 1.000 | 0.999 | 0.970 |

Claude-SD tracks mumax3 to ≤ 0.006 across the full sweep.

### 9.4 Standard Problem #3 (hysteresis)

**Geometry**: 1 µm × 1 µm × 20 nm Permalloy (100×100×2 cells, 10 nm). **Protocol**: relax per field, sweep.

| Quantity | Claude-SD | mumax3 | µMAG (dynamic) |
|---|---|---|---|
| Switching field H_sw | −13.8 mT | −13.3 mT | −20 mT |

Energy-minimization H_sw (≈ −13 to −14 mT) differs from the dynamic µMAG reference (−20 mT) by protocol,
not implementation; both codes agree within their protocol. Finer (5 nm) cells improve quantitative accuracy.

### 9.5 Standard Problem #5 (Zhang–Li vortex-core gyration)

**Geometry**: 100×100×10 nm Permalloy, relaxed vortex + in-plane current. **Observable**: core trajectory.
Claude-SD reproduces the vortex-core gyration; core displacement at 8 ns ≈ (−3.5, −14.5) nm.

### 9.6 Slonczewski STT switching

**Geometry**: 160×80×4 nm Permalloy (40×20×1 cells). **Protocol**: J < 0 (AP→P), P = 0.5, d = 4 nm, α = 0.01.

| Quantity | Value |
|---|---|
| Critical current |J_c| | ≈ 0.9×10¹² A/m² |
| Switching time (J = −3×10¹² A/m²) | < 1 ns |
| Final ⟨mx⟩ | +0.997 |

### 9.7 Bloch domain-wall width

`bloch_dw.exe`: measured λ ≈ theory λ = π√(A/K) within 10 % (K = 1e5–4e6 J/m³, dx = 0.5–1 nm).

---

## 10. Performance benchmarks

Measured on NVIDIA RTX 5060 Ti (Blackwell, CUDA 13.2), Windows 11 / MSVC. Full methodology and the
four-solver campaign are in `benchmarks/RESULTS_2026.md`.

### 10.1 CPU vs GPU scaling

Full LLG step (Exchange + Demag + Zeeman, RK4, fixed Δt):

| Grid | Cells | CPU (ms/step) | GPU (ms/step) | Speedup | VRAM |
|------|-------|--------------|--------------|---------|------|
| SP#4 (200×50×1) | 10 K | 10.5 | 1.5 | **7×** | 10 MB |
| Medium (200×200×5) | 200 K | 362 | 22 | **17×** | 193 MB |
| Large (500×500×10) | 2.5 M | ~4500 (est.) | 290 | **~16×** | 2.4 GB |

### 10.2 Adaptive vs fixed-step (GPU)

SP#4 Field A, 0.3 ns: GPU RK45 (1047 acc. steps, 2.2 s) is **3.7× faster** than GPU RK4 (6000 steps,
8.2 s) and **7.4× faster** than CPU RK45, at identical accuracy.

### 10.3 Cross-solver throughput (ms per field-eval)

Because solvers use different-order integrators (RK4 = 4, DOPRI5 = 6, Heun = 2 evals/step), the fair metric
is ms per field-evaluation. A crossover sits near **0.1–0.5 M cells**:

| Scenario | cells | CS cuFFT_f32 | mumax3 | MuMax-CO |
|---|---|---|---|---|
| SP#4 2-D | 10 K | **0.042** | 0.223 | 0.211 |
| medium 3-D | 540 K | 2.61 | 1.71 | **1.66** |
| large 3-D | 2.5 M | 12.7 | 11.3 | **11.1** |

- Small / 2-D → Claude-SD `f32` (CUDA-Graph) is fastest (~5× vs mumax3).
- Large 3-D → mumax3 / MuMax-CO lead (cuFFT-bound); Claude-SD within ~1.1×.
- `f32` is 4–6× faster than `f64` at large 3-D sizes (Blackwell Tensor-Core FFT).
- Finite-T (Heun↔Heun): Claude-SD 2.1× faster than mumax3 at SP#4, ≈ parity at 0.2 M cells.

---

## 11. Example gallery

The `notebooks/` directory holds runnable `*.py` scripts (many mirror the mumax3 examples):

| Notebook | Topic |
|---|---|
| `01_sp4_dynamics.py` | SP#4 CPU dynamics + ⟨m⟩(t) trajectory |
| `02_sp1_phase_diagram.py` | SP#1 vortex/single-domain phase diagram |
| `03_thermal_sp4.py` | SP#4 at T = 300 K (SLLG, Heun) |
| `04_sp4_gpu.py` | SP#4 GPU + CPU/GPU comparison |
| `05_thermal_gpu.py` | GPU SLLG (HeunIntegratorGPU) |
| `06_sp3_hysteresis.py` | SP#3 hysteresis loop |
| `07_hysteresis.py` | mumax3 "hysteresis" example (512×128×4 nm strip) |
| `08_stt_switching.py` | Slonczewski STT switching (|J_c| ≈ 0.9×10¹² A/m²) |
| `09_initial_magnetization.py` | four initial states relaxed to local minima |
| `10_skyrmion_dynamics.py` | DMI skyrmion |
| `13_fmr_spectrum.py` | FMR spectrum |
| `15_spinwave_dispersion.py` | spin-wave dispersion |
| `17_afm_and_zhangli.py` | AFM + Zhang–Li |
| `18_periodic_demag_gpu_bench.py` | periodic-BC demag GPU benchmark |

See the `notebooks/` folder for the full set; mumax3 `.mx3` equivalents live in `notebooks/mx3/`.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `import micromag` fails on GPU build | Call `os.add_dll_directory(...CUDA/bin/x64)` **before** importing. |
| `mm.cuda_available()` is `False` | You imported the CPU build's module; point `sys.path` at a `windows-msvc-cuda*` build. |
| CMake configure fails: generator not found | Edit `"generator"` in `CMakePresets.json` to your VS version (e.g. `Visual Studio 17 2022`). |
| Skyrmion charge `Q` differs run-to-run / by build | Near the metastability boundary `Q` is precision-sensitive — use `f64` (cuFFT) as reference; see `benchmarks/RESULTS_2026.md`. |
| Hysteresis / relaxation never converges | Use `RelaxGPU` (torque threshold) for non-uniform states; for stiff DMI textures use `MinimizeGPU` or raise `max_steps`. |
| RK45 step size reaches minimum | Solution too stiff: loosen `atol`/`rtol`, reduce `dt_max`; with α ≥ 0.5, `dt_max = 5×10⁻¹²` s is reliable. |
| STT switching not observed | Check current sign (J < 0 ⇒ AP→P) and |J| > |J_c|; with low α, precessional switching takes longer. |
| GPU hangs / wrong results | Verify CUDA DLLs on path; check VRAM (200 K cells ≈ 200 MB, 2.5 M ≈ 2.4 GB); a `cudaDeviceSynchronize` error means VRAM exhaustion or launch failure. |
| mumax3 thermal run panics (`CURAND_LENGTH_NOT_MULTIPLE`) | mumax3 needs an even cell count for `Temp>0`; Claude-SD handles a single cell directly. |
| FFTW plan creation fails (CPU) | Ensure nx, ny, nz ≥ 2; FFTW_ESTIMATE + FFTW_UNALIGNED are used. |

---

## Appendix A — mumax3 `.mx3` API coverage

`micromag.mx3` (§4.6) interprets a **practical subset** of the mumax3 scripting language on the Claude-SD
engine — it does **not** run mumax3. This appendix lists the mumax3 API the interpreter **does not (yet)
implement**, and whether the capability is available through the Claude-SD **Python API** instead.
Unsupported script statements emit a `[mx3 warning]` and are skipped, so a partially-covered script still
runs as far as it can.

### Supported (for reference)

Grid / `SetPBC`; `Msat` / `Aex` / `alpha` / `Ku1` / `anisU` / `Dind` / `Dbulk` / `B_ext` / `EnableDemag`;
`m` init (`uniform` / `vortex` / `random` / `twodomain` / `neelskyrmion` / `blochskyrmion`); regions
(`defregion`, `*.SetRegion`); spin-torque params (`J`, `Pol`, `xi`); shapes + `setgeom` (CPU); solver
(`MaxErr` / `MaxDt` / `MinDt` / `SetSolver`); `relax` / `minimize` / `run` / `steps`; output
(`save` / `saveas` / `tableSave` / `tableAdd` / `autosave` / `tableAutoSave`); `for` / `if`-`else`; `print`.
(Full list: [`examples/mx3/README.md`](../examples/mx3/README.md).)

### Not implemented (or partial) in the `.mx3` interpreter

| mumax3 API | Status in `micromag.mx3` | Available via Claude-SD Python API? |
|---|---|---|
| `setgeom(...)` on the **GPU** backend | Partial — **CPU backend only** | Yes — geometry masks (`set_mask`) on CPU & GPU |
| `ext_makegrains` / grain generation in script | Not implemented | Yes — `voronoi_grains()`, `poisson_disk_grains()` |
| `Crop`, `Resize`, `Expand` (geometry transforms) | Not implemented | No |
| `Temp` (finite temperature inside a script) | Not implemented in the runner | Yes — `HeunIntegrator(GPU)` with `T_K` (SLLG) |
| Full `Slonczewski` params (`FixedLayer`, `Lambda`, `EpsilonPrime`, field-like) | Partial — basic STT / Zhang-Li via `J` / `Pol` / `xi` | Yes — `SlonczewskiSTT`, `SpinOrbitTorque`, `ZhangLiTorque` |
| VCMA / `VoltageController` anisotropy | Not implemented | No |
| `AddFieldTerm` / `AddEdensTerm` (custom field/energy terms) | Not implemented | Via C++ (`IEffectiveField`) + binding |
| `ext_*` extensions (`ext_topologicalcharge`, `ext_centerWall`, `ext_bubble`, `ext_rotate`, …) | Not implemented in the runner | Topological charge: yes (Python helper); others: no |
| Per-region `alpha` / `xi` / `Pol` on the **GPU** integrators | Not implemented (CPU OK) | Partial (CPU per-region) |
| `RunWhile(cond)` / advanced run controls | Not implemented | Loop in Python instead |
| `for … range`, user-defined `func`, `expect` / `expectV` | Not implemented (use C-style `for`) | N/A |
| Output options: `OutputFormat`, `SnapshotFormat`, save-`Crop` / slices | Not implemented (OVF 2.0 + `.txt` table only) | Use the Python API + NumPy |

> The Claude-SD **Python API** (§8) is the recommended route for anything the script interpreter doesn't
> cover — it exposes the full engine (all effective fields, spin torques, integrators, per-cell materials,
> geometry, and analysis helpers). For features absent from both (e.g. VCMA, geometry transforms), see
> [§5 — extending the engine with Claude Code](#5-using--extending-claude-sd-with-claude-code).

---

*License: GPLv3. Issues & contributions welcome. Build/architecture reference: `CLAUDE.md`. Full benchmark
methodology and results: `benchmarks/RESULTS_2026.md`.*
