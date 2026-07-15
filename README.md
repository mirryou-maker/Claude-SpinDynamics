# Claude-SpinDynamics

**C++20/CUDA micromagnetic simulator with Python bindings**

Claude-SpinDynamics is an open-source micromagnetics library that solves the
Landau–Lifshitz–Gilbert (LLG) equation on structured grids using a CPU (FFTW)
backend and GPU (cuFFT / VkFFT) backends in **single and double precision**. It
is validated against the µMAG standard problems and cross-checked against
mumax3, mumax+, MuMax-CO and OOMMF.

📖 **New here? Start with the [User Guide](docs/USER_GUIDE.md)** — advantages, installation,
beginner & advanced tutorials, and how to extend Claude-SD with Claude Code.

---

## ⬇️ Download & run — no build, no source required (Windows x64)

Don't want to compile? **Download only the prebuilt binary** (not the source) from the
**[Releases page](https://github.com/mirryou-maker/Claude-SpinDynamics/releases/latest)**, unzip, and run —
two packages:

| Package | For | Size |
|---------|-----|------|
| `Claude-SD-…-cpu-py313.zip` | **CPU** — runs on any x64 Windows, no GPU | ~3 MB |
| `Claude-SD-…-gpu-py313.zip` | **GPU (CUDA)** — needs an NVIDIA GPU; all four variants (cuFFT/VkFFT × f32/f64), CUDA runtime **bundled** (no CUDA Toolkit install) | ~280 MB |

Each package contains **every executable from the build** plus the `micromag` Python module:

- **Standalone apps — no Python needed:** run `bin\sp4.exe` / `bin\sp4_gpu.exe`, `bloch_dw.exe`, … (every
  µMAG problem, demo, and thermal app), plus `unit_tests.exe` to self-check. (GPU: run `add_dll_to_path.bat`
  first.)
- **Python API:** `py -3.13 -m pip install numpy matplotlib`, then point `sys.path` at the `python\` folder
  and `import micromag`. See each zip's `README.txt` for a copy-paste quick start.

Requires **Windows 10/11 x64** (+ **Python 3.13** for the Python API; **NVIDIA GPU + driver** for the GPU
package). To build from source instead, see [Quick Start](#quick-start).

---

## Key Features

| Category | Feature |
|----------|---------|
| **Physics** | LLG, SLLG (stochastic), Slonczewski STT, Spin-Orbit Torque, Zhang–Li |
| **Effective fields** | Zeeman, Exchange (Neumann/Periodic), Demag (open & periodic BC), Uniaxial/Cubic/Surface anisotropy, DMI (bulk & interfacial), RKKY, magnetoelastic |
| **Precision / FFT** | float32 **and** float64; cuFFT **and** VkFFT demag backends |
| **CPU integrators** | RK4 (fixed-step), RK45/DOPRI5 (adaptive, FSAL), Heun (Stratonovich SLLG) |
| **GPU integrators** | RK4GPU, RK45GPU (adaptive DOPRI5/FSAL), HeunGPU (cuRAND), Relax/Minimize |
| **GPU fields** | all effective fields on GPU (cuFFT/VkFFT demag); per-cell materials, geometry masks, region maps |
| **Python API** | Full pybind11 bindings + NumPy bridge + mx3 runner + `recommend_integrator()` |
| **Validation** | µMAG SP#1–5; cross-checked vs mumax3 / mumax+ / MuMax-CO / OOMMF |

---

## Performance

Full LLG step time (Exchange + Demag + Zeeman, RK4):

| Grid | Cells | CPU (ms/step) | GPU (ms/step) | Speedup |
|------|-------|--------------|--------------|---------|
| SP#4 (200×50×1) | 10 K | 10.5 | 1.5 | **7.0×** |
| Medium (200×200×5) | 200 K | 362 | 22 | **16.6×** |
| Large (500×500×10) | 2.5 M | ~4 500 (est.) | 290 | **~15.6×** |

GPU adaptive RK45 vs. fixed-step RK4 on SP#4 (0.3 ns):
1 047 accepted steps × 2.1 ms = **2.2 s** vs. 6 000 steps × 1.4 ms = 8.2 s → **3.7× faster, identical accuracy**.

---

## µMAG Validation

| Standard Problem | Quantity | Claude-SpinDynamics | Reference | Note |
|-----------------|---------|-----------------|---------------|-------|
| SP#4 Field A | ⟨mx⟩ (1 ns) | −0.982 | µMAG −0.9862 | **0.4 %** |
| SP#4 Field A | t_switch | 175 ps | µMAG 174–176 ps | < 1 % |
| SP#1 (t = 10 nm) | L_c | ~100–115 nm | 110–120 nm | within range |
| SP#2 | remanent ⟨mx⟩/Ms | matches mumax3 | mumax3 | ≤ 0.006 across sweep |
| SP#3 | H_sw (energy-min.) | −13.8 mT | mumax3 −13.3 mT | protocol-consistent |
| SP#5 | vortex-core gyration | reproduced | mumax3 | ✓ |

Full cross-solver numbers (incl. OOMMF f64 anchor) are in
[`benchmarks/RESULTS_2026.md`](benchmarks/RESULTS_2026.md) and [User Guide §9](docs/USER_GUIDE.md).

---

## Quick Start

### CPU build
```powershell
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release
ctest --preset windows-msvc   # 232 tests
```

### GPU build (CUDA required)
```powershell
cmake --preset windows-msvc-cuda
cmake --build build/windows-msvc-cuda --config Release
.\build\windows-msvc-cuda\bin\Release\unit_tests_gpu.exe   # 113 tests
```

### Linux / WSL build (CPU)
```bash
# Deps (Ubuntu): sudo apt install build-essential cmake ninja-build libfftw3-dev
#                pip install --user pybind11 numpy      # Catch2 auto-fetched
cmake --preset linux-gcc
cmake --build build/linux-gcc -j$(nproc)
./build/linux-gcc/bin/unit_tests                        # 231/232 pass*
```
FFTW is found via pkg-config and pybind11 via `python -m pybind11 --cmakedir`
when no CMake config package is present (`linux-gcc-cuda` adds the CUDA backend
on a host with the NVIDIA toolkit). CPU performance is on par with the Windows
build — see [CPU parity benchmark](benchmarks/linux_cpu_parity.md). *The one
skipped assertion is an FFTW-implementation-sensitive MFM peak location, not a
portability defect.*

### Build variants (CPU / GPU · precision · FFT backend)

The build is chosen at **configure time** by a CMake preset — precision and FFT backend are compile-time,
not run-time. Full guidance: [User Guide §2.5](docs/USER_GUIDE.md).

| Preset | Target | Precision | FFT | Pick it for |
|--------|--------|-----------|-----|-------------|
| `windows-msvc` | CPU | f64 | FFTW | development, portability, reference (no GPU) |
| `linux-gcc` | CPU | f64 | FFTW | Linux / WSL CPU build (`linux-gcc-cuda` for GPU) |
| `windows-msvc-cuda` | GPU | f64 | cuFFT | reference accuracy, topology-sensitive (skyrmion *Q*) |
| `windows-msvc-cuda-f32` | GPU | f32 | cuFFT | fastest small / 2-D runs |
| `windows-msvc-cuda-vkfft` | GPU | f64 | VkFFT | large 3-D, f64 accuracy |
| `windows-msvc-cuda-vkfft-f32` | GPU | f32 | VkFFT | fastest large 3-D production |

Rule of thumb: CPU for portability, GPU for production; f64 for accuracy, f32 for speed; cuFFT for
small / power-of-two, VkFFT for large 3-D / non-power-of-two. In Python, select a build by pointing
`sys.path` at its `build/<preset>/python/` directory (CPU vs GPU is confirmed by `mm.cuda_available()`).

### Python (minimal example)
```python
import sys
sys.path.insert(0, 'build/windows-msvc/python')
import micromag as mm
import numpy as np

grid   = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
mat    = mm.Material.permalloy()
H_ext  = mm.Vec3(-24.6e3, 4.3e3, 0.0)          # SP#4 Field A

demag  = mm.DemagField(grid)
heff   = mm.EffectiveFieldSum()
heff.add(mm.ExchangeField())
heff.add(demag)
heff.add(mm.ZeemanField(H_ext))

m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(1.0, 0.1, 0.0)); m.normalize()

integ = mm.RK45Integrator()
t = 0.0
while t < 1e-9:
    t += integ.step(m, mat, heff)

mx, my, mz = mm.mean_magnetization(m)
print(f"⟨mx⟩(1 ns) = {mx:.4f}")   # expect ~ -0.982
```

---

## Recommended: drive Claude-SD with Claude Code

Claude-SD was built end-to-end with **[Claude Code](https://claude.com/claude-code)** and is designed to be
*used* with it too. The repo ships a [`CLAUDE.md`](CLAUDE.md) playbook (architecture, SI conventions,
build/test commands, the full API surface) that an agent reads first, so it stays consistent with the
codebase from the start. Typical things to ask Claude Code:

- **Run & script simulations** — *"relax a 256 nm Permalloy disk and plot the vortex core"*, or *"sweep DMI
  from 2–5 mJ/m² and report the skyrmion charge"* — it writes the Python, runs it, and shows the result.
- **Pick the right build & integrator** — it knows the preset matrix and `recommend_integrator()`.
- **Explain & debug** — *"how does the periodic demag kernel work?"*, or diagnose a run that won't converge.
- **Extend the engine** — add an effective-field term / standard problem / kernel with a CPU reference and a
  test, verified across all four CUDA builds.

You don't need Claude Code to use Claude-SD, but it is the fastest path from a physics question to a running,
validated simulation. See [User Guide §5](docs/USER_GUIDE.md) for the workflow and guardrails.

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | User guide & reference manual: getting started, theory, full API, µMAG validation, benchmarks, extending with Claude Code |
| [`CLAUDE.md`](CLAUDE.md) | Developer reference: build commands, architecture, GPU internals |
| [`notebooks/`](notebooks/) | Jupyter-compatible Python notebooks (SP#4, SP#1, thermal, GPU, hysteresis, STT) |

---

## Repository Structure

```
Claude-SpinDynamics/
├── include/micromag/   # Public headers (C++20)
├── src/                # C++ / CUDA implementation
├── apps/               # Standalone simulation executables
├── tests/              # Catch2 unit tests (CPU + GPU)
├── python/             # pybind11 bindings + micromag package
├── notebooks/          # Example notebooks (07-09: mumax3 replications)
└── docs/               # User guide & reference manual + design documents
```

---

## Toolchain

- **OS**: Windows 11 (MSVC, UTF-8). Presets use the `Visual Studio 18 2026`
  generator — for VS 2022 change `"generator"` in `CMakePresets.json` to
  `Visual Studio 17 2022`.
- **C++ standard**: C++20
- **Dependencies**: FFTW3, pybind11, Catch2 v3 (all via vcpkg)
- **GPU**: CUDA 13.2, cuFFT, cuRAND (`cmake --preset windows-msvc-cuda`)
- **Python**: 3.13, NumPy, Matplotlib

---

## Citation

> If you use Claude-SpinDynamics in your research, please cite:
>
> *[Paper in preparation — Claude-SpinDynamics: A GPU-accelerated micromagnetics
> simulator with adaptive time-stepping and Python bindings]*

---

## Authors & Acknowledgements

- **Chun-Yeol You** (DGIST) — author, physics direction, validation.
- **Claude Code** (Anthropic; Claude Opus) — AI pair-programmer. Claude-SD was designed,
  implemented, tested, and benchmarked end-to-end with Claude Code under a specification-and-test
  workflow; commits carry `Co-Authored-By: Claude`. See [CONTRIBUTORS.md](CONTRIBUTORS.md).

---

## License

GNU General Public License v3.0 (GPLv3) — see [LICENSE](LICENSE).
