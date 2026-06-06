# NanoSpinDynamics

**C++20/CUDA micromagnetic simulator with Python bindings**

NanoSpinDynamics is an open-source micromagnetics library that solves the
Landau–Lifshitz–Gilbert (LLG) equation on structured grids using CPU (FFTW)
and GPU (cuFFT/CUDA) backends.  It is validated against the µMAG standard
problems and designed for research-scale simulations of nanomagnetic devices.

---

## Key Features

| Category | Feature |
|----------|---------|
| **Physics** | LLG, SLLG (stochastic), Slonczewski STT, Spin-Orbit Torque |
| **Effective fields** | Zeeman, Exchange (Neumann/Periodic BC), Demag (open BC), Demag (periodic BC), Uniaxial anisotropy |
| **CPU integrators** | RK4 (fixed-step), RK45/DOPRI5 (adaptive, FSAL), Heun (Stratonovich SLLG) |
| **GPU integrators** | RK4GPU, RK45GPU (adaptive DOPRI5/FSAL), HeunGPU (cuRAND noise) |
| **GPU fields** | DemagFieldGPU (cuFFT), ExchangeFieldGPU, ZeemanFieldGPU, UniaxialAnisotropyFieldGPU |
| **Python API** | Full pybind11 bindings + NumPy bridge + Jupyter notebooks |
| **Validation** | µMAG SP#1, SP#3, SP#4 (< 0.4 % error vs. reference) |

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

| Standard Problem | Quantity | NanoSpinDynamics | µMAG Reference | Error |
|-----------------|---------|-----------------|---------------|-------|
| SP#4 Field A | ⟨mx⟩ (1 ns) | −0.982 | −0.9862 | **0.4 %** |
| SP#4 Field A | t_switch | 175 ps | 174–176 ps | < 1 % |
| SP#1 (t=10 nm) | L_c | 115 nm | 110–120 nm | < 5 % |
| SP#3 (512×128 nm) | H_sw | −25 mT | −20 to −30 mT | within range |

---

## Quick Start

### CPU build
```powershell
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release
ctest --preset windows-msvc   # 87 tests
```

### GPU build (CUDA required)
```powershell
cmake --preset windows-msvc-cuda
cmake --build build/windows-msvc-cuda --config Release
.\build\windows-msvc-cuda\bin\Release\unit_tests_gpu.exe   # 53 tests
```

### Python (minimal example)
```python
import sys
sys.path.insert(0, 'build/windows-msvc/python')
import _micromag as mm
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

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/user_manual.md`](docs/user_manual.md) | Full user manual: theory, API reference, examples, validation |
| [`CLAUDE.md`](CLAUDE.md) | Developer reference: build commands, architecture, GPU internals |
| [`notebooks/`](notebooks/) | Jupyter-compatible Python notebooks (SP#4, SP#1, thermal, GPU, hysteresis, STT) |

---

## Repository Structure

```
NanoSpinDynamics/
├── include/micromag/   # Public headers (C++20)
├── src/                # C++ / CUDA implementation
├── apps/               # Standalone simulation executables
├── tests/              # Catch2 unit tests (CPU + GPU)
├── python/             # pybind11 bindings + micromag package
├── notebooks/          # Example notebooks (07-09: mumax3 replications)
└── docs/               # User manual and design documents
```

---

## Toolchain

- **OS**: Windows 11 (MSVC 2022, UTF-8)
- **C++ standard**: C++20
- **Dependencies**: FFTW3, pybind11, Catch2 v3 (all via vcpkg)
- **GPU**: CUDA 13.2, cuFFT, cuRAND (`cmake --preset windows-msvc-cuda`)
- **Python**: 3.13, NumPy, Matplotlib

---

## Citation

> If you use NanoSpinDynamics in your research, please cite:
>
> *[Paper in preparation — NanoSpinDynamics: A GPU-accelerated micromagnetics
> simulator with adaptive time-stepping and Python bindings]*

---

## License

TBD
