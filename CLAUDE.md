# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

### CPU build (default)

```powershell
# Configure (only needed once, or after CMakeLists changes)
cmake --preset windows-msvc

# Build
cmake --build build/windows-msvc --config Release

# Run all CPU tests (87 tests)
ctest --preset windows-msvc

# Run a specific test tag
.\build\windows-msvc\bin\Release\unit_tests.exe "[demag]"

# Run a specific test by name substring
.\build\windows-msvc\bin\Release\unit_tests.exe "Demag: cube macrospin"
```

### GPU build (CUDA required)

```powershell
# Configure
cmake --preset windows-msvc-cuda

# Build
cmake --build build/windows-msvc-cuda --config Release

# Run GPU tests (118 tests)
.\build\windows-msvc-cuda\bin\Release\unit_tests_gpu.exe

# Run µMAG benchmarks
.\build\windows-msvc-cuda\bin\Release\sp4_full_gpu_bench.exe   # G7: RK4 vs GPU RK4
.\build\windows-msvc-cuda\bin\Release\sp4_rk45_gpu.exe         # G9: GPU RK45 adaptive
.\build\windows-msvc-cuda\bin\Release\sp4_gpu_1ns.exe          # SP#4 1 ns GPU validation
.\build\windows-msvc-cuda\bin\Release\llg_large_bench.exe      # 2.5 M cell scaling
```

### Linux / WSL build (CPU)

```bash
# Ubuntu deps: sudo apt install build-essential cmake ninja-build libfftw3-dev
#              pip install --user "pybind11>=2.12,<4" numpy  # Catch2 auto-fetched
cmake --preset linux-gcc
cmake --build build/linux-gcc -j$(nproc)
./build/linux-gcc/bin/unit_tests                        # 231/232 pass (1 MFM peak-loc, FFTW-sensitive)
```

FFTW resolves via pkg-config (`libfftw3` + `libfftw3_threads`) when no
`FFTW3Config.cmake` exists; pybind11 via `python -m pybind11 --cmakedir`; Catch2
via FetchContent. `linux-gcc-cuda` adds the CUDA backend on a host with the
NVIDIA toolkit. CPU speed is on par with Windows (Linux single-thread is
actually faster) — see `benchmarks/linux_cpu_parity.md` and regenerate with
`benchmarks/cpu_parity_bench.py --threads 1,2,4,8`.

### CPU validation apps

```powershell
# Bloch DW width validation (Priority 2: µMAG extension)
.\build\windows-msvc\bin\Release\bloch_dw.exe
# Expected: λ_meas ≈ λ_theory = π√(A/K) within 10% for K=1e5–1e6 J/m³
```

Debug build uses preset `windows-msvc-debug` and outputs to `build/windows-msvc-debug`.  
Toolchain: MSVC + vcpkg at `C:/vcpkg`, triplet `x64-windows`.  
CUDA toolkit: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2`.

### Python modules

| Build | Module location | CUDA available |
|-------|----------------|----------------|
| CPU   | `build/windows-msvc/python/` | `False` |
| GPU   | `build/windows-msvc-cuda/python/` | `True` |

GPU module requires `os.add_dll_directory('...CUDA/v13.2/bin/x64')` before import.

## Architecture

### Layer model (bottom → top)

```
types.hpp          Vec3, Real, Index, physical constants (gamma_0 = 1.76e11 rad/(T·s))
grid.hpp           StructuredGrid — cell geometry only
field.hpp          VectorField3D — owns data[], wraps grid
material.hpp       Material — Ms, A, K, alpha, easy_axis (static factories: permalloy/cobalt/iron)
effective_field.hpp  IEffectiveField (accumulate/energy/name) + EffectiveFieldSum (compositor)
spin_torque.hpp    ISpinTorque (accumulate/name) + SpinTorqueSum (compositor)
integrator.hpp     RK4Integrator, RK45Integrator (adaptive DOPRI5), HeunIntegrator (SLLG)
```

### Effective field implementations (CPU)

Each `IEffectiveField` **adds** to `H_out` (never zeros it):

| Class | File | Physics |
|-------|------|---------|
| `ZeemanField` | zeeman.cpp | Uniform H_ext |
| `UniaxialAnisotropyField` | anisotropy.cpp | H_ani = (2K/μ₀Ms)(m·û)û |
| `ExchangeField` | exchange.cpp | 6-point Laplacian, Neumann or Periodic BC |
| `DemagField` | demag.cpp | FFT convolution with Newell tensor, zero-padded to 2N |
| `DemagFieldPeriodic` | demag_periodic.cpp | Periodic-BC demag, no padding, image-sum kernel |

### Spin torque implementations

Each `ISpinTorque` **adds** to `dm_out` in [1/s]:

| Class | Physics |
|-------|---------|
| `SlonczewskiSTT` | CPP-STT: a_J[m×(m×p̂)] + b_J[m×p̂] |
| `SpinOrbitTorque` | SOT (spin Hall): a_SOT[η_DL m×(m×σ̂) + η_FL (m×σ̂)] |

### CPU integrators

| Class | Algorithm | Use case |
|-------|-----------|----------|
| `RK4Integrator` | Fixed-step RK4 | Fast, simple; needs small dt |
| `RK45Integrator` | Adaptive DOPRI5/FSAL | Accurate, auto dt control |
| `HeunIntegrator` | Stratonovich Heun | SLLG (finite temperature, fixed dt) |

### GPU layer (MICROMAG_CUDA=1)

All GPU classes are in `include/micromag/*_gpu.hpp` and compiled into `micromag_cuda`.

**GPU field drop-ins** (same `IEffectiveField` interface as CPU):

| Class | File | Notes |
|-------|------|-------|
| `DemagFieldGPU` | demag_gpu.hpp | cuFFT, batch, stream; GPU precompute |
| `ExchangeFieldGPU` | exchange_gpu.hpp | 6-point Laplacian CUDA kernel |
| `ZeemanFieldGPU` | field_kernels_gpu.hpp | Uniform field kernel |
| `UniaxialAnisotropyFieldGPU` | field_kernels_gpu.hpp | Per-cell dot-product kernel |

**GPU integrators** (zero PCIe per step after initial upload):

| Class | File | Algorithm |
|-------|------|-----------|
| `RK4IntegratorGPU` | rk4_integrator_gpu.hpp | Fixed-step RK4, FSAL |
| `RK45IntegratorGPU` | rk45_integrator_gpu.hpp | Adaptive DOPRI5/FSAL, 1 D2H scalar/trial step |
| `HeunIntegratorGPU` | heun_integrator_gpu.hpp | Stratonovich Heun, cuRAND noise |

GPU usage pattern:
```cpp
RK4IntegratorGPU integ(grid, dt);
integ.upload(m0);
for (int k = 0; k < N; ++k)
    integ.step(mat, demag, exch, zeeman);   // all GPU, zero PCIe
integ.download(m_out);
```

**GPU state**: `GPUMagState` owns 5×[3N] double buffers + pinned staging.  
`RK45IntegratorGPU` additionally owns 9×[3N] scratch buffers (k1-k7, m5, err).

### Memory layout

`linear_index(i, j, k) = i + nx*(j + ny*k)` — **x is fastest**.  
Applies to `VectorField3D`, FFTW buffers in `demag.cpp`, and all GPU buffers.  
FFTW plans use `(nz, ny, nx)` argument order (C row-major → x is last → x-fastest).

### LLG equation (Landau-Lifshitz form)

```
dm/dt = -γ'μ₀(m×H) - γ'αμ₀ m×(m×H),   γ' = γ₀/(1+α²),   H in A/m
```

Spin torques are added as extra `dm/dt` terms at each integrator stage.

### Demag pipeline (open BC — DemagField)

1. **Construction**: pad to 2N, create FFTW plans, call `precompute_kernel()`.
2. **`precompute_kernel()`**: fills 6 arrays `K_xx_…K_yz_` using Newell (1993) closed-form integrals (`nxx`/`newell_f`, `nxy`/`newell_g`). Even/odd symmetry fills padded kernel.
3. **`accumulate()`**: 3× forward FFT, 3× pointwise K·M + IFFT, `H_demag = -N·M`.

Normalisation: divide by `pad_nx*pad_ny*pad_nz` after IFFT.

### Demag pipeline (periodic BC — DemagFieldPeriodic)

No zero-padding (FFT size = grid size → **8× smaller FFT** than open BC).

1. **`precompute_kernel()`**: periodic Newell sum: `N^per(r) = Σ_n N^Newell(r + n·L)` over ±`n_rep` image cells per dimension (default `n_rep=2`, 125 images). Then FFT. **k=0 mode zeroed** (uniform m → H=0, toroidal convention).
2. **`accumulate()`**: identical pipeline to `DemagField` but with unpadded buffers.

## Error-handling policy

- **Constructors validate their arguments** and throw `std::invalid_argument`
  on nonsense input (non-positive grid dims/cell sizes, `dt <= 0`, inconsistent
  RK45 tolerances). Resource failures (FFTW plan creation) throw
  `std::runtime_error`. pybind11 translates both to Python exceptions.
- **Hot loops do not validate** — per-cell code assumes constructor-checked
  state; use `assert` for internal invariants there, never `throw`.
- GPU code: every CUDA call goes through `CUDA_CHECK`; kernel launches are
  followed by a `cudaGetLastError` check. Set `MICROMAG_SYNC_DEBUG=1` (env var)
  to force a device-synchronize after every checked launch when hunting
  stream races (slow; debugging only).

## SI conventions

- H in A/m, M = Ms·m (m is unit vector)
- `H_demag = -N·M` (N is positive-definite demag tensor)
- Energy: `E = -μ₀/2 · Ms · Σ m·H_demag · dV`
- Newell normalisation: `1/(4π·dx·dy·dz)` applied inside `nxx`/`nxy`

## GPU performance (measured)

| Grid | Cells | CPU ms/step | GPU ms/step | Speedup |
|------|-------|-------------|-------------|---------|
| SP#4 (200×50×1) | 10K | 10.5 | 1.5 | 7× |
| Medium (200×200×5) | 200K | 362 | 22 | 17× |
| Large (500×500×10) | 2.5M | ~4500 (est) | 290 | ~16× |

GPU RK45 vs GPU RK4 on SP#4 (0.3 ns): 1047 steps @ 2.1 ms vs 6000 steps @ 1.4 ms → **3.7× fewer steps, 3.7× faster wall time**.

## µMAG benchmarks (validated)

| App | Standard | Result |
|-----|----------|--------|
| `sp4`, `sp4_rk45` | SP#4 Field A | t_switch ≈ 175 ps, ⟨mx⟩(1ns) = −0.982 (0.4% vs µMAG) |
| `sp1`, `sp1_phase` | SP#1 | L_c = 115 nm, vortex/S-state phase diagram |
| `sp3` | SP#3 | Hysteresis: H_sw ≈ −20 mT (1 µm × 1 µm × 20 nm, 10 nm cells) |
| `sp4_gpu_1ns` | SP#4 GPU | ⟨mx⟩(1ns) = −0.944 (GPU RK4 fixed dt=5e-14 s) |
| `bloch_dw` | Bloch DW | λ_meas ≈ λ_theory = π√(A/K) within 10% (K=1e5–4e6 J/m³, dx=0.5–1 nm) |

## GPU per-cell field API

All GPU fields support uniform and per-cell material parameters:

```python
import micromag as mm

# Per-cell exchange (harmonic mean at grain boundaries)
exch = mm.ExchangeFieldGPU(grid)
exch.set_material_field(matf)   # MaterialField3D
print(exch.has_material_field)  # True

# Per-cell uniaxial anisotropy (K, easy_axis per cell)
ani = mm.UniaxialAnisotropyFieldGPU(grid)
ani.set_material_field(matf)

# Per-cell DMI
dmi = mm.InterfacialDMIFieldGPU(grid, D_uniform)
dmi.set_D_field(D_sf, Ms_sf)   # ScalarField3D × 2
print(dmi.has_D_field)

# Per-cell cubic anisotropy
cubic = mm.CubicAnisotropyFieldGPU(grid, Kc1, Kc2)
cubic.set_Kc_field(Kc1_sf, Kc2_sf, Ms_sf)

# Per-cell surface anisotropy
surf = mm.SurfaceAnisotropyFieldGPU(grid, Ks, t_film)
surf.set_Ks_field(Ks_sf, Ms_sf)

# Per-cell dispatch is consistent across BOTH call paths: the host-side
# accumulate(m, mat, H) and the integrator accumulate_gpu_ptr path route
# through the same kernel dispatch, so a set_*_field() always takes effect.

# FieldSumGPU: pass extra fields to all GPU integrators
fields = mm.FieldSumGPU()
fields.add(exch); fields.add(ani); fields.add(dmi)

# RK4 / RK45 / Heun all accept FieldSumGPU overload
rk4   = mm.RK4IntegratorGPU(grid, dt);   rk4.step(mat, demag, fields)
rk45  = mm.RK45IntegratorGPU(grid);      rk45.step(mat, demag, fields)
heun  = mm.HeunIntegratorGPU(grid, dt);  heun.step(mat, demag, fields)
relax = mm.RelaxGPU(grid);               relax.run(mat, demag, fields, opts)
```

## Python utility functions

```python
# Animation
arr    = mm.batch_to_numpy(frames)               # (n_frames, nz, ny, nx, 3) ndarray
mm.save_animation(frames, "out.gif", component="z", fps=10)

# DW width (Lilley definition: λ = π/max|dm/dx|)
lam, x0 = mm.bloch_dw_width(m, axis=0, comp=2)  # [m], [m]

# Skyrmion phase diagram D×K sweep (GPU, convenience wrapper)
res = mm.skyrmion_phase_diagram_gpu(D_vals, K_vals, grid, mat)

# Parameter sweep (multiprocessing-safe, n_jobs>1 uses spawn context)
results = mm.parameter_sweep(fn, {"D": D_vals, "K": K_vals}, n_jobs=4)
```

## Response Size Constraints
- Never produce a single response large enough to hit the 32,000 output token limit.
- If a task requires a massive code change or a large GUI implementation, break it down into multiple smaller steps.
- Do not output the entire file if only a specific part needs modification.

## Test structure

**CPU** (`tests/unit_tests`, 232 tests): Catch2 v3, tags below.  
**GPU** (`tests/unit_tests_gpu`, 118 tests): same runner, all tagged `[gpu]`.

| Tag | Scope |
|-----|-------|
| `[demag]` | DemagField (open BC), FFTW sanity |
| `[demag_periodic]` | DemagFieldPeriodic (periodic BC) |
| `[exchange]` | ExchangeField (Neumann + periodic BC) |
| `[llg]` | LLG torque, RK4, RK45 integrators |
| `[spin_torque]` | STT, SOT |
| `[zeeman]`, `[anisotropy]` | Field kernels |
| `[grid]`, `[field]` | StructuredGrid, VectorField3D |
| `[gpu]` | All GPU tests (unit_tests_gpu only) |


## 진도 관리 (GitHub Projects)

- 트래커 CLI: `python "d:\Claude-Code-R\Project_Tracker\tracker.py"`
- **이 저장소의 프로젝트 코드: `P5`** ← 이 값만 프로젝트마다 다르다

### 세션 시작 시
`tracker.py show P5` 를 실행하고, State / Phase / Next / Blocker 를 3줄 이내로 요약해 보고한다.

### 세션 종료 시 또는 의미 있는 작업 완료 시
`tracker.py update P5 --state <상태> --phase "<단계>" --next "<다음 액션>" --log "<한 줄 요약>"` 를 실행한다. 바뀐 항목만 인자로 넘긴다.

### 규칙
- **다른 코드(P1, P3 등)는 절대 건드리지 않는다.**
- 로그는 한 줄, 사실 위주로. 상세 내용은 커밋 메시지에 남긴다.
- 막혔을 때는 `--blocker "<이유>"` 와 함께 `--state 블로커` 를 설정한다. 해소되면 `--blocker ""` 로 비운다.
- `PROJECTS.md` 는 자동 생성 파일이므로 직접 수정하지 않는다.
