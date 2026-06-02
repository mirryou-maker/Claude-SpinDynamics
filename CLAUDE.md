# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```powershell
# Configure (only needed once, or after CMakeLists changes)
cmake --preset windows-msvc

# Build
cmake --build build/windows-msvc --config Release

# Run all tests
ctest --preset windows-msvc

# Run a specific test tag (e.g. demag tests only)
.\build\windows-msvc\bin\Release\unit_tests.exe "[demag]"

# Run a specific test by name substring
.\build\windows-msvc\bin\Release\unit_tests.exe "Demag: cube macrospin"
```

Debug build uses preset `windows-msvc-debug` and outputs to `build/windows-msvc-debug`.  
Toolchain: MSVC + vcpkg at `C:/vcpkg`, triplet `x64-windows`.  
Python venv: `.venv/`, compiled module lands at `build/windows-msvc/python/`.

## Architecture

### Layer model (bottom → top)

```
types.hpp          Vec3, Real, Index, physical constants
grid.hpp           StructuredGrid — cell geometry only
field.hpp          VectorField3D — owns data[], wraps grid
material.hpp       Material — Ms, A, K, alpha, easy_axis (static factories: permalloy/cobalt/iron)
effective_field.hpp  IEffectiveField (accumulate/energy/name) + EffectiveFieldSum (compositor)
spin_torque.hpp    ISpinTorque (accumulate/name) + SpinTorqueSum (compositor)
integrator.hpp     RK4Integrator — drives m forward using H_eff + spin torques
```

### Effective field implementations

Each `IEffectiveField` **adds** its contribution to `H_out` (never zeros it — the caller or `EffectiveFieldSum::compute` does that):

| Class | File | Physics |
|-------|------|---------|
| `ZeemanField` | zeeman.cpp | Uniform H_ext |
| `UniaxialAnisotropyField` | anisotropy.cpp | H_ani = (2K/μ₀Ms)(m·û)û |
| `ExchangeField` | exchange.cpp | 6-point Laplacian, Neumann or Periodic BC |
| `DemagField` | demag.cpp | FFT convolution with Newell tensor (Phase 2, currently buggy) |

### Spin torque implementations

Each `ISpinTorque` **adds** to `dm_out` in [1/s]:

| Class | Physics |
|-------|---------|
| `SlonczewskiSTT` | CPP-STT: a_J[m×(m×p̂)] + b_J[m×p̂] |
| `SpinOrbitTorque` | SOT (spin Hall): a_SOT[η_DL m×(m×σ̂) + η_FL (m×σ̂)] |

`SpinTorqueSum` is the compositor for spin torques, analogous to `EffectiveFieldSum`.

### Memory layout

`linear_index(i, j, k) = i + nx*(j + ny*k)` — **x is fastest**.  
This applies to both `VectorField3D` and all padded FFTW buffers in `demag.cpp`.  
FFTW plans are created with `(pad_nz, pad_ny, pad_nx)` argument order (C row-major: last index varies fastest = x-fastest matches our layout).

### LLG equation (Landau-Lifshitz form)

```
dm/dt = -γ'μ₀(m×H) - γ'αμ₀ m×(m×H),   γ' = γ₀/(1+α²),   H in A/m
```

`RK4Integrator::step` evaluates this at 4 RK4 stages, normalises `|m|=1` after each full step.  
Spin torques are added as extra `dm/dt` terms evaluated at each stage.

### Demag (Phase 2) — FFT convolution pipeline

`DemagField` uses zero-padding to 2N in each dimension (linear convolution via circular):

1. **Construction**: allocate scratch `r_buf_` (real, padded) and `c_buf_` (complex, r2c output), create forward/inverse FFTW plans, call `precompute_kernel()`.
2. **`precompute_kernel()`**: fills 6 kernel arrays `K_xx_…K_yz_` in frequency space using Newell (1993) analytical formulas (`nxx` uses `newell_f`, `nxy` uses `newell_g`). Diagonal components have even symmetry; off-diagonal have mixed parity (encoded by `sx/sy/sz` signs).
3. **`accumulate()`**: 3× forward FFT (Mx, My, Mz), 3× pointwise kernel product + IFFT, writes `H_demag = -N·M` into `H_out`.

Normalisation: FFTW is unnormalized, so divide by `pad_nx*pad_ny*pad_nz` after IFFT.

## SI conventions

- H in A/m, M = Ms·m (m is unit vector)
- `H_demag = -N·M` (negative sign, N is positive-definite demag tensor)
- Energy: `E = -μ₀/2 · Ms · Σ m·H_demag · dV`
- Newell tensor normalisation: `-1/(4π·dx·dy·dz)` applied inside `nxx`/`nxy`

## Response Size Constraints
- Never produce a single response large enough to hit the 32,000 output token limit.
- If a task requires a massive code change or a large GUI implementation, break it down into multiple smaller steps.
- Do not output the entire file if only a specific part needs modification.

## Test structure

All tests live in `tests/unit_tests` (single executable, Catch2 v3).  
Tags: `[demag]`, `[exchange]`, `[llg]`, `[spin_torque]`, `[zeeman]`, `[anisotropy]`, `[grid]`, `[field]`.
