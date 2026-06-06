# NanoSpinDynamics — User Manual

**Version**: 1.0  |  **Toolchain**: MSVC 2022, CUDA 13.2, Python 3.13

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Physical Model](#2-physical-model)
3. [Numerical Methods](#3-numerical-methods)
4. [Installation and Build](#4-installation-and-build)
5. [Python API Reference](#5-python-api-reference)
6. [Simulation Workflow](#6-simulation-workflow)
7. [µMAG Validation](#7-µmag-validation)
8. [Performance Benchmarks](#8-performance-benchmarks)
9. [Example Gallery](#9-example-gallery)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Introduction

NanoSpinDynamics is a C++20/CUDA micromagnetic simulation library that solves
the Landau–Lifshitz–Gilbert (LLG) equation on three-dimensional structured
grids.  It is designed for research on nanomagnetic devices (spin-transfer
torque MRAM, domain-wall racetracks, thermal stability of magnetic bits) and
validated against the international µMAG standard problems.

### 1.1 Design Goals

- **Accuracy**: Validated against µMAG SP#1, SP#3, SP#4 reference results.
- **Performance**: GPU acceleration via CUDA gives 7–17× speedup over CPU;
  adaptive time-stepping (DOPRI5) gives a further 3–4× reduction in step count.
- **Usability**: Full Python bindings (pybind11 + NumPy) enable interactive
  exploration in Jupyter notebooks without sacrificing C++ performance.
- **Extensibility**: Clean layer model; new effective-field terms are added
  by subclassing `IEffectiveField`.

### 1.2 Comparison with Existing Codes

| Feature | NanoSpinDynamics | OOMMF | mumax³ |
|---------|-----------------|-------|--------|
| Language | C++20 / Python | C++ / Tcl | Go / CUDA |
| GPU acceleration | Yes (CUDA) | No | Yes (CUDA) |
| Adaptive integrator (GPU) | Yes (DOPRI5) | No | No |
| Periodic demag BC | Yes | Yes | Yes |
| Thermal (SLLG) on GPU | Yes | No | Yes |
| Python API | Yes | Partial | No |

---

## 2. Physical Model

### 2.1 Landau–Lifshitz–Gilbert Equation

The magnetization dynamics are governed by the LLG equation in
Landau–Lifshitz form:

```
dm/dt = -γ' μ₀ (m × H_eff) - γ' α μ₀ m × (m × H_eff)
```

where:
- **m** = M/Ms is the unit magnetization vector
- **H_eff** [A/m] is the effective field (sum of all contributions)
- **γ₀** = 1.7609×10¹¹ rad T⁻¹ s⁻¹ is the gyromagnetic ratio
- **γ'** = γ₀ / (1 + α²) is the reduced gyromagnetic ratio
- **α** is the dimensionless Gilbert damping parameter
- **μ₀** = 4π×10⁻⁷ T·m/A is the permeability of free space

The two terms represent precessional motion (first) and energy dissipation
toward the effective-field direction (second).

### 2.2 Effective Field Contributions

The total effective field is:

```
H_eff = H_exchange + H_demag + H_Zeeman + H_aniso [+ H_thermal]
```

#### 2.2.1 Exchange Field

```
H_exchange = (2A / μ₀ Ms) ∇²m
```

Discretized as a 6-point Laplacian on the structured grid (x-fastest layout):

```
∇²m ≈ [m(i+1) + m(i-1) + m(j+1) + ... - 6m(i,j,k)] / dx²
```

Boundary conditions: **Neumann** (∂m/∂n = 0 at surfaces, default) or
**Periodic** (wraps across cell boundaries).

Parameters: exchange stiffness **A** [J/m].

#### 2.2.2 Demagnetization Field (Open BC)

```
H_demag = -N · M = -N · (Ms m)
```

where **N** is the symmetric demagnetization tensor computed via the
Newell (1993) analytical formulas.  The convolution is evaluated in
frequency space using the FFT (FFTW on CPU, cuFFT on GPU):

1. Zero-pad magnetization to 2N in each dimension.
2. Forward FFT → pointwise multiplication with kernel → inverse FFT.
3. Divide by (2N)³ (FFTW unnormalized convention).

The zero-padding converts the circular convolution of the FFT into the
required linear (aperiodic) convolution.

#### 2.2.3 Demagnetization Field (Periodic BC)

For simulations with periodic boundary conditions in all three dimensions,
the periodic demag kernel is obtained by summing standard Newell tensors
over ±n_rep image cells per dimension:

```
N^periodic(r) = Σ_{n ∈ Z³, |n| ≤ n_rep} N^Newell(r + n·L)
```

The k = 0 Fourier mode is set to zero (toroidal convention: uniform
magnetization contributes no demagnetizing field).

#### 2.2.4 Zeeman Field

```
H_Zeeman = H_ext   (uniform applied field, A/m)
```

For spatially varying external fields, use `ZeemanFieldSpatial`
(to be implemented in Phase A4).

#### 2.2.5 Uniaxial Magnetocrystalline Anisotropy

```
H_aniso = (2K / μ₀ Ms) (m · û) û
```

where **K** [J/m³] is the anisotropy constant and **û** is the easy-axis
unit vector.

### 2.3 Spin Transfer Torque (Slonczewski)

Current-perpendicular-to-plane (CPP) spin-transfer torque:

```
τ_STT = a_J [m × (m × p̂)] + b_J [m × p̂]
```

where:
- **p̂** is the fixed-layer spin-polarization direction
- **a_J** = γ₀ ħ J P / (2 e Ms d)   [rad/s]  (damping-like)
- **b_J** = −β a_J                             (field-like)
- **J** [A/m²] is the current density (signed; J < 0 switches antiparallel → parallel)
- **P** ∈ [0, 1] is the spin polarization efficiency
- **d** [m] is the free-layer thickness

### 2.4 Spin-Orbit Torque

```
τ_SOT = a_SOT [ η_DL m×(m×σ̂) + η_FL (m×σ̂) ]
a_SOT = γ₀ ħ |J_c| |θ_SH| / (2 e Ms d_FM)
```

where **θ_SH** is the spin Hall angle and **σ̂** = ẑ × Ĵ_c is the
spin-polarization direction set by the heavy-metal geometry.

### 2.5 Stochastic LLG (Thermal Fluctuations)

The SLLG equation adds a Langevin noise term to model thermal fluctuations:

```
dm/dt = τ_LLG(m, H_eff + H_th)
H_th ~ N(0, σ),  σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V Δt))
```

The Heun (Stratonovich) scheme preserves the correct Boltzmann distribution
in thermal equilibrium.  The noise amplitude σ scales with temperature **T**,
damping **α**, cell volume **V**, and time step **Δt**.

---

## 3. Numerical Methods

### 3.1 Fixed-Step RK4 (CPU and GPU)

Classic 4-stage Runge–Kutta with fixed time step Δt:

```
k1 = f(m)
k2 = f(m + Δt/2 k1)
k3 = f(m + Δt/2 k2)
k4 = f(m + Δt k3)
m_new = m + (Δt/6)(k1 + 2k2 + 2k3 + k4)
```

followed by per-cell normalization |**m**| = 1.

Stability limit from the exchange field:
```
Δt_max ≈ 2.83 / (γ₀ × (2A/Ms) × (π/dx)²)
```

For Permalloy at dx = 5 nm: Δt_max ≈ 1.25 ps.

### 3.2 Adaptive DOPRI5 / RK45 (CPU and GPU)

Dormand–Prince embedded Runge–Kutta pair (5th/4th order) with
First-Same-As-Last (FSAL) property:

- 7 function evaluations per trial step; 6 effective per accepted step (FSAL).
- Error estimate: e = h Σᵢ eᵢ kᵢ  (Butcher error coefficients).
- Step control: h_new = h × clip(safety × (1/‖e‖)^0.2, fac_min, fac_max).
- Error norm: ‖e‖ = sqrt[(1/3N) Σ (eᵢ/scᵢ)²],  scᵢ = atol + rtol × max(|mᵢ|, |m̃ᵢ|).

GPU implementation: error reduction uses a block-parallel GPU kernel with
`atomicAdd`; one device-to-host copy of a single scalar per trial step
(negligible vs. H_eff evaluation cost).

Default tolerances: rtol = 10⁻⁴, atol = 10⁻⁶.

### 3.3 Stratonovich Heun (SLLG)

Predictor–corrector Heun scheme for the Stratonovich SDE:

```
Predictor:  m* = normalize(m + Δt f(m, H_eff + η))
Corrector:  m_new = normalize(m + (Δt/2)[f(m, H+η) + f(m*, H_eff(m*)+η)])
```

where η is the **same** noise sample in both evaluations (Stratonovich
convention).  Noise is generated per-step on GPU using cuRAND (normal
distribution).

### 3.4 Exchange-Stability Time Step Guideline

| dx (nm) | l_ex / dx | Δt_max (ps) | Recommended Δt (fs) |
|---------|----------|------------|---------------------|
| 10 | 0.57 | 5 011 | 50 |
| 5 | 1.14 | 1 253 | 50 |
| 3 | 1.90 | 451 | 18 |
| 2 | 2.84 | 200 | 8 |

(Permalloy: Ms = 800 kA/m, A = 13 pJ/m)

---

## 4. Installation and Build

### 4.1 Prerequisites

- Windows 11, MSVC 2022 (C++20)
- CMake ≥ 3.24
- vcpkg at `C:/vcpkg` (triplet `x64-windows`)
- Dependencies (installed via vcpkg): `fftw3`, `pybind11`, `catch2`
- GPU: CUDA 13.2 (`cmake --preset windows-msvc-cuda`)

### 4.2 Build Commands

```powershell
# CPU build
cmake --preset windows-msvc
cmake --build build/windows-msvc --config Release

# Run tests (87 tests)
.\build\windows-msvc\bin\Release\unit_tests.exe

# GPU build
cmake --preset windows-msvc-cuda
cmake --build build/windows-msvc-cuda --config Release

# Run GPU tests (53 tests)
.\build\windows-msvc-cuda\bin\Release\unit_tests_gpu.exe
```

### 4.3 Python Setup

```python
import sys, os

# CPU module
sys.path.insert(0, 'build/windows-msvc/python')

# GPU module (requires CUDA DLLs in path)
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')
sys.path.insert(0, 'build/windows-msvc-cuda/python')

import _micromag as mm
print(mm.cuda_available())   # True for GPU build
```

---

## 5. Python API Reference

### 5.1 Grid and Field

```python
# Structured uniform grid
grid = mm.StructuredGrid(nx, ny, nz, dx, dy, dz)   # dx,dy,dz in meters
grid.nx, grid.ny, grid.nz    # cell counts
grid.dx, grid.dy, grid.dz    # cell sizes [m]
grid.size                    # total cell count

# 3D vector field
m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(mx, my, mz))
m.set_vortex(cx, cy, core_radius)   # counter-clockwise in-plane vortex
m.normalize()                        # enforce |m| = 1 per cell

# NumPy conversion
arr = mm.to_numpy(m)        # returns (nz, ny, nx, 3) float64 array
mm.from_numpy(m, arr)       # copy numpy array into VectorField3D
mx, my, mz = mm.mean_magnetization(m)
```

### 5.2 Material

```python
mat = mm.Material.permalloy()   # Ms=800 kA/m, A=13 pJ/m, K=0, α=0.02
mat = mm.Material.cobalt()      # Ms=1400 kA/m, A=28 pJ/m, K=520 kJ/m³
mat = mm.Material.iron()        # Ms=1710 kA/m, A=20 pJ/m, K=48 kJ/m³

mat.Ms           # saturation magnetization [A/m]
mat.A_exchange   # exchange stiffness [J/m]
mat.K_uniaxial   # uniaxial anisotropy constant [J/m³]
mat.easy_axis    # easy-axis unit vector (Vec3)
mat.alpha        # Gilbert damping parameter
```

### 5.3 Effective Fields

```python
# Uniform external field
zeeman = mm.ZeemanField(mm.Vec3(Hx, Hy, Hz))   # H in A/m
zeeman.H_ext = mm.Vec3(Hx, Hy, Hz)              # update at runtime

# Exchange field
exch = mm.ExchangeField()                        # Neumann BC (default)
exch = mm.ExchangeField(mm.BoundaryCondition.Periodic)

# Demagnetization field (open BC, zero-padded)
demag = mm.DemagField(grid)                      # constructor is slow (FFTW)

# Demagnetization field (periodic BC, no padding)
demag_p = mm.DemagFieldPeriodic(grid, n_rep=2)   # n_rep: image cells per side

# Uniaxial anisotropy
aniso = mm.UniaxialAnisotropyField()             # uses mat.K_uniaxial, mat.easy_axis

# Composite effective field
heff = mm.EffectiveFieldSum()
heff.add(exch); heff.add(demag); heff.add(zeeman)
heff.total_energy(m, mat)    # total magnetostatic energy [J]
```

### 5.4 Integrators (CPU)

```python
# Fixed-step RK4
integ = mm.RK4Integrator(dt=1e-13)
integ.step(m, mat, heff)                 # advance by dt
integ.step(m, mat, heff, stt_sum)        # with spin torque

# Adaptive DOPRI5
opts = mm.RK45Options()
opts.rtol = 1e-4; opts.atol = 1e-6
opts.dt_init = 5e-14; opts.dt_max = 1e-11
integ = mm.RK45Integrator(opts)
dt = integ.step(m, mat, heff)            # returns actual dt taken

# Stratonovich Heun (SLLG, fixed Δt)
thermal = mm.ThermalField(grid, T_K=300.0, dt=1e-13, seed=42)
integ = mm.HeunIntegrator(dt=1e-13)
integ.step(m, mat, heff, thermal)
```

### 5.5 GPU Integrators

```python
# Fixed-step GPU RK4
integ = mm.RK4IntegratorGPU(grid, dt=5e-14)
integ.upload(m)                           # copy CPU → GPU (once)
for _ in range(N):
    integ.step(mat, demag_gpu, exch_gpu, zeeman_gpu)   # zero PCIe/step
integ.download(m)                         # copy GPU → CPU (for monitoring)

# Adaptive GPU DOPRI5
opts = mm.RK45GPUOptions()
opts.rtol = 1e-4; opts.dt_init = 5e-14
integ = mm.RK45IntegratorGPU(grid, opts)
integ.upload(m)
t = 0.0
while t < t_end:
    t += integ.step(mat, demag_gpu, exch_gpu, zeeman_gpu)
print(integ.n_accepted, integ.n_rejected)

# GPU SLLG (Stratonovich Heun + cuRAND)
integ = mm.HeunIntegratorGPU(grid, dt=1e-13, seed=42)
integ.upload(m)
integ.step(mat, demag_gpu, exch_gpu, zeeman_gpu, T_K=300.0)
```

### 5.6 GPU Field Drop-ins

```python
demag_gpu  = mm.DemagFieldGPU(grid)                  # cuFFT demag
exch_gpu   = mm.ExchangeFieldGPU(grid)               # CUDA Laplacian
zeeman_gpu = mm.ZeemanFieldGPU(grid, mm.Vec3(Hx,Hy,Hz))
aniso_gpu  = mm.UniaxialAnisotropyFieldGPU(grid)

zeeman_gpu.H_ext = mm.Vec3(Hx2, Hy2, Hz2)           # update at runtime
```

### 5.7 Spin Torques

```python
# Slonczewski STT (CPP geometry)
# J < 0: antiparallel(-p) → parallel(+p) switching
stt = mm.SlonczewskiSTT(J=-3e12, P=0.5, d=4e-9, p=mm.Vec3(1,0,0))
stt_sum = mm.SpinTorqueSum(); stt_sum.add(stt)
integ.step(m, mat, heff, stt_sum)         # with CPU integrator

# Spin-orbit torque (in-plane current, HM/FM bilayer)
sot = mm.SpinOrbitTorque(J_c=1e12, theta_SH=0.12, d_fm=3e-9,
                          sigma=mm.Vec3(0,1,0))   # Pt: θ_SH=0.12
```

---

## 6. Simulation Workflow

### 6.1 SP#4-Style Dynamics (Standard Pattern)

```python
import sys; sys.path.insert(0, 'build/windows-msvc/python')
import _micromag as mm
import numpy as np

# 1. Define geometry
grid = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
mat  = mm.Material.permalloy()

# 2. Build effective field
heff = mm.EffectiveFieldSum()
heff.add(mm.ExchangeField())
heff.add(mm.DemagField(grid))            # expensive constructor
heff.add(mm.ZeemanField(mm.Vec3(-24.6e3, 4.3e3, 0.0)))

# 3. Initialise magnetization
m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(1.0, 0.1, 0.0)); m.normalize()

# 4. Integrate
integ = mm.RK45Integrator()
t, ts, mxs = 0.0, [], []
while t < 1e-9:
    dt = integ.step(m, mat, heff)
    t += dt
    ts.append(t); mxs.append(mm.mean_magnetization(m)[0])
```

### 6.2 Hysteresis Loop

```python
zeeman = mm.ZeemanField()
heff   = mm.EffectiveFieldSum()
heff.add(mm.ExchangeField()); heff.add(mm.DemagField(grid)); heff.add(zeeman)

mat.alpha = 0.5    # overdamped: fast convergence to local minimum
m.set_uniform(mm.Vec3(1,0,0)); m.normalize()

mu0 = 4*np.pi*1e-7
for H_mT in np.arange(100, -105, -5):
    zeeman.H_ext = mm.Vec3(H_mT*1e-3/mu0, 0, 0)
    integ = mm.RK45Integrator(); t = 0
    while t < 2e-9: t += integ.step(m, mat, heff)
    mx = mm.mean_magnetization(m)[0]
```

### 6.3 GPU Simulation

```python
import os
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64')
sys.path.insert(0, 'build/windows-msvc-cuda/python')
import _micromag as mm

grid = mm.StructuredGrid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9)
mat  = mm.Material.permalloy()

demag  = mm.DemagFieldGPU(grid)
exch   = mm.ExchangeFieldGPU(grid)
zeeman = mm.ZeemanFieldGPU(grid, mm.Vec3(-24.6e3, 4.3e3, 0.0))

m = mm.VectorField3D(grid)
m.set_uniform(mm.Vec3(1.0, 0.1, 0.0)); m.normalize()

integ = mm.RK45IntegratorGPU(grid)     # adaptive DOPRI5 on GPU
integ.upload(m)
t = 0.0
while t < 1e-9:
    t += integ.step(mat, demag, exch, zeeman)
integ.download(m)
```

---

## 7. µMAG Validation

### 7.1 Standard Problem #4

**Geometry**: 500×125×3 nm Permalloy thin film (200×50×1 cells, 2.5 nm).  
**Protocol**: Start saturated in +x, apply Field A (H = (−24.6, 4.3, 0) kA/m) at t = 0.  
**Observable**: ⟨mx⟩(t), switching time t_switch.

| Method | ⟨mx⟩(1 ns) | t_switch | vs. µMAG |
|--------|-----------|---------|---------|
| CPU RK45 | −0.982 | 175 ps | **0.4 %** |
| GPU RK4 (dt=5×10⁻¹⁴ s) | −0.944 | ~125 ps | 4.2 %* |
| GPU RK45 (adaptive) | −0.982 | 175 ps | **0.4 %** |
| µMAG reference | −0.9862 | 174–176 ps | — |

*Fixed-step RK4 oscillates near equilibrium; RK45 eliminates this artifact.

### 7.2 Standard Problem #1

**Geometry**: Square Permalloy elements (L×L×10 nm, 5 nm cells, α = 0.5).  
**Protocol**: Relax from (a) uniform +x, (b) vortex initialization; record energy.  
**Observable**: Critical length L_c at which vortex becomes ground state.

| Quantity | NanoSpinDynamics | Literature |
|---------|-----------------|-----------|
| L_c (t = 10 nm) | 115 nm | 110–120 nm |
| L_c scaling | L_c ∝ t^{−0.78} | L_c ∝ t^{−0.5 to −1.0} |
| Vortex ground state (500 nm) | ✓ | ✓ |
| S-state ground state (200 nm) | ✓ | ✓ |

### 7.3 Standard Problem #3 (Hysteresis)

**Geometry**: 1 µm × 1 µm × 20 nm Permalloy (100×100×2 cells, 10 nm).  
**Protocol**: RK45 relax at each field; sweep +150 → −150 mT (5 mT steps).

| Quantity | NanoSpinDynamics | Expected range |
|---------|-----------------|----------------|
| Nucleation onset H_nuc | −10 mT | −10 to −20 mT |
| Switching field H_sw (⟨mx⟩ = 0) | −20 mT | −15 to −30 mT |
| Reversal complete H_ann | −25 mT | −20 to −40 mT |
| Hysteresis window ΔH | 15 mT | 10–30 mT |

*Note: 10 nm cells (l_ex/dx = 1.8) give qualitative results;
 5 nm cells on GPU are needed for quantitative µMAG SP#3 accuracy.*

### 7.4 Slonczewski STT Switching

**Geometry**: 160×80×4 nm Permalloy rectangle (40×20×1 cells, 4 nm).  
**Protocol**: J < 0 (antiparallel → parallel), P = 0.5, d = 4 nm, α = 0.01.

| Quantity | Value |
|---------|-------|
| Critical current density |J_c| | ~0.9×10¹² A/m² |
| Switching time (J = −3×10¹² A/m²) | < 1 ns |
| Final ⟨mx⟩ after switching | +0.997 |
| a_J at J_c | ~7.2×10⁸ rad/s ≈ 13 % of ω_precession |

---

## 8. Performance Benchmarks

All measurements on: Windows 11, MSVC 2022 Release, CUDA 13.2.

### 8.1 CPU vs. GPU Scaling

Full LLG step (Exchange + Demag + Zeeman, RK4, fixed Δt = 5×10⁻¹⁴ s):

| Grid | Cells | CPU (ms/step) | GPU (ms/step) | Speedup | VRAM |
|------|-------|--------------|--------------|---------|------|
| SP#4 (200×50×1) | 10 K | 10.49 | 1.51 | **7.0×** | 10 MB |
| Medium (200×200×5) | 200 K | 362.5 | 21.89 | **16.6×** | 193 MB |
| Large (500×500×10) | 2.5 M | ~4 531 (est.) | 290.1 | **~15.6×** | 2 403 MB |

### 8.2 Adaptive vs. Fixed-Step (GPU)

SP#4 Field A, 0.3 ns simulation (switching event at ~175 ps):

| Integrator | Steps | Wall time | ⟨mx⟩(0.3 ns) |
|-----------|-------|-----------|-------------|
| GPU RK4 (fixed Δt = 5×10⁻¹⁴ s) | 6 000 | 8.19 s | −0.694 |
| GPU RK45 (adaptive) | 1 047 acc. + 11 rej. | **2.21 s** | −0.693 |
| CPU RK45 (reference) | 1 046 | 16.45 s | −0.693 |

GPU RK45 is **3.7× faster than GPU RK4** (82.6 % fewer steps) and
**7.4× faster than CPU RK45**, with identical accuracy.

### 8.3 DemagField Construction Time

| Grid | Cells | CPU (FFTW) | GPU (cuFFT) |
|------|-------|-----------|------------|
| 200×50×1 | 10 K | 118 ms | 187 ms |
| 200×200×5 | 200 K | 2 489 ms | 166 ms |
| 500×500×10 | 2.5 M | — | 1 685 ms |

GPU construction includes precomputing the Newell kernel on GPU.

---

## 9. Example Gallery

### 9.1 Hysteresis Loop (notebooks/07_hysteresis.py)
Replication of the mumax3 "hysteresis" example.  
512×128×4 nm Permalloy strip, field swept +100 → −100 mT with 5 mT steps.  
Overdamped relaxation (α = 0.5) at each field step replaces mumax3's `Minimize()`.

Results: switching at H_sw ≈ −25 mT; |⟨mx⟩| = 0.96 at remanence (H = 0).

### 9.2 Slonczewski STT Switching (notebooks/08_stt_switching.py)
Replication of the mumax3 "Slonczewski" example.  
160×80×4 nm free layer, fixed layer along +x, J < 0 drives antiparallel → parallel.  
Critical current |J_c| ≈ 0.9×10¹² A/m²; complete switching within 1 ns for J = −3×10¹² A/m².

### 9.3 Initial Magnetization States (notebooks/09_initial_magnetization.py)
Replication of the mumax3 "Initial Magnetization" example.  
500×500×10 nm Permalloy, four initial states relaxed to local minima.

| State | Final energy (aJ) | Ground state? |
|-------|-----------------|---------------|
| Uniform +x | (higher) | No |
| Vortex CCW | (lowest) | **Yes** |
| Two-domain | (medium) | No |
| Random | (medium) | No |

### 9.4 Further Examples
See `notebooks/` for:
- `01_sp4_dynamics.py` — SP#4 CPU simulation + trajectory plot
- `02_sp1_phase_diagram.py` — SP#1 vortex nucleation phase diagram
- `03_thermal_sp4.py` — SP#4 at T = 300 K (SLLG, Heun integrator)
- `04_sp4_gpu.py` — SP#4 GPU simulation + CPU/GPU comparison
- `05_thermal_gpu.py` — GPU SLLG (HeunIntegratorGPU)
- `06_sp3_hysteresis.py` — SP#3 hysteresis loop analysis

---

## 10. Troubleshooting

### FFTW plan creation fails
Ensure the grid dimensions are reasonable (nx, ny, nz ≥ 2).  
FFTW_ESTIMATE is used (no overwrite during planning); FFTW_UNALIGNED
is set because `std::vector` alignment is not guaranteed.

### GPU simulation hangs / incorrect results
- Verify CUDA DLLs are in path: `os.add_dll_directory('...CUDA/v13.2/bin/x64')`
- Check VRAM: 200K cells needs ~200 MB; 2.5M cells needs ~2.4 GB.
- `cudaDeviceSynchronize()` errors indicate VRAM exhaustion or kernel launch failure.

### RK45 step size reaches minimum
The solution is too stiff for the current settings.  
- Reduce `atol`/`rtol` to allow larger error tolerance.
- Reduce `dt_max` if the exchange field causes instability at large steps.
- With α ≥ 0.5 (overdamped), `dt_max = 5×10⁻¹²` s works reliably.

### Hysteresis loop does not show switching
- Ensure α is large enough (α = 0.5 for quasi-static Minimize equivalent).
- Increase `t_max` per field step (2 ns is usually sufficient for α = 0.5).
- Check that the initial state is saturated at the starting field.

### STT switching not observed
- Check current sign: J < 0 switches antiparallel → parallel.
- Verify |J| > |J_c|: for 160×80×4 nm Permalloy with α = 0.01, |J_c| ≈ 0.9×10¹² A/m².
- Increase simulation time: with low α (0.01), switching via precessional dynamics takes longer.
