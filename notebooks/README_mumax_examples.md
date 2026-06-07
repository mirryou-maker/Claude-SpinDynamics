# mumax3 Examples — Implementation Status

Reference: https://mumax.github.io/examples.html

Updated 2026-06-07 after Phase C1 (per-cell material parameters) — see `CLAUDE.md`
for the full class/architecture reference and the project memory for phase history.

## Implemented (notebook + library both ready)

| Notebook | mumax3 Example | Status | Notes |
|----------|---------------|--------|-------|
| `01_sp4_dynamics.py` | Standard Problem #4 | ✅ Full | CPU RK45, 0.4% vs µMAG |
| `02_sp1_phase_diagram.py` | Standard Problem #1 | ✅ Full | Vortex/S-state phase diagram, L_c = 115 nm |
| `03_thermal_sp4.py` | SP#4 at finite temperature | ✅ Full | SLLG (Heun + ThermalField) vs deterministic RK45 |
| `04_sp4_gpu.py` | Standard Problem #4 GPU | ✅ Full | GPU RK4, 7.4× vs CPU |
| `05_thermal_gpu.py` | SLLG on GPU | ✅ Full | HeunIntegratorGPU (cuRAND), CPU/GPU/deterministic comparison |
| `06_sp3_hysteresis.py` | Standard Problem #3 | ✅ Full | Hysteresis loop, H_sw ≈ −20 mT (10 nm cells) |
| `07_hysteresis.py` | Hysteresis | ✅ Full | 512×128 nm, α=0.5 replaces Minimize() |
| `08_stt_switching.py` | Slonczewski STT | ✅ Full | Rectangle (no ellipse), J_c sweep |
| `09_initial_magnetization.py` | Initial Magnetization | ✅ Partial | uniform/vortex/two-domain/random; no Skyrmion/VortexWall |

## Library ready — example notebook not yet written

These mumax3 examples were previously blocked on missing C++ features. Phases
A–C closed every one of those gaps at the library/Python-binding level; what
remains is writing a dedicated replication notebook for each (Todo_list.txt #3).

| mumax3 Example | Feature now available | Class / API | Added in |
|----------------|----------------------|-------------|----------|
| **Slicing Output** | Output cropping / component extraction | `VectorField3D::crop()`, `::component()`, `ScalarField3D`, `to_numpy_scalar()` | Phase A1 |
| **RKKY** | Inter-layer exchange coupling | `RKKYField(ref_m, J, d)` | Phase A2 |
| **PMA Racetrack** | Sliding simulation window + domain-wall tracking | `VectorField3D::shift_x/y()`, `DomainWallTracker` | Phase A3 |
| **Permalloy Racetrack** | Sliding window + notched geometry | Same as above + `GeomMask` (notch via `Rect`/boolean ops) | Phase A3 + B1 |
| **Geometry** | Shape API: primitives, booleans, transforms, mask-aware fields | `GeomMask`, `Ellipse`/`Circle`/`Rect`/`Cylinder`, `union`/`sub`/`intersect`, `translate`/`rotate`, `apply_mask`, mask-aware Neumann BC in `ExchangeField` | Phase B1 |
| **Magnetic Force Microscopy** | Fourier-space stray-field imaging | `MFMImage(grid, lift_nm, tip)` — monopole/dipole tip models | Phase B2 |
| **Regions** | Per-region material parameters | `MaterialField3D` (per-cell Ms/A/K/α/easy_axis) attached via `set_material_field()` to Exchange/Demag/Anisotropy/Integrators | Phase C1 |
| **Voronoi Tessellation** | Per-grain randomized anisotropy | `voronoi_grains(grid, n_grains, base, sigma_K, seed)` | Phase C1 |

### SP#2 (Standard Problem #2)
mumax3 SP#2 involves adaptive mesh sizing based on exchange length (`d/lex=30` criterion).
Our code uses fixed uniform grids, so this example cannot be directly replicated.
The underlying physics (equilibrium state of ellipsoidal Ni particle) could be approximated
with a cuboid geometry (`exchange_length()`/`optimal_dx()`/`sp2_grid()` utilities exist,
Phase A5), but the adaptive meshing aspect itself is not available.

## Not implemented — still needs library work (Phase D, lowest priority)

| mumax3 Example | Missing Feature | Would Need |
|----------------|----------------|-----------|
| **Rotating Cheese** | Dynamic geometry at runtime | Time-varying `GeomMask` recompute/apply each step (mostly a usage pattern on top of `GeomMask`, no deep changes — Phase D1) |
| **Spinning Hard Disk** | Moving write field + granular hard disk | Combines `shift_x/y` (A3) + `ZeemanFieldSpatial` (A4) + `GeomMask` (B1) + `MaterialField3D`/`voronoi_grains` (C1) — substantial integration work even though every prerequisite now exists (Phase D2) |

Both items require only Phase D-level integration work — every underlying
primitive (geometry masks, spatial Zeeman fields, sliding windows, per-cell
material/grains) has existed since the end of Phase C.
