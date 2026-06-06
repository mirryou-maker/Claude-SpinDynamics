# mumax3 Examples — Implementation Status

Reference: https://mumax.github.io/examples.html

## Implemented

| Notebook | mumax3 Example | Status | Notes |
|----------|---------------|--------|-------|
| `01_sp4_dynamics.py` | Standard Problem #4 | ✅ Full | CPU RK45, 0.4% vs µMAG |
| `04_sp4_gpu.py` | Standard Problem #4 GPU | ✅ Full | GPU RK4, 7.4× vs CPU |
| `07_hysteresis.py` | Hysteresis | ✅ Full | 512×128 nm, α=0.5 replaces Minimize() |
| `08_stt_switching.py` | Slonczewski STT | ✅ Full | Rectangle (no ellipse), J_c sweep |
| `09_initial_magnetization.py` | Initial Magnetization | ✅ Partial | uniform/vortex/two-domain/random; no Skyrmion/VortexWall |

## Not Implemented — Missing Features

| mumax3 Example | Missing Feature | Would Need |
|----------------|----------------|-----------|
| **Geometry** | Shape API (cylinders, ellipsoids, boolean ops) | `SetGeom()`, `Ellipse()`, `Rect()` etc. |
| **Rotating Cheese** | Dynamic geometry at runtime | `SetGeom()` with time-varying function |
| **Regions** | Per-region material parameters | `defRegion()`, per-cell Ms/Ku/Aex |
| **Slicing Output** | Output cropping/component filtering | `Crop()`, `CropY()`, `Comp()` |
| **Magnetic Force Microscopy** | MFM stray field calculation | Tip model, near-field integration |
| **PMA Racetrack** | Sliding simulation window | `ext_centerWall()`, infinite-wire BC |
| **Permalloy Racetrack** | Sliding window + notched geometry | Same + shape API for notch |
| **Voronoi Tessellation** | Per-grain material regions | `ext_makegrains()`, spatial Ms/Ku variation |
| **RKKY** | Inter-layer exchange coupling | Layer-specific exchange tensor |
| **Spinning Hard Disk** | Moving write field + granular hard disk | `Shift()`, PMA grains, time-varying geometry |

### SP#2 (Standard Problem #2)
mumax3 SP#2 involves adaptive mesh sizing based on exchange length (`d/lex=30` criterion).
Our code uses fixed uniform grids, so this example cannot be directly replicated.
The underlying physics (equilibrium state of ellipsoidal Ni particle) could be approximated
with a cuboid geometry, but the adaptive meshing aspect is not available.

## Key Design Gap: Geometry API

Most of the unimplementable examples require a geometry API that allows defining
non-rectangular simulation domains. mumax3 has:
- Primitive shapes: `Rect`, `Circle`, `Ellipse`, `Cylinder`, `Sphere`, `Cone`
- Boolean operations: `Add`, `Sub`, `Intersect`, `Xor`
- Transformations: `Translate`, `Rotate`, `Scale`

To implement these, we would need a per-cell mask (0/1 occupancy), and the effective-field
kernels would need to respect the mask (zero Ms for empty cells).

## Key Design Gap: Sliding Window

The racetrack examples use `ext_centerWall()` to keep the domain wall centred as it
propagates under STT drive. This requires:
- Periodic shifting of the entire magnetization array
- Tracking a spatial feature (wall position) dynamically

## Key Design Gap: Per-Cell Material Parameters

The Regions and Voronoi examples require spatially-varying material parameters.
Currently all cells share a single `Material` struct (Ms, A, K, alpha, easy_axis).
Per-cell material maps would need a `MaterialField3D` analogous to `VectorField3D`.
