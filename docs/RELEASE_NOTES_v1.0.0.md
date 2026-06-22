# Claude-SpinDynamics v1.0.0 — Release Notes

First archival release accompanying the *npj Computational Materials* manuscript.

## Highlights
- **Dual precision** (float32 + float64) GPU micromagnetics — unique among the
  mumax family (all float32-only).
- **Two FFT demag backends**: cuFFT (CUDA-Graph replay, best on small grids) and
  VkFFT (best on large 3D).
- **Native physics**: Slonczewski STT, Spin-Orbit Torque, Zhang-Li, bulk &
  interfacial DMI, RKKY, magnetoelastic, surface anisotropy; per-cell materials,
  geometry masks, region maps.
- **Integrators**: RK4, RK45-DP (DOPRI5), Heun (SLLG), Relax/Minimize; with
  `recommend_integrator()` auto-selection.
- **Validation**: µMAG SP#1–5; cross-checked vs mumax3, mumax+, MuMax-CO (+OOMMF).
- **Reproducible benchmark suite** (`benchmarks/`) → one-command `make_report.py`.

## Verification
- 232 CPU + 113 GPU Catch2 tests; all four CUDA build variants green
  (cuFFT/VkFFT × f32/f64).

## Notable fix in this release
- **DMI GPU stream race**: `BulkDMIFieldGPU`/`InterfacialDMIFieldGPU` lacked a
  `set_stream` override, so in `FieldSumGPU` single-stream mode they accumulated
  into `d_H_out` on a separate stream concurrently with the other fields — a
  read-modify-write race that made damped-LLG relaxation of DMI textures
  non-deterministic near the skyrmion metastability boundary. Exposed by
  multi-build/cross-code validation and fixed (deterministic restored).

## Known limitations
- `RelaxGPU` (fixed-step damped LLG) converges slowly for stiff DMI skyrmion
  relaxation relative to adaptive minimizers; prefer `MinimizeGPU` or increase
  `max_steps` for topological-charge studies near phase boundaries.

## How to cite
See `CITATION.cff`. Archived on Zenodo (DOI assigned on release).
