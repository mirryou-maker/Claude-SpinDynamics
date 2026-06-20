# Cross-solver benchmark — 5-way

Five configurations compared on µMAG standard problems:

| ID | tool | hardware | precision | input |
|----|------|----------|-----------|-------|
| C1 | mumax3 3.11.1 | GPU (RTX) | float32 | `.mx3` |
| C2 | OOMMF 1.2.1.0 | CPU (8 threads) | double | `.mif` |
| C3 | ours float32 (`windows-msvc-cuda-f32`) | GPU | float32 | `.mx3` |
| C4 | ours double (`windows-msvc-cuda`) | GPU | double | `.mx3` |
| C5 | MuMax-CO (CUDA-Graph-optimised mumax3) | GPU | float32 | `.mx3` |

All four `.mx3` configs run the *identical* script (our runner is mumax3-compatible);
only OOMMF uses a hand-written `.mif`.

## SP#4 — dynamic field-A reversal (500×125×3 nm, 250×64×1, 2 nm cells)

Adaptive RK45 (Dormand–Prince), `MaxErr = 1e-5`. Trajectory ⟨m⟩(t) over 1 ns,
sampled every 5 ps; OOMMF (double) taken as the accuracy reference.

| config | ⟨mx⟩(1ns) | ⟨my⟩(1ns) | ⟨mz⟩(1ns) | t_switch (ps) | mx RMS vs OOMMF | my RMS |
|--------|-----------|-----------|-----------|---------------|-----------------|--------|
| mumax3      | −0.9845 | 0.1289 | 0.0428 | 135.7 | 0.0152 | 0.0204 |
| MuMax-CO    | −0.9845 | 0.1289 | 0.0428 | 135.7 | 0.0152 | 0.0204 |
| OOMMF (ref) | −0.9849 | 0.1230 | 0.0434 | 135.7 | —      | —      |
| ours-double | −0.9678 | 0.0115 | −0.0120 | 170.9 | 0.2275 | 0.2902 |
| ours-f32    | _step-collapse — see below_ | | | | | |

µMAG reference: ⟨mx⟩(1ns) ≈ −0.98, t_switch ≈ 0.14 ns.

ours-double wall ≈ 610 s vs mumax3 ≈ 16 s — the adaptive RK45 takes many small
steps on the stiff 2 nm mesh. **ours-f32 did not finish:** with `MaxErr = 1e-5`
the relative tolerance is below the float32 epsilon (~1e-7), so the DOPRI5 step
controller collapses near the fast switching event (it reached only ~0.6 ns
after far longer than the double run and then stalled). This is a genuine
benchmark finding: **our adaptive RK45 is impractical in float32 at tol 1e-5** —
an f32 run needs a looser tolerance (e.g. `MaxErr = 1e-4`) or fixed-step
integration. (The fixed-step performance sweep below is the apples-to-apples
float32 comparison.)

**Findings (SP#4):**
- **mumax3, MuMax-CO and OOMMF agree tightly.** mumax3 and its CUDA-Graph
  optimisation MuMax-CO are bit-identical; both match the double-precision OOMMF
  reference to mx-RMS ≈ 0.015 over the whole 1 ns, with the same t_switch
  (135.7 ps). This validates the harness and the float32 ≈ double agreement for
  mumax3.
- **Our solver reaches a switched state but its trajectory deviates.**
  ours-double ends at ⟨mx⟩ = −0.968 (cf. −0.985) but switches later
  (171 ps vs 136 ps) and the ⟨my⟩ path is off (mx-RMS ≈ 0.23). This is a real
  finding to investigate — likely a demag-accuracy / S-state-chirality / RK45
  step-control difference rather than a precision effect.

**Robustness issues found & fixed in our `.mx3` runner during this benchmark:**
- The runner used a fixed relax/RK4 timestep tuned for ~5 nm cells; on the 2 nm
  SP#4 mesh the exchange-limited rate (∝ 1/dx²) is ~6× higher, so RelaxGPU
  diverged to a garbage S-state and RK45 stalled. Added a cell-size-aware
  `_safe_dt()` and wired `MaxErr` into the RK45 options (commit 2379cfd).
- OOMMF MIF gotchas (documented in `sp4/`): `multiplier [expr …]` inside a
  `Specify` block and `Oxs_FileVectorField` both crash this OOMMF build with a
  bare "child process exited abnormally"; the working SP#4 MIF uses a single
  2-stage `Oxs_UZeeman` (Hrange in A/m) + `Oxs_TimeDriver`.

## Remaining (planned)

SP#1 (hysteresis), SP#3 (energetics), SP#5 (STT), and the fixed-step performance
sweep (wall / ms-step / throughput / VRAM across the GPU configs) — pending.
