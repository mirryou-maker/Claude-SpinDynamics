# Cross-solver benchmark

Configurations compared on µMAG standard problems:

| ID | tool | hardware | precision |
|----|------|----------|-----------|
| C1 | mumax3 3.11.1 | GPU (RTX) | float32 |
| C2 | OOMMF 1.2.1.0 | CPU (8 threads) | double |
| C3 | MuMax-CO (CUDA-Graph-optimised mumax3) | GPU | float32 |
| C4 | **Claude-SD** double (`windows-msvc-cuda`) | GPU | double |
| C5 | **Claude-SD** float32 (`windows-msvc-cuda-f32`) | GPU | float32 |

Each Claude-SD config is run with BOTH a fixed-step RK4 and the adaptive
RK45 (DOPRI5) GPU integrator.  mumax3/MuMax-CO use their RK45DP, OOMMF its
RKF54 — all adaptive at `MaxErr = 1e-5`.

## SP#4 — dynamic field-A reversal (500×125×3 nm, 250×64×1, 2 nm cells)

Trajectory ⟨m⟩(t) over 1 ns, sampled every 5 ps; OOMMF (double, adaptive)
taken as the accuracy reference. Each solver relaxes its own S-state first
(Claude-SD reaches ⟨m⟩ = (0.967, 0.126, 0) = mumax3).

| config | ⟨mx⟩(1ns) | ⟨my⟩(1ns) | ⟨mz⟩(1ns) | t_switch (ps) | mx RMS vs OOMMF | wall |
|--------|-----------|-----------|-----------|---------------|-----------------|------|
| mumax3 (adapt)        | −0.9845 | 0.1289 | 0.0428 | 135.7 | 0.0152 | ~16 s |
| MuMax-CO (adapt)      | −0.9845 | 0.1289 | 0.0428 | 135.7 | 0.0152 | ~14 s |
| OOMMF (adapt, ref)    | −0.9849 | 0.1230 | 0.0434 | 135.7 | —      | CPU   |
| Claude-SD f64 adapt   | −0.9851 | 0.1215 | 0.0434 | 135.7 | 0.0184 | 14 s  |
| Claude-SD f32 adapt   | −0.9849 | 0.1234 | 0.0433 | 135.7 | 0.0157 | **7 s** |
| Claude-SD f64 fixed   | −0.9840 | 0.1324 | 0.0429 | 135.7 | 0.0114 | 83 s  |
| Claude-SD f32 fixed   | −0.9840 | 0.1323 | 0.0429 | 135.7 | 0.0114 | 38 s  |

µMAG reference: ⟨mx⟩(1ns) ≈ −0.98, t_switch ≈ 0.14 ns.

**All seven configurations now agree to the mumax3↔OOMMF spread** (mx-RMS
0.011–0.018 over the full 1 ns, identical t_switch 135.7 ps). Both Claude-SD
precisions and both integrators match the references; the adaptive RK45 is
~6× faster than fixed RK4 (14 s vs 83 s at f64; 7 s vs 38 s at f32) with no
step-collapse (5.6k accepted / ~20 rejected steps), and float32 is ~2× faster
than double throughout.

**Root cause that was found & fixed (this is why the earlier draft showed
Claude-SD deviating).** The first benchmark draft had Claude-SD ending at
⟨mx⟩ = −0.968, t_switch 171 ps, mx-RMS 0.23, and the f32 RK45 collapsing. The
cause was a **race condition in the GPU effective-field summation**: every GPU
field (exchange, zeeman, …) runs on its own cuda stream and does
`d_H += H_field` in place; with no barrier between fields the read-modify-write
of `d_H` raced, so some cells lost a field's contribution — worst at high-field
boundary cells — injecting spurious numerical energy dissipation that slowed the
dynamics and broke the adaptive step controller. The physics was always correct
(demag verified against the **Aharoni exact** prism formula to 0.2%, exchange/
LLG to <0.1%; the CPU solver always matched mumax3 to 5 digits). Fixed by
serialising the field streams in `FieldSumGPU` and the Relax/Minimize/Heun field
assembly (commits 46766a2, e9dee9c). After the fix the GPU α=0 energy drift went
from −1.71 %/100 ps to −0.000 %, and RelaxGPU reaches the correct S-state.

**OOMMF MIF gotchas** (documented in `sp4/`): `multiplier [expr …]` inside a
`Specify` block and `Oxs_FileVectorField` both crash this OOMMF build with a
bare "child process exited abnormally", and any vector-field (`Oxs_Demag::Field`)
schedule also crashes it; the working SP#4 MIF uses a single 2-stage
`Oxs_UZeeman` (Hrange in A/m) + `Oxs_TimeDriver`.

![SP#4 trajectories](sp4/sp4_trajectories.png)

## Remaining (planned)

SP#1 (hysteresis), SP#3 (energetics), SP#5 (STT), and a dedicated fixed-step
performance sweep (ms/step / throughput / VRAM across the GPU configs) — pending.
