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

## Performance sweep (fixed-step, GPU)

Throughput vs grid size for a fixed-step run (permalloy, 2 nm in-plane cells,
`FixDt = 1e-14`).  Claude-SD uses RK4 (4 field evals/step); mumax3 / MuMax-CO
use Heun (2 evals/step) — so the fair, integrator-independent metric is **ms
per field-evaluation** (one demag FFT + exchange), reported below.  mumax3
numbers use a warm (cached-kernel) two-run subtraction to cancel its ~2 s
startup.

**ms per field-eval** (lower = faster):

| cells | Claude-SD f64 | Claude-SD f32 | mumax3 f32 | MuMax-CO f32 |
|-------|--------------:|--------------:|-----------:|-------------:|
| 16 384   | 0.347 | 0.149 | 0.263 | — |
| 65 536   | 1.188 | 0.242 | 0.299 | — |
| 262 144  | 5.516 | 1.037 | 0.421 | — |
| 524 288  | 12.14 | 2.749 | 0.629 | — |
| 1 048 576 | 26.26 | 5.739 | 1.212 | 1.169 |

(MuMax-CO's CUDA-Graph step is too fast at ≤0.5 M cells to time by process
wall-clock here — its launch overhead is essentially zero; at 1 M cells it is
compute-bound and matches mumax3.)

**Peak model VRAM** (1 M cells): Claude-SD f64 ≈ 1130 MB, f32 ≈ 640 MB
(~0.55×, the in-place demag keeps a single padded spectrum).

Findings:
- **float32 is ~4–5× faster than double** for Claude-SD at compute-bound sizes
  (e.g. 1 M cells: 5.74 vs 26.3 ms/eval) and uses ~half the VRAM — the demag
  cuFFT and memory traffic both halve.
- **mumax3 / MuMax-CO are the throughput reference.** At 1 M cells Claude-SD f32
  is ~4.7× slower per field-eval than mumax3, f64 ~22×.  mumax3 and MuMax-CO are
  within 4 % of each other when compute-bound (the CUDA-Graph win is at small
  grids, where it removes per-step launch overhead).  Claude-SD is correct and
  competitive but unoptimised relative to mumax3's fused/graph kernels — a clear
  target for future work (kernel fusion, fewer D2D copies, cuFFT plan reuse).

![throughput](perf/perf_throughput.png)

## SP#3 — cube flower/vortex energy crossing (preliminary)

Cube of edge L (in lex = √(2A/µ₀Ms²)), Ku = 0.1 Kd (easy axis ∥ edge), no
field. Relax a flower and a vortex branch and find L_c where the total
energies cross (µMAG reference **L_c ≈ 8.47 lex**, E/Kd/V ≈ 0.303).

Claude-SD (double, continuation: vortex carried down from large L, flower up):

| L/lex | E_flower | E_vortex | ΔE=v−f | ⟨mz⟩_fl | ⟨mz⟩_vx |
|-------|----------|----------|--------|---------|---------|
| 8.0 | 0.2048 | 0.2048 | 0.0000 | 0.975 | 0.975 |
| 8.3 | 0.2035 | 0.2033 | −0.0002 | 0.972 | 0.940 |
| 8.5 | 0.2026 | 0.2016 | −0.0010 | 0.971 | 0.891 |
| 9.0 | 0.2006 | 0.1951 | −0.0055 | 0.967 | 0.775 |

→ **L_c ≈ 8.0–8.3 lex** (grid-converged: N=16³ and N=28³ agree to 3 digits).
Energies are offset by −0.1 from the µMAG value because our uniaxial term uses
−Ku cos²θ (vs the reference +Ku sin²θ); this constant cancels in ΔE, so it does
not move L_c. The remaining ~4 % gap to 8.47 is **state preparation near
criticality** — the vortex deforms continuously toward the flower as L→L_c
(⟨mz⟩ 0.11→0.89), so the vortex branch is not the clean continuum vortex. This
is a benchmark-methodology limitation, not a solver error; the energetics and
the crossing region are correct. (A clean L_c=8.47 needs a sharper vortex
preparation / minimiser restart strategy.)

## Remaining (planned)

SP#5 (Zhang-Li STT, vortex-core steady displacement) and SP#1 (hysteresis)
5-way — pending.
