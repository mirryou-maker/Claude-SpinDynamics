# Plan: matched-protocol skyrmion D–K phase diagram on iREMB

## Goal
The showcase skyrmion phase-diagram figure (`paper/showcase/figure/phase_cmp.png`)
compares **CS damped-LLG relax()** vs **mumax3 relax()**. Because the two use
different relaxation dynamics (CS = damped LLG, mumax3 relax() = energy descent),
the intermediate cells near D_c settle into different metastable basins — the
"basin-selection" difference discussed in the manuscript.

Add a **matched-protocol** panel where both codes use the SAME relaxation
philosophy, so the phase maps agree. Two equivalent ways:
- **Preferred**: CS `MinimizeGPU` (Barzilai–Borwein/FIRE energy minimiser, the
  direct analogue of mumax3's `Minimize()`/`relax()`) vs mumax3 `relax()`.
- Alternative: drive both with the same fixed-step damped LLG (mumax3
  `RunWhile`/`Run` with high alpha, no relax) — slower, less clean.

We already have `paper/showcase/figure/make_phase_cmp.py`; the matched variant
swaps `RelaxGPU` → `MinimizeGPU` on the CS side (a one-line change, cached in
`phase_cmp_cache.json`).

## Why iREMB (and the hard constraint)
iREMB GPUs are **Tesla P100 (CC 6.0, sm_60)** and **V100 (CC 7.0, sm_70)**.
The released v1.0.3 binaries target **sm_75…sm_120 + PTX** — so they will NOT
run on P100 or V100 ("no kernel image is available"). Running CS on iREMB
therefore requires a **CUDA rebuild that adds sm_70 (v100q) and/or sm_60
(P100q)**. This must be planned before any qsub.

Recommended queue: **v100q** (sm_70, Tensor Cores, currently empty). Single
node, `ncpus=20:ngpus=1`. The sweep is 8×8 = 64 relaxations per code — minutes
of GPU time, so walltime 00:30:00 is ample.

## Step-by-step (login node first — compute nodes have NO internet)

1. **On the login node** (`ssh iremb`):
   - `git -C /home/cyyou68/repos/Claude-SpinDynamics pull` (or clone).
   - Build the CUDA variant with sm_70 added:
     `cmake --preset linux-gcc-cuda -DMICROMAG_CUDA_ARCHS=70`
     (CMAKE toolkit on iREMB; verify `module load CUDA/12.x`). If a login-node
     GPU is unavailable for the build, build on a v100q interactive node.
   - Activate `sci` env: `. /home/cyyou68/activate_sci.sh` (matplotlib Agg,
     headless — the compute node has no display).
   - Stage a mumax3 Linux binary + the .mx3 phase scripts into
     `/scratch/cyyou68/phase_matched/`. (Check `module av mumax` first;
     mumax3 may not be installed — if not, copy the Windows-tested scripts and
     a Linux mumax3 build, or run the mumax3 side locally and only run CS on
     iREMB.)

2. **Copy** repo + built `_micromag*.so` + `benchmarks/` to
   `/scratch/cyyou68/phase_matched/` (run from scratch, never /home).

3. **qsub** `phase_matched.sh` (PBS Pro):
   ```sh
   #PBS -N phase_matched
   #PBS -q v100q
   #PBS -l select=1:ncpus=20:ngpus=1
   #PBS -l walltime=00:30:00
   #PBS -j oe
   #PBS -V
   . /etc/profile.d/modules.sh
   . /home/cyyou68/activate_sci.sh
   cd /scratch/cyyou68/phase_matched
   # PBS does not set CUDA_VISIBLE_DEVICES; pick a free GPU:
   FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | awk -F', ' '$2 < 100 {print $1; exit}')
   export CUDA_VISIBLE_DEVICES=$FREE
   py -3 benchmarks/make_phase_cmp_matched.py   # CS MinimizeGPU vs mumax3 relax
   ```

4. **Retrieve** `phase_cmp_matched.png` + `phase_cmp_matched_cache.json`,
   drop into `paper/showcase/figure/`, add as a showcase panel/section next to
   the existing (mismatched-protocol) figure.

## Deliverable
`make_phase_cmp_matched.py` (MinimizeGPU on CS side; otherwise identical to
make_phase_cmp.py) + the resulting `phase_cmp_matched.png` showing the two
codes' phase maps agreeing under matched relaxation, alongside the existing
figure that shows the basin difference under mismatched protocols.

## Note
This can also run LOCALLY (RTX 5060 Ti, ~10 min) since the matched variant is a
one-line change and the local binaries already run. iREMB is requested for
V100 validation / offload; if the sm_70 rebuild proves costly, run locally and
reserve iREMB for a larger sweep.
