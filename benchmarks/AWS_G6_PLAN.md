# AWS g6.xlarge — GPU parity run plan (Phase 2)

Ready-to-execute plan for confirming the **Linux GPU build** matches the Windows
GPU numbers on identical CUDA source. Everything below is committed, so on the
instance you only `git pull` and run one script.

- **Instance:** `g6.xlarge` (NVIDIA L4, Ada `sm_89`, 24 GB) — spot recommended.
- **AMI:** latest Ubuntu "Deep Learning" AMI (ships driver + nvcc).
- **Runner:** [`scripts/aws_g6_gpu_parity.sh`](../scripts/aws_g6_gpu_parity.sh)
- **Harness:** [`gpu_parity_bench.py`](gpu_parity_bench.py)

## Before you launch (one-time account setup)
1. **Quota:** Service Quotas → EC2 → "Running On-Demand G and VT instances"
   ≥ 4 vCPUs (g6.xlarge = 4). New accounts default to 0 — request ahead of time.
2. **Key pair + security group** allowing inbound SSH (22) from your IP.
3. Pick a region with g6 capacity (us-east-1 / us-west-2 are safe).

## Launch → run (copy-paste)
```bash
# 1. SSH in
ssh -i my-key.pem ubuntu@<PUBLIC_IP>

# 2. Get the code (private repo — use a PAT or deploy key, or scp a tarball)
git clone https://github.com/mirryou-maker/Claude-SpinDynamics.git
cd Claude-SpinDynamics

# 3. One-shot: deps → build (CUDA sm_89) → GPU tests → parity bench
bash scripts/aws_g6_gpu_parity.sh
#    overrides if needed:  CUDA_ARCH=89  STEPS=300  SKIP_APT=1

# 4. Pull the result JSON back to your machine
#    (from your laptop)
scp -i my-key.pem \
  ubuntu@<PUBLIC_IP>:~/Claude-SpinDynamics/benchmarks/_parity_gpu_linux_*.json .
```

## What the script does
1. `nvidia-smi` sanity + records GPU name/driver/VRAM.
2. apt: `build-essential cmake ninja-build libfftw3-dev git python3-pip`.
3. Ensures `nvcc` on PATH (DLAMI: `/usr/local/cuda/bin`).
4. pip `--user`: `pybind11 numpy matplotlib`.
5. `cmake --preset linux-gcc-cuda -DCMAKE_CUDA_ARCHITECTURES=89` + build.
6. `unit_tests_gpu` (full GPU suite).
7. `import micromag; cuda_available()` check.
8. `gpu_parity_bench.py --steps 300 --json …` → SP#4 / Medium / Large ms/step
   with a Lin/Win ratio column.

## Reading the result
`gpu_parity_bench.py` prints `ms/step`, the Windows baseline (from CLAUDE.md:
SP#4 1.51, Medium 21.89, Large 290 ms/step), and their ratio.

| ratio (Lin/Win) | meaning |
|---|---|
| ≈ 0.8–1.2 | **PASS** — Linux GPU build on par; port healthy |
| L4 slower on Large | expected — L4 is a smaller GPU than the Windows dev card; compare *shape*/scaling, not just absolute |

**Caveats baked into the verdict:**
- This validates the **f64** path and that the CUDA build works end-to-end on
  Linux. It does **not** reproduce Blackwell-specific f32 Tensor-Core-FFT
  speedups — L4 has no Blackwell FFT path. For those, a `p6-b200` instance
  (`CUDA_ARCH=100`) would be needed; not required for a parity check.
- Absolute ms/step depends on the GPU model; the **ratio and the CPU→GPU
  crossover shape** are the portable conclusions.

## After the run
1. Add a "GPU parity" section to `benchmarks/linux_cpu_parity.md` with the L4
   numbers + verdict.
2. Commit the `_parity_gpu_linux_*.json` and the doc update.
3. **Terminate the instance** (spot: it stops billing on terminate).

## Cost
~$0.80/hr on-demand, ~$0.25–0.35/hr spot. The whole run (build + tests + bench)
is well under an hour → a few cents to ~$1.
