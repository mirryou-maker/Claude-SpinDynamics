# benchmarks/ — index

Entry point for every performance/accuracy claim: **which claim → which document
→ which raw data → how to regenerate.**

## Claims → evidence

| # | Claim | Document | Raw data | Regenerate |
|---|-------|----------|----------|------------|
| 1 | Claude-SD matches µMAG/OOMMF/mumax3 to reference accuracy (SP#1/3/4/5, energies < 3e-4) | [`REPORT.md`](REPORT.md) | tables inline + `results/all_solvers.json` | `make_report.py` |
| 2 | CS f32 faster than mumax3 at small grids, competitive at large (Windows/Blackwell campaign) | [`REPORT.md`](REPORT.md), [`RESULTS_2026.md`](RESULTS_2026.md) | `results/all_solvers.json` | `run_throughput_cs.py`, `run_throughput_mumax.py`, `run_throughput_mumaxplus.py` |
| 3 | CS performance is platform-independent (Linux == Windows, CPU & GPU ~1 %) | [`linux_cpu_parity.md`](linux_cpu_parity.md) | `_parity_linux.json`, `_parity_windows.json`, `_parity_gpu_linux_NVIDIA_L4.json` | `cpu_parity_bench.py`, `gpu_parity_bench.py` |
| 4 | Small-grid win over mumax3 holds on **native Linux** (Ada/L4: 2.4× @ 10 K, par @ 65 K, ~1.1× @ 4.2 M) | [`linux_crosssolver_results.md`](linux_crosssolver_results.md) | `_crosssolver_linux_L4.json`, `_crosssolver_linux_L4_large.json` | `linux_crosssolver_bench.py` |
| 5 | What is (and is not yet) claimable on Linux — paper scoping | [`linux_competitive_claim.md`](linux_competitive_claim.md) | (analysis of #3+#4) | — |
| 6 | Comprehensive CS-vs-mumax3 advantages (capability superset, f64, methodology) | [`csd_vs_mumax_advantages.md`](csd_vs_mumax_advantages.md) | (synthesis) | — |
| 7 | Headline f32/**Blackwell** numbers on native Linux | *deferred* | — | `../scripts/local_linux_blackwell_bench.sh` (run on the RTX 5060 Ti under Linux) |

Planning/history: [`BENCHMARK_PLAN.md`](BENCHMARK_PLAN.md) (campaign design, fairness
rules), [`AWS_G6_PLAN.md`](AWS_G6_PLAN.md) (the L4 run procedure).

## Harness quick reference

```bash
# CPU parity (any OS; identical script both sides)
python benchmarks/cpu_parity_bench.py --threads 1,2,4,8 --json out.json

# GPU parity (CUDA build)
python benchmarks/gpu_parity_bench.py --json out.json

# Regression smoke: fail if >15% slower than a stored baseline (same machine!)
python benchmarks/cpu_parity_bench.py --baseline benchmarks/_parity_windows.json
python benchmarks/gpu_parity_bench.py --baseline benchmarks/_parity_gpu_<GPU>.json

# Cross-solver CS vs mumax3 (Linux, same host)
python3 benchmarks/linux_crosssolver_bench.py \
    --mumax <mumax3-binary> \
    --cs-f64 build/linux-gcc-cuda/python --cs-f32 build/linux-gcc-cuda-f32/python \
    --json out.json
```

Conventions: `_*.json` files are raw measurement samples (committed for
provenance; absolute numbers are hardware-specific — **ratios** are the portable
result). Baselines for the smoke check are only meaningful on the same machine.
