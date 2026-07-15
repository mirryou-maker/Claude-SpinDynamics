#!/usr/bin/env bash
# =============================================================================
# Claude-SpinDynamics — AWS g6.xlarge (NVIDIA L4) GPU parity, one-shot runner.
#
# Purpose: on a fresh Ubuntu GPU instance, install deps, build the CUDA target,
# run the GPU test suite, and run the GPU parity benchmark — Phase 2 of the
# Linux benchmarking plan (see benchmarks/AWS_G6_PLAN.md).
#
# Usage (from the repo root, after `git clone` / `scp`):
#     bash scripts/aws_g6_gpu_parity.sh
#
# Env overrides:
#     CUDA_ARCH=89   # L4/L40S=89, A10G=86, T4=75, A100=80, H100=90, B200=100
#     REPO=$HOME/Claude-SpinDynamics
#     SKIP_APT=1      # skip apt (deps already present)
#     STEPS=300       # measured steps per grid
# =============================================================================
set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-89}"                 # g6.xlarge = L4 = sm_89
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
STEPS="${STEPS:-300}"
BUILD="$REPO/build/linux-gcc-cuda"

echo "=============================================================="
echo " Claude-SpinDynamics GPU parity runner"
echo "   repo      : $REPO"
echo "   cuda arch : sm_$CUDA_ARCH"
echo "=============================================================="

# --- 0. GPU sanity ----------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — no GPU driver. Use a GPU AMI (DLAMI)." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# --- 1. System deps ---------------------------------------------------------
if [ "${SKIP_APT:-0}" != "1" ]; then
    echo "--- installing apt deps (sudo) ---"
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential cmake ninja-build \
        libfftw3-dev git python3-pip python3-dev
fi

# --- 2. CUDA toolkit check --------------------------------------------------
# A DLAMI ships nvcc. The project targets CUDA 13.2 but builds with 12.x too.
if ! command -v nvcc >/dev/null 2>&1; then
    # DLAMI usually puts nvcc under /usr/local/cuda/bin
    export PATH="/usr/local/cuda/bin:$PATH"
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Install the CUDA toolkit or use a DLAMI." >&2
    echo "       (driver alone is not enough — need nvcc to compile.)" >&2
    exit 1
fi
echo "nvcc: $(nvcc --version | tail -1)"

# --- 3. Python deps (user site) ---------------------------------------------
echo "--- pip deps ---"
python3 -m pip install --user --quiet --break-system-packages \
    "pybind11>=2.12,<4" numpy matplotlib

# --- 4. Configure + build (CUDA) -------------------------------------------
echo "--- cmake configure (linux-gcc-cuda, sm_$CUDA_ARCH) ---"
cd "$REPO"
cmake --preset linux-gcc-cuda -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
echo "--- build (this compiles CUDA — a few minutes) ---"
cmake --build "$BUILD" -j"$(nproc)"

# --- 5. GPU unit tests ------------------------------------------------------
echo "--- GPU unit tests ---"
if [ -x "$BUILD/bin/unit_tests_gpu" ]; then
    "$BUILD/bin/unit_tests_gpu" || echo "WARN: some GPU tests failed (see above)"
else
    echo "WARN: unit_tests_gpu not built; skipping."
fi

# --- 6. Import check --------------------------------------------------------
echo "--- python import check ---"
PYTHONPATH="$BUILD/python" python3 -c \
    "import micromag as mm; print('cuda_available =', mm.cuda_available())"

# --- 7. GPU parity benchmark ------------------------------------------------
echo "--- GPU parity benchmark (steps=$STEPS) ---"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT="$REPO/benchmarks/_parity_gpu_linux_${GPU_NAME}.json"
python3 "$REPO/benchmarks/gpu_parity_bench.py" --steps "$STEPS" --json "$OUT"

echo "=============================================================="
echo " DONE. Raw JSON: $OUT"
echo " Compare ms/step vs the Windows GPU column (CLAUDE.md table)."
echo " ratio ~1.0 => Linux GPU build on par. Update"
echo " benchmarks/linux_cpu_parity.md with a GPU section, then commit."
echo "=============================================================="
