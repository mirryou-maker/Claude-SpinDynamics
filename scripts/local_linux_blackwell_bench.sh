#!/usr/bin/env bash
# =============================================================================
# Claude-SpinDynamics — LOCAL native-Linux Blackwell cross-solver runner.
#
# Purpose: on your own Linux box with the CAMPAIGN GPU (RTX 5060 Ti, Blackwell
# GB206, sm_120 — or any NVIDIA card), reproduce the headline f32/Blackwell
# competitive numbers on native Linux. This is the exact-silicon match the AWS
# L4 (Ada) run could not provide. See benchmarks/linux_competitive_claim.md.
#
# Steps: detect GPU/arch -> deps -> build CS f64 + f32 -> install mumax3 ->
#        GPU parity (vs Windows) -> cross-solver sweep (CS f32 vs mumax3 f32).
#
# Usage (from the repo root, on the Linux machine):
#     bash scripts/local_linux_blackwell_bench.sh
#
# Env overrides:
#     CUDA_ARCH=120        # RTX 50-series=120, B200=100, L4=89, A100=80 (auto-detected)
#     MUMAX_URL=<url>       # override the mumax3 linux tarball URL
#     REPO=$PWD             # repo root
#     SKIP_APT=1            # deps already installed
#     STEPS=300             # parity steps
# =============================================================================
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
STEPS="${STEPS:-300}"
BUILD64="$REPO/build/linux-gcc-cuda"
BUILD32="$REPO/build/linux-gcc-cuda-f32"
MUMAX_DIR="$HOME/mumax3"
# mumax3 3.12 linux (cuda12.9 build; JIT-PTX runs on Blackwell sm_120)
MUMAX_URL="${MUMAX_URL:-https://github.com/mumax/3/releases/download/v3.12/mumax3.12_linux_cuda12.9.tar.gz}"

echo "=============================================================="
echo " Claude-SpinDynamics — local Linux Blackwell cross-solver"
echo "   repo : $REPO"
echo "=============================================================="

# --- 0. GPU + arch detection ------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — install the NVIDIA driver first." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
if [ -z "${CUDA_ARCH:-}" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
    CUDA_ARCH="$CC"     # "12.0" -> "120", "8.9" -> "89"
fi
echo "  building for CUDA arch: sm_$CUDA_ARCH"

# --- 1. Toolchain check -----------------------------------------------------
export PATH="/usr/local/cuda/bin:$PATH"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Install the CUDA toolkit (>=12.8 for Blackwell sm_120)." >&2
    exit 1
fi
nvcc --version | tail -1
GCCMAJ=$(g++ -dumpversion | cut -d. -f1)
echo "  host g++: $(g++ --version | head -1)"
if [ "$CUDA_ARCH" = "120" ] || [ "$CUDA_ARCH" = "100" ]; then
    echo "  NOTE: Blackwell needs CUDA >= 12.8. If nvcc is older, sm_$CUDA_ARCH will fail."
fi
# CUDA<->GCC matching guard (avoids the GCC15 test-link issue etc.)
if [ "$GCCMAJ" -ge 15 ]; then
    echo "  WARN: GCC $GCCMAJ may be newer than your CUDA supports; if the build"
    echo "        fails, install g++-13 and re-run with"
    echo "        CMAKE_ARGS='-DCMAKE_CUDA_HOST_COMPILER=g++-13'."
fi

# --- 2. Deps ----------------------------------------------------------------
if [ "${SKIP_APT:-0}" != "1" ]; then
    echo "--- apt deps (sudo) ---"
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential cmake ninja-build \
        libfftw3-dev git python3-pip python3-dev curl
fi
echo "--- pip deps ---"
python3 -m pip install --user --quiet --break-system-packages "pybind11>=2.12,<4" numpy matplotlib || \
python3 -m pip install --user --quiet "pybind11>=2.12,<4" numpy matplotlib

# --- 3. Build CS f64 + f32 --------------------------------------------------
cd "$REPO"
echo "--- configure + build CS f64 (linux-gcc-cuda) ---"
cmake --preset linux-gcc-cuda -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" ${CMAKE_ARGS:-}
cmake --build "$BUILD64" --target _micromag -j"$(nproc)"
echo "--- configure + build CS f32 (linux-gcc-cuda-f32, tests off) ---"
cmake -S . -B "$BUILD32" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMICROMAG_USE_CUDA=ON -DMICROMAG_FLOAT32=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DMICROMAG_BUILD_TESTS=OFF ${CMAKE_ARGS:-}
cmake --build "$BUILD32" --target _micromag -j"$(nproc)"

# --- 4. Install mumax3 ------------------------------------------------------
MUMAX_BIN=$(find "$MUMAX_DIR" -name mumax3 -type f 2>/dev/null | head -1 || true)
if [ -z "$MUMAX_BIN" ]; then
    echo "--- downloading mumax3 ---"
    mkdir -p "$MUMAX_DIR"
    curl -sL -o "$MUMAX_DIR/mumax3.tar.gz" "$MUMAX_URL"
    tar xzf "$MUMAX_DIR/mumax3.tar.gz" -C "$MUMAX_DIR"
    MUMAX_BIN=$(find "$MUMAX_DIR" -name mumax3 -type f | head -1)
fi
echo "  mumax3: $MUMAX_BIN"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
"$MUMAX_BIN" -v 2>&1 | head -4 || { echo "ERROR: mumax3 failed to launch (check CUDA libs)." >&2; exit 1; }

# --- 5. Import check --------------------------------------------------------
PYTHONPATH="$BUILD32/python" python3 -c \
    "import micromag as mm; print('CS f32 cuda_available =', mm.cuda_available())"

# --- 6. GPU parity (CS Linux vs the Windows table) --------------------------
echo "--- GPU parity benchmark (f64) ---"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
PYTHONPATH="$BUILD64/python" python3 "$REPO/benchmarks/gpu_parity_bench.py" \
    --steps "$STEPS" --json "$REPO/benchmarks/_parity_gpu_linux_${GPU}.json" || true

# --- 7. Cross-solver sweep: CS f32 vs mumax3 f32 ----------------------------
echo "--- cross-solver sweep (this is the headline Blackwell-Linux run) ---"
python3 "$REPO/benchmarks/linux_crosssolver_bench.py" \
    --mumax "$MUMAX_BIN" \
    --cs-f64 "$BUILD64/python" --cs-f32 "$BUILD32/python" \
    --json "$REPO/benchmarks/_crosssolver_linux_${GPU}.json"

echo "--- optional large-grid extension ---"
python3 "$REPO/benchmarks/linux_crosssolver_bench.py" \
    --mumax "$MUMAX_BIN" --cs-f32 "$BUILD32/python" \
    --grids "L1M:256,256,16,3D L4M:512,512,16,3D" \
    --json "$REPO/benchmarks/_crosssolver_linux_${GPU}_large.json"

echo "=============================================================="
echo " DONE. Results:"
echo "   benchmarks/_parity_gpu_linux_${GPU}.json"
echo "   benchmarks/_crosssolver_linux_${GPU}.json  (+ _large)"
echo " Compare the CS_f32/mumax3 ratios against the Ada/L4 table in"
echo " benchmarks/linux_crosssolver_results.md — on Blackwell, CS's mid/large"
echo " position should improve (Tensor-Core FFT). Commit the JSONs + a results"
echo " section, then you have the headline claim on native Linux."
echo "=============================================================="
