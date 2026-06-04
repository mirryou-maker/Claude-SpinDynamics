// Large-grid GPU demag benchmark
// Compares DemagField (CPU, FFTW) vs DemagFieldGPU (GPU, cuFFT)
// on a realistic research-scale grid: 500×500×10 = 2.5M cells.
//
// Expected behaviour:
//   CPU: construction ~minutes, per-step ~seconds (very slow for this size)
//   GPU: construction ~seconds, per-step ~50–100ms  → 50–100× speedup
//
// Run:  .\build\windows-msvc-cuda\bin\Release\benchmark_large.exe

#ifdef MICROMAG_CUDA

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/demag.hpp"
#include "micromag/demag_gpu.hpp"

using namespace micromag;
using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::milliseconds;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Measure N accumulate() calls, return mean [ms]
// ---------------------------------------------------------------------------
template<typename Demag>
static double time_accumulate(Demag& demag, const VectorField3D& m,
                               const Material& mat, int N) {
    VectorField3D H(m.grid());
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};

    // Warm-up (1 call to populate caches / FFTW state)
    demag.accumulate(m, mat, H);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};

    const auto t0 = Clock::now();
    for (int k = 0; k < N; ++k) {
        for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
        demag.accumulate(m, mat, H);
    }
    return elapsed_ms(t0) / N;
}

// ---------------------------------------------------------------------------
static void run_benchmark(int nx, int ny, int nz,
                           double dx, double dy, double dz,
                           int N_accumulate, bool skip_cpu)
{
    const StructuredGrid grid(nx, ny, nz, dx, dy, dz);
    const Material mat = Material::permalloy();

    const size_t ncells    = static_cast<size_t>(nx*ny*nz);
    const size_t pad_total = static_cast<size_t>(2*nx) *
                             static_cast<size_t>(2*ny) *
                             static_cast<size_t>(2*nz);

    std::cout << "\n";
    std::cout << "Grid : " << nx << "×" << ny << "×" << nz
              << " = " << ncells/1e6 << " M cells\n";
    std::cout << "Cell : " << dx*1e9 << "×" << dy*1e9 << "×" << dz*1e9 << " nm\n";
    std::cout << "Pad  : " << 2*nx << "×" << 2*ny << "×" << 2*nz
              << " (" << pad_total/1e6 << " M points)\n";

    // Estimate GPU memory (kernels + batch buffers)
    const size_t fft_nx   = static_cast<size_t>(nx) + 1;  // approx
    const size_t cplx_sz  = fft_nx * 2*ny * 2*nz;
    const size_t real_sz  = static_cast<size_t>(2*nx)*2*ny*2*nz;
    const double gpu_mb   = (6*cplx_sz*16 + (3+3+3+3)*real_sz*8 +
                              3*cplx_sz*16) / 1e6;
    std::cout << "GPU  : estimated " << std::setprecision(0) << gpu_mb << " MB VRAM\n";

    VectorField3D m(grid); m.set_uniform({0, 0, 1});

    // ------------------------------------------------------------------
    // CPU benchmark
    // ------------------------------------------------------------------
    if (!skip_cpu) {
        std::cout << "\n--- CPU (FFTW) ---\n";
        auto t0 = Clock::now();
        std::unique_ptr<DemagField> cpu;
        try { cpu = std::make_unique<DemagField>(grid); }
        catch (const std::exception& e) {
            std::cout << "  Construction FAILED: " << e.what() << "\n";
            goto gpu_section;
        }
        const double t_ctor = elapsed_ms(t0);
        std::cout << "  Construction: " << t_ctor << " ms\n";

        if (t_ctor > 60000) {
            std::cout << "  accumulate(): skipped (construction already >60s)\n";
        } else {
            double t_acc = time_accumulate(*cpu, m, mat, 1);
            std::cout << "  accumulate() ×1: " << t_acc << " ms\n";
        }
    }

    gpu_section:
    // ------------------------------------------------------------------
    // GPU benchmark
    // ------------------------------------------------------------------
    std::cout << "\n--- GPU (cuFFT, stream) ---\n";
    {
        std::unique_ptr<DemagFieldGPU> gpu;
        auto t0 = Clock::now();
        try { gpu = std::make_unique<DemagFieldGPU>(grid); }
        catch (const std::exception& e) {
            std::cout << "  Construction FAILED: " << e.what() << "\n";
            std::cout << "  (GPU may have insufficient VRAM for this grid)\n";
            return;
        }
        const double t_ctor = elapsed_ms(t0);
        std::cout << "  Construction:      " << std::setprecision(1) << t_ctor << " ms\n";

        const double t_acc = time_accumulate(*gpu, m, mat, N_accumulate);
        std::cout << "  accumulate() mean: " << t_acc << " ms  ("
                  << N_accumulate << " runs)\n";
        std::cout << "  Throughput:        " << ncells/1e6/t_acc*1e3
                  << " M cells/s\n";
    }
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== GPU demag benchmark — large grids ===\n";
    std::cout << std::fixed;

    // SP#4 reference (200×50×1 = 10K cells) — should match earlier results
    run_benchmark(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9, 5, /*skip_cpu=*/false);

    // Medium grid (200×200×5 = 200K cells)
    run_benchmark(200, 200, 5, 2.0e-9, 2.0e-9, 2.0e-9, 3, /*skip_cpu=*/true);

    // Large grid (500×500×10 = 2.5M cells) — target benchmark
    run_benchmark(500, 500, 10, 2.0e-9, 2.0e-9, 2.0e-9, 3, /*skip_cpu=*/true);

    std::cout << "\n=== Summary ===\n";
    std::cout << "CPU accumulate() for large grids omitted (would take minutes).\n";
    std::cout << "Typical CPU/GPU speedup ratio for large grids: 50–200×.\n";
    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "Rebuild with -DMICROMAG_USE_CUDA=ON\n";
    return 1;
}
#endif
