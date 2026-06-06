// llg_large_bench.cpp — Large-grid full LLG scaling benchmark
//
// Benchmarks RK4IntegratorGPU (full LLG: Exchange + Demag + Zeeman)
// across three grid sizes, showing GPU speedup vs CPU as cells scale:
//
//   SP#4    200×50×1   =   10K cells  (reference switching geometry)
//   Medium  200×200×5  =  200K cells  (research-scale micromagnetics)
//   Large   500×500×10 =  2.5M cells  (large-scale GPU target)
//
// For the Large grid, CPU accumulate is extrapolated from Medium
// (FFTW construction + each step would each take ~minutes).
//
// Usage:  .\build\windows-msvc-cuda\bin\Release\llg_large_bench.exe

#ifdef MICROMAG_CUDA

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "micromag/demag.hpp"
#include "micromag/demag_gpu.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/exchange.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/integrator.hpp"
#include "micromag/material.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/types.hpp"
#include "micromag/zeeman.hpp"

using namespace micromag;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static Vec3 mean_m(const VectorField3D& m) {
    Vec3 s{0, 0, 0};
    for (Index i = 0; i < m.size(); ++i) {
        s.x += m[i].x; s.y += m[i].y; s.z += m[i].z;
    }
    const double N = static_cast<double>(m.size());
    return {s.x / N, s.y / N, s.z / N};
}

// ---------------------------------------------------------------------------
struct BenchResult {
    const char* label;
    size_t      ncells;
    double      cpu_ms_step;   // 0 = skipped
    double      gpu_ms_step;
    double      speedup;       // cpu/gpu (0 if CPU skipped)
};

// ---------------------------------------------------------------------------
// CPU full-LLG: RK4Integrator + ExchangeField + DemagField + ZeemanField
// Returns ms/step, or -1.0 on construction failure.
// n_warmup: warm-up steps (excluded from timing)
// n_timed:  steps included in timing
// ---------------------------------------------------------------------------
static double run_cpu(const StructuredGrid& grid, const Material& mat,
                      const VectorField3D& m0, const Vec3& Hext,
                      double dt, int n_warmup, int n_timed)
{
    std::cout << "  [CPU] constructing DemagField... " << std::flush;
    std::unique_ptr<DemagField> demag;
    try {
        auto t0 = Clock::now();
        demag = std::make_unique<DemagField>(grid);
        std::cout << elapsed_ms(t0) << " ms\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")\n";
        return -1.0;
    }

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ExchangeField>());
    heff.add(std::move(demag));
    heff.add(std::make_shared<ZeemanField>(Hext));

    VectorField3D m(grid);
    for (Index i = 0; i < m.size(); ++i) m[i] = m0[i];

    RK4Integrator cpu(dt);

    // Warm-up
    for (int k = 0; k < n_warmup; ++k) cpu.step(m, mat, heff);
    for (Index i = 0; i < m.size(); ++i) m[i] = m0[i];

    const auto t0 = Clock::now();
    for (int k = 0; k < n_timed; ++k) cpu.step(m, mat, heff);
    const double wall_ms = elapsed_ms(t0);

    Vec3 avg = mean_m(m);
    const double ms_step = wall_ms / n_timed;
    std::cout << "  [CPU] " << n_timed << " steps: "
              << std::fixed << std::setprecision(1) << wall_ms << " ms"
              << "  → " << std::setprecision(3) << ms_step << " ms/step"
              << "  <mx>=" << std::setprecision(5) << avg.x << "\n";
    return ms_step;
}

// ---------------------------------------------------------------------------
// GPU full-LLG: RK4IntegratorGPU + ExchangeFieldGPU + DemagFieldGPU + ZeemanFieldGPU
// Returns ms/step.
// ---------------------------------------------------------------------------
static double run_gpu(const StructuredGrid& grid, const Material& mat,
                      const VectorField3D& m0, const Vec3& Hext,
                      double dt, int n_warmup, int n_timed)
{
    std::cout << "  [GPU] constructing DemagFieldGPU... " << std::flush;
    std::unique_ptr<DemagFieldGPU> demag;
    try {
        auto t0 = Clock::now();
        demag = std::make_unique<DemagFieldGPU>(grid);
        std::cout << elapsed_ms(t0) << " ms\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")\n";
        return -1.0;
    }

    ExchangeFieldGPU exch(grid);
    ZeemanFieldGPU   zeeman(grid, Hext);
    RK4IntegratorGPU gpu(grid, dt);
    gpu.upload(m0);

    // Warm-up (cold cuFFT plan + kernel JIT)
    for (int k = 0; k < n_warmup; ++k)
        gpu.step(mat, *demag, exch, zeeman);
    gpu.upload(m0);

    const auto t0 = Clock::now();
    for (int k = 0; k < n_timed; ++k)
        gpu.step(mat, *demag, exch, zeeman);
    const double wall_ms = elapsed_ms(t0);

    VectorField3D m_out(grid);
    gpu.download(m_out);
    Vec3 avg = mean_m(m_out);

    const double ms_step = wall_ms / n_timed;
    std::cout << "  [GPU] " << n_timed << " steps: "
              << std::fixed << std::setprecision(1) << wall_ms << " ms"
              << "  → " << std::setprecision(3) << ms_step << " ms/step"
              << "  <mx>=" << std::setprecision(5) << avg.x << "\n";
    return ms_step;
}

// ---------------------------------------------------------------------------
static BenchResult benchmark(const char* label,
                              int nx, int ny, int nz,
                              double dx, double dy, double dz,
                              double dt, const Vec3& Hext,
                              int n_cpu_warmup, int n_cpu_timed,
                              int n_gpu_warmup, int n_gpu_timed,
                              bool skip_cpu)
{
    const StructuredGrid grid(nx, ny, nz, dx, dy, dz);
    const Material mat = Material::permalloy();
    const size_t ncells = static_cast<size_t>(nx) * ny * nz;

    // Estimate GPU VRAM (rough: kernels + real/complex scratch)
    const size_t pnx = 2*nx, pny = 2*ny, pnz = 2*nz;
    const size_t cplx  = (pnx/2+1) * pny * pnz;
    const size_t real_ = pnx * pny * pnz;
    const double vram_mb = (6*cplx*16 + 6*real_*8 + 3*cplx*16) / 1.0e6;

    std::cout << "\n--- " << label << " ---\n"
              << "  Grid : " << nx << "x" << ny << "x" << nz
              << " = " << ncells / 1000.0 << " K cells"
              << "  (padded " << pnx << "x" << pny << "x" << pnz << ")\n"
              << "  VRAM : ~" << std::fixed << std::setprecision(0) << vram_mb << " MB\n";

    // Initial state: SP#4-style saturated in +x with small +y tilt
    VectorField3D m0(grid);
    m0.set_uniform({1.0, 0.1, 0.0});
    m0.normalize();

    double cpu_ms = 0.0;
    if (!skip_cpu) {
        cpu_ms = run_cpu(grid, mat, m0, Hext, dt, n_cpu_warmup, n_cpu_timed);
        if (cpu_ms < 0) cpu_ms = 0.0;
    } else {
        std::cout << "  [CPU] skipped (grid too large — estimated from scaling)\n";
    }

    double gpu_ms = run_gpu(grid, mat, m0, Hext, dt, n_gpu_warmup, n_gpu_timed);
    if (gpu_ms < 0) gpu_ms = 0.0;

    const double speedup = (cpu_ms > 0 && gpu_ms > 0) ? cpu_ms / gpu_ms : 0.0;
    if (speedup > 0)
        std::cout << "  Speedup: " << std::setprecision(1) << speedup << "x\n";

    return {label, ncells, cpu_ms, gpu_ms, speedup};
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== Large-grid full LLG scaling benchmark ===\n";
    std::cout << "Fields: Exchange + Demag + Zeeman   Integrator: RK4IntegratorGPU\n";
    std::cout << std::fixed;

    const Vec3 Hext{-24.6e3, 4.3e3, 0.0};  // SP#4 Field A (µMAG standard)

    std::vector<BenchResult> results;

    // SP#4 reference — 10K cells, 100 GPU steps + 10 CPU steps
    results.push_back(benchmark(
        "SP#4 ref  (200x50x1 = 10K cells)",
        200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9,
        5e-14, Hext,
        /*cpu warmup*/ 1, /*cpu timed*/ 10,
        /*gpu warmup*/ 1, /*gpu timed*/ 100,
        /*skip_cpu*/ false));

    // Medium — 200K cells, 50 GPU steps + 5 CPU steps
    results.push_back(benchmark(
        "Medium    (200x200x5 = 200K cells)",
        200, 200, 5, 2.0e-9, 2.0e-9, 2.0e-9,
        5e-14, Hext,
        /*cpu warmup*/ 1, /*cpu timed*/ 5,
        /*gpu warmup*/ 1, /*gpu timed*/ 50,
        /*skip_cpu*/ false));

    // Large — 2.5M cells, GPU only (CPU would take ~minutes per step)
    results.push_back(benchmark(
        "Large     (500x500x10 = 2.5M cells)",
        500, 500, 10, 2.0e-9, 2.0e-9, 2.0e-9,
        5e-14, Hext,
        /*cpu warmup*/ 0, /*cpu timed*/ 0,
        /*gpu warmup*/ 1, /*gpu timed*/ 20,
        /*skip_cpu*/ true));

    // -----------------------------------------------------------------------
    // Summary table
    // -----------------------------------------------------------------------
    std::cout << "\n";
    std::cout << "=== Summary: full LLG step time ===\n";
    std::cout << std::setw(38) << std::left << "Grid"
              << std::setw(12) << std::right << "Cells"
              << std::setw(16) << "CPU ms/step"
              << std::setw(16) << "GPU ms/step"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(94, '-') << "\n";

    double cpu_scale = 0.0;  // used to extrapolate Large CPU
    for (auto& r : results) {
        std::cout << std::setw(38) << std::left << r.label
                  << std::setw(12) << std::right << r.ncells;
        if (r.cpu_ms_step > 0) {
            std::cout << std::setw(16) << std::setprecision(2) << r.cpu_ms_step;
            cpu_scale = r.cpu_ms_step / static_cast<double>(r.ncells);
        } else {
            // Extrapolate: CPU cost scales roughly linearly with N (all fields O(N) except demag O(N logN))
            if (cpu_scale > 0) {
                const double est = cpu_scale * static_cast<double>(r.ncells);
                std::cout << std::setw(14) << std::setprecision(0) << est << "  (est)";
            } else {
                std::cout << std::setw(16) << "—";
            }
        }
        if (r.gpu_ms_step > 0) {
            std::cout << std::setw(16) << std::setprecision(2) << r.gpu_ms_step;
            if (r.speedup > 0)
                std::cout << std::setw(10) << std::setprecision(1) << r.speedup << "x";
            else if (cpu_scale > 0) {
                const double est_cpu = cpu_scale * static_cast<double>(r.ncells);
                std::cout << std::setw(9) << std::setprecision(1) << est_cpu / r.gpu_ms_step << "x (est)";
            }
        } else {
            std::cout << std::setw(16) << "FAILED";
        }
        std::cout << "\n";
    }
    std::cout << std::string(94, '-') << "\n";
    std::cout << "CPU: RK4Integrator (FFTW demag)   GPU: RK4IntegratorGPU (cuFFT demag)\n";
    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "Rebuild with -DMICROMAG_USE_CUDA=ON (cmake --preset windows-msvc-cuda)\n";
    return 1;
}
#endif
