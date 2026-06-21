// sp4_full_gpu_bench.cpp — G7: Full-GPU LLG benchmark
//
// Compares CPU RK4Integrator vs GPU RK4IntegratorGPU for two grid sizes:
//   SP#4   (200×50×1   =  10K cells)  — classic switching benchmark
//   Medium (200×200×5  = 200K cells)  — research-scale comparison
//
// Both integrators use a FIXED time step dt so the step count is identical
// and the comparison is apples-to-apples.
//
// Usage:  .\build\windows-msvc-cuda\bin\Release\sp4_full_gpu_bench.exe

#ifdef MICROMAG_CUDA

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

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

static double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static Vec3 mean_m(const VectorField3D& m) {
    Vec3 s{0,0,0};
    for (Index i=0; i<m.size(); ++i) { s.x+=m[i].x; s.y+=m[i].y; s.z+=m[i].z; }
    const double N = static_cast<double>(m.size());
    return {s.x/N, s.y/N, s.z/N};
}

// ---------------------------------------------------------------------------
static void benchmark(const char* label,
                       int nx, int ny, int nz,
                       double dx, double dy, double dz,
                       int n_steps, double dt,
                       const Vec3& Hext)
{
    const StructuredGrid grid(nx, ny, nz, dx, dy, dz);
    const Material mat = Material::permalloy();
    const size_t ncells = static_cast<size_t>(nx*ny*nz);

    // Initial state: SP#4-style saturated + small tilt
    VectorField3D m0(grid);
    m0.set_uniform({1.0, 0.1, 0.0});
    m0.normalize();

    std::cout << "\n=== " << label << " ===\n"
              << "Grid  : " << nx << "×" << ny << "×" << nz
              << " = " << ncells << " cells\n"
              << "Steps : " << n_steps << " × dt=" << dt << " s"
              << "  →  t_sim=" << n_steps*dt*1e9 << " ns\n";

    // ------------------------------------------------------------------
    // CPU benchmark
    // ------------------------------------------------------------------
    {
        EffectiveFieldSum heff;
        heff.add(std::make_shared<ExchangeField>());
        heff.add(std::make_shared<DemagField>(grid));
        heff.add(std::make_shared<ZeemanField>(Hext));

        VectorField3D m(grid);
        for (Index i=0; i<m.size(); ++i) m[i] = m0[i];

        // Warm-up: 1 step
        RK4Integrator cpu(dt);
        cpu.step(m, mat, heff);
        for (Index i=0; i<m.size(); ++i) m[i] = m0[i];

        auto t0 = Clock::now();
        for (int k=0; k<n_steps; ++k) cpu.step(m, mat, heff);
        const double wall = elapsed_s(t0);

        Vec3 avg = mean_m(m);
        std::cout << "CPU   : " << std::fixed << std::setprecision(2)
                  << wall << " s  ("
                  << std::setprecision(3) << wall*1000/n_steps << " ms/step)"
                  << "  <mx>=" << std::setprecision(5) << avg.x << "\n";
    }

    // ------------------------------------------------------------------
    // GPU benchmark
    // ------------------------------------------------------------------
    {
        DemagFieldGPU    demag(grid);
        ExchangeFieldGPU exch(grid);
        ZeemanFieldGPU   zeeman(grid, Hext);

        RK4IntegratorGPU gpu(grid, dt);
        gpu.upload(m0);

        // Warm-up: 1 step (cold cuFFT / kernel JIT)
        gpu.step(mat, demag, exch, zeeman);
        gpu.upload(m0);

        auto t0 = Clock::now();
        for (int k=0; k<n_steps; ++k)
            gpu.step(mat, demag, exch, zeeman);
        cudaDeviceSynchronize();
        const double wall = elapsed_s(t0);

        VectorField3D m_out(grid);
        gpu.download(m_out);
        Vec3 avg = mean_m(m_out);

        std::cout << "GPU   : " << std::fixed << std::setprecision(2)
                  << wall << " s  ("
                  << std::setprecision(3) << wall*1000/n_steps << " ms/step)"
                  << "  <mx>=" << std::setprecision(5) << avg.x << "\n";

        // Compute speedup from the line above (re-run CPU timing is cached from above scope)
        // — just print the per-step times; speedup visible from comparison
    }
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== G7: Full-GPU LLG benchmark (RK4IntegratorGPU) ===\n";
    std::cout << "Exchange + Demag + Zeeman, fixed-step RK4\n";

    const Vec3 Hext{-24.6e3, 4.3e3, 0.0};   // SP#4 Field A

    // SP#4 reference grid — 10K cells, 6000 steps = 0.3 ns
    // (capture the switching event at ~175 ps)
    benchmark("SP#4 reference  (200×50×1 = 10K cells)",
              200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9,
              6000, 5e-14, Hext);

    // Medium grid — 200K cells, 200 steps to measure per-step time
    benchmark("Medium grid  (200×200×5 = 200K cells)",
              200, 200, 5, 2.0e-9, 2.0e-9, 2.0e-9,
              200, 5e-14, Hext);

    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "Rebuild with -DMICROMAG_USE_CUDA=ON\n";
    return 1;
}
#endif
