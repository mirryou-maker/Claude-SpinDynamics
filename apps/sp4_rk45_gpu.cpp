// sp4_rk45_gpu.cpp — G9: GPU adaptive RK45 benchmark
//
// Compares three integrators on SP#4 Field A (200×50×1 = 10K cells):
//   CPU RK45          — adaptive DOPRI5 reference
//   GPU RK4 (fixed)   — fixed dt=5e-14 s, 6000 steps
//   GPU RK45 (adaptive) — DOPRI5 with same rtol/atol as CPU
//
// Expected: GPU RK45 takes fewer steps than GPU RK4 (same accuracy)
//           and is faster than CPU RK45 (GPU evaluates H_eff faster).
//
// Run: .\build\windows-msvc-cuda\bin\Release\sp4_rk45_gpu.exe

#ifdef MICROMAG_CUDA

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

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
#include "micromag/rk45_integrator_gpu.hpp"
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
    const double N=static_cast<double>(m.size());
    return {s.x/N, s.y/N, s.z/N};
}

// ---------------------------------------------------------------------------
int main() {
    const StructuredGrid grid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9);
    const Material mat = Material::permalloy();
    const Vec3 Hext{-24.6e3, 4.3e3, 0.0};   // SP#4 Field A

    VectorField3D m0(grid);
    m0.set_uniform({1.0, 0.1, 0.0});
    m0.normalize();

    const double t_end = 3e-10;   // 0.3 ns — captures the switching event

    std::cout << "=== G9: GPU RK45IntegratorGPU vs RK4 vs CPU RK45 ===\n"
              << "SP#4 Field A  (200x50x1 = 10K cells,  t_sim = "
              << t_end*1e9 << " ns)\n\n";
    std::cout << std::fixed;

    // -----------------------------------------------------------------------
    // 1. CPU RK45 — reference
    // -----------------------------------------------------------------------
    {
        EffectiveFieldSum heff;
        heff.add(std::make_shared<ExchangeField>());
        heff.add(std::make_shared<DemagField>(grid));
        heff.add(std::make_shared<ZeemanField>(Hext));

        VectorField3D m(grid);
        for (Index i=0; i<m.size(); ++i) m[i] = m0[i];
        RK45Integrator integ;

        // Warm-up
        integ.step(m, mat, heff);
        for (Index i=0; i<m.size(); ++i) m[i] = m0[i];

        double t=0.0; int n=0;
        auto t0 = Clock::now();
        while (t < t_end) { t += integ.step(m, mat, heff); ++n; }
        const double wall = elapsed_s(t0);
        Vec3 avg = mean_m(m);

        std::cout << "CPU RK45 (adaptive):\n"
                  << "  steps   = " << n << "  (accepted)\n"
                  << "  wall    = " << std::setprecision(2) << wall << " s"
                  << "  (" << std::setprecision(3) << wall*1000/n << " ms/step)\n"
                  << "  <mx>    = " << std::setprecision(5) << avg.x << "\n\n";
    }

    // Build shared GPU fields (used by both GPU benchmarks)
    DemagFieldGPU    demag(grid);
    ExchangeFieldGPU exch(grid);
    ZeemanFieldGPU   zeeman(grid, Hext);

    // -----------------------------------------------------------------------
    // 2. GPU RK4 (fixed step) — 6000 steps × 5e-14 s = 0.3 ns
    // -----------------------------------------------------------------------
    {
        const int n_steps = 6000;
        const double dt   = 5e-14;

        RK4IntegratorGPU gpu(grid, dt);
        gpu.upload(m0);
        gpu.step(mat, demag, exch, zeeman);    // warm-up
        gpu.upload(m0);

        auto t0 = Clock::now();
        for (int k=0; k<n_steps; ++k) gpu.step(mat, demag, exch, zeeman);
        const double wall = elapsed_s(t0);

        VectorField3D m(grid); gpu.download(m);
        Vec3 avg = mean_m(m);

        std::cout << "GPU RK4  (fixed dt=" << dt << " s):\n"
                  << "  steps   = " << n_steps << "\n"
                  << "  wall    = " << std::setprecision(2) << wall << " s"
                  << "  (" << std::setprecision(3) << wall*1000/n_steps << " ms/step)\n"
                  << "  <mx>    = " << std::setprecision(5) << avg.x << "\n\n";
    }

    // -----------------------------------------------------------------------
    // 3. GPU RK45 (adaptive DOPRI5)
    // -----------------------------------------------------------------------
    {
        RK45IntegratorGPU::Options opts;
        opts.rtol    = 1e-4;
        opts.atol    = 1e-6;
        opts.dt_init = 5e-14;
        opts.dt_max  = 1e-11;

        RK45IntegratorGPU gpu(grid, opts);
        gpu.upload(m0);

        // Warm-up (1 step)
        gpu.step(mat, demag, exch, zeeman);
        gpu.upload(m0);

        double t=0.0;
        auto t0 = Clock::now();
        while (t < t_end) t += gpu.step(mat, demag, exch, zeeman);
        const double wall = elapsed_s(t0);

        const int n_acc = gpu.n_accepted();
        const int n_rej = gpu.n_rejected();
        const int n_tot = n_acc + n_rej;

        VectorField3D m(grid); gpu.download(m);
        Vec3 avg = mean_m(m);

        std::cout << "GPU RK45 (adaptive DOPRI5):\n"
                  << "  accepted = " << n_acc
                  << "  rejected = " << n_rej
                  << "  total    = " << n_tot << "\n"
                  << "  wall     = " << std::setprecision(2) << wall << " s"
                  << "  (" << std::setprecision(3) << wall*1000/n_acc << " ms/accepted)\n"
                  << "  <mx>     = " << std::setprecision(5) << avg.x << "\n\n";

        // Speedup vs GPU RK4 (same step count as baseline)
        const int rk4_steps = 6000;
        const double rk4_ms_step = 1.4;   // from G7 benchmark
        const double rk4_equiv_wall = rk4_steps * rk4_ms_step / 1000.0;
        std::cout << "  vs GPU RK4 (6000 steps):  RK45 used "
                  << std::setprecision(1)
                  << 100.0 * n_acc / rk4_steps << "% as many accepted steps\n";
    }

    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "Rebuild with -DMICROMAG_USE_CUDA=ON (cmake --preset windows-msvc-cuda)\n";
    return 1;
}
#endif
