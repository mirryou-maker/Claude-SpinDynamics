#include "micromag/main_guard.hpp"
// SP#4 GPU benchmark (Phase 3 Step 6)
// Runs the SP#4 Field A scenario with DemagFieldGPU and compares
// accuracy + wall-clock time against the CPU DemagField.
//
// Build:  cmake --preset windows-msvc-cuda
//         cmake --build build/windows-msvc-cuda --config Release
// Run:    .\build\windows-msvc-cuda\bin\Release\sp4_gpu.exe

#ifdef MICROMAG_CUDA

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/exchange.hpp"
#include "micromag/demag.hpp"
#include "micromag/demag_gpu.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;
using Clock = std::chrono::steady_clock;

static Vec3 avg_m(const VectorField3D& m) {
    Real sx=0,sy=0,sz=0;
    for (Index i=0;i<m.size();++i){ sx+=m[i].x; sy+=m[i].y; sz+=m[i].z; }
    const Real N=static_cast<Real>(m.size());
    return {sx/N,sy/N,sz/N};
}

static void run(const char* label, bool use_gpu,
                const StructuredGrid& grid, const Material& mat,
                const Vec3& H_app, Real dt, int N_steps)
{
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(H_app));
    heff.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));
    if (use_gpu)
        heff.add(std::make_shared<DemagFieldGPU>(grid));
    else
        heff.add(std::make_shared<DemagField>(grid));

    constexpr Real tilt = 0.5 * constants::pi / 180.0;
    VectorField3D m(grid);
    for (Index i=0;i<m.size();++i) m[i] = {std::cos(tilt), std::sin(tilt), 0.0};

    RK4Integrator rk4(dt);
    bool switched = false; double t_sw = -1;

    auto t0 = Clock::now();
    for (int step = 0; step < N_steps; ++step) {
        rk4.step(m, mat, heff);
        if (!switched && avg_m(m).x < 0.0) {
            switched = true;
            t_sw = step * dt * 1e9;
        }
    }
    const double wall = std::chrono::duration<double>(Clock::now()-t0).count();

    Vec3 a = avg_m(m);
    std::cout << std::fixed << std::setprecision(5);
    std::cout << label << "\n";
    std::cout << "  t_switch  = " << (switched ? std::to_string(t_sw) : "none") << " ns\n";
    std::cout << "  <mx>_final= " << a.x << "  (ref ~ -0.9862)\n";
    std::cout << "  wall time = " << std::setprecision(1) << wall << " s\n\n";
}

static int run_main() {
    const StructuredGrid grid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9);
    const Material mat = Material::permalloy();

    constexpr Real B_mT  = 25e-3;
    constexpr Real theta = 170.0 * constants::pi / 180.0;
    const Vec3 H_app = {
        B_mT*std::cos(theta)/constants::mu_0,
        B_mT*std::sin(theta)/constants::mu_0, 0.0
    };
    constexpr Real dt     = 5e-14;   // 0.05 ps
    constexpr int  N_steps = 4000;   // 0.2 ns  (enough to see switching)

    std::cout << "=== SP#4 GPU vs CPU benchmark ===\n";
    std::cout << "Grid: 200x50x1  dt=" << dt*1e12 << " ps  steps=" << N_steps << "\n\n";

    run("CPU (DemagField, FFTW)",         false, grid, mat, H_app, dt, N_steps);
    run("GPU (DemagFieldGPU, cuFFT)",     true,  grid, mat, H_app, dt, N_steps);
    return 0;
}


MICROMAG_GUARDED_MAIN(run_main)

#else
#include <iostream>
int main() {
    std::cout << "Rebuild with -DMICROMAG_USE_CUDA=ON to enable GPU support.\n";
    return 1;
}
#endif
