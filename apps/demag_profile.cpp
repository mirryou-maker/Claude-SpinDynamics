// demag_profile.cpp — GPU demag phase-breakdown profiler using the GPU pipeline.
// Set MICROMAG_DEMAG_PROFILE=1 before running to get per-phase timing output.
// Uses RK4IntegratorGPU so that accumulate_gpu_ptr() is called each step.
#ifdef MICROMAG_CUDA

#include <cstdlib>
#include <iostream>
#include "micromag/demag_gpu.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/types.hpp"

using namespace micromag;

static void profile_grid(const char* label,
                          int nx, int ny, int nz,
                          double dx, double dy, double dz,
                          int n_steps)
{
    const StructuredGrid grid(nx, ny, nz, dx, dy, dz);
    const Material mat = Material::permalloy();

    std::cout << "\n--- " << label << " (" << nx << "x" << ny << "x" << nz
              << " = " << nx*ny*nz << " cells, padded "
              << 2*nx << "x" << 2*ny << "x" << (nz==1?1:2*nz) << ") ---\n"
              << std::flush;

    DemagFieldGPU    demag(grid);
    ExchangeFieldGPU exch(grid);
    ZeemanFieldGPU   zeeman(grid, Vec3{-24.6e3, 4.3e3, 0.0});

    VectorField3D m0(grid);
    m0.set_uniform({1.0, 0.1, 0.0}); m0.normalize();

    constexpr Real dt = 5e-14;
    RK4IntegratorGPU integ(grid, dt);
    integ.upload(m0);

    // With MICROMAG_DEMAG_PROFILE=1, do_capture falls back to direct execution
    // so accumulate_gpu_ptr is called each step (profiling events fire correctly).
    for (int k = 0; k < n_steps; ++k)
        integ.step(mat, demag, exch, zeeman);

    VectorField3D m_out(grid);
    integ.download(m_out);
    std::cout << "  done (" << n_steps << " steps)\n" << std::flush;
}

int main() {
    if (!std::getenv("MICROMAG_DEMAG_PROFILE"))
        std::cerr << "Warning: set MICROMAG_DEMAG_PROFILE=1 for phase timing output.\n";

    // n_steps chosen so profiler (every 20th accumulate_gpu_ptr call, 4 calls per
    // RK4 step) fires ~2 times per grid: 20 steps * 4 = 80 calls -> 4 reports
    profile_grid("SP#4",   200,  50,  1, 2.5e-9, 2.5e-9, 3.0e-9, 20);
    profile_grid("Medium", 200, 200,  5, 2.0e-9, 2.0e-9, 2.0e-9, 20);
    profile_grid("Large",  500, 500, 10, 2.0e-9, 2.0e-9, 2.0e-9, 10);
    return 0;
}

#else
#include <iostream>
int main() { std::cout << "Rebuild with -DMICROMAG_USE_CUDA=ON\n"; return 1; }
#endif
