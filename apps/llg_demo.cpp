#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/exchange.hpp"
#include "micromag/integrator.hpp"
#include "micromag/vtk_writer.hpp"

int main() {
    using namespace micromag;

    // 32 nm × 32 nm × 4 nm permalloy slab, 2 nm cells
    StructuredGrid grid(16, 16, 2, 2e-9, 2e-9, 2e-9);
    VectorField3D m(grid);
    m.set_vortex(16e-9, 16e-9, 4e-9);

    Material mat = Material::permalloy();
    mat.alpha = 0.5;  // high damping for fast relaxation demo

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 0}));
    heff.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));

    const Real dt         = 5e-13;   // 0.5 ps per step
    const int  n_steps    = 2000;
    const int  out_every  = 200;

    RK4Integrator rk4(dt);

    std::cout << "=== Phase 1c llg_demo ===\n";
    std::cout << "Grid: " << grid.nx() << "x" << grid.ny() << "x" << grid.nz()
              << "  (" << grid.size() << " cells)\n";
    std::cout << "alpha=" << mat.alpha << "  dt=" << dt << " s\n\n";
    std::cout << std::left
              << std::setw(8)  << "step"
              << std::setw(12) << "t [ps]"
              << std::setw(16) << "E [J]"
              << "<m_z>\n";

    write_vtk_legacy("llg_t0000.vtk", m, "m");

    for (int step = 0; step <= n_steps; ++step) {
        if (step % out_every == 0) {
            Real E    = heff.total_energy(m, mat);
            Real mz   = 0;
            for (Index i = 0; i < m.size(); ++i) mz += m[i].z;
            mz /= static_cast<Real>(m.size());

            std::cout << std::left
                      << std::setw(8)  << step
                      << std::setw(12) << step * dt * 1e12
                      << std::setw(16) << E
                      << mz << "\n";

            if (step > 0) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "llg_t%04d.vtk", step);
                write_vtk_legacy(buf, m, "m");
            }
        }
        if (step < n_steps) rk4.step(m, mat, heff);
    }
    std::cout << "\nDone. Open llg_t*.vtk files in ParaView to animate.\n";
    return 0;
}
