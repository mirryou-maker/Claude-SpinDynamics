#include <iostream>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/exchange.hpp"
#include "micromag/vtk_writer.hpp"

int main() {
    using namespace micromag;

    // 64 nm × 64 nm × 8 nm cobalt-like slab, 2 nm cells
    StructuredGrid grid(32, 32, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(grid), H(grid);

    Vec3 ext = grid.extent();
    m.set_vortex(ext.x * 0.5, ext.y * 0.5, 8e-9);

    Material mat = Material::cobalt();

    EffectiveFieldSum sum;
    sum.add(std::make_shared<ZeemanField>(Vec3{0, 0, 5e4}));
    sum.add(std::make_shared<UniaxialAnisotropyField>());
    sum.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));

    sum.compute(m, mat, H);

    write_vtk_legacy("vortex_m.vtk",  m, "m");
    write_vtk_legacy("vortex_H.vtk",  H, "H_eff");

    std::cout << "=== Phase 1b field_demo ===\n";
    std::cout << "Grid: " << grid.nx() << " x " << grid.ny() << " x " << grid.nz()
              << "  (" << grid.size() << " cells)\n";
    std::cout << "Material: cobalt  Ms=" << mat.Ms
              << "  A=" << mat.A_exchange
              << "  K=" << mat.K_uniaxial << "\n\n";
    for (const auto& term : sum.terms())
        std::cout << "  E[" << term->name() << "] = " << term->energy(m, mat) << " J\n";
    std::cout << "  E[total]          = " << sum.total_energy(m, mat) << " J\n\n";
    std::cout << "Wrote vortex_m.vtk and vortex_H.vtk\n";
    return 0;
}
