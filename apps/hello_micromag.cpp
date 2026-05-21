#include <iostream>
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/vtk_writer.hpp"

int main() {
    using namespace micromag;

    // 64 nm x 64 nm x 8 nm, 2 nm cells -> 32 x 32 x 4
    StructuredGrid grid(32, 32, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(grid);

    Vec3 ext = grid.extent();
    m.set_vortex(ext.x * 0.5, ext.y * 0.5, /*core_radius=*/8e-9);

    write_vtk_legacy("vortex.vtk", m);

    std::cout << "Wrote vortex.vtk: "
              << grid.nx() << " x " << grid.ny() << " x " << grid.nz()
              << " cells (" << grid.size() << " total)\n";
    std::cout << "Open in ParaView to visualize.\n";
    return 0;
}
