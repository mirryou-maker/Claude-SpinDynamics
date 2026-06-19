#include "micromag/topological_charge.hpp"
#include "micromag/detail/grad_helpers.hpp"
#include <numbers>

namespace micromag {

namespace {
// Compute q(ix,iy,iz) = m · (dm_dx × dm_dy) using central differences (Neumann BC).
inline Real q_cell(const VectorField3D& m, const StructuredGrid& g,
                   Index ix, Index iy, Index iz) {
    const Vec3 gx = detail::grad_x(m, g, ix, iy, iz);
    const Vec3 gy = detail::grad_y(m, g, ix, iy, iz);
    return m[g.linear_index(ix, iy, iz)].dot(gx.cross(gy));
}
}  // anonymous namespace

ScalarField3D topological_charge_density(const VectorField3D& m) {
    const StructuredGrid& g = m.grid();
    ScalarField3D dens(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix)
        dens[g.linear_index(ix, iy, iz)] = q_cell(m, g, ix, iy, iz);
    return dens;
}

std::pair<Real, ScalarField3D> topological_charge(const VectorField3D& m) {
    const StructuredGrid& g = m.grid();
    ScalarField3D dens = topological_charge_density(m);
    const Real factor = g.dx() * g.dy() / (4.0 * std::numbers::pi);
    Real Q = 0.0;
    for (Index i = 0; i < dens.size(); ++i)
        Q += dens[i];
    return {Q * factor, dens};
}

Real topological_charge_Q(const VectorField3D& m) {
    return topological_charge(m).first;
}

}  // namespace micromag
