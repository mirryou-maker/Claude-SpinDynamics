#include "micromag/dmi.hpp"
#include "micromag/detail/grad_helpers.hpp"
#include "micromag/grid.hpp"

namespace micromag {

using detail::grad_x;
using detail::grad_y;
using detail::grad_z;

// ---------------------------------------------------------------------------
// BulkDMIField
// ---------------------------------------------------------------------------

void BulkDMIField::accumulate(const VectorField3D& m,
                               const Material& mat,
                               VectorField3D& H_out) const {
    if (D_ == 0.0) return;

    const StructuredGrid& g = m.grid();
    const Real prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 gz = grad_z(m, g, ix, iy, iz);

        // curl m = (∂mz/∂y - ∂my/∂z, ∂mx/∂z - ∂mz/∂x, ∂my/∂x - ∂mx/∂y)
        Vec3 curl_m{ gy.z - gz.y, gz.x - gx.z, gx.y - gy.x };
        H_out[idx] += curl_m * prefac;
    }
}

Real BulkDMIField::energy(const VectorField3D& m, const Material& mat) const {
    if (D_ == 0.0) return 0.0;

    const StructuredGrid& g = m.grid();
    const Real dV = g.cell_volume();
    Real E = 0.0;

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 gz = grad_z(m, g, ix, iy, iz);
        Vec3 curl_m{ gy.z - gz.y, gz.x - gx.z, gx.y - gy.x };
        E += m[idx].dot(curl_m);
    }
    return D_ * E * dV;
}

ScalarField3D BulkDMIField::energy_density(const VectorField3D& m,
                                            const Material& mat) const {
    ScalarField3D edens(m.grid());
    if (D_ == 0.0) return edens;

    const StructuredGrid& g = m.grid();
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 gz = grad_z(m, g, ix, iy, iz);
        Vec3 curl_m{ gy.z - gz.y, gz.x - gx.z, gx.y - gy.x };
        edens[idx] = D_ * m[idx].dot(curl_m);
    }
    return edens;
}

// ---------------------------------------------------------------------------
// InterfacialDMIField
// ---------------------------------------------------------------------------

void InterfacialDMIField::accumulate(const VectorField3D& m,
                                      const Material& mat,
                                      VectorField3D& H_out) const {
    if (D_ == 0.0) return;

    const StructuredGrid& g = m.grid();
    const Real prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 H_dmi{ gx.z, gy.z, -(gx.x + gy.y) };
        H_out[idx] += H_dmi * prefac;
    }
}

Real InterfacialDMIField::energy(const VectorField3D& m,
                                  const Material& mat) const {
    if (D_ == 0.0) return 0.0;

    const StructuredGrid& g = m.grid();
    const Real dV = g.cell_volume();
    Real E = 0.0;

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 mi = m[idx];
        Real div_xy = gx.x + gy.y;
        Real m_dot_grad_mz = mi.x * gx.z + mi.y * gy.z;
        E += mi.z * div_xy - m_dot_grad_mz;
    }
    return D_ * E * dV;
}

ScalarField3D InterfacialDMIField::energy_density(const VectorField3D& m,
                                                   const Material& mat) const {
    ScalarField3D edens(m.grid());
    if (D_ == 0.0) return edens;

    const StructuredGrid& g = m.grid();
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 mi = m[idx];
        Real div_xy = gx.x + gy.y;
        Real m_dot_grad_mz = mi.x * gx.z + mi.y * gy.z;
        edens[idx] = D_ * (mi.z * div_xy - m_dot_grad_mz);
    }
    return edens;
}

}  // namespace micromag
