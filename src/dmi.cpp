#include "micromag/dmi.hpp"
#include "micromag/grid.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// Shared finite-difference helpers (central diff, Neumann BC)
// ---------------------------------------------------------------------------

namespace {

// Central difference in x: (f(i+1) - f(i-1)) / (2*dx)
// At left boundary (i=0): one-sided forward (f(1)-f(0))/dx
// At right boundary (i=nx-1): one-sided backward (f(nx-1)-f(nx-2))/dx
inline Vec3 grad_x(const VectorField3D& m, const StructuredGrid& g,
                   Index ix, Index iy, Index iz) {
    const Index nx = g.nx();
    if (nx == 1) return {0, 0, 0};
    const Real  dx = g.dx();
    if (ix == 0) {
        return (m[g.linear_index(1,  iy, iz)] - m[g.linear_index(0, iy, iz)]) / dx;
    }
    if (ix == nx - 1) {
        return (m[g.linear_index(nx-1, iy, iz)] - m[g.linear_index(nx-2, iy, iz)]) / dx;
    }
    return (m[g.linear_index(ix+1, iy, iz)] - m[g.linear_index(ix-1, iy, iz)]) / (2.0 * dx);
}

inline Vec3 grad_y(const VectorField3D& m, const StructuredGrid& g,
                   Index ix, Index iy, Index iz) {
    const Index ny = g.ny();
    if (ny == 1) return {0, 0, 0};
    const Real  dy = g.dy();
    if (iy == 0) {
        return (m[g.linear_index(ix, 1,    iz)] - m[g.linear_index(ix, 0, iz)]) / dy;
    }
    if (iy == ny - 1) {
        return (m[g.linear_index(ix, ny-1, iz)] - m[g.linear_index(ix, ny-2, iz)]) / dy;
    }
    return (m[g.linear_index(ix, iy+1, iz)] - m[g.linear_index(ix, iy-1, iz)]) / (2.0 * dy);
}

inline Vec3 grad_z(const VectorField3D& m, const StructuredGrid& g,
                   Index ix, Index iy, Index iz) {
    const Index nz = g.nz();
    if (nz == 1) return {0, 0, 0};
    const Real  dz = g.dz();
    if (iz == 0) {
        return (m[g.linear_index(ix, iy, 1   )] - m[g.linear_index(ix, iy, 0)]) / dz;
    }
    if (iz == nz - 1) {
        return (m[g.linear_index(ix, iy, nz-1)] - m[g.linear_index(ix, iy, nz-2)]) / dz;
    }
    return (m[g.linear_index(ix, iy, iz+1)] - m[g.linear_index(ix, iy, iz-1)]) / (2.0 * dz);
}

}  // namespace

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
        Vec3 curl_m{
            gy.z - gz.y,
            gz.x - gx.z,
            gx.y - gy.x
        };

        H_out[idx] += curl_m * prefac;
    }
}

Real BulkDMIField::energy(const VectorField3D& m,
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
        Vec3 gz = grad_z(m, g, ix, iy, iz);

        Vec3 curl_m{
            gy.z - gz.y,
            gz.x - gx.z,
            gx.y - gy.x
        };

        E += m[idx].dot(curl_m);
    }

    return D_ * E * dV;
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

        // H_x =  ∂mz/∂x  (= gx.z)
        // H_y =  ∂mz/∂y  (= gy.z)
        // H_z = -(∂mx/∂x + ∂my/∂y)  (= -(gx.x + gy.y))
        Vec3 H_dmi{
             gx.z,
             gy.z,
            -(gx.x + gy.y)
        };

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

        // e = D [mz(∇·m) - m·∇mz]  in xy only
        // ∇·m (xy) = ∂mx/∂x + ∂my/∂y = gx.x + gy.y
        // m·∇mz    = mx*(∂mz/∂x) + my*(∂mz/∂y) = m[idx].x*gx.z + m[idx].y*gy.z
        Vec3 mi = m[idx];
        Real div_xy = gx.x + gy.y;
        Real m_dot_grad_mz = mi.x * gx.z + mi.y * gy.z;

        E += mi.z * div_xy - m_dot_grad_mz;
    }

    return D_ * E * dV;
}

}  // namespace micromag
