#pragma once
// Shared central-difference gradient helpers with Neumann BC.
// Included by dmi.cpp, exchange.cpp, init_mag.cpp, etc.

#include "micromag/field.hpp"
#include "micromag/grid.hpp"

namespace micromag::detail {

// Central diff in x: (m[i+1]-m[i-1])/(2dx).
// Boundary: one-sided (m[1]-m[0])/dx or (m[n-1]-m[n-2])/dx.
inline Vec3 grad_x(const VectorField3D& m, const StructuredGrid& g,
                   Index ix, Index iy, Index iz) {
    const Index nx = g.nx();
    if (nx == 1) return {0, 0, 0};
    const Real dx = g.dx();
    if (ix == 0)
        return (m[g.linear_index(1, iy, iz)] - m[g.linear_index(0, iy, iz)]) / dx;
    if (ix == nx - 1)
        return (m[g.linear_index(nx-1, iy, iz)] - m[g.linear_index(nx-2, iy, iz)]) / dx;
    return (m[g.linear_index(ix+1, iy, iz)] - m[g.linear_index(ix-1, iy, iz)]) / (2.0 * dx);
}

inline Vec3 grad_y(const VectorField3D& m, const StructuredGrid& g,
                   Index ix, Index iy, Index iz) {
    const Index ny = g.ny();
    if (ny == 1) return {0, 0, 0};
    const Real dy = g.dy();
    if (iy == 0)
        return (m[g.linear_index(ix, 1,    iz)] - m[g.linear_index(ix, 0, iz)]) / dy;
    if (iy == ny - 1)
        return (m[g.linear_index(ix, ny-1, iz)] - m[g.linear_index(ix, ny-2, iz)]) / dy;
    return (m[g.linear_index(ix, iy+1, iz)] - m[g.linear_index(ix, iy-1, iz)]) / (2.0 * dy);
}

inline Vec3 grad_z(const VectorField3D& m, const StructuredGrid& g,
                   Index ix, Index iy, Index iz) {
    const Index nz = g.nz();
    if (nz == 1) return {0, 0, 0};
    const Real dz = g.dz();
    if (iz == 0)
        return (m[g.linear_index(ix, iy, 1   )] - m[g.linear_index(ix, iy, 0)]) / dz;
    if (iz == nz - 1)
        return (m[g.linear_index(ix, iy, nz-1)] - m[g.linear_index(ix, iy, nz-2)]) / dz;
    return (m[g.linear_index(ix, iy, iz+1)] - m[g.linear_index(ix, iy, iz-1)]) / (2.0 * dz);
}

}  // namespace micromag::detail
