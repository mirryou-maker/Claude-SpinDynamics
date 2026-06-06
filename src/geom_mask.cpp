#include "micromag/geom_mask.hpp"

namespace micromag {

// All factory functions use a coordinate system centred at the geometric
// centre of the simulation box — matching mumax3's convention where (0,0,0)
// maps to the box centre.
//
// For cell (i,j,k) the centred coordinates are:
//   x = (i + 0.5) * dx - nx*dx/2
//   y = (j + 0.5) * dy - ny*dy/2
//   z = (k + 0.5) * dz - nz*dz/2

GeomMask ellipse(const StructuredGrid& grid, Real a, Real b)
{
    const Real cx = Real{0.5} * static_cast<Real>(grid.nx()) * grid.dx();
    const Real cy = Real{0.5} * static_cast<Real>(grid.ny()) * grid.dy();
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + Real{0.5}) * grid.dx() - cx;
        const Real y = (static_cast<Real>(iy) + Real{0.5}) * grid.dy() - cy;
        mask(ix, iy, iz) = ((x*x)/(a*a) + (y*y)/(b*b) <= Real{1}) ? Real{1} : Real{0};
    }
    return mask;
}

GeomMask circle(const StructuredGrid& grid, Real r)
{
    return ellipse(grid, r, r);
}

GeomMask rect(const StructuredGrid& grid, Real lx, Real ly)
{
    const Real cx  = Real{0.5} * static_cast<Real>(grid.nx()) * grid.dx();
    const Real cy  = Real{0.5} * static_cast<Real>(grid.ny()) * grid.dy();
    const Real hlx = Real{0.5} * lx;
    const Real hly = Real{0.5} * ly;
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + Real{0.5}) * grid.dx() - cx;
        const Real y = (static_cast<Real>(iy) + Real{0.5}) * grid.dy() - cy;
        mask(ix, iy, iz) = (std::abs(x) <= hlx && std::abs(y) <= hly) ? Real{1} : Real{0};
    }
    return mask;
}

GeomMask cylinder(const StructuredGrid& grid, Real r, Real h)
{
    const Real cx = Real{0.5} * static_cast<Real>(grid.nx()) * grid.dx();
    const Real cy = Real{0.5} * static_cast<Real>(grid.ny()) * grid.dy();
    const Real cz = Real{0.5} * static_cast<Real>(grid.nz()) * grid.dz();
    const Real r2 = r * r;
    const Real hz = Real{0.5} * h;
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + Real{0.5}) * grid.dx() - cx;
        const Real y = (static_cast<Real>(iy) + Real{0.5}) * grid.dy() - cy;
        const Real z = (static_cast<Real>(iz) + Real{0.5}) * grid.dz() - cz;
        mask(ix, iy, iz) = (x*x + y*y <= r2 && std::abs(z) <= hz) ? Real{1} : Real{0};
    }
    return mask;
}

}  // namespace micromag
