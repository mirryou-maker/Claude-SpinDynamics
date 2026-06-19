#include "micromag/geom_mask.hpp"
#include <cmath>

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

GeomMask translate(const GeomMask& src, Real shift_x, Real shift_y)
{
    const auto& g = src.grid();
    // Round physical shift to nearest cell offset
    const Index di = static_cast<Index>(std::round(shift_x / g.dx()));
    const Index dj = static_cast<Index>(std::round(shift_y / g.dy()));

    GeomMask result(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index src_ix = ix - di;
        const Index src_iy = iy - dj;
        if (src_ix >= 0 && src_ix < g.nx() && src_iy >= 0 && src_iy < g.ny())
            result(ix, iy, iz) = src(src_ix, src_iy, iz);
        // else stays 0 (default)
    }
    return result;
}

GeomMask rotate(const GeomMask& src, Real theta)
{
    const auto& g = src.grid();
    const Real  c = std::cos(theta);
    const Real  s = std::sin(theta);
    // Half-extents in fractional index units (box centre as origin)
    const Real  half_nx = Real{0.5} * static_cast<Real>(g.nx());
    const Real  half_ny = Real{0.5} * static_cast<Real>(g.ny());

    GeomMask result(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        // Fractional centred index: fi = ix + 0.5 - nx/2
        const Real fi = static_cast<Real>(ix) + Real{0.5} - half_nx;
        const Real fj = static_cast<Real>(iy) + Real{0.5} - half_ny;

        // Inverse rotation (by -theta) to find source fractional centred index
        const Real src_fi = fi * c + fj * s;
        const Real src_fj = -fi * s + fj * c;

        // Convert back to absolute fractional grid index
        const Real sx = src_fi + half_nx - Real{0.5};
        const Real sy = src_fj + half_ny - Real{0.5};

        // Bilinear interpolation from source mask
        const Index ix0 = static_cast<Index>(std::floor(sx));
        const Index iy0 = static_cast<Index>(std::floor(sy));
        const Index ix1 = ix0 + 1;
        const Index iy1 = iy0 + 1;
        const Real  tx  = sx - static_cast<Real>(ix0);
        const Real  ty  = sy - static_cast<Real>(iy0);

        // Sample with out-of-bounds → 0
        auto sample = [&](Index ii, Index jj) -> Real {
            if (ii < 0 || ii >= g.nx() || jj < 0 || jj >= g.ny())
                return Real{0};
            return src(ii, jj, iz);
        };

        result(ix, iy, iz) =
            (Real{1} - tx) * (Real{1} - ty) * sample(ix0, iy0) +
            tx             * (Real{1} - ty) * sample(ix1, iy0) +
            (Real{1} - tx) * ty             * sample(ix0, iy1) +
            tx             * ty             * sample(ix1, iy1);
    }
    return result;
}

// ---------------------------------------------------------------------------
// New shape factories
// ---------------------------------------------------------------------------

GeomMask square(const StructuredGrid& grid, Real side)
{
    return rect(grid, side, side);
}

GeomMask cuboid(const StructuredGrid& grid, Real lx, Real ly, Real lz)
{
    const Real cx  = Real{0.5} * static_cast<Real>(grid.nx()) * grid.dx();
    const Real cy  = Real{0.5} * static_cast<Real>(grid.ny()) * grid.dy();
    const Real cz  = Real{0.5} * static_cast<Real>(grid.nz()) * grid.dz();
    const Real hlx = Real{0.5} * lx;
    const Real hly = Real{0.5} * ly;
    const Real hlz = Real{0.5} * lz;
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + Real{0.5}) * grid.dx() - cx;
        const Real y = (static_cast<Real>(iy) + Real{0.5}) * grid.dy() - cy;
        const Real z = (static_cast<Real>(iz) + Real{0.5}) * grid.dz() - cz;
        mask(ix, iy, iz) = (std::abs(x) <= hlx &&
                            std::abs(y) <= hly &&
                            std::abs(z) <= hlz) ? Real{1} : Real{0};
    }
    return mask;
}

GeomMask sphere(const StructuredGrid& grid, Real r)
{
    const Real cx = Real{0.5} * static_cast<Real>(grid.nx()) * grid.dx();
    const Real cy = Real{0.5} * static_cast<Real>(grid.ny()) * grid.dy();
    const Real cz = Real{0.5} * static_cast<Real>(grid.nz()) * grid.dz();
    const Real r2 = r * r;
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + Real{0.5}) * grid.dx() - cx;
        const Real y = (static_cast<Real>(iy) + Real{0.5}) * grid.dy() - cy;
        const Real z = (static_cast<Real>(iz) + Real{0.5}) * grid.dz() - cz;
        mask(ix, iy, iz) = (x*x + y*y + z*z <= r2) ? Real{1} : Real{0};
    }
    return mask;
}

GeomMask layer(const StructuredGrid& grid, Index n)
{
    return layers(grid, n, n);
}

GeomMask layers(const StructuredGrid& grid, Index n1, Index n2)
{
    GeomMask mask(grid);
    const Index lo = std::max(Index{0}, n1);
    const Index hi = std::min(grid.nz() - 1, n2);
    for (Index iz = lo; iz <= hi; ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix)
        mask(ix, iy, iz) = Real{1};
    return mask;
}

GeomMask x_range(const StructuredGrid& grid, Real x1, Real x2)
{
    const Real cx = Real{0.5} * static_cast<Real>(grid.nx()) * grid.dx();
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + Real{0.5}) * grid.dx() - cx;
        mask(ix, iy, iz) = (x >= x1 && x <= x2) ? Real{1} : Real{0};
    }
    return mask;
}

GeomMask y_range(const StructuredGrid& grid, Real y1, Real y2)
{
    const Real cy = Real{0.5} * static_cast<Real>(grid.ny()) * grid.dy();
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real y = (static_cast<Real>(iy) + Real{0.5}) * grid.dy() - cy;
        mask(ix, iy, iz) = (y >= y1 && y <= y2) ? Real{1} : Real{0};
    }
    return mask;
}

GeomMask z_range(const StructuredGrid& grid, Real z1, Real z2)
{
    const Real cz = Real{0.5} * static_cast<Real>(grid.nz()) * grid.dz();
    GeomMask mask(grid);
    for (Index iz = 0; iz < grid.nz(); ++iz)
    for (Index iy = 0; iy < grid.ny(); ++iy)
    for (Index ix = 0; ix < grid.nx(); ++ix) {
        const Real z = (static_cast<Real>(iz) + Real{0.5}) * grid.dz() - cz;
        mask(ix, iy, iz) = (z >= z1 && z <= z2) ? Real{1} : Real{0};
    }
    return mask;
}

}  // namespace micromag
