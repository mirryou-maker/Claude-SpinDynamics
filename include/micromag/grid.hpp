#pragma once

#include "types.hpp"

namespace micromag {

// Cell-centered uniform Cartesian grid.
// Cell (i,j,k) occupies [i*dx, (i+1)*dx] x [j*dy, (j+1)*dy] x [k*dz, (k+1)*dz].
class StructuredGrid {
public:
    StructuredGrid(Index nx, Index ny, Index nz, Real dx, Real dy, Real dz);

    Index nx() const { return nx_; }
    Index ny() const { return ny_; }
    Index nz() const { return nz_; }

    Real dx() const { return dx_; }
    Real dy() const { return dy_; }
    Real dz() const { return dz_; }

    Index size() const { return nx_ * ny_ * nz_; }
    Real cell_volume() const { return dx_ * dy_ * dz_; }

    // x is fastest (i varies first), matches numpy/Fortran convention.
    Index linear_index(Index i, Index j, Index k) const {
        return i + nx_ * (j + ny_ * k);
    }

    Vec3 cell_center(Index i, Index j, Index k) const {
        return {(static_cast<Real>(i) + Real{0.5}) * dx_,
                (static_cast<Real>(j) + Real{0.5}) * dy_,
                (static_cast<Real>(k) + Real{0.5}) * dz_};
    }

    Vec3 extent() const {
        return {static_cast<Real>(nx_) * dx_,
                static_cast<Real>(ny_) * dy_,
                static_cast<Real>(nz_) * dz_};
    }

private:
    Index nx_, ny_, nz_;
    Real dx_, dy_, dz_;
};

}  // namespace micromag
