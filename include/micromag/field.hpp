#pragma once

#include <vector>
#include "types.hpp"
#include "grid.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// ScalarField3D — per-cell Real-valued field on a StructuredGrid.
// Same memory layout as VectorField3D: linear_index = i + nx*(j + ny*k).
// ---------------------------------------------------------------------------
class ScalarField3D {
public:
    explicit ScalarField3D(const StructuredGrid& grid)
        : grid_(&grid), data_(static_cast<std::size_t>(grid.size()), Real{0}) {}

    const StructuredGrid& grid() const { return *grid_; }
    Index size() const { return static_cast<Index>(data_.size()); }

    Real& operator[](Index i)             { return data_[static_cast<std::size_t>(i)]; }
    const Real& operator[](Index i) const { return data_[static_cast<std::size_t>(i)]; }

    Real& at(Index i, Index j, Index k) {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }
    const Real& at(Index i, Index j, Index k) const {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }

    void set_uniform(Real v) { std::fill(data_.begin(), data_.end(), v); }

    const std::vector<Real>& data() const { return data_; }
    std::vector<Real>&       data()       { return data_; }

private:
    const StructuredGrid* grid_;
    std::vector<Real>     data_;
};

// ---------------------------------------------------------------------------

class VectorField3D {
public:
    explicit VectorField3D(const StructuredGrid& grid)
        : grid_(&grid), data_(static_cast<std::size_t>(grid.size())) {}

    const StructuredGrid& grid() const { return *grid_; }

    Vec3& at(Index i, Index j, Index k) {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }
    const Vec3& at(Index i, Index j, Index k) const {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }

    Vec3& operator[](Index linear) {
        return data_[static_cast<std::size_t>(linear)];
    }
    const Vec3& operator[](Index linear) const {
        return data_[static_cast<std::size_t>(linear)];
    }

    Index size() const { return static_cast<Index>(data_.size()); }
    const std::vector<Vec3>& data() const { return data_; }
    std::vector<Vec3>& data() { return data_; }

    // Set all cells to the same vector.
    void set_uniform(const Vec3& m);

    // Normalize every vector to unit length. No-op for zero vectors.
    void normalize();

    // Set magnetization to a 2D vortex around the z-axis through (cx, cy).
    // Out-of-plane core (m_z) is +1 inside core_radius, smoothly transitioning.
    void set_vortex(Real cx, Real cy, Real core_radius);

    // Extract one Cartesian component (c=0 → x, 1 → y, 2 → z).
    ScalarField3D component(int c) const;

    // Copy sub-region into dst.
    // dst must have size (ix1-ix0+1) × (iy1-iy0+1) × (iz1-iz0+1).
    // Indices are inclusive and zero-based.
    void crop_into(VectorField3D& dst,
                   Index ix0, Index ix1,
                   Index iy0, Index iy1,
                   Index iz0, Index iz1) const;

private:
    const StructuredGrid* grid_;
    std::vector<Vec3> data_;
};

}  // namespace micromag
