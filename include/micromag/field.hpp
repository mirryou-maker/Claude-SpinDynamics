#pragma once

#include <vector>
#include <version>
// std::span accessors are C++20; the CUDA TUs compile this header as C++17
// (CMAKE_CUDA_STANDARD 17), so they are feature-guarded. Device code never
// touches host cell storage, so the narrower C++17 view is fine there.
#if defined(__cpp_lib_span)
#include <span>
#endif
#include "types.hpp"
#include "grid.hpp"

namespace micromag {

// Forward declaration — avoids including geom_mask.hpp here.
class GeomMask;

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

#if defined(__cpp_lib_span)
    // Preferred cell-data accessors: views that do not expose the container.
    std::span<const Real> span() const { return data_; }
    std::span<Real>       span()       { return data_; }
#endif

    // Container access — kept for existing call sites; prefer span() in new
    // code so the storage strategy stays an implementation detail.
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

#if defined(__cpp_lib_span)
    // Preferred cell-data accessors: views that do not expose the container.
    std::span<const Vec3> span() const { return data_; }
    std::span<Vec3>       span()       { return data_; }
#endif

    // Container access — kept for existing call sites; prefer span() in new
    // code so the storage strategy stays an implementation detail.
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

    // Shift the field by n cells along x (positive = shift right).
    // Cells shifted in from the left/right edge are filled with fill_m.
    // n = 0: no-op.  |n| >= nx: fills entire field with fill_m.
    void shift_x(Index n, const Vec3& fill_m);

    // Shift the field by n cells along y (positive = shift in +y direction).
    void shift_y(Index n, const Vec3& fill_m);

    // Find the x-index where component c crosses zero (sign change), searching
    // in the row (iy, iz). Returns -1 if no crossing found.
    // Useful for domain-wall tracking (c=0 for mz in PMA, c=2 for mx in Py).
    Index zero_crossing_x(int c, Index iy, Index iz) const;

    // Zero m at every cell where mask < threshold (default 0.5).
    // For edge-smoothed masks the magnetization is multiplied by the mask value.
    void apply_mask(const GeomMask& mask);

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
