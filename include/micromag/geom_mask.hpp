#pragma once

#include <algorithm>
#include <vector>
#include "grid.hpp"
#include "types.hpp"

namespace micromag {

// GeomMask — per-cell occupancy mask for geometry definition.
//
// Each cell stores a float in [0, 1]:
//   1.0 = fully inside,  0.0 = fully outside,  (0,1) = edge-smoothed.
//
// Factory functions (ellipse, circle, rect, cylinder) return a GeomMask with
// binary 0/1 values; edge-smoothing can be added later.
//
// Usage:
//   auto mask = ellipse(grid, 100e-9, 50e-9);
//   m.apply_mask(mask);   // zero m outside ellipse
//   exch.set_mask(&mask); // Neumann BC at mask=0 boundary
class GeomMask {
public:
    explicit GeomMask(const StructuredGrid& grid)
        : grid_(&grid), data_(static_cast<std::size_t>(grid.size()), Real{0}) {}

    const StructuredGrid& grid() const { return *grid_; }
    Index size() const { return static_cast<Index>(data_.size()); }

    Real& operator()(Index i, Index j, Index k) {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }
    const Real& operator()(Index i, Index j, Index k) const {
        return data_[static_cast<std::size_t>(grid_->linear_index(i, j, k))];
    }
    Real& operator[](Index linear) {
        return data_[static_cast<std::size_t>(linear)];
    }
    const Real& operator[](Index linear) const {
        return data_[static_cast<std::size_t>(linear)];
    }

    const std::vector<Real>& data() const { return data_; }
    std::vector<Real>&       data()       { return data_; }

    void set_uniform(Real v) { std::fill(data_.begin(), data_.end(), v); }

    // In-place invert: each cell value v → 1 - v.
    void invert() {
        for (auto& v : data_) v = Real{1} - v;
    }

private:
    const StructuredGrid* grid_;
    std::vector<Real>     data_;
};

// ---------------------------------------------------------------------------
// Boolean combinators (return new GeomMask; do not modify inputs)
// ---------------------------------------------------------------------------

// Union: max(A, B) per cell.
inline GeomMask union_(const GeomMask& a, const GeomMask& b) {
    GeomMask result(a.grid());
    for (Index i = 0; i < a.size(); ++i)
        result[i] = std::max(a[i], b[i]);
    return result;
}

// Subtraction: max(A - B, 0) per cell.
inline GeomMask sub_(const GeomMask& a, const GeomMask& b) {
    GeomMask result(a.grid());
    for (Index i = 0; i < a.size(); ++i)
        result[i] = std::max(a[i] - b[i], Real{0});
    return result;
}

// Intersection: min(A, B) per cell.
inline GeomMask intersect_(const GeomMask& a, const GeomMask& b) {
    GeomMask result(a.grid());
    for (Index i = 0; i < a.size(); ++i)
        result[i] = std::min(a[i], b[i]);
    return result;
}

}  // namespace micromag
