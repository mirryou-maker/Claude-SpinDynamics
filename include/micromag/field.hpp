#pragma once

#include <vector>
#include "types.hpp"
#include "grid.hpp"

namespace micromag {

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

private:
    const StructuredGrid* grid_;
    std::vector<Vec3> data_;
};

}  // namespace micromag
