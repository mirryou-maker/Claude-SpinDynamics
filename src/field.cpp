#include "micromag/field.hpp"

#include <cmath>

namespace micromag {

void VectorField3D::set_uniform(const Vec3& m) {
    for (auto& v : data_) {
        v = m;
    }
}

void VectorField3D::normalize() {
    for (auto& v : data_) {
        Real n = v.norm();
        if (n > Real{1e-30}) {
            v /= n;
        }
    }
}

void VectorField3D::set_vortex(Real cx, Real cy, Real core_radius) {
    for (Index k = 0; k < grid_->nz(); ++k) {
        for (Index j = 0; j < grid_->ny(); ++j) {
            for (Index i = 0; i < grid_->nx(); ++i) {
                Vec3 c = grid_->cell_center(i, j, k);
                Real rx = c.x - cx;
                Real ry = c.y - cy;
                Real r = std::sqrt(rx * rx + ry * ry);

                if (r < Real{1e-30}) {
                    at(i, j, k) = {0, 0, 1};
                    continue;
                }

                // Smooth core: m_z drops from 1 -> 0 over [0, core_radius].
                Real mz = (r < core_radius)
                              ? std::cos(Real{0.5} * Real{3.14159265358979323846} *
                                         (r / core_radius))
                              : Real{0};
                Real m_inplane = std::sqrt(std::max(Real{0}, Real{1} - mz * mz));
                Real mx = -ry / r * m_inplane;
                Real my = rx / r * m_inplane;

                at(i, j, k) = {mx, my, mz};
            }
        }
    }
}

}  // namespace micromag
