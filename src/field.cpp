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

ScalarField3D VectorField3D::component(int c) const {
    ScalarField3D out(*grid_);
    for (Index i = 0; i < size(); ++i)
        out[i] = (c == 0 ? data_[i].x : c == 1 ? data_[i].y : data_[i].z);
    return out;
}

void VectorField3D::crop_into(VectorField3D& dst,
                               Index ix0, Index ix1,
                               Index iy0, Index iy1,
                               Index iz0, Index iz1) const
{
    const Index dnx = ix1 - ix0 + 1;
    const Index dny = iy1 - iy0 + 1;
    const Index dnz = iz1 - iz0 + 1;
    (void)dnx; (void)dny; (void)dnz;   // used only in debug assertion
    for (Index kz = iz0; kz <= iz1; ++kz)
    for (Index ky = iy0; ky <= iy1; ++ky)
    for (Index kx = ix0; kx <= ix1; ++kx) {
        Index src_idx = kx + grid_->nx() * (ky + grid_->ny() * kz);
        Index dst_idx = (kx - ix0) + dnx * ((ky - iy0) + dny * (kz - iz0));
        dst[dst_idx] = data_[static_cast<std::size_t>(src_idx)];
    }
}

}  // namespace micromag
