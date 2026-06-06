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

void VectorField3D::shift_x(Index n, const Vec3& fill_m) {
    if (n == 0) return;
    const Index nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
    for (Index iz = 0; iz < nz; ++iz)
    for (Index iy = 0; iy < ny; ++iy) {
        if (n > 0) {
            // shift right: ix=nx-1..n get data from ix-n; ix=0..n-1 get fill
            for (Index ix = nx - 1; ix >= n; --ix)
                at(ix, iy, iz) = at(ix - n, iy, iz);
            for (Index ix = 0; ix < n && ix < nx; ++ix)
                at(ix, iy, iz) = fill_m;
        } else {
            // shift left (n < 0): ix=0..nx-1+n get data from ix-n; tail gets fill
            const Index shift = -n;
            for (Index ix = 0; ix < nx - shift; ++ix)
                at(ix, iy, iz) = at(ix + shift, iy, iz);
            for (Index ix = nx - shift; ix < nx; ++ix)
                at(ix, iy, iz) = fill_m;
        }
    }
}

void VectorField3D::shift_y(Index n, const Vec3& fill_m) {
    if (n == 0) return;
    const Index nx = grid_->nx(), ny = grid_->ny(), nz = grid_->nz();
    for (Index iz = 0; iz < nz; ++iz)
    for (Index ix = 0; ix < nx; ++ix) {
        if (n > 0) {
            for (Index iy = ny - 1; iy >= n; --iy)
                at(ix, iy, iz) = at(ix, iy - n, iz);
            for (Index iy = 0; iy < n && iy < ny; ++iy)
                at(ix, iy, iz) = fill_m;
        } else {
            const Index shift = -n;
            for (Index iy = 0; iy < ny - shift; ++iy)
                at(ix, iy, iz) = at(ix, iy + shift, iz);
            for (Index iy = ny - shift; iy < ny; ++iy)
                at(ix, iy, iz) = fill_m;
        }
    }
}

Index VectorField3D::zero_crossing_x(int c, Index iy, Index iz) const {
    const Index nx = grid_->nx();
    for (Index ix = 0; ix < nx - 1; ++ix) {
        const Vec3& v0 = at(ix,     iy, iz);
        const Vec3& v1 = at(ix + 1, iy, iz);
        const Real s0 = (c == 0 ? v0.x : c == 1 ? v0.y : v0.z);
        const Real s1 = (c == 0 ? v1.x : c == 1 ? v1.y : v1.z);
        if (s0 * s1 < Real{0})   // sign change
            return ix;
    }
    return Index{-1};
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
