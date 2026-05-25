#include "micromag/anisotropy.hpp"

namespace micromag {

void UniaxialAnisotropyField::accumulate(const VectorField3D& m,
                                          const Material& mat,
                                          VectorField3D& H_out) const {
    if (mat.K_uniaxial == 0) return;

    Vec3 u = mat.easy_axis;
    Real u_norm = u.norm();
    if (u_norm < 1e-30) return;
    u = u / u_norm;

    const Real prefactor = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);

    for (Index idx = 0; idx < m.size(); ++idx)
        H_out[idx] += u * (prefactor * m[idx].dot(u));
}

Real UniaxialAnisotropyField::energy(const VectorField3D& m,
                                      const Material& mat) const {
    if (mat.K_uniaxial == 0) return 0;

    Vec3 u = mat.easy_axis;
    Real u_norm = u.norm();
    if (u_norm < 1e-30) return 0;
    u = u / u_norm;

    Real sum_sq = 0;
    for (Index idx = 0; idx < m.size(); ++idx) {
        Real c = m[idx].dot(u);
        sum_sq += c * c;
    }
    return -mat.K_uniaxial * sum_sq * m.grid().cell_volume();
}

}  // namespace micromag
