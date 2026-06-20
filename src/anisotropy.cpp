#include "micromag/anisotropy.hpp"
#include "micromag/material_field.hpp"

namespace micromag {

void UniaxialAnisotropyField::accumulate(const VectorField3D& m,
                                          const Material& mat,
                                          VectorField3D& H_out) const {
    const Index N = m.size();
    if (matf_) {
        // Per-cell: K1 and easy_axis from matf_; Ku2 always from uniform mat.
        const Real k2_mat = mat.Ku2;
        #pragma omp parallel for schedule(static) if(N > 4096)
        for (Index idx = 0; idx < N; ++idx) {
            const Real K1 = matf_->K_uniaxial(idx);
            if (K1 == 0 && k2_mat == 0) continue;

            Vec3 u = matf_->easy_axis(idx);
            const Real u_norm = u.norm();
            if (u_norm < 1e-30) continue;
            u = u / u_norm;

            const Real inv_mu0Ms = 1.0 / (constants::mu_0 * matf_->Ms(idx));
            const Real c = m[idx].dot(u);
            H_out[idx] += u * ((2.0 * K1 * c + 4.0 * k2_mat * c * c * c) * inv_mu0Ms);
        }
        return;
    }

    if (mat.K_uniaxial == 0 && mat.Ku2 == 0) return;

    Vec3 u = mat.easy_axis;
    Real u_norm = u.norm();
    if (u_norm < 1e-30) return;
    u = u / u_norm;

    const Real inv_mu0Ms = 1.0 / (constants::mu_0 * mat.Ms);
    const Real k1_pre = 2.0 * mat.K_uniaxial * inv_mu0Ms;
    const Real k2_pre = 4.0 * mat.Ku2 * inv_mu0Ms;

    #pragma omp parallel for schedule(static) if(N > 4096)
    for (Index idx = 0; idx < N; ++idx) {
        const Real c = m[idx].dot(u);
        H_out[idx] += u * (k1_pre * c + k2_pre * c * c * c);
    }
}

Real UniaxialAnisotropyField::energy(const VectorField3D& m,
                                      const Material& mat) const {
    const Real dV = m.grid().cell_volume();

    if (matf_) {
        const Real k2_mat = mat.Ku2;
        Real E = 0;
        for (Index idx = 0; idx < m.size(); ++idx) {
            const Real K1 = matf_->K_uniaxial(idx);
            if (K1 == 0 && k2_mat == 0) continue;

            Vec3 u = matf_->easy_axis(idx);
            const Real u_norm = u.norm();
            if (u_norm < 1e-30) continue;
            u = u / u_norm;

            const Real c = m[idx].dot(u);
            E -= (K1 * c * c + k2_mat * c * c * c * c) * dV;
        }
        return E;
    }

    if (mat.K_uniaxial == 0 && mat.Ku2 == 0) return 0;

    Vec3 u = mat.easy_axis;
    Real u_norm = u.norm();
    if (u_norm < 1e-30) return 0;
    u = u / u_norm;

    Real E = 0;
    for (Index idx = 0; idx < m.size(); ++idx) {
        const Real c = m[idx].dot(u);
        E -= (mat.K_uniaxial * c * c + mat.Ku2 * c * c * c * c) * dV;
    }
    return E;
}

ScalarField3D UniaxialAnisotropyField::energy_density(const VectorField3D& m,
                                                       const Material& mat) const {
    ScalarField3D edens(m.grid());

    if (matf_) {
        const Real k2_mat = mat.Ku2;
        for (Index idx = 0; idx < m.size(); ++idx) {
            const Real K1 = matf_->K_uniaxial(idx);
            Vec3 u = matf_->easy_axis(idx);
            const Real u_norm = u.norm();
            if (u_norm < 1e-30) continue;
            u = u / u_norm;
            const Real c = m[idx].dot(u);
            edens[idx] = -(K1 * c * c + k2_mat * c * c * c * c);
        }
        return edens;
    }

    Vec3 u = mat.easy_axis;
    Real u_norm = u.norm();
    if (u_norm < 1e-30) return edens;
    u = u / u_norm;

    for (Index idx = 0; idx < m.size(); ++idx) {
        const Real c = m[idx].dot(u);
        edens[idx] = -(mat.K_uniaxial * c * c + mat.Ku2 * c * c * c * c);
    }
    return edens;
}

}  // namespace micromag
