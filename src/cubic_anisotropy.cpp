#include "micromag/cubic_anisotropy.hpp"

namespace micromag {

namespace {

// Per-cell cubic anisotropy field contribution.
// Returns H to add for one cell.
inline Vec3 cubic_field(Real a1, Real a2, Real a3,
                        const Vec3& c1, const Vec3& c2, const Vec3& c3,
                        Real pre1, Real pre2) {
    Vec3 H{0, 0, 0};
    // Kc1 term: -2Kc1/mu0Ms * [a1(a2²+a3²)c1 + a2(a1²+a3²)c2 + a3(a1²+a2²)c3]
    const Real a12 = a1*a1, a22 = a2*a2, a32 = a3*a3;
    H += c1 * (pre1 * a1 * (a22 + a32));
    H += c2 * (pre1 * a2 * (a12 + a32));
    H += c3 * (pre1 * a3 * (a12 + a22));
    // Kc2 term: -2Kc2/mu0Ms * [a1*a2²*a3² c1 + a1²*a2*a3² c2 + a1²*a2²*a3 c3]
    H += c1 * (pre2 * a1 * a22 * a32);
    H += c2 * (pre2 * a12 * a2 * a32);
    H += c3 * (pre2 * a12 * a22 * a3);
    return H;
}

}  // namespace

void CubicAnisotropyField::accumulate(const VectorField3D& m,
                                       const Material& mat,
                                       VectorField3D& H_out) const {
    if (Kc1_ == 0 && Kc2_ == 0) return;

    Vec3 c3 = c1_.cross(c2_);
    const Real inv_mu0Ms = 1.0 / (constants::mu_0 * mat.Ms);
    const Real pre1 = -2.0 * Kc1_ * inv_mu0Ms;
    const Real pre2 = -2.0 * Kc2_ * inv_mu0Ms;

    const Index N = m.size();
    #pragma omp parallel for schedule(static) if(N > 4096)
    for (Index idx = 0; idx < N; ++idx) {
        const Vec3& mi = m[idx];
        const Real a1 = mi.dot(c1_);
        const Real a2 = mi.dot(c2_);
        const Real a3 = mi.dot(c3);
        H_out[idx] += cubic_field(a1, a2, a3, c1_, c2_, c3, pre1, pre2);
    }
}

Real CubicAnisotropyField::energy(const VectorField3D& m,
                                   [[maybe_unused]] const Material& mat) const {
    if (Kc1_ == 0 && Kc2_ == 0) return 0;

    Vec3 c3 = c1_.cross(c2_);
    const Real dV = m.grid().cell_volume();
    Real E = 0;

    for (Index idx = 0; idx < m.size(); ++idx) {
        const Vec3& mi = m[idx];
        const Real a1 = mi.dot(c1_);
        const Real a2 = mi.dot(c2_);
        const Real a3 = mi.dot(c3);
        const Real a12 = a1*a1, a22 = a2*a2, a32 = a3*a3;
        E += Kc1_ * (a12*a22 + a22*a32 + a32*a12);
        E += Kc2_ * (a12*a22*a32);
    }
    return E * dV;
}

ScalarField3D CubicAnisotropyField::energy_density(const VectorField3D& m,
                                                    [[maybe_unused]] const Material& mat) const {
    ScalarField3D edens(m.grid());
    if (Kc1_ == 0 && Kc2_ == 0) return edens;

    Vec3 c3 = c1_.cross(c2_);
    for (Index idx = 0; idx < m.size(); ++idx) {
        const Vec3& mi = m[idx];
        const Real a1 = mi.dot(c1_);
        const Real a2 = mi.dot(c2_);
        const Real a3 = mi.dot(c3);
        const Real a12 = a1*a1, a22 = a2*a2, a32 = a3*a3;
        edens[idx] = Kc1_ * (a12*a22 + a22*a32 + a32*a12)
                   + Kc2_ * (a12*a22*a32);
    }
    return edens;
}

}  // namespace micromag
