#include "micromag/spin_torque.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// SlonczewskiSTT
// ---------------------------------------------------------------------------

SlonczewskiSTT::SlonczewskiSTT(Real J, Real P, Real d, Vec3 p, Real beta)
    : J_(J), P_(P), d_(d), beta_(beta) {
    Real n = p.norm();
    p_ = (n > 1e-30) ? p / n : Vec3{0, 0, 1};
}

Real SlonczewskiSTT::a_J(Real Ms) const {
    // a_J = γ₀ ħ J P / (2 e Ms d)  [1/s]
    return constants::gamma_0 * constants::hbar * J_ * P_
           / (2.0 * constants::e_charge * Ms * d_);
}

void SlonczewskiSTT::accumulate(const VectorField3D& m,
                                 const Material& mat,
                                 VectorField3D& dm_out) const {
    const Real aJ = a_J(mat.Ms);
    const Real bJ = -beta_ * aJ;

    for (Index i = 0; i < m.size(); ++i) {
        Vec3 mi      = m[i];
        Vec3 mxp     = mi.cross(p_);
        Vec3 mxmxp   = mi.cross(mxp);
        dm_out[i]   += mxmxp * aJ + mxp * bJ;
    }
}

// ---------------------------------------------------------------------------
// SpinOrbitTorque
// ---------------------------------------------------------------------------

SpinOrbitTorque::SpinOrbitTorque(Real J_c, Real theta_SH, Real d_fm,
                                   Vec3 sigma, Real eta_DL, Real eta_FL)
    : J_c_(J_c), theta_SH_(theta_SH), d_fm_(d_fm),
      eta_DL_(eta_DL), eta_FL_(eta_FL) {
    Real n = sigma.norm();
    sigma_ = (n > 1e-30) ? sigma / n : Vec3{0, 1, 0};
}

Real SpinOrbitTorque::a_SOT(Real Ms) const {
    // a_SOT = γ₀ ħ J_c θ_SH / (2 e Ms d_fm)  [1/s]
    return constants::gamma_0 * constants::hbar * J_c_ * theta_SH_
           / (2.0 * constants::e_charge * Ms * d_fm_);
}

void SpinOrbitTorque::accumulate(const VectorField3D& m,
                                  const Material& mat,
                                  VectorField3D& dm_out) const {
    const Real a = a_SOT(mat.Ms);

    for (Index i = 0; i < m.size(); ++i) {
        Vec3 mi      = m[i];
        Vec3 mxs     = mi.cross(sigma_);
        Vec3 mxmxs   = mi.cross(mxs);
        dm_out[i]   += mxmxs * (a * eta_DL_) + mxs * (a * eta_FL_);
    }
}

// ---------------------------------------------------------------------------
// SpinTorqueSum
// ---------------------------------------------------------------------------

void SpinTorqueSum::add(std::shared_ptr<ISpinTorque> term) {
    if (term) terms_.push_back(std::move(term));
}

void SpinTorqueSum::accumulate(const VectorField3D& m, const Material& mat,
                                VectorField3D& dm_out) const {
    for (const auto& t : terms_)
        t->accumulate(m, mat, dm_out);
}

}  // namespace micromag
