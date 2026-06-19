#include "micromag/spin_torque.hpp"
#include "micromag/grid.hpp"

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
// ZhangLiSTT
// ---------------------------------------------------------------------------

ZhangLiSTT::ZhangLiSTT(Vec3 J, Real P, Real xi)
    : J_(J), P_(P), xi_(xi) {}

Real ZhangLiSTT::u(Real Ms) const {
    // u = P μ_B |J| / (e Ms)  [m/s]
    // |J| is the scalar magnitude of the current density vector
    const Real Jmag = J_.norm();
    return P_ * constants::mu_B * Jmag / (constants::e_charge * Ms);
}

void ZhangLiSTT::accumulate(const VectorField3D& m,
                              const Material& mat,
                              VectorField3D& dm_out) const {
    const Real u_val = u(mat.Ms);
    if (u_val == 0.0) return;

    const StructuredGrid& g = m.grid();
    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    const Real  dx = g.dx(), dy = g.dy(), dz = g.dz();
    // Unit current direction
    const Real Jmag = J_.norm();
    if (Jmag < 1e-30) return;
    const Vec3 Jhat = J_ / Jmag;

    // Compute (ĵ·∇)m  at each cell via finite differences (Neumann BC)
    // Then add u*(grad_m - xi*m×grad_m) to dm_out

    for (Index iz = 0; iz < nz; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 mi = m[idx];

        // ∂m/∂x
        Vec3 dm_dx;
        if (Jhat.x != 0.0) {
            if (ix == 0)      dm_dx = (m[g.linear_index(1,  iy, iz)] - m[g.linear_index(0, iy, iz)]) / dx;
            else if (ix == nx-1) dm_dx = (m[g.linear_index(nx-1, iy, iz)] - m[g.linear_index(nx-2, iy, iz)]) / dx;
            else               dm_dx = (m[g.linear_index(ix+1, iy, iz)] - m[g.linear_index(ix-1, iy, iz)]) / (2.0*dx);
        }

        // ∂m/∂y
        Vec3 dm_dy;
        if (Jhat.y != 0.0) {
            if (iy == 0)      dm_dy = (m[g.linear_index(ix, 1,  iz)] - m[g.linear_index(ix, 0, iz)]) / dy;
            else if (iy == ny-1) dm_dy = (m[g.linear_index(ix, ny-1, iz)] - m[g.linear_index(ix, ny-2, iz)]) / dy;
            else               dm_dy = (m[g.linear_index(ix, iy+1, iz)] - m[g.linear_index(ix, iy-1, iz)]) / (2.0*dy);
        }

        // ∂m/∂z
        Vec3 dm_dz;
        if (Jhat.z != 0.0) {
            if (iz == 0)      dm_dz = (m[g.linear_index(ix, iy, 1   )] - m[g.linear_index(ix, iy, 0)]) / dz;
            else if (iz == nz-1) dm_dz = (m[g.linear_index(ix, iy, nz-1)] - m[g.linear_index(ix, iy, nz-2)]) / dz;
            else               dm_dz = (m[g.linear_index(ix, iy, iz+1)] - m[g.linear_index(ix, iy, iz-1)]) / (2.0*dz);
        }

        // (ĵ·∇)m = Jhat.x * ∂m/∂x + Jhat.y * ∂m/∂y + Jhat.z * ∂m/∂z
        Vec3 grad_m = dm_dx * Jhat.x + dm_dy * Jhat.y + dm_dz * Jhat.z;

        // τ_ZL = u [(ĵ·∇)m − ξ m×(ĵ·∇)m]
        dm_out[idx] += (grad_m - mi.cross(grad_m) * xi_) * u_val;
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
