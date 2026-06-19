#pragma once

#include <memory>
#include <vector>

#include "field.hpp"
#include "material.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// Abstract interface for non-conservative spin torques
//
// Each term adds its contribution directly to dm/dt [1/s].
// Convention (Landau-Lifshitz form, H in A/m):
//
//   dm/dt_total = τ_LLG(m, H_eff)  +  Σ τ_spin(m)
//
// Both STT and SOT are evaluated at the intermediate RK4 stages (they depend
// on m through the cross-products, but NOT on H_eff).
// ---------------------------------------------------------------------------
class ISpinTorque {
public:
    virtual ~ISpinTorque() = default;

    // Accumulate contribution into dm_out [1/s].
    virtual void accumulate(const VectorField3D& m,
                            const Material& mat,
                            VectorField3D& dm_out) const = 0;
    virtual const char* name() const = 0;
};

// ---------------------------------------------------------------------------
// Slonczewski Spin Transfer Torque (STT)
//
// Geometry: current flows perpendicular to the FM film (CPP).
// Reference layer fixes the spin polarisation direction p̂.
//
//   τ_STT = a_J [m×(m×p̂)] + b_J [m×p̂]
//
//   a_J = γ₀ħ J P / (2 e Ms d)   [1/s]   (damping-like; sign follows sign(J))
//   b_J = −β a_J                           (field-like; β ~ 0–0.5)
//
// sign(a_J) > 0  →  antidamping (can switch m away from p̂)
// sign(a_J) < 0  →  damping     (stabilises m toward p̂)
// ---------------------------------------------------------------------------
class SlonczewskiSTT : public ISpinTorque {
public:
    // J   : current density [A/m²]  (signed; positive = e⁻ from FL to RL)
    // P   : spin polarisation efficiency [0,1]
    // d   : free-layer thickness [m]
    // p   : reference-layer polarisation direction (normalised internally)
    // beta: field-like / damping-like ratio (b_J = −β a_J)
    SlonczewskiSTT(Real J, Real P, Real d, Vec3 p, Real beta = 0.0);

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& dm_out) const override;

    const char* name() const override { return "SlonczewskiSTT"; }

    // Compute a_J for a given Ms [1/s]
    Real a_J(Real Ms) const;

    Real J()    const { return J_; }
    Real P()    const { return P_; }
    Real d()    const { return d_; }
    Real beta() const { return beta_; }
    Vec3 p()    const { return p_; }

    void set_J(Real J)       { J_ = J; }
    void set_P(Real P)       { P_ = P; }
    void set_beta(Real beta) { beta_ = beta; }

private:
    Real J_, P_, d_, beta_;
    Vec3 p_;  // unit polarisation vector
};

// ---------------------------------------------------------------------------
// Spin-Orbit Torque (SOT)  — spin Hall effect in an adjacent heavy metal
//
// Geometry: charge current J_c flows in-plane (x-direction typical).
//           Interface normal along ẑ  →  spin polarisation σ̂ = ẑ × Ĵ_c.
//           Caller supplies σ̂ directly (already accounts for geometry).
//
//   τ_SOT = a_SOT [ η_DL m×(m×σ̂) + η_FL (m×σ̂) ]
//
//   a_SOT = γ₀ħ |J_c| |θ_SH| / (2 e Ms d_FM)   [1/s]
//           (overall sign = sign(J_c) × sign(θ_SH))
//
// Typical values: θ_SH ≈ 0.07 (Ta), 0.12 (Pt), 0.30 (W)
//                 η_DL ~ 1,  η_FL ~ 0.0–0.3
// ---------------------------------------------------------------------------
class SpinOrbitTorque : public ISpinTorque {
public:
    // J_c     : charge current density [A/m²] (signed)
    // theta_SH: spin Hall angle (signed; negative for Ta, positive for Pt)
    // d_fm    : ferromagnet thickness [m]
    // sigma   : spin-Hall polarisation direction (normalised internally)
    // eta_DL  : damping-like efficiency (default 1)
    // eta_FL  : field-like efficiency   (default 0)
    SpinOrbitTorque(Real J_c, Real theta_SH, Real d_fm, Vec3 sigma,
                    Real eta_DL = 1.0, Real eta_FL = 0.0);

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& dm_out) const override;

    const char* name() const override { return "SpinOrbitTorque"; }

    Real a_SOT(Real Ms) const;

    Real J_c()      const { return J_c_; }
    Real theta_SH() const { return theta_SH_; }
    Real d_fm()     const { return d_fm_; }
    Real eta_DL()   const { return eta_DL_; }
    Real eta_FL()   const { return eta_FL_; }
    Vec3 sigma()    const { return sigma_; }

    void set_J_c(Real J_c)         { J_c_ = J_c; }
    void set_theta_SH(Real th)     { theta_SH_ = th; }
    void set_eta_DL(Real e)        { eta_DL_ = e; }
    void set_eta_FL(Real e)        { eta_FL_ = e; }

private:
    Real J_c_, theta_SH_, d_fm_, eta_DL_, eta_FL_;
    Vec3 sigma_;  // unit spin-polarisation vector
};

// ---------------------------------------------------------------------------
// Zhang-Li Spin Transfer Torque (CIP-STT)
//
// Describes current-driven domain-wall motion for current flowing in-plane.
// Corresponds to mumax3 J + Pol + xi (DisableZhangLiTorque=false).
//
//   τ_ZL = u [(ĵ·∇)m − ξ m×(ĵ·∇)m]    [1/s]
//
//   u  = P μ_B |J| / (e Ms)   [m/s]  (spin-drift velocity)
//   ξ  = xi (non-adiabaticity parameter, typically 0.01–0.1)
//
// Gradient (ĵ·∇)m is computed by finite differences (central, Neumann BC).
// J is a 3-component vector [A/m²]; direction encodes current orientation.
// ---------------------------------------------------------------------------
class ZhangLiSTT : public ISpinTorque {
public:
    // J  : current density vector [A/m²], e.g. {1e12, 0, 0} for +x current
    // P  : spin polarisation [0,1]
    // xi : non-adiabaticity (β in some notations); typical 0.01–0.1
    ZhangLiSTT(Vec3 J, Real P, Real xi);

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& dm_out) const override;

    const char* name() const override { return "ZhangLiSTT"; }

    Vec3  J()    const { return J_; }
    Real  P()    const { return P_; }
    Real  xi()   const { return xi_; }

    void  set_J(Vec3 J)   { J_  = J;  }
    void  set_P(Real P)   { P_  = P;  }
    void  set_xi(Real xi) { xi_ = xi; }

    // Spin-drift velocity [m/s] for a given Ms
    Real u(Real Ms) const;

private:
    Vec3 J_;
    Real P_, xi_;
};

// ---------------------------------------------------------------------------
// Compositor: accumulates all spin torque terms
// ---------------------------------------------------------------------------
class SpinTorqueSum {
public:
    void add(std::shared_ptr<ISpinTorque> term);

    // Accumulate all terms into dm_out.
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& dm_out) const;

    std::size_t num_terms() const { return terms_.size(); }
    const std::vector<std::shared_ptr<ISpinTorque>>& terms() const { return terms_; }

private:
    std::vector<std::shared_ptr<ISpinTorque>> terms_;
};

}  // namespace micromag
