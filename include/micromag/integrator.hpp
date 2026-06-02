#pragma once

#include <memory>
#include "field.hpp"
#include "material.hpp"
#include "effective_field.hpp"
#include "spin_torque.hpp"

namespace micromag {

// Per-cell LLG torque (Landau-Lifshitz form):
//   dm/dt = -γ'μ₀(m×H) - γ'αμ₀ m×(m×H),   γ' = γ₀/(1+α²),   H in [A/m]
inline Vec3 llg_torque(Vec3 m, Vec3 H, Real alpha) noexcept {
    const Real gp  = constants::gamma_0 * constants::mu_0 / (1.0 + alpha * alpha);
    const Vec3 mxH = m.cross(H);
    return (mxH + m.cross(mxH) * alpha) * (-gp);
}

// Classic RK4 integrator for LLG.
// Scratch fields are allocated once on the first step and reused.
class RK4Integrator {
public:
    explicit RK4Integrator(Real dt = 1e-13);

    // Advance m by dt. Normalises |m|=1 after the update.
    // stt: optional spin torques (STT, SOT) evaluated at each RK4 stage.
    void step(VectorField3D& m, const Material& mat,
              const EffectiveFieldSum& heff,
              const SpinTorqueSum* stt = nullptr);

    Real dt()          const { return dt_; }
    void set_dt(Real dt)     { dt_ = dt;  }

private:
    Real dt_;
    std::unique_ptr<VectorField3D> H_, m_tmp_, k1_, k2_, k3_, k4_;
    void ensure_scratch(const StructuredGrid& g);
};

// ---------------------------------------------------------------------------
// Dormand-Prince RK4(5) adaptive integrator
// Advances m by one step whose size is chosen to keep the per-cell RMS
// error below (rtol * |m| + atol).  Uses the FSAL property so only
// 6 effective field evaluations are needed per accepted step.
// ---------------------------------------------------------------------------
class RK45Integrator {
public:
    struct Options {
        Real rtol    = 1e-4;    // relative tolerance on m components
        Real atol    = 1e-6;    // absolute tolerance
        Real dt_init = 5e-14;   // initial step size [s]
        Real dt_min  = 1e-16;   // minimum allowed step [s]
        Real dt_max  = 1e-11;   // maximum allowed step [s]
        Real safety  = 0.9;     // safety factor
        Real fac_min = 0.2;     // minimum step-reduction factor per rejection
        Real fac_max = 5.0;     // maximum step-growth factor per acceptance
    };

    explicit RK45Integrator(Options opts = {});

    // Advance m by one adaptive step; returns dt actually used.
    Real step(VectorField3D& m, const Material& mat,
              const EffectiveFieldSum& heff,
              const SpinTorqueSum* stt = nullptr);

    Real  dt_current()  const { return dt_; }
    Index n_accepted()  const { return n_accepted_; }
    Index n_rejected()  const { return n_rejected_; }

private:
    Options opts_;
    Real    dt_;
    Index   n_accepted_{0};
    Index   n_rejected_{0};

    bool k1_valid_{false};  // true when k1_ already holds f(m) from prev FSAL

    std::unique_ptr<VectorField3D> H_;
    std::unique_ptr<VectorField3D> m_tmp_;
    std::unique_ptr<VectorField3D> k1_, k2_, k3_, k4_, k5_, k6_, k7_;
    std::unique_ptr<VectorField3D> m5_;   // 5th-order solution
    std::unique_ptr<VectorField3D> err_;  // local error estimate

    void ensure_scratch(const StructuredGrid& g);
    Real error_norm(const VectorField3D& m,
                    const VectorField3D& m5,
                    const VectorField3D& e) const;
};

}  // namespace micromag
