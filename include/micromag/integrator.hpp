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

}  // namespace micromag
