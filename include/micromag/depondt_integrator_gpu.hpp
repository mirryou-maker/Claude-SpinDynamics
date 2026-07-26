#pragma once

// depondt_integrator_gpu.hpp — Task 1 (finite-T acceleration roadmap):
// Depondt–Mertens rotation integrator for the (stochastic) LLG on GPU.
//
// WHY THIS INTEGRATOR
//   The magnetisation is advanced by an explicit ROTATION (Rodrigues formula)
//   about the instantaneous angular velocity, so |m| = 1 is preserved to machine
//   precision BY CONSTRUCTION — no post-step renormalisation (roadmap invariant
//   #1). This is the property adaptive SLLG stepping (Berkov–Gorn / Leliaert)
//   relies on: the drift term touches only |m|, which a rotation fixes exactly.
//
//   Predictor–corrector (Heun-type, Depondt & Mertens 2009):
//     ω(m,H) = γ₀μ₀/(1+α²) · ( H + α (m×H) )        [rad/s], so ṁ = ω×m = LLG
//     ω1 = ω(mⁿ,      H(mⁿ))
//     m* = R(ω1, Δt) mⁿ                              (predictor rotation)
//     ω2 = ω(m*,      H(m*))
//     mⁿ⁺¹ = R(½(ω1+ω2), Δt) mⁿ                      (corrector rotation)
//
// STATUS (increment 1-A): deterministic core (T_K = 0) is implemented; it is a
//   drop-in, norm-exact alternative to RK4/Heun for T=0 dynamics. The finite-T
//   noise path (device-side Philox, roadmap 1-B) and adaptive step control
//   (step-doubling + PI controller, 1-C) plug into the hooks marked below.
//   Calling step() with T_K > 0 currently THROWS rather than silently running a
//   deterministic (wrong) finite-T step — finite-T code fails silently by
//   default, so we make it fail loudly instead.
//
// BATCH-READY (roadmap invariant #2): the public API and the internal kernels
//   are written so a leading replica dimension can be added in Task 2 without a
//   retrofit — dt is carried as a member (per-replica dt[R] later), and the
//   per-cell kernels index a flat [3·N] buffer that becomes [R·3·N].
//
// OPERATOR CHOICE: like RK4/RK45/Heun this integrates the *real* LLG (precession
//   + Gilbert damping from Material::alpha). It is NOT an energy minimiser; see
//   docs/USER_GUIDE.md §4.4. HeunIntegratorGPU remains the fixed-step baseline
//   used for the speedup comparison in the roadmap DoD.
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "demag_gpu_iface.hpp"
#include "effective_field_gpu_iface.hpp"
#include "gpu_real.hpp"
#include "gpu_state.hpp"
#include "material.hpp"
#include "spin_torque_gpu.hpp"
#include "types.hpp"

namespace micromag {

// Adaptive-stepping controls. Present now (1-A) so the API is stable; the
// controller that consumes them lands in increment 1-C. With adaptive == false
// the integrator takes fixed steps of `dt` (the current 1-A behaviour).
struct DepondtGPUOptions {
    bool adaptive = false;    // 1-C: enable step-doubling error control
    Real rtol     = 1e-4;     // relative tolerance (per-step local error)
    Real atol     = 1e-6;     // absolute tolerance
    Real dt_min   = 1e-16;    // [s]
    Real dt_max   = 1e-11;    // [s]
    Real safety   = 0.9;      // PI-controller safety factor
    Real fac_min  = 0.2;      // min step-shrink factor
    Real fac_max  = 2.0;      // max step-grow factor
};

class DepondtMertensGPU {
public:
    using Options = DepondtGPUOptions;

    // seed: reserved for the device-side Philox stream (increment 1-B); it is
    // the key together with the (future) replica id. Stored now so the ctor
    // signature is stable across increments.
    DepondtMertensGPU(const StructuredGrid& grid, Real dt, unsigned seed = 42);
    ~DepondtMertensGPU();

    DepondtMertensGPU(const DepondtMertensGPU&)            = delete;
    DepondtMertensGPU& operator=(const DepondtMertensGPU&) = delete;

    void upload(const VectorField3D& m)   { state_.upload(m);   }
    void download(VectorField3D& m) const { state_.download(m); }

    // One Depondt–Mertens step. Returns the dt actually taken (== dt_ while
    // fixed-step; adaptive control in 1-C will return the accepted step).
    // T_K > 0 is rejected until the Philox noise path (1-B) is wired.
    Real step(const Material& mat, IDemagGPU& demag,
              FieldSumGPU& extra_fields,
              Real T_K = 0.0,
              SpinTorqueSumGPU* torques = nullptr);

    Real dt() const      { return dt_; }
    void set_dt(Real dt) { dt_ = dt; }

    Options& options()             { return opts_; }
    const Options& options() const { return opts_; }

    // |m|=1 conservation is intrinsic; exposed for the drift regression test.
    double max_angle_gpu() const { return state_.max_angle_gpu(); }

    // Canonical thermal-field standard deviation (roadmap 1-D): the SINGLE point
    // where σ ∝ 1/√Δt is computed. Every dt change must route through here so a
    // rescale can never be silently forgotten (roadmap §5.1). Public so the
    // dedicated rescale regression test can call it directly.
    static double therm_sigma(const Material& mat, double dt,
                              double dx, double dy, double dz, double T_K);

private:
    // One Depondt–Mertens predictor–corrector sub-step m_in → m_out over `dt`.
    // Reads m_in (const), writes m_out (may alias m_in — mⁿ is saved first).
    // Adds the SAME Philox thermal field (offset `noise_offset`) in BOTH stages
    // when T_K>0 (Stratonovich). Used by the fixed-step path.
    void substep(const Material& mat, IDemagGPU& demag, FieldSumGPU& fields,
                 const GReal* m_in, GReal* m_out, double dt, Real T_K,
                 unsigned long long noise_offset);

    // 1-C: one adaptive step. Error is the embedded predictor–corrector estimate
    // ‖ω2−ω1‖·dt (no step-doubling, no rejection → no Wiener-path bias, roadmap
    // §5.3 policy (b)); a PI controller sets dt for the NEXT step. Returns the dt
    // just taken.
    Real step_adaptive(const Material& mat, IDemagGPU& demag,
                       FieldSumGPU& fields, Real T_K);

    GPUMagState state_;
    Real        dt_;
    Real        dx_, dy_, dz_;   // cell dims [m] — needed for σ
    unsigned    seed_;           // Philox key seed (1-B)
    unsigned long long step_index_ = 0;   // Philox counter component (1-B)
    Options     opts_;
    double*     d_err_ = nullptr;         // [1] device scalar, adaptive error norm
};

}  // namespace micromag

#endif // MICROMAG_CUDA
