#pragma once

// heun_integrator_gpu.hpp — G8: GPU Heun integrator for finite-T SLLG
//
// Implements the Stratonovich Heun scheme on GPU:
//
//   Resample noise: η^n  ~  N(0, σ)  per cell per component
//     σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V Δt))
//
//   Predictor:
//     H_eff(m^n) + η^n  →  k1 = f(m^n, H)
//     m_pred = normalize(m^n + dt·k1)
//
//   Corrector:
//     H_eff(m_pred) + η^n  ←  SAME noise (Stratonovich convention)
//     k2 = f(m_pred, H)
//     m^{n+1} = normalize(m^n + dt/2·(k1 + k2))
//
// T_K=0 → σ=0 → no noise → identical to a 2-stage (Heun ODE) integrator.
//
// cuRAND is used for bulk Gaussian generation on GPU.
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "demag_gpu.hpp"
#include "demag_gpu_iface.hpp"
#include "effective_field_gpu_iface.hpp"
#include "exchange_gpu.hpp"
#include "field_kernels_gpu.hpp"
#include "gpu_real.hpp"
#include "gpu_state.hpp"
#include "material.hpp"
#include "spin_torque_gpu.hpp"
#include "types.hpp"

namespace micromag {

class HeunIntegratorGPU {
public:
    // seed: RNG seed for reproducible noise (same convention as ThermalField)
    HeunIntegratorGPU(const StructuredGrid& grid, Real dt,
                       unsigned seed = 42);
    ~HeunIntegratorGPU();

    HeunIntegratorGPU(const HeunIntegratorGPU&)            = delete;
    HeunIntegratorGPU& operator=(const HeunIntegratorGPU&) = delete;

    // CPU ↔ GPU transfer
    void upload(const VectorField3D& m)   { state_.upload(m);   }
    void download(VectorField3D& m) const { state_.download(m); }

    // One Stratonovich Heun step (fixed-field overload).
    // T_K = 0 disables noise (σ = 0, cuRAND skipped).
    // demag may be DemagFieldGPU (open BC) or DemagFieldPeriodicGPU (periodic BC).
    void step(const Material&               mat,
              IDemagGPU&                    demag,
              ExchangeFieldGPU&             exch,
              ZeemanFieldGPU&               zeeman,
              Real                          T_K  = 0.0,
              UniaxialAnisotropyFieldGPU*   aniso = nullptr);

    // Flexible overload: arbitrary FieldSumGPU + optional spin torques.
    // Spin torques are applied AFTER LLG torque (same convention as RK4/RK45).
    void step(const Material& mat, IDemagGPU& demag,
              FieldSumGPU& extra_fields,
              Real T_K = 0.0,
              SpinTorqueSumGPU* torques = nullptr);

    Real dt() const      { return dt_; }
    void set_dt(Real dt) {
        if (dt != dt_) { gs1_.valid = gs2_.valid = false; }
        dt_ = dt;
    }

    // Max misalignment angle between adjacent spins (GPU-side, no D2H transfer).
    // Used by run_until_converged_gpu to check convergence without downloading m.
    double max_angle_gpu() const { return state_.max_angle_gpu(); }

    // Force CUDA Graph re-capture on next T=0 step() call.
    // Call after changing the field set (e.g. new ZeemanFieldGPU).
    void invalidate_graph() { gs1_.valid = gs2_.valid = false; }

private:
    GPUMagState state_;
    Real        dt_;
    Real        dx_, dy_, dz_;   // cell dimensions [m] — needed for sigma

    // Thermal noise: d_noise_[N_pad] — filled by cuRAND, added to d_H
    void*  d_noise_  = nullptr;
    size_t N_pad_;       // >= 3*N, rounded up to even for cuRAND requirement

    // cuRAND generator (opaque void* to avoid curand.h in header)
    void*  curand_gen_ = nullptr;

    // P4-style CUDA Graph state — one per step() overload, T=0 only.
    struct GraphState {
        void*    exec  = nullptr;
        bool     valid = false;
        Material mat   = {};
        Real     dt    = Real{0};
        Vec3     hext  = {};             // H_ext baked into captured kernel args
        bool     exch_percell  = false;  // per-cell material mode at capture time
        bool     aniso_percell = false;
        unsigned long long fs_rev = ~0ull;  // FieldSumGPU revision at capture time
    };
    GraphState gs1_;  // step(mat, demag, exch, zeeman, T_K=0, aniso)
    GraphState gs2_;  // step(mat, demag, extra_fields, T_K=0, torques)

    // Fixed-field run_half
    // P11: all [3×N] device buffer pointers use GReal (float or double per build flag).
    void run_half(const Material& mat,
                   const GReal*    d_m_in,
                   GReal*          d_H,
                   GReal*          d_ki,
                   IDemagGPU& demag, ExchangeFieldGPU& exch,
                   ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
                   bool add_noise);

    // FieldSumGPU run_half overload
    void run_half(const Material& mat,
                   const GReal*    d_m_in,
                   GReal*          d_H,
                   GReal*          d_ki,
                   IDemagGPU& demag, FieldSumGPU& extra_fields,
                   bool add_noise,
                   SpinTorqueSumGPU* torques = nullptr);

};

}  // namespace micromag

#endif // MICROMAG_CUDA
