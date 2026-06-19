#pragma once

// rk45_integrator_gpu.hpp — GPU Dormand-Prince adaptive integrator (DOPRI5/FSAL)
//
// Adaptive-step LLG integration on GPU.  The error estimation and step-size
// control run on the CPU; all H_eff evaluations and stage arithmetic run on GPU.
//
// Algorithm: DOPRI5 (Dormand-Prince) with FSAL (First Same As Last).
//   7 stages per trial step; on acceptance k7 = f(m5) is reused as k1
//   of the next step, reducing effective evaluations to 6/step.
//   Step-size adjusted with PI controller: h_new = h * clip(fac, fac_min, fac_max)
//   where fac = safety * (1/err_norm)^(1/5).
//
// One D2H copy per trial step (single double — error norm scalar).
// Zero PCIe overhead otherwise.
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "demag_gpu.hpp"
#include "demag_gpu_iface.hpp"
#include "effective_field_gpu_iface.hpp"
#include "exchange_gpu.hpp"
#include "field_kernels_gpu.hpp"
#include "gpu_state.hpp"
#include "material.hpp"
#include "spin_torque_gpu.hpp"
#include "types.hpp"

namespace micromag {

class RK45IntegratorGPU {
public:
    struct Options {
        Real rtol    = 1e-4;
        Real atol    = 1e-6;
        Real dt_init = 5e-14;
        Real dt_min  = 1e-16;
        Real dt_max  = 1e-11;
        Real safety  = 0.9;
        Real fac_min = 0.2;
        Real fac_max = 5.0;
    };

    explicit RK45IntegratorGPU(const StructuredGrid& grid, Options opts = {});
    ~RK45IntegratorGPU();

    RK45IntegratorGPU(const RK45IntegratorGPU&)            = delete;
    RK45IntegratorGPU& operator=(const RK45IntegratorGPU&) = delete;

    void upload(const VectorField3D& m)        { state_.upload(m);   }
    void download(VectorField3D& m) const      { state_.download(m); }

    // One adaptive DOPRI5 step.  Returns the dt actually taken.
    // demag may be DemagFieldGPU (open BC) or DemagFieldPeriodicGPU (periodic BC).
    Real step(const Material&               mat,
              IDemagGPU&                    demag,
              ExchangeFieldGPU&             exch,
              ZeemanFieldGPU&               zeeman,
              UniaxialAnisotropyFieldGPU*   aniso = nullptr);

    // Flexible overload: demag + arbitrary FieldSumGPU.
    Real step(const Material& mat, IDemagGPU& demag, FieldSumGPU& extra_fields);

    // FieldSumGPU + spin torques.
    Real step(const Material& mat, IDemagGPU& demag,
              FieldSumGPU& extra_fields, SpinTorqueSumGPU& torques);

    Real dt() const         { return dt_; }
    Real dt_current() const { return dt_; }
    int  n_accepted() const { return n_accepted_; }
    int  n_rejected() const { return n_rejected_; }

    // Maximum misalignment angle between adjacent spins (degrees).
    // Computed entirely on GPU — only 1 double transferred D2H per call.
    double max_angle_gpu() const { return state_.max_angle_gpu(); }

private:
    GPUMagState state_;
    Options     opts_;
    Real        dt_;
    bool        k1_valid_ = false;

    // ki slopes [3×N], owned by this integrator (not GPUMagState)
    double* d_k1_ = nullptr;
    double* d_k2_ = nullptr;
    double* d_k3_ = nullptr;
    double* d_k4_ = nullptr;
    double* d_k5_ = nullptr;
    double* d_k6_ = nullptr;
    double* d_k7_ = nullptr;   // FSAL: f(m5) → k1 of next step

    double* d_m5_       = nullptr;   // 5th-order candidate solution [3×N]
    double* d_err_      = nullptr;   // error estimate [3×N]
    double* d_err_sum_  = nullptr;   // single double on device (error norm reduction)
    double* d_m_stage_  = nullptr;   // intermediate m for stage evaluations [3×N]

    int n_accepted_ = 0;
    int n_rejected_ = 0;

    void alloc_scratch(size_t N);

    // Compute H_eff(d_m_in) + LLG torque → d_ki_out (fixed-field overload).
    void eval_ki(const Material& mat,
                 IDemagGPU& demag, ExchangeFieldGPU& exch,
                 ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
                 const double* d_m_in, double* d_ki_out);

    // Compute H_eff(d_m_in) + LLG torque → d_ki_out (FieldSumGPU overload).
    // Optional torques are added to d_ki_out after LLG torque.
    void eval_ki(const Material& mat, IDemagGPU& demag, FieldSumGPU& extra_fields,
                 const double* d_m_in, double* d_ki_out,
                 SpinTorqueSumGPU* torques = nullptr);
};

}  // namespace micromag

#endif // MICROMAG_CUDA
