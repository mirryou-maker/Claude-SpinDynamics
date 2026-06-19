#pragma once

// rk4_integrator_gpu.hpp — G6: Full-GPU RK4 LLG integrator.
//
// RK4IntegratorGPU owns a GPUMagState and assembles the G1–G5 building blocks
// into a complete Butcher step:
//
//   for stage 1..4:
//     zero_H()
//     exchange  .accumulate_gpu_ptr(d_m, mat, d_H)   ← G1
//     demag     .accumulate_gpu_ptr(d_m, mat, d_H)   ← G3 add-on
//     zeeman    .accumulate_gpu_ptr(d_m, mat, d_H)   ← G2
//     aniso     .accumulate_gpu_ptr(d_m, mat, d_H)   ← G2 (optional)
//     launch_llg_torque(d_ki, d_m, d_H, ...)         ← G4
//     launch_rk4_accumulate(d_k_acc, d_ki, weight)   ← G5
//     if not last: launch_rk4_stage(d_m, d_m0, d_ki, scale)  ← G5
//   launch_rk4_finalize(d_m, d_m0, d_k_acc, dt)     ← G5
//   launch_normalize(d_m)                             ← G5
//
// All kernels run on GPUMagState::stream_ — fields have their internal streams
// redirected via set_stream() at the beginning of each step().
// Zero PCIe per step; download only when the caller calls download().
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "demag_gpu.hpp"
#include "demag_gpu_iface.hpp"
#include "exchange_gpu.hpp"
#include "field_kernels_gpu.hpp"
#include "gpu_state.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

class RK4IntegratorGPU {
public:
    // Allocates GPUMagState (5 × [3×N] GPU buffers + pinned staging).
    RK4IntegratorGPU(const StructuredGrid& grid, Real dt);

    // Non-copyable
    RK4IntegratorGPU(const RK4IntegratorGPU&)            = delete;
    RK4IntegratorGPU& operator=(const RK4IntegratorGPU&) = delete;

    // Upload initial m; download current m for monitoring.
    void upload(const VectorField3D& m)        { state_.upload(m);   }
    void download(VectorField3D& m) const      { state_.download(m); }

    // One complete RK4 step — entirely on GPU, no PCIe.
    // aniso may be nullptr (skipped if not present).
    // demag may be DemagFieldGPU (open BC) or DemagFieldPeriodicGPU (periodic BC).
    void step(const Material&               mat,
              IDemagGPU&                    demag,
              ExchangeFieldGPU&             exch,
              ZeemanFieldGPU&               zeeman,
              UniaxialAnisotropyFieldGPU*   aniso = nullptr);

    Real dt() const        { return dt_; }
    void set_dt(Real dt)   { dt_ = dt;  }

private:
    GPUMagState state_;
    Real        dt_;

    // Run one RK4 stage: accumulate fields → ki → update k_acc → (optional) m_stage.
    void run_stage(const Material& mat,
                   IDemagGPU& demag, ExchangeFieldGPU& exch,
                   ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
                   double stage_scale,
                   double accum_weight);
};

}  // namespace micromag

#endif // MICROMAG_CUDA
