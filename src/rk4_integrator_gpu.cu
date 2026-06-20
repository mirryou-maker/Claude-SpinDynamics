// rk4_integrator_gpu.cu ??G6: Full-GPU RK4 LLG integrator
//
// Assembles G1?밎5 building blocks into a single step() with zero PCIe overhead.
// All kernels run on GPUMagState::stream_; field streams are redirected via
// set_stream() so the entire pipeline is serialised on one CUDA stream.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "micromag/gpu_real.hpp"
#include "micromag/rk4_gpu.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA(integ): ")      \
                                   + cudaGetErrorString(_e));           \
    } while (0)

namespace micromag {

RK4IntegratorGPU::RK4IntegratorGPU(const StructuredGrid& grid, Real dt)
    : state_(grid), dt_(dt) {}

// ---------------------------------------------------------------------------
// run_stage ??one of the four RK4 stages
//
// On entry:  state_.d_m()  holds the stage's starting magnetization
//            state_.d_m0() holds m at the beginning of the step
//            state_.d_k_acc() accumulates the weighted sum
//
// Actions:
//   1. zero d_H
//   2. accumulate all field contributions into d_H
//   3. compute ki = llg_torque(d_m, d_H)
//   4. k_acc += accum_weight * ki
//   5. if stage_scale != 0: d_m = d_m0 + stage_scale * ki
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::run_stage(
    const Material& mat,
    IDemagGPU& demag, ExchangeFieldGPU& exch,
    ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
    double stage_scale, double accum_weight)
{
    void* s = state_.stream();
    GReal* dm  = state_.d_m();
    GReal* dH  = state_.d_H();
    GReal* dm0 = state_.d_m0();
    GReal* dki = state_.d_ki();
    GReal* dka = state_.d_k_acc();
    const int N = static_cast<int>(state_.N());

    // 1. Zero effective field accumulator.
    // All fields are on state_.stream() (set_stream called in step()); stream
    // ordering guarantees zero_H completes before any subsequent kernel.
    state_.zero_H();

    // 2. Accumulate each field on the shared stream — no barriers needed.
    exch.accumulate_gpu_ptr(dm, mat, dH);
    zeeman.accumulate_gpu_ptr(dm, mat, dH);
    if (aniso)
        aniso->accumulate_gpu_ptr(dm, mat, dH);
    demag.accumulate_gpu_ptr(dm, mat, dH);
    // d_H = H_exch + H_zeeman + H_aniso + H_demag (all ordered on stream_).

    // 3. LLG torque: ki = f(m, H)
    launch_llg_torque(dki, dm, dH, mat.alpha, N, s);

    // 4. Accumulate into weighted sum
    launch_rk4_accumulate(dka, dki, accum_weight, N, s);

    // 5. Stage update: d_m = d_m0 + stage_scale * ki  (skip for last stage)
    if (stage_scale != 0.0)
        launch_rk4_stage(dm, dm0, dki, stage_scale, N, s);
}

// ---------------------------------------------------------------------------
// step ??complete 4-stage RK4 step
//
// Butcher tableau (classic RK4):
//   stage 1: eval at m0,          accumulate 1/6,  next = m0 + dt/2*k1
//   stage 2: eval at m0+dt/2*k1,  accumulate 2/6,  next = m0 + dt/2*k2
//   stage 3: eval at m0+dt/2*k2,  accumulate 2/6,  next = m0 + dt*k3
//   stage 4: eval at m0+dt*k3,    accumulate 1/6,  no stage update
//   finalize: m = m0 + dt * k_acc
//   normalize: |m| = 1
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::step(
    const Material& mat,
    IDemagGPU&                    demag,
    ExchangeFieldGPU&             exch,
    ZeemanFieldGPU&               zeeman,
    UniaxialAnisotropyFieldGPU*   aniso)
{
    void* s = state_.stream();

    // Route all fields to the integrator's stream so run_stage() needs no syncs.
    demag.set_stream(s);
    exch.set_stream(s);
    zeeman.set_stream(s);
    if (aniso) aniso->set_stream(s);

    // Save m0 and zero the accumulator
    state_.save_m0();
    state_.zero_k_acc();

    const double h = static_cast<double>(dt_);

    run_stage(mat, demag, exch, zeeman, aniso, h * 0.5,  1.0/6.0); // k1
    run_stage(mat, demag, exch, zeeman, aniso, h * 0.5,  2.0/6.0); // k2
    run_stage(mat, demag, exch, zeeman, aniso, h * 1.0,  2.0/6.0); // k3
    run_stage(mat, demag, exch, zeeman, aniso, 0.0,      1.0/6.0); // k4

    // Finalize: m = m0 + dt * k_acc  (state_ accessors return GReal*)
    launch_rk4_finalize(state_.d_m(), state_.d_m0(), state_.d_k_acc(),
                         h, static_cast<int>(state_.N()), s);

    // Normalize: |m| = 1
    launch_normalize(state_.d_m(), static_cast<int>(state_.N()), s);

    // Synchronise: caller's next download() or upload() will also sync,
    // but explicit sync here keeps step() self-contained.
    state_.sync();
}

// ---------------------------------------------------------------------------
// run_stage ??FieldSumGPU overload (optional spin torques)
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::run_stage(
    const Material& mat,
    IDemagGPU& demag, FieldSumGPU& extra_fields,
    double stage_scale, double accum_weight,
    SpinTorqueSumGPU* torques)
{
    void* s = state_.stream();
    GReal* dm  = state_.d_m();
    GReal* dH  = state_.d_H();
    GReal* dm0 = state_.d_m0();
    GReal* dki = state_.d_ki();
    GReal* dka = state_.d_k_acc();
    const int N = static_cast<int>(state_.N());

    // All fields are on state_.stream() (set_stream called in step()).
    state_.zero_H();
    extra_fields.accumulate_gpu_ptr(dm, mat, dH);
    demag.accumulate_gpu_ptr(dm, mat, dH);

    launch_llg_torque(dki, dm, dH, mat.alpha, N, s);

    // Spin torques: add dm/dt contributions AFTER LLG torque (also on state_.stream()).
    if (torques && torques->size() > 0)
        torques->accumulate_gpu_ptr(dm, mat, dki);

    launch_rk4_accumulate(dka, dki, accum_weight, N, s);

    if (stage_scale != 0.0)
        launch_rk4_stage(dm, dm0, dki, stage_scale, N, s);
}

// ---------------------------------------------------------------------------
// step ??FieldSumGPU overload (no spin torques)
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::step(
    const Material& mat, IDemagGPU& demag, FieldSumGPU& extra_fields)
{
    void* s = state_.stream();
    demag.set_stream(s);
    extra_fields.set_stream(s);

    state_.save_m0();
    state_.zero_k_acc();

    const double h = static_cast<double>(dt_);

    run_stage(mat, demag, extra_fields, h * 0.5, 1.0/6.0); // k1
    run_stage(mat, demag, extra_fields, h * 0.5, 2.0/6.0); // k2
    run_stage(mat, demag, extra_fields, h * 1.0, 2.0/6.0); // k3
    run_stage(mat, demag, extra_fields, 0.0,     1.0/6.0); // k4

    launch_rk4_finalize(state_.d_m(), state_.d_m0(), state_.d_k_acc(),
                         h, static_cast<int>(state_.N()), s);
    launch_normalize(state_.d_m(), static_cast<int>(state_.N()), s);
    state_.sync();
}

// ---------------------------------------------------------------------------
// step ??FieldSumGPU + SpinTorqueSumGPU overload
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::step(
    const Material& mat, IDemagGPU& demag,
    FieldSumGPU& extra_fields, SpinTorqueSumGPU& torques)
{
    void* s = state_.stream();
    demag.set_stream(s);
    extra_fields.set_stream(s);
    torques.set_stream(s);

    state_.save_m0();
    state_.zero_k_acc();

    const double h = static_cast<double>(dt_);

    run_stage(mat, demag, extra_fields, h * 0.5, 1.0/6.0, &torques); // k1
    run_stage(mat, demag, extra_fields, h * 0.5, 2.0/6.0, &torques); // k2
    run_stage(mat, demag, extra_fields, h * 1.0, 2.0/6.0, &torques); // k3
    run_stage(mat, demag, extra_fields, 0.0,     1.0/6.0, &torques); // k4

    launch_rk4_finalize(state_.d_m(), state_.d_m0(), state_.d_k_acc(),
                         h, static_cast<int>(state_.N()), s);
    launch_normalize(state_.d_m(), static_cast<int>(state_.N()), s);
    state_.sync();
}

}  // namespace micromag

#endif // MICROMAG_CUDA

