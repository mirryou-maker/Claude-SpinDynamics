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
// OPERATOR CHOICE: this is a *real* LLG integrator — full precession + Gilbert
// damping taken from Material::alpha. Use it (or RK45/Heun) for dynamics,
// switching, FMR, and for any relaxation/metastability question where the
// physical path matters (a fixed dt makes it simplest but needs dt small
// enough for stability). Do NOT substitute RelaxGPU/MinimizeGPU there: those
// are precession-free energy minimisers that ignore mat.alpha (RelaxGPU) or
// pin the seeded topology (MinimizeGPU). See docs/USER_GUIDE.md §4.4.
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

class RK4IntegratorGPU {
public:
    // Allocates GPUMagState (5 × [3×N] GPU buffers + pinned staging).
    RK4IntegratorGPU(const StructuredGrid& grid, Real dt);
    ~RK4IntegratorGPU();

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

    // Flexible overload: demag runs first (cuFFT pipeline), then extra_fields.
    // extra_fields may contain ExchangeFieldGPU, ZeemanFieldGPU,
    // UniaxialAnisotropyFieldGPU, BulkDMIFieldGPU, InterfacialDMIFieldGPU, etc.
    void step(const Material& mat, IDemagGPU& demag, FieldSumGPU& extra_fields);

    // FieldSumGPU + spin torques (STT/SOT/Zhang-Li).
    // torques are applied AFTER LLG torque at each stage.
    void step(const Material& mat, IDemagGPU& demag,
              FieldSumGPU& extra_fields, SpinTorqueSumGPU& torques);

    Real dt() const { return dt_; }
    void set_dt(Real dt);  // also invalidates any captured graph

    // Maximum misalignment angle between adjacent spins (degrees).
    // Computed entirely on GPU — only 1 double transferred D2H per call.
    double max_angle_gpu() const { return state_.max_angle_gpu(); }

    // P4: Force graph re-capture on next step() call.
    // Call after modifying the field set (e.g. adding a field to FieldSumGPU).
    void invalidate_graph();

private:
    GPUMagState state_;
    Real        dt_;

    // P4: CUDA Graph state — one per step() overload.
    // exec is stored as void* to keep cuda_runtime.h out of this header.
    struct GraphState {
        void*    exec  = nullptr;  // cudaGraphExec_t; nullptr = not captured
        bool     valid = false;    // false = need (re-)capture on next step
        Material mat   = {};       // material snapshot baked into the graph
        Real     dt    = Real{0};  // dt snapshot
        Vec3     hext  = {};       // H_ext baked into the captured kernel args
        bool     exch_percell  = false;  // per-cell material mode at capture time
        bool     aniso_percell = false;  // (changing mode invalidates the graph)
        unsigned long long fs_rev = ~0ull;  // FieldSumGPU revision at capture time
    };
    GraphState gs1_;  // step(mat, demag, exch, zeeman, aniso)
    GraphState gs2_;  // step(mat, demag, extra_fields)
    GraphState gs3_;  // step(mat, demag, extra_fields, torques)

    // Run one RK4 stage (fixed-field overload).
    void run_stage(const Material& mat,
                   IDemagGPU& demag, ExchangeFieldGPU& exch,
                   ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
                   double stage_scale, double accum_weight);

    // Run one RK4 stage (FieldSumGPU overload, optional spin torques).
    void run_stage(const Material& mat, IDemagGPU& demag,
                   FieldSumGPU& extra_fields,
                   double stage_scale, double accum_weight,
                   SpinTorqueSumGPU* torques = nullptr);
};

}  // namespace micromag

#endif // MICROMAG_CUDA
