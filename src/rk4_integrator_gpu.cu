// rk4_integrator_gpu.cu — G6: Full-GPU RK4 LLG integrator
//
// Assembles G1-G5 building blocks into a single step() with zero PCIe overhead.
// All kernels run on GPUMagState::stream_; field streams are redirected via
// set_stream() so the entire pipeline is serialised on one CUDA stream.
//
// P4: CUDA Graphs — graph captured on first step() call; replayed on subsequent
// calls to eliminate kernel-launch CPU overhead (~50-100 us/step for SP#4).
// Re-captured automatically when material parameters or dt change.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include <cmath>
#include "micromag/exchange.hpp"
#include "micromag/field_kernels_gpu.hpp"
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

// ---------------------------------------------------------------------------
// P4 helpers
// ---------------------------------------------------------------------------

static bool mat_eq(const Material& a, const Material& b) {
    return a.Ms == b.Ms && a.A_exchange == b.A_exchange &&
           a.K_uniaxial == b.K_uniaxial && a.Ku2 == b.Ku2 &&
           a.alpha == b.alpha &&
           a.easy_axis.x == b.easy_axis.x &&
           a.easy_axis.y == b.easy_axis.y &&
           a.easy_axis.z == b.easy_axis.z;
}

static void free_graph_exec(void*& exec_v) {
    if (exec_v) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_v));
        exec_v = nullptr;
    }
}

// Capture body() as a CUDA graph and instantiate a cudaGraphExec_t.
// Returns true on success (exec_out holds the new exec).
// On any failure: exec_out stays null, body() is run directly as a fallback.
// Capture body() onto stream s, instantiate a cudaGraphExec_t.
// Returns true on success.  On failure, body() is run directly as a fallback.
//
// ALWAYS calls cudaStreamEndCapture after a successful BeginCapture so the
// stream is never left in capture mode (which would poison the recycled handle
// for subsequent tests / objects).
template<class F>
static bool do_capture(cudaStream_t s, void*& exec_out, F body) {
    free_graph_exec(exec_out);

    // VkFFT uses internally-managed CUDA state that cannot be captured into a
    // CUDA Graph (cuLaunchKernel returns CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 906).
    // Skip capture entirely for VkFFT builds; the FFT speedup from VkFFT is
    // more valuable than graph-replay savings for Medium/Large grids.
#ifdef MICROMAG_VKFFT
    body();
    return false;
#endif

    // When MICROMAG_DEMAG_PROFILE is set, the demag profiler uses CUDA events
    // inside body() and calls cudaEventElapsedTime() which requires the events
    // to be completed.  During capture, GPU ops are queued but not executed, so
    // event queries crash.  Fall back to direct execution so profiling works.
    static const bool profiling_active =
        (std::getenv("MICROMAG_DEMAG_PROFILE") != nullptr);
    if (profiling_active) {
        body();   // direct execution; events fire and are readable
        return false;
    }

    cudaError_t begin_err = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (begin_err != cudaSuccess) {
        cudaGetLastError();
        body();    // fallback: execute normally
        return false;
    }

    // body() may throw if it calls non-capturable ops (e.g. cudaStreamSynchronize).
    // Wrap in try-catch so EndCapture is ALWAYS called after Begin.
    bool body_ok = true;
    try {
        body();    // all GPU ops captured into the graph; not executed yet
    } catch (...) {
        body_ok = false;
    }

    cudaGraph_t g = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(s, &g);
    // Stream is now out of capture mode regardless of body_ok / end_err.

    if (!body_ok || end_err != cudaSuccess || !g) {
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();  // clear any sticky error
        body();    // fallback: ops not executed during capture; run directly now
        return false;
    }

    cudaGraphExec_t ge = nullptr;
    cudaError_t inst_err = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);    // template no longer needed after instantiation
    if (inst_err != cudaSuccess) {
        cudaGetLastError();
        body();    // fallback
        return false;
    }

    exec_out = static_cast<void*>(ge);
    return true;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

RK4IntegratorGPU::RK4IntegratorGPU(const StructuredGrid& grid, Real dt)
    : state_(grid), dt_(dt) {}

RK4IntegratorGPU::~RK4IntegratorGPU() {
    free_graph_exec(gs1_.exec);
    free_graph_exec(gs2_.exec);
    free_graph_exec(gs3_.exec);
}

void RK4IntegratorGPU::set_dt(Real dt) {
    if (dt != dt_)
        gs1_.valid = gs2_.valid = gs3_.valid = false;
    dt_ = dt;
}

void RK4IntegratorGPU::invalidate_graph() {
    gs1_.valid = gs2_.valid = gs3_.valid = false;
}

// ---------------------------------------------------------------------------
// run_stage — one of the four RK4 stages
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

    state_.zero_H();

    // Fuse exchange + zeeman + anisotropy into one kernel pass when all three
    // fields are in uniform-material mode (no per-cell data on device).
    // Saves 36% of global-memory ops vs three separate launches.
    const bool aniso_percell = aniso && aniso->has_material_field();
    if (!exch.has_material_field() && !aniso_percell) {
        const double mu0Ms = constants::mu_0 * mat.Ms;
        const double dx = exch.dx(), dy = exch.dy(), dz = exch.dz();
        const double fx = (dx > 0) ? 2.0 * mat.A_exchange / (mu0Ms * dx * dx) : 0.0;
        const double fy = (dy > 0) ? 2.0 * mat.A_exchange / (mu0Ms * dy * dy) : 0.0;
        const double fz = (dz > 0) ? 2.0 * mat.A_exchange / (mu0Ms * dz * dz) : 0.0;

        double aniso_factor = 0.0, ux = 0, uy = 0, uz = 1;
        if (aniso && mat.K_uniaxial != 0.0 && mu0Ms > 0) {
            Vec3 u = mat.easy_axis;
            const double unorm = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
            if (unorm > 1e-30) {
                ux = u.x/unorm; uy = u.y/unorm; uz = u.z/unorm;
                aniso_factor = 2.0 * mat.K_uniaxial / mu0Ms;
            }
        }

        const Vec3& Hext = zeeman.H_ext();
        const bool periodic = (exch.bc() == BoundaryCondition::Periodic);
        launch_fused_local_fields(
            dm, dH,
            state_.nx(), state_.ny(), state_.nz(),
            fx, fy, fz,
            Hext.x, Hext.y, Hext.z,
            aniso_factor, ux, uy, uz,
            periodic, s);
    } else {
        // Per-cell mode: fall back to individual kernels
        exch.accumulate_gpu_ptr(dm, mat, dH);
        zeeman.accumulate_gpu_ptr(dm, mat, dH);
        if (aniso)
            aniso->accumulate_gpu_ptr(dm, mat, dH);
    }
    demag.accumulate_gpu_ptr(dm, mat, dH);

    launch_llg_torque(dki, dm, dH, mat.alpha, N, s);
    launch_rk4_accumulate(dka, dki, accum_weight, N, s);

    if (stage_scale != 0.0)
        launch_rk4_stage(dm, dm0, dki, stage_scale, N, s);
}

// ---------------------------------------------------------------------------
// step — fixed-field overload
//
// P4: On first call (or after material/dt change), the entire GPU pipeline
// is captured as a CUDA graph.  All subsequent calls replay the graph,
// eliminating kernel-launch CPU overhead.
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::step(
    const Material& mat,
    IDemagGPU&                    demag,
    ExchangeFieldGPU&             exch,
    ZeemanFieldGPU&               zeeman,
    UniaxialAnisotropyFieldGPU*   aniso)
{
    void* sv = state_.stream();
    cudaStream_t s = static_cast<cudaStream_t>(sv);

    // CPU-only setup — not in capture window.
    demag.set_stream(sv);
    exch.set_stream(sv);
    zeeman.set_stream(sv);
    if (aniso) aniso->set_stream(sv);

    bool stale = !gs1_.valid || !mat_eq(gs1_.mat, mat) || gs1_.dt != dt_;

    if (stale) {
        const double h = static_cast<double>(dt_);
        auto body = [&] {
            state_.save_m0();
            state_.zero_k_acc();
            run_stage(mat, demag, exch, zeeman, aniso, h * 0.5,  1.0/6.0);
            run_stage(mat, demag, exch, zeeman, aniso, h * 0.5,  2.0/6.0);
            run_stage(mat, demag, exch, zeeman, aniso, h * 1.0,  2.0/6.0);
            run_stage(mat, demag, exch, zeeman, aniso, 0.0,      1.0/6.0);
            launch_rk4_finalize(state_.d_m(), state_.d_m0(), state_.d_k_acc(),
                                 h, static_cast<int>(state_.N()), sv);
            launch_normalize(state_.d_m(), static_cast<int>(state_.N()), sv);
        };

        gs1_.valid = do_capture(s, gs1_.exec, body);
        gs1_.mat   = mat;
        gs1_.dt    = dt_;

        if (!gs1_.valid) {
            // body() ran directly in do_capture fallback
            state_.sync();
            return;
        }
    }

    CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(gs1_.exec), s));
    state_.sync();
}

// ---------------------------------------------------------------------------
// run_stage — FieldSumGPU overload (optional spin torques)
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

    state_.zero_H();
    extra_fields.accumulate_gpu_ptr(dm, mat, dH);
    demag.accumulate_gpu_ptr(dm, mat, dH);

    launch_llg_torque(dki, dm, dH, mat.alpha, N, s);

    if (torques && torques->size() > 0)
        torques->accumulate_gpu_ptr(dm, mat, dki);

    launch_rk4_accumulate(dka, dki, accum_weight, N, s);

    if (stage_scale != 0.0)
        launch_rk4_stage(dm, dm0, dki, stage_scale, N, s);
}

// ---------------------------------------------------------------------------
// step — FieldSumGPU overload (no spin torques)
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::step(
    const Material& mat, IDemagGPU& demag, FieldSumGPU& extra_fields)
{
    void* sv = state_.stream();
    cudaStream_t s = static_cast<cudaStream_t>(sv);
    demag.set_stream(sv);
    extra_fields.set_stream(sv);

    bool stale = !gs2_.valid || !mat_eq(gs2_.mat, mat) || gs2_.dt != dt_;

    if (stale) {
        const double h = static_cast<double>(dt_);
        auto body = [&] {
            state_.save_m0();
            state_.zero_k_acc();
            run_stage(mat, demag, extra_fields, h * 0.5, 1.0/6.0);
            run_stage(mat, demag, extra_fields, h * 0.5, 2.0/6.0);
            run_stage(mat, demag, extra_fields, h * 1.0, 2.0/6.0);
            run_stage(mat, demag, extra_fields, 0.0,     1.0/6.0);
            launch_rk4_finalize(state_.d_m(), state_.d_m0(), state_.d_k_acc(),
                                 h, static_cast<int>(state_.N()), sv);
            launch_normalize(state_.d_m(), static_cast<int>(state_.N()), sv);
        };

        gs2_.valid = do_capture(s, gs2_.exec, body);
        gs2_.mat   = mat;
        gs2_.dt    = dt_;

        if (!gs2_.valid) {
            state_.sync();
            return;
        }
    }

    CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(gs2_.exec), s));
    state_.sync();
}

// ---------------------------------------------------------------------------
// step — FieldSumGPU + SpinTorqueSumGPU overload
//
// CUDA Graph capture is intentionally skipped here.  Torque magnitudes
// (J, J_c, etc.) are computed at kernel-launch time from the torque object's
// current field values; if they were captured into a graph the baked-in
// constants would become stale the moment the caller changes stt.J or
// sot.J_c between steps — giving silently wrong physics.
// Direct execution adds ~50-100 µs/step of kernel-launch overhead vs the
// graphed field-only overload, but is always correct.
// ---------------------------------------------------------------------------
void RK4IntegratorGPU::step(
    const Material& mat, IDemagGPU& demag,
    FieldSumGPU& extra_fields, SpinTorqueSumGPU& torques)
{
    void* sv = state_.stream();
    demag.set_stream(sv);
    extra_fields.set_stream(sv);
    torques.set_stream(sv);

    const double h = static_cast<double>(dt_);
    state_.save_m0();
    state_.zero_k_acc();
    run_stage(mat, demag, extra_fields, h * 0.5, 1.0/6.0, &torques);
    run_stage(mat, demag, extra_fields, h * 0.5, 2.0/6.0, &torques);
    run_stage(mat, demag, extra_fields, h * 1.0, 2.0/6.0, &torques);
    run_stage(mat, demag, extra_fields, 0.0,     1.0/6.0, &torques);
    launch_rk4_finalize(state_.d_m(), state_.d_m0(), state_.d_k_acc(),
                         h, static_cast<int>(state_.N()), sv);
    launch_normalize(state_.d_m(), static_cast<int>(state_.N()), sv);
    state_.sync();
}

}  // namespace micromag

#endif // MICROMAG_CUDA

