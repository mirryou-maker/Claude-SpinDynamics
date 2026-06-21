// heun_integrator_gpu.cu ??G8: GPU Heun integrator (SLLG finite-T)
//
// Stratonovich Heun scheme:
//   1. Generate 管^n once per step: N(0, ?) via cuRAND
//      ? = sqrt(2慣 k_B T / (關? Ms 款? V ?t))  [A/m]
//
//   Predictor:
//     H = H_eff(m^n) + 管^n
//     k1 = f(m^n, H)
//     m_pred = normalize(m^n + dt쨌k1)
//
//   Corrector (SAME 管^n ??Stratonovich):
//     H = H_eff(m_pred) + 管^n
//     k2 = f(m_pred, H)
//     m^{n+1} = normalize(m^n + dt/2쨌(k1 + k2))
//
// T_K=0: ?=0, cuRAND skipped, gives deterministic Heun ODE integrator.
//
// GPUMagState buffer reuse:
//   d_m_     ??m^n
//   d_m0_    ??m_pred
//   d_ki_    ??k1
//   d_k_acc_ ??k2

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/gpu_real.hpp"
#include "micromag/heun_integrator_gpu.hpp"
#include "micromag/rk4_gpu.hpp"
#include "micromag/spin_torque_gpu.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA(heun): ")       \
                                   + cudaGetErrorString(_e));           \
    } while (0)
#define CURAND_CHECK(call)                                              \
    do {                                                                \
        curandStatus_t _r = (call);                                     \
        if (_r != CURAND_STATUS_SUCCESS)                                \
            throw std::runtime_error("cuRAND error " +                  \
                                     std::to_string((int)_r));          \
    } while (0)

namespace micromag {

// ---------------------------------------------------------------------------
// P4-style CUDA Graph helpers (same logic as rk4_integrator_gpu.cu)
// ---------------------------------------------------------------------------
static bool heun_mat_eq(const Material& a, const Material& b) {
    return a.Ms == b.Ms && a.A_exchange == b.A_exchange &&
           a.K_uniaxial == b.K_uniaxial && a.Ku2 == b.Ku2 &&
           a.alpha == b.alpha &&
           a.easy_axis.x == b.easy_axis.x &&
           a.easy_axis.y == b.easy_axis.y &&
           a.easy_axis.z == b.easy_axis.z;
}

static void heun_free_graph(void*& exec_v) {
    if (exec_v) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_v));
        exec_v = nullptr;
    }
}

template<class F>
static bool heun_do_capture(cudaStream_t s, void*& exec_out, F body) {
    heun_free_graph(exec_out);
#ifdef MICROMAG_VKFFT
    body();
    return false;
#endif
    static const bool profiling_active =
        (std::getenv("MICROMAG_DEMAG_PROFILE") != nullptr);
    if (profiling_active) { body(); return false; }

    cudaError_t begin_err = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (begin_err != cudaSuccess) { cudaGetLastError(); body(); return false; }
    bool body_ok = true;
    try { body(); } catch (...) { body_ok = false; }
    cudaGraph_t g = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(s, &g);
    if (!body_ok || end_err != cudaSuccess || !g) {
        if (g) cudaGraphDestroy(g);
        cudaGetLastError(); body(); return false;
    }
    cudaGraphExec_t ge = nullptr;
    cudaError_t inst_err = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
    if (inst_err != cudaSuccess) { cudaGetLastError(); body(); return false; }
    exec_out = static_cast<void*>(ge);
    return true;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
HeunIntegratorGPU::HeunIntegratorGPU(const StructuredGrid& grid,
                                       Real dt, unsigned seed)
    : state_(grid), dt_(dt),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz())
{
    // N_pad: 3*N rounded up to even (cuRAND requirement)
    const size_t N3 = 3 * state_.N();
    N_pad_ = (N3 % 2 == 0) ? N3 : N3 + 1;
    CUDA_CHECK(cudaMalloc(&d_noise_, N_pad_ * sizeof(GReal)));  // P11: GReal noise buffer

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)seed));
    curand_gen_ = static_cast<void*>(gen);
}

HeunIntegratorGPU::~HeunIntegratorGPU() {
    heun_free_graph(gs1_.exec);
    heun_free_graph(gs2_.exec);
    if (curand_gen_)
        curandDestroyGenerator(static_cast<curandGenerator_t>(curand_gen_));
    cudaFree(d_noise_);
}

// ---------------------------------------------------------------------------
// run_half ??accumulate fields into d_H, compute torque into d_ki
// ---------------------------------------------------------------------------
void HeunIntegratorGPU::run_half(
    const Material& mat,
    const GReal*    d_m_in,
    GReal*          d_H,
    GReal*          d_ki,
    IDemagGPU& demag, ExchangeFieldGPU& exch,
    ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
    bool add_noise)
{
    void* s = state_.stream();
    const int N = static_cast<int>(state_.N());

    // All fields are on state_.stream() (set by step()); stream ordering suffices.
    state_.zero_H();
    exch.accumulate_gpu_ptr(d_m_in, mat, d_H);
    zeeman.accumulate_gpu_ptr(d_m_in, mat, d_H);
    if (aniso)
        aniso->accumulate_gpu_ptr(d_m_in, mat, d_H);
    demag.accumulate_gpu_ptr(d_m_in, mat, d_H);

    if (add_noise)
        launch_add_3N(d_H, static_cast<const GReal*>(d_noise_), N, s);

    launch_llg_torque(d_ki, d_m_in, d_H, mat.alpha, N, s);
}

// ---------------------------------------------------------------------------
// step — one complete Stratonovich Heun step (fixed-field overload)
//
// T_K=0: deterministic Heun ODE, eligible for CUDA Graph replay.
// T_K>0: cuRAND noise varies each step — Graph capture skipped.
// ---------------------------------------------------------------------------
void HeunIntegratorGPU::step(
    const Material& mat,
    IDemagGPU& demag, ExchangeFieldGPU& exch,
    ZeemanFieldGPU& zeeman, Real T_K,
    UniaxialAnisotropyFieldGPU* aniso)
{
    const int  N = static_cast<int>(state_.N());
    const Real h = dt_;
    void*      sv = state_.stream();
    cudaStream_t s = static_cast<cudaStream_t>(sv);

    demag.set_stream(sv);
    exch.set_stream(sv);
    zeeman.set_stream(sv);
    if (aniso) aniso->set_stream(sv);

    const bool thermal = (T_K > 0.0 && mat.Ms > 0.0);

    // T=0: try CUDA Graph replay
    if (!thermal) {
        bool stale = !gs1_.valid || !heun_mat_eq(gs1_.mat, mat) || gs1_.dt != dt_;
        if (stale) {
            auto body = [&] {
                run_half(mat, state_.d_m(), state_.d_H(), state_.d_ki(),
                         demag, exch, zeeman, aniso, false);
                launch_rk4_stage(state_.d_m0(), state_.d_m(), state_.d_ki(), h, N, sv);
                launch_normalize(state_.d_m0(), N, sv);
                run_half(mat, state_.d_m0(), state_.d_H(), state_.d_k_acc(),
                         demag, exch, zeeman, aniso, false);
                launch_heun_corrector(state_.d_m(), state_.d_ki(), state_.d_k_acc(),
                                      h * 0.5, N, sv);
                launch_normalize(state_.d_m(), N, sv);
            };
            gs1_.valid = heun_do_capture(s, gs1_.exec, body);
            gs1_.mat   = mat;
            gs1_.dt    = dt_;
            if (!gs1_.valid) { state_.sync(); return; }
        }
        CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(gs1_.exec), s));
        state_.sync();
        return;
    }

    // T>0: thermal noise — cannot use Graph (cuRAND must run each step)
    if (thermal) {
        // ? = sqrt(2慣 k_B T / (關? Ms 款? V dt))
        const Real V   = dx_ * dy_ * dz_;
        const Real num = 2.0 * mat.alpha * constants::k_B * T_K;
        const Real den = constants::mu_0 * mat.Ms * constants::gamma_0 * V * h;
        const double sig = std::sqrt(num / den);

        // cuRAND: fills d_noise_[N_pad_] with N(0, sig) ??type matches GReal.
#ifdef MICROMAG_FLOAT32
        CURAND_CHECK(curandGenerateNormal(
            static_cast<curandGenerator_t>(curand_gen_),
            static_cast<float*>(d_noise_),
            static_cast<size_t>(N_pad_), 0.0f, static_cast<float>(sig)));
#else
        CURAND_CHECK(curandGenerateNormalDouble(
            static_cast<curandGenerator_t>(curand_gen_),
            static_cast<double*>(d_noise_),
            static_cast<size_t>(N_pad_), 0.0, sig));
#endif
    }

    // Predictor: k1 = f(m^n, H_eff + η^n);  m_pred = normalize(m^n + dt·k1)
    run_half(mat, state_.d_m(), state_.d_H(), state_.d_ki(),
             demag, exch, zeeman, aniso, thermal);
    launch_rk4_stage(state_.d_m0(), state_.d_m(), state_.d_ki(), h, N, sv);
    launch_normalize(state_.d_m0(), N, sv);

    // Corrector: k2 = f(m_pred, H_eff + η^n)  [SAME noise — Stratonovich]
    run_half(mat, state_.d_m0(), state_.d_H(), state_.d_k_acc(),
             demag, exch, zeeman, aniso, thermal);
    launch_heun_corrector(state_.d_m(), state_.d_ki(), state_.d_k_acc(),
                          h * 0.5, N, sv);
    launch_normalize(state_.d_m(), N, sv);
    state_.sync();
}

// ---------------------------------------------------------------------------
// run_half ??FieldSumGPU overload
// ---------------------------------------------------------------------------
void HeunIntegratorGPU::run_half(
    const Material& mat,
    const GReal*    d_m_in,
    GReal*          d_H,
    GReal*          d_ki,
    IDemagGPU& demag, FieldSumGPU& extra_fields,
    bool add_noise,
    SpinTorqueSumGPU* torques)
{
    void* s = state_.stream();
    const int N = static_cast<int>(state_.N());

    // All fields on state_.stream() (set by step()); stream ordering suffices.
    state_.zero_H();
    extra_fields.accumulate_gpu_ptr(d_m_in, mat, d_H);
    demag.accumulate_gpu_ptr(d_m_in, mat, d_H);

    if (add_noise)
        launch_add_3N(d_H, static_cast<const GReal*>(d_noise_), N, s);

    launch_llg_torque(d_ki, d_m_in, d_H, mat.alpha, N, s);

    if (torques && torques->size() > 0)
        torques->accumulate_gpu_ptr(d_m_in, mat, d_ki);
}

// ---------------------------------------------------------------------------
// step — FieldSumGPU overload (optional spin torques, optional noise)
//
// T_K=0 without torques: CUDA Graph eligible.
// T_K=0 with torques: also Graph eligible (torques are deterministic).
// T_K>0: cuRAND noise — no Graph.
// ---------------------------------------------------------------------------
void HeunIntegratorGPU::step(
    const Material& mat, IDemagGPU& demag,
    FieldSumGPU& extra_fields,
    Real T_K, SpinTorqueSumGPU* torques)
{
    const int  N = static_cast<int>(state_.N());
    const Real h = dt_;
    void*      sv = state_.stream();
    cudaStream_t s = static_cast<cudaStream_t>(sv);

    demag.set_stream(sv);
    extra_fields.set_stream(sv);
    if (torques) torques->set_stream(sv);

    const bool thermal = (T_K > 0.0 && mat.Ms > 0.0);

    // T=0: try CUDA Graph replay
    if (!thermal) {
        bool stale = !gs2_.valid || !heun_mat_eq(gs2_.mat, mat) || gs2_.dt != dt_;
        if (stale) {
            auto body = [&] {
                run_half(mat, state_.d_m(), state_.d_H(), state_.d_ki(),
                         demag, extra_fields, false, torques);
                launch_rk4_stage(state_.d_m0(), state_.d_m(), state_.d_ki(), h, N, sv);
                launch_normalize(state_.d_m0(), N, sv);
                run_half(mat, state_.d_m0(), state_.d_H(), state_.d_k_acc(),
                         demag, extra_fields, false, torques);
                launch_heun_corrector(state_.d_m(), state_.d_ki(), state_.d_k_acc(),
                                      h * 0.5, N, sv);
                launch_normalize(state_.d_m(), N, sv);
            };
            gs2_.valid = heun_do_capture(s, gs2_.exec, body);
            gs2_.mat   = mat;
            gs2_.dt    = dt_;
            if (!gs2_.valid) { state_.sync(); return; }
        }
        CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(gs2_.exec), s));
        state_.sync();
        return;
    }

    // T>0: generate thermal noise then run directly
    const Real V   = dx_ * dy_ * dz_;
    const Real num = 2.0 * mat.alpha * constants::k_B * T_K;
    const Real den = constants::mu_0 * mat.Ms * constants::gamma_0 * V * h;
    const double sig = std::sqrt(num / den);
#ifdef MICROMAG_FLOAT32
    CURAND_CHECK(curandGenerateNormal(
        static_cast<curandGenerator_t>(curand_gen_),
        static_cast<float*>(d_noise_),
        static_cast<size_t>(N_pad_), 0.0f, static_cast<float>(sig)));
#else
    CURAND_CHECK(curandGenerateNormalDouble(
        static_cast<curandGenerator_t>(curand_gen_),
        static_cast<double*>(d_noise_),
        static_cast<size_t>(N_pad_), 0.0, sig));
#endif

    run_half(mat, state_.d_m(), state_.d_H(), state_.d_ki(),
             demag, extra_fields, true, torques);
    launch_rk4_stage(state_.d_m0(), state_.d_m(), state_.d_ki(), h, N, sv);
    launch_normalize(state_.d_m0(), N, sv);
    run_half(mat, state_.d_m0(), state_.d_H(), state_.d_k_acc(),
             demag, extra_fields, true, torques);
    launch_heun_corrector(state_.d_m(), state_.d_ki(), state_.d_k_acc(),
                          h * 0.5, N, sv);
    launch_normalize(state_.d_m(), N, sv);
    state_.sync();
}

}  // namespace micromag

#endif // MICROMAG_CUDA

