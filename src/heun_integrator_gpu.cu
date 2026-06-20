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
// step ??one complete Stratonovich Heun step
// ---------------------------------------------------------------------------
void HeunIntegratorGPU::step(
    const Material& mat,
    IDemagGPU& demag, ExchangeFieldGPU& exch,
    ZeemanFieldGPU& zeeman, Real T_K,
    UniaxialAnisotropyFieldGPU* aniso)
{
    const int  N = static_cast<int>(state_.N());
    const Real h = dt_;
    void*      s = state_.stream();

    demag.set_stream(s);
    exch.set_stream(s);
    zeeman.set_stream(s);
    if (aniso) aniso->set_stream(s);

    // ---- Generate thermal noise 管^n (once per step) -------------------------
    const bool thermal = (T_K > 0.0 && mat.Ms > 0.0);
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

    // ---- Predictor -----------------------------------------------------------
    // k1 = f(m^n, H_eff + 管^n);  m_pred = normalize(m^n + dt쨌k1)
    run_half(mat,
             state_.d_m(),
             state_.d_H(),
             state_.d_ki(),
             demag, exch, zeeman, aniso, thermal);

    // Single-stream: stage + normalize ordered before corrector run_half on s.
    launch_rk4_stage(state_.d_m0(), state_.d_m(), state_.d_ki(), h, N, s);
    launch_normalize(state_.d_m0(), N, s);

    // ---- Corrector -----------------------------------------------------------
    // k2 = f(m_pred, H_eff + 管^n)  [SAME noise ??Stratonovich]
    run_half(mat,
             state_.d_m0(),
             state_.d_H(),
             state_.d_k_acc(),
             demag, exch, zeeman, aniso, thermal);

    // m^{n+1} = normalize(m^n + dt/2쨌(k1 + k2))
    launch_heun_corrector(state_.d_m(), state_.d_ki(), state_.d_k_acc(),
                           h * 0.5, N, s);
    launch_normalize(state_.d_m(), N, s);
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
// step ??FieldSumGPU overload (optional spin torques, optional noise)
// ---------------------------------------------------------------------------
void HeunIntegratorGPU::step(
    const Material& mat, IDemagGPU& demag,
    FieldSumGPU& extra_fields,
    Real T_K, SpinTorqueSumGPU* torques)
{
    const int  N = static_cast<int>(state_.N());
    const Real h = dt_;
    void*      s = state_.stream();

    demag.set_stream(s);
    extra_fields.set_stream(s);
    if (torques) torques->set_stream(s);

    const bool thermal = (T_K > 0.0 && mat.Ms > 0.0);
    if (thermal) {
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
    }

    // Predictor
    run_half(mat,
             state_.d_m(),
             state_.d_H(),
             state_.d_ki(),
             demag, extra_fields, thermal, torques);

    // Single-stream: stage + normalize ordered before corrector on s.
    launch_rk4_stage(state_.d_m0(), state_.d_m(), state_.d_ki(), h, N, s);
    launch_normalize(state_.d_m0(), N, s);

    // Corrector (same noise ??Stratonovich)
    run_half(mat,
             state_.d_m0(),
             state_.d_H(),
             state_.d_k_acc(),
             demag, extra_fields, thermal, torques);

    launch_heun_corrector(state_.d_m(), state_.d_ki(), state_.d_k_acc(),
                           h * 0.5, N, s);
    launch_normalize(state_.d_m(), N, s);
    state_.sync();
}

}  // namespace micromag

#endif // MICROMAG_CUDA

