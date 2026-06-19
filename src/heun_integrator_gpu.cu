// heun_integrator_gpu.cu — G8: GPU Heun integrator (SLLG finite-T)
//
// Stratonovich Heun scheme:
//   1. Generate η^n once per step: N(0, σ) via cuRAND
//      σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V Δt))  [A/m]
//
//   Predictor:
//     H = H_eff(m^n) + η^n
//     k1 = f(m^n, H)
//     m_pred = normalize(m^n + dt·k1)
//
//   Corrector (SAME η^n — Stratonovich):
//     H = H_eff(m_pred) + η^n
//     k2 = f(m_pred, H)
//     m^{n+1} = normalize(m^n + dt/2·(k1 + k2))
//
// T_K=0: σ=0, cuRAND skipped, gives deterministic Heun ODE integrator.
//
// GPUMagState buffer reuse:
//   d_m_     → m^n
//   d_m0_    → m_pred
//   d_ki_    → k1
//   d_k_acc_ → k2

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/heun_integrator_gpu.hpp"
#include "micromag/rk4_gpu.hpp"
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
    CUDA_CHECK(cudaMalloc(&d_noise_, N_pad_ * sizeof(double)));

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
// run_half — accumulate fields into d_H, compute torque into d_ki
// ---------------------------------------------------------------------------
void HeunIntegratorGPU::run_half(
    const Material& mat,
    const double*   d_m_in,
    double*         d_H,
    double*         d_ki,
    IDemagGPU& demag, ExchangeFieldGPU& exch,
    ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
    bool add_noise)
{
    void* s = state_.stream();
    const int N = static_cast<int>(state_.N());

    CUDA_CHECK(cudaMemsetAsync(d_H, 0, 3*N*sizeof(double),
                               static_cast<cudaStream_t>(s)));

    exch.accumulate_gpu_ptr(d_m_in, mat, d_H);
    CUDA_CHECK(cudaDeviceSynchronize());

    zeeman.accumulate_gpu_ptr(d_m_in, mat, d_H);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (aniso) {
        aniso->accumulate_gpu_ptr(d_m_in, mat, d_H);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    demag.accumulate_gpu_ptr(d_m_in, mat, d_H);   // syncs internally

    if (add_noise)
        launch_add_3N(d_H,
                      reinterpret_cast<const double*>(d_noise_),
                      N, s);

    launch_llg_torque(d_ki, d_m_in, d_H, mat.alpha, N, s);
}

// ---------------------------------------------------------------------------
// step — one complete Stratonovich Heun step
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

    // ---- Generate thermal noise η^n (once per step) -------------------------
    const bool thermal = (T_K > 0.0 && mat.Ms > 0.0);
    if (thermal) {
        // σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V dt))
        const Real V   = dx_ * dy_ * dz_;
        const Real num = 2.0 * mat.alpha * constants::k_B * T_K;
        const Real den = constants::mu_0 * mat.Ms * constants::gamma_0 * V * h;
        const double sig = std::sqrt(num / den);

        // cuRAND: fills d_noise_[N_pad_] with N(0, sig) doubles
        CURAND_CHECK(curandGenerateNormalDouble(
            static_cast<curandGenerator_t>(curand_gen_),
            reinterpret_cast<double*>(d_noise_),
            static_cast<size_t>(N_pad_), 0.0, sig));
    }

    // ---- Predictor -----------------------------------------------------------
    // k1 = f(m^n, H_eff + η^n);  m_pred = normalize(m^n + dt·k1)
    run_half(mat, state_.d_m(), state_.d_H(), state_.d_ki(),
             demag, exch, zeeman, aniso, thermal);

    launch_rk4_stage(state_.d_m0(), state_.d_m(), state_.d_ki(), h, N, s);
    launch_normalize(state_.d_m0(), N, s);
    state_.sync();

    // ---- Corrector -----------------------------------------------------------
    // k2 = f(m_pred, H_eff + η^n)  [SAME noise — Stratonovich]
    run_half(mat, state_.d_m0(), state_.d_H(), state_.d_k_acc(),
             demag, exch, zeeman, aniso, thermal);

    // m^{n+1} = normalize(m^n + dt/2·(k1 + k2))
    launch_heun_corrector(state_.d_m(), state_.d_ki(), state_.d_k_acc(),
                           h * 0.5, N, s);
    launch_normalize(state_.d_m(), N, s);
    state_.sync();
}

}  // namespace micromag

#endif // MICROMAG_CUDA
