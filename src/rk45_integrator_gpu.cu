// rk45_integrator_gpu.cu — GPU DOPRI5 adaptive LLG integrator
//
// DOPRI5 Butcher tableau (Dormand-Prince, FSAL):
//   k1 = f(m)
//   k2 = f(m + h/5*k1)
//   k3 = f(m + h*(3/40*k1 + 9/40*k2))
//   k4 = f(m + h*(44/45*k1 - 56/15*k2 + 32/9*k3))
//   k5 = f(m + h*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
//   k6 = f(m + h*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))
//   m5 = m + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
//   k7 = f(m5)                            ← FSAL: reused as k1 of next step
//   err = h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)
//
// Per-step D2H: one scalar double (error norm) — negligible vs. demag cost.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/rk45_integrator_gpu.hpp"
#include "micromag/rk4_gpu.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA(rk45): ") + cudaGetErrorString(_e)); \
} while(0)

namespace micromag {

// ---------------------------------------------------------------------------
RK45IntegratorGPU::RK45IntegratorGPU(const StructuredGrid& grid, Options opts)
    : state_(grid), opts_(opts), dt_(opts.dt_init)
{
    alloc_scratch(grid.size());
}

RK45IntegratorGPU::~RK45IntegratorGPU() {
    cudaFree(d_k1_);  cudaFree(d_k2_);  cudaFree(d_k3_);  cudaFree(d_k4_);
    cudaFree(d_k5_);  cudaFree(d_k6_);  cudaFree(d_k7_);
    cudaFree(d_m5_);  cudaFree(d_err_); cudaFree(d_err_sum_);
    cudaFree(d_m_stage_);
}

void RK45IntegratorGPU::alloc_scratch(size_t N) {
    const size_t bytes3N = 3 * N * sizeof(double);
    auto alloc3N = [&](double*& ptr) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes3N));
    };
    alloc3N(d_k1_);  alloc3N(d_k2_);  alloc3N(d_k3_);  alloc3N(d_k4_);
    alloc3N(d_k5_);  alloc3N(d_k6_);  alloc3N(d_k7_);
    alloc3N(d_m5_);  alloc3N(d_err_); alloc3N(d_m_stage_);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_err_sum_), sizeof(double)));
}

// ---------------------------------------------------------------------------
// eval_ki: compute H_eff(d_m_in) and LLG torque → d_ki_out
// ---------------------------------------------------------------------------
void RK45IntegratorGPU::eval_ki(
    const Material& mat,
    IDemagGPU& demag, ExchangeFieldGPU& exch,
    ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
    const double* d_m_in, double* d_ki_out)
{
    void*  s  = state_.stream();
    double* dH = state_.d_H();
    const int N = static_cast<int>(state_.N());

    state_.zero_H();
    state_.sync();

    exch.accumulate_gpu_ptr(d_m_in, mat, dH);
    CUDA_CHECK(cudaDeviceSynchronize());

    zeeman.accumulate_gpu_ptr(d_m_in, mat, dH);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (aniso) {
        aniso->accumulate_gpu_ptr(d_m_in, mat, dH);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // demag syncs internally
    demag.accumulate_gpu_ptr(d_m_in, mat, dH);

    launch_llg_torque(d_ki_out, d_m_in, dH, mat.alpha, N, s);
    state_.sync();
}

// ---------------------------------------------------------------------------
// step — one DOPRI5 trial-accept loop; returns dt taken
// ---------------------------------------------------------------------------
Real RK45IntegratorGPU::step(
    const Material& mat,
    IDemagGPU& demag, ExchangeFieldGPU& exch,
    ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso)
{
    const double* dm = state_.d_m();
    const int N = static_cast<int>(state_.N());
    void*  s = state_.stream();

    // k1 = f(m)  — skip after an accepted step (FSAL: k1 = previous k7)
    if (!k1_valid_)
        eval_ki(mat, demag, exch, zeeman, aniso, dm, d_k1_);

    for (;;) {
        const double h = static_cast<double>(dt_);

        // ----- Stage 2 -----
        // m_stage = m + h/5 * k1  (reuse existing kernel)
        launch_rk4_stage(d_m_stage_, dm, d_k1_, h/5.0, N, s);
        state_.sync();
        eval_ki(mat, demag, exch, zeeman, aniso, d_m_stage_, d_k2_);

        // ----- Stage 3 -----
        launch_dopri5_stage3(d_m_stage_, dm, h, d_k1_, d_k2_, N, s);
        state_.sync();
        eval_ki(mat, demag, exch, zeeman, aniso, d_m_stage_, d_k3_);

        // ----- Stage 4 -----
        launch_dopri5_stage4(d_m_stage_, dm, h, d_k1_, d_k2_, d_k3_, N, s);
        state_.sync();
        eval_ki(mat, demag, exch, zeeman, aniso, d_m_stage_, d_k4_);

        // ----- Stage 5 -----
        launch_dopri5_stage5(d_m_stage_, dm, h, d_k1_, d_k2_, d_k3_, d_k4_, N, s);
        state_.sync();
        eval_ki(mat, demag, exch, zeeman, aniso, d_m_stage_, d_k5_);

        // ----- Stage 6 -----
        launch_dopri5_stage6(d_m_stage_, dm, h, d_k1_, d_k2_, d_k3_, d_k4_, d_k5_, N, s);
        state_.sync();
        eval_ki(mat, demag, exch, zeeman, aniso, d_m_stage_, d_k6_);

        // ----- 5th-order solution -----
        launch_dopri5_m5(d_m5_, dm, h, d_k1_, d_k3_, d_k4_, d_k5_, d_k6_, N, s);
        state_.sync();

        // ----- Stage 7 (FSAL): k7 = f(m5) -----
        eval_ki(mat, demag, exch, zeeman, aniso, d_m5_, d_k7_);

        // ----- Error estimate and RMS norm -----
        launch_dopri5_err(d_err_, h, d_k1_, d_k3_, d_k4_, d_k5_, d_k6_, d_k7_, N, s);
        const double err_norm = launch_dopri5_err_norm(
            d_err_sum_, d_err_, dm, d_m5_,
            static_cast<double>(opts_.rtol),
            static_cast<double>(opts_.atol), N, s);

        // ----- Step-size factor (same formula as CPU RK45) -----
        const double fac_raw = opts_.safety * std::pow(1.0 / std::max(err_norm, 1e-20), 0.2);

        if (err_norm <= 1.0) {
            // Accept step: m ← normalize(m5)
            CUDA_CHECK(cudaMemcpyAsync(
                state_.d_m(), d_m5_, 3 * N * sizeof(double),
                cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(s)));
            launch_normalize(state_.d_m(), N, s);
            state_.sync();

            // FSAL: k7 → k1 for next step
            std::swap(d_k1_, d_k7_);
            k1_valid_ = true;

            dt_ = static_cast<Real>(std::min(
                static_cast<double>(opts_.dt_max),
                dt_ * std::min(static_cast<double>(opts_.fac_max), fac_raw)));
            ++n_accepted_;
            return static_cast<Real>(h);
        }

        // Reject step: reduce dt, retry (k1 still valid — m unchanged)
        dt_ = static_cast<Real>(std::max(
            static_cast<double>(opts_.dt_min),
            dt_ * std::max(static_cast<double>(opts_.fac_min), fac_raw)));
        k1_valid_ = true;
        ++n_rejected_;

        if (dt_ <= opts_.dt_min)
            throw std::runtime_error(
                "RK45IntegratorGPU: step size reached minimum — solution may be stiff");
    }
}

}  // namespace micromag

#endif // MICROMAG_CUDA
