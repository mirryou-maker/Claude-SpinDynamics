// depondt_integrator_gpu.cu — Task 1-A: Depondt–Mertens rotation integrator.
//
// Increment 1-A scope: deterministic (T_K = 0) norm-exact LLG step. The finite-T
// Philox noise path (1-B) and the step-doubling/PI adaptive controller (1-C)
// attach at the hooks marked TODO(1-B)/TODO(1-C). See the header and
// CLAUDE-SD_FINITE_TEMP_ROADMAP.md (Task 1) for the full plan.
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/depondt_integrator_gpu.hpp"
#include "micromag/gpu_real.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA error: ") +      \
                                     cudaGetErrorString(_e));           \
    } while (0)

namespace micromag {

namespace {

constexpr int TPB = 256;
inline int nblocks(int n) { return (n + TPB - 1) / TPB; }

// ω(m,H) = gp·(H + α (m×H)),  gp = γ₀μ₀/(1+α²).  Then ṁ = ω×m reproduces the
// codebase LLG (dm/dt = -gp[m×H + α m×(m×H)]). One thread per cell; math in
// double for accuracy regardless of the GReal storage type.
// Buffers are flat [3·N] (x0..,y..? no: interleaved xyz per cell, i.e. 3*i+{0,1,2}
// matching GPUMagState layout). Task 2 prepends a replica stride of 3·N.
__global__ void omega_kernel(GReal* __restrict__ omega,
                             const GReal* __restrict__ m,
                             const GReal* __restrict__ H,
                             double gp, double alpha, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Component-major layout: buf[c*N + i].
    const double mx = m[i], my = m[N+i], mz = m[2*N+i];
    const double hx = H[i], hy = H[N+i], hz = H[2*N+i];
    // m × H
    const double cx = my*hz - mz*hy;
    const double cy = mz*hx - mx*hz;
    const double cz = mx*hy - my*hx;
    omega[i]     = static_cast<GReal>(gp * (hx + alpha * cx));
    omega[N+i]   = static_cast<GReal>(gp * (hy + alpha * cy));
    omega[2*N+i] = static_cast<GReal>(gp * (hz + alpha * cz));
}

// Rodrigues rotation of m_in about axis ω̂ by angle θ = |ω|·dt → m_out.
// Exact norm preservation. |ω|→0 falls back to identity.
__global__ void rotate_kernel(GReal* __restrict__ m_out,
                             const GReal* __restrict__ m_in,
                             const GReal* __restrict__ omega,
                             double dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Component-major layout: buf[c*N + i].
    const double wx = omega[i], wy = omega[N+i], wz = omega[2*N+i];
    const double wnorm = sqrt(wx*wx + wy*wy + wz*wz);
    const double mx = m_in[i], my = m_in[N+i], mz = m_in[2*N+i];
    if (wnorm < 1e-30) {
        m_out[i]     = static_cast<GReal>(mx);
        m_out[N+i]   = static_cast<GReal>(my);
        m_out[2*N+i] = static_cast<GReal>(mz);
        return;
    }
    const double th = wnorm * dt;
    const double s = sin(th), c = cos(th);
    const double ex = wx / wnorm, ey = wy / wnorm, ez = wz / wnorm;
    // e × m
    const double kx = ey*mz - ez*my;
    const double ky = ez*mx - ex*mz;
    const double kz = ex*my - ey*mx;
    const double edotm = ex*mx + ey*my + ez*mz;
    // Rodrigues: m cosθ + (e×m) sinθ + e (e·m)(1-cosθ)
    m_out[i]     = static_cast<GReal>(mx*c + kx*s + ex*edotm*(1.0 - c));
    m_out[N+i]   = static_cast<GReal>(my*c + ky*s + ey*edotm*(1.0 - c));
    m_out[2*N+i] = static_cast<GReal>(mz*c + kz*s + ez*edotm*(1.0 - c));
}

// out = ½(a + b), elementwise over [3·N].
__global__ void avg_kernel(GReal* __restrict__ out,
                          const GReal* __restrict__ a,
                          const GReal* __restrict__ b, int N3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    out[i] = static_cast<GReal>(0.5 * (double(a[i]) + double(b[i])));
}

// 1-B: device-side Philox thermal field.  H += σ·η, η ~ N(0,1) per component.
// Counter-based (curandStatePhilox4_32_10_t): fully reproducible from
// (seed, cell, offset) with NO stored state — calling with the same offset in
// both predictor and corrector reuses the SAME realisation (Stratonovich).
// Component-major layout buf[c*N+i]. `offset` separates steps/sub-steps.
// Task 2 will fold the replica id into the subsequence argument.
__global__ void add_thermal_noise_kernel(GReal* __restrict__ H, double sigma,
                                        unsigned seed, unsigned long long offset,
                                        int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandStatePhilox4_32_10_t st;
    curand_init((unsigned long long)seed, /*subsequence=*/(unsigned long long)i,
                /*offset=*/offset, &st);
    const double ex = curand_normal_double(&st);
    const double ey = curand_normal_double(&st);
    const double ez = curand_normal_double(&st);
    H[i]     = static_cast<GReal>(double(H[i])     + sigma * ex);
    H[N+i]   = static_cast<GReal>(double(H[N+i])   + sigma * ey);
    H[2*N+i] = static_cast<GReal>(double(H[2*N+i]) + sigma * ez);
}

// Σ (a−b)² over [3·N] → *out (double). One block-reduction then atomicAdd.
// Used for the adaptive embedded error norm (1-C). Small D2H (one scalar/step).
__global__ void sumsq_diff_kernel(const GReal* __restrict__ a,
                                 const GReal* __restrict__ b,
                                 double* __restrict__ out, int N3)
{
    __shared__ double sh[TPB];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double v = 0.0;
    if (i < N3) { const double d = double(a[i]) - double(b[i]); v = d * d; }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

}  // namespace

// ---------------------------------------------------------------------------
DepondtMertensGPU::DepondtMertensGPU(const StructuredGrid& grid,
                                     Real dt, unsigned seed)
    : state_(grid), dt_(dt),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      seed_(seed)
{
    if (dt <= Real{0})
        throw std::invalid_argument("DepondtMertensGPU: dt must be > 0");
    CUDA_CHECK(cudaMalloc(&d_err_, sizeof(double)));
}

DepondtMertensGPU::~DepondtMertensGPU() {
    if (d_err_) cudaFree(d_err_);
}

// Roadmap 1-D: the ONE place σ ∝ 1/√Δt is formed.  σ = √(2α k_B T /(μ₀ Ms γ₀ V Δt)).
double DepondtMertensGPU::therm_sigma(const Material& mat, double dt,
                                      double dx, double dy, double dz, double T_K)
{
    if (T_K <= 0.0 || mat.Ms <= 0.0) return 0.0;
    const double V = dx * dy * dz;
    // σ_H (A/m) — standard Brown thermal field with the A/m unit correction:
    //
    //   σ = √( 2α k_B T / (μ₀² Ms γ₀ V dt) )
    //
    // The single correction vs the form previously used in the codebase is
    // μ₀ → μ₀²: the field enters the LLG in A/m (B = μ₀H), so H_th = B_th/μ₀
    // carries an extra 1/μ₀² in its variance. (The bare-μ₀ form made the
    // field-coupled thermal ~1/μ₀ too weak — <mz>=1 where L(ξ)≈0.67.)
    // Validated against the Langevin law ⟨m_z⟩ = L(ξ), ξ = μ₀ Ms V H/k_B T for
    // ξ = 1,3,6. Same formula as HeunIntegratorGPU / ThermalField (an earlier
    // apparent "factor-2 scheme difference" was a d_H stream race, since fixed).
    const double num = 2.0 * mat.alpha * constants::k_B * T_K;
    const double den = constants::mu_0 * constants::mu_0
                       * mat.Ms * constants::gamma_0 * V * dt;
    return std::sqrt(num / den);
}

// ---------------------------------------------------------------------------
// One Depondt–Mertens predictor–corrector sub-step m_in → m_out over `dt`.
void DepondtMertensGPU::substep(const Material& mat, IDemagGPU& demag,
                                FieldSumGPU& fields, const GReal* m_in,
                                GReal* m_out, double dt, Real T_K,
                                unsigned long long noise_offset)
{
    auto stream = reinterpret_cast<cudaStream_t>(state_.stream());
    const int N  = static_cast<int>(state_.N());
    const int N3 = 3 * N;
    const double gp = constants::gamma_0 * constants::mu_0
                      / (1.0 + double(mat.alpha) * double(mat.alpha));
    const double sigma = therm_sigma(mat, dt, dx_, dy_, dz_, double(T_K));

    GReal* d_m0 = state_.d_m0();     // saved mⁿ (scratch)
    GReal* d_H  = state_.d_H();
    GReal* d_w1 = state_.d_ki();     // ω1 / ω̄
    GReal* d_w2 = state_.d_k_acc();  // ω2

    // save mⁿ so m_out may alias m_in
    CUDA_CHECK(cudaMemcpyAsync(d_m0, m_in, N3 * sizeof(GReal),
                               cudaMemcpyDeviceToDevice, stream));
    // predictor: ω1 at mⁿ,  m* = R(ω1,dt) mⁿ
    state_.zero_H();
    fields.accumulate_gpu_ptr(d_m0, mat, d_H);
    demag.accumulate_gpu_ptr(d_m0, mat, d_H);
    if (sigma > 0.0)
        add_thermal_noise_kernel<<<nblocks(N), TPB, 0, stream>>>(
            d_H, sigma, seed_, noise_offset, N);
    omega_kernel<<<nblocks(N), TPB, 0, stream>>>(d_w1, d_m0, d_H, gp, mat.alpha, N);
    rotate_kernel<<<nblocks(N), TPB, 0, stream>>>(m_out, d_m0, d_w1, dt, N);
    // corrector: ω2 at m*,  mⁿ⁺¹ = R(½(ω1+ω2),dt) mⁿ  (SAME noise realisation)
    state_.zero_H();
    fields.accumulate_gpu_ptr(m_out, mat, d_H);
    demag.accumulate_gpu_ptr(m_out, mat, d_H);
    if (sigma > 0.0)
        add_thermal_noise_kernel<<<nblocks(N), TPB, 0, stream>>>(
            d_H, sigma, seed_, noise_offset, N);
    omega_kernel<<<nblocks(N), TPB, 0, stream>>>(d_w2, m_out, d_H, gp, mat.alpha, N);
    avg_kernel<<<nblocks(N3), TPB, 0, stream>>>(d_w1, d_w1, d_w2, N3);
    rotate_kernel<<<nblocks(N), TPB, 0, stream>>>(m_out, d_m0, d_w1, dt, N);
}

// ---------------------------------------------------------------------------
// 1-C: adaptive step. Embedded error = ‖ω2−ω1‖·dt (no step-doubling, no
// rejection — always accept the corrector, adapt dt forward; roadmap §5.3 (b)).
Real DepondtMertensGPU::step_adaptive(const Material& mat, IDemagGPU& demag,
                                      FieldSumGPU& fields, Real T_K)
{
    auto stream = reinterpret_cast<cudaStream_t>(state_.stream());
    const int N  = static_cast<int>(state_.N());
    const int N3 = 3 * N;
    const double gp = constants::gamma_0 * constants::mu_0
                      / (1.0 + double(mat.alpha) * double(mat.alpha));
    const double h = static_cast<double>(dt_);
    const double sigma = therm_sigma(mat, h, dx_, dy_, dz_, double(T_K));
    const unsigned long long offset = step_index_ * 3ull;

    GReal* d_m  = state_.d_m();
    GReal* d_m0 = state_.d_m0();
    GReal* d_H  = state_.d_H();
    GReal* d_w1 = state_.d_ki();     // ω1 / ω̄
    GReal* d_w2 = state_.d_k_acc();  // ω2

    CUDA_CHECK(cudaMemcpyAsync(d_m0, d_m, N3 * sizeof(GReal),
                               cudaMemcpyDeviceToDevice, stream));
    // predictor
    state_.zero_H();
    fields.accumulate_gpu_ptr(d_m0, mat, d_H);
    demag.accumulate_gpu_ptr(d_m0, mat, d_H);
    if (sigma > 0.0)
        add_thermal_noise_kernel<<<nblocks(N), TPB, 0, stream>>>(
            d_H, sigma, seed_, offset, N);
    omega_kernel<<<nblocks(N), TPB, 0, stream>>>(d_w1, d_m0, d_H, gp, mat.alpha, N);
    rotate_kernel<<<nblocks(N), TPB, 0, stream>>>(d_m, d_m0, d_w1, h, N);
    // corrector ω2 (SAME noise)
    state_.zero_H();
    fields.accumulate_gpu_ptr(d_m, mat, d_H);
    demag.accumulate_gpu_ptr(d_m, mat, d_H);
    if (sigma > 0.0)
        add_thermal_noise_kernel<<<nblocks(N), TPB, 0, stream>>>(
            d_H, sigma, seed_, offset, N);
    omega_kernel<<<nblocks(N), TPB, 0, stream>>>(d_w2, d_m, d_H, gp, mat.alpha, N);
    // embedded error ‖ω2−ω1‖ (before averaging destroys ω1)
    CUDA_CHECK(cudaMemsetAsync(d_err_, 0, sizeof(double), stream));
    sumsq_diff_kernel<<<nblocks(N3), TPB, 0, stream>>>(d_w2, d_w1, d_err_, N3);
    // finalise mⁿ⁺¹
    avg_kernel<<<nblocks(N3), TPB, 0, stream>>>(d_w1, d_w1, d_w2, N3);
    rotate_kernel<<<nblocks(N), TPB, 0, stream>>>(d_m, d_m0, d_w1, h, N);

    double sumsq = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&sumsq, d_err_, sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    const double err = std::sqrt(sumsq / double(N3)) * h;   // ~rad
    const Real taken = dt_;
    ++step_index_;

    // PI-lite controller (order-2 heuristic on the angular discrepancy).
    const double tol = double(opts_.atol) + double(opts_.rtol);
    double fac = (err <= 1e-300) ? double(opts_.fac_max)
                                 : double(opts_.safety) * std::sqrt(tol / err);
    fac = std::min(double(opts_.fac_max), std::max(double(opts_.fac_min), fac));
    double dt_new = double(dt_) * fac;
    dt_new = std::min(double(opts_.dt_max), std::max(double(opts_.dt_min), dt_new));
    dt_ = static_cast<Real>(dt_new);
    return taken;
}

// ---------------------------------------------------------------------------
Real DepondtMertensGPU::step(const Material& mat, IDemagGPU& demag,
                             FieldSumGPU& extra_fields,
                             Real T_K, SpinTorqueSumGPU* torques)
{
    if (torques && torques->size() > 0)
        throw std::runtime_error(
            "DepondtMertensGPU: spin-torque path not yet wired (Task 1 follow-up).");

    // Bind all field evaluations to the integrator's stream, else the field
    // kernels race with the rotation/noise kernels on d_H (non-deterministic
    // results even with a fixed seed). Mirrors HeunIntegratorGPU.
    demag.set_stream(state_.stream());
    extra_fields.set_stream(state_.stream());

    const double h = static_cast<double>(dt_);
    if (!opts_.adaptive) {
        substep(mat, demag, extra_fields, state_.d_m(), state_.d_m(), h, T_K,
                /*noise_offset=*/step_index_ * 3ull);
        CUDA_CHECK(cudaGetLastError());
        ++step_index_;
        return dt_;
    }

    // --- 1-C: adaptive step-doubling + PI controller ---------------------
    return step_adaptive(mat, demag, extra_fields, T_K);
}

}  // namespace micromag

#endif // MICROMAG_CUDA
