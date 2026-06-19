// rk4_gpu.cu — G4 (LLG torque) + G5 (RK4 stage kernels)
//
// Memory layout: [3×N] component-major, buf[c*N + idx]
//   idx = ix + nx*(iy + ny*iz),   c ∈ {0=x, 1=y, 2=z}
//
// All kernels use 1-D launch with 256 threads/block.
// LLG torque loops over N cells (one per thread).
// Stage/accumulate/finalize loop over 3N elements (flat, avoids component decode).
// Normalize loops over N cells (needs per-cell sqrt).

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "micromag/rk4_gpu.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA(rk4): ")        \
                                   + cudaGetErrorString(_e));           \
    } while (0)

namespace micromag {

// ===========================================================================
// G4: LLG torque kernel
//
// Landau-Lifshitz form:
//   dm/dt = -γ'μ₀ [(m×H) + α m×(m×H)]
//   γ'    = γ₀/(1+α²)   (already multiplied by μ₀ in gp)
//
// One thread per cell; SETS dm_out (does not add).
// ===========================================================================
__global__ static void llg_torque_kernel(
    double* __restrict__       dm_out,  // [3×N] SET
    const double* __restrict__ m,       // [3×N]
    const double* __restrict__ H,       // [3×N]
    double gp,                          // γ₀μ₀ / (1 + α²)
    double alpha,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
    const double Hx = H[0*N+idx], Hy = H[1*N+idx], Hz = H[2*N+idx];

    // mxH = m × H
    const double mxHx = my*Hz - mz*Hy;
    const double mxHy = mz*Hx - mx*Hz;
    const double mxHz = mx*Hy - my*Hx;

    // m × (m × H)
    const double mmxHx = my*mxHz - mz*mxHy;
    const double mmxHy = mz*mxHx - mx*mxHz;
    const double mmxHz = mx*mxHy - my*mxHx;

    // dm/dt = -gp * [(mxH) + alpha*(m×mxH)]
    dm_out[0*N+idx] = -gp * (mxHx + alpha*mmxHx);
    dm_out[1*N+idx] = -gp * (mxHy + alpha*mmxHy);
    dm_out[2*N+idx] = -gp * (mxHz + alpha*mmxHz);
}

// ===========================================================================
// G5: RK4 stage kernels — flat 3N operations (no component decode needed)
// ===========================================================================

// m_out = m0 + scale * ki
__global__ static void rk4_stage_kernel(
    double* __restrict__       m_out,
    const double* __restrict__ m0,
    const double* __restrict__ ki,
    double scale, int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m_out[i] = m0[i] + scale * ki[i];
}

// k_acc += weight * ki
__global__ static void rk4_accumulate_kernel(
    double* __restrict__       k_acc,
    const double* __restrict__ ki,
    double weight, int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    k_acc[i] += weight * ki[i];
}

// m_new = m0 + dt * k_acc
__global__ static void rk4_finalize_kernel(
    double* __restrict__       m_new,
    const double* __restrict__ m0,
    const double* __restrict__ k_acc,
    double dt, int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m_new[i] = m0[i] + dt * k_acc[i];
}

// Normalise per cell: m /= |m|
__global__ static void normalize_kernel(double* m, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
    const double inv = 1.0 / sqrt(mx*mx + my*my + mz*mz);
    m[0*N+idx] = mx * inv;
    m[1*N+idx] = my * inv;
    m[2*N+idx] = mz * inv;
}

// ===========================================================================
// Launcher functions (called from C++ test / G6 code)
// ===========================================================================

void launch_llg_torque(double* d_ki, const double* d_m, const double* d_H,
                        double alpha, int N, void* stream)
{
    const double gp = constants::gamma_0 * constants::mu_0 / (1.0 + alpha*alpha);
    const int blk = 256;
    const int grd = (N + blk - 1) / blk;
    llg_torque_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        d_ki, d_m, d_H, gp, alpha, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_rk4_stage(double* m_out, const double* m0, const double* ki,
                       double scale, int N, void* stream)
{
    const int N3  = 3 * N;
    const int blk = 256;
    const int grd = (N3 + blk - 1) / blk;
    rk4_stage_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        m_out, m0, ki, scale, N3);
    CUDA_CHECK(cudaGetLastError());
}

void launch_rk4_accumulate(double* k_acc, const double* ki,
                             double weight, int N, void* stream)
{
    const int N3  = 3 * N;
    const int blk = 256;
    const int grd = (N3 + blk - 1) / blk;
    rk4_accumulate_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        k_acc, ki, weight, N3);
    CUDA_CHECK(cudaGetLastError());
}

void launch_rk4_finalize(double* m_new, const double* m0, const double* k_acc,
                           double dt, int N, void* stream)
{
    const int N3  = 3 * N;
    const int blk = 256;
    const int grd = (N3 + blk - 1) / blk;
    rk4_finalize_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        m_new, m0, k_acc, dt, N3);
    CUDA_CHECK(cudaGetLastError());
}

void launch_normalize(double* m, int N, void* stream)
{
    const int blk = 512;   // normalize is pure memory-bound; 512 → better SM occupancy
    const int grd = (N + blk - 1) / blk;
    normalize_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(m, N);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Heun-specific kernels
// ===========================================================================

// dst[i] += src[i]  — flat 3N  (adds thermal noise d_noise_ to d_H)
__global__ static void add_3N_kernel(
    double* __restrict__       dst,
    const double* __restrict__ src,
    int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    dst[i] += src[i];
}

// m[i] += dt_half * (k1[i] + k2[i])  — flat 3N  (Heun trapezoidal corrector)
__global__ static void heun_corrector_kernel(
    double* __restrict__       m,
    const double* __restrict__ k1,
    const double* __restrict__ k2,
    double dt_half, int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m[i] += dt_half * (k1[i] + k2[i]);
}

void launch_add_3N(double* dst, const double* src, int N, void* stream)
{
    const int N3  = 3 * N;
    const int blk = 512;   // memory-bound add; 512 improves occupancy
    const int grd = (N3 + blk - 1) / blk;
    add_3N_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(dst, src, N3);
    CUDA_CHECK(cudaGetLastError());
}

void launch_heun_corrector(double* m, const double* k1, const double* k2,
                             double dt_half, int N, void* stream)
{
    const int N3  = 3 * N;
    const int blk = 256;
    const int grd = (N3 + blk - 1) / blk;
    heun_corrector_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        m, k1, k2, dt_half, N3);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// DOPRI5 (Dormand-Prince RK45) stage and error kernels — flat 3N operations
//
// Double atomicAdd requires compute 6.0+ (Pascal+).  For older GPUs the
// CAS-based fallback below is used automatically via __CUDA_ARCH__ guard.
// ===========================================================================

#if __CUDA_ARCH__ < 600
__device__ static double atomicAddDouble(double* addr, double val) {
    unsigned long long int* a = (unsigned long long int*)addr;
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD_DOUBLE atomicAddDouble
#else
#define ATOMIC_ADD_DOUBLE atomicAdd
#endif

// --------------------------------------------------------------------------
// Stage-specific linear combination kernels
// All operate on flat N3 = 3*N elements.
// --------------------------------------------------------------------------

// Stage 3:  m_s = m0 + h*(3/40 k1 + 9/40 k2)
__global__ static void dopri5_stage3_kernel(
    double* __restrict__ m_s, const double* __restrict__ m0,
    const double* __restrict__ k1, const double* __restrict__ k2,
    double h, int N3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m_s[i] = m0[i] + h * (3.0/40.0*k1[i] + 9.0/40.0*k2[i]);
}

// Stage 4:  m_s = m0 + h*(44/45 k1 - 56/15 k2 + 32/9 k3)
__global__ static void dopri5_stage4_kernel(
    double* __restrict__ m_s, const double* __restrict__ m0,
    const double* __restrict__ k1, const double* __restrict__ k2,
    const double* __restrict__ k3, double h, int N3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m_s[i] = m0[i] + h * ( 44.0/45.0*k1[i]
                           - 56.0/15.0*k2[i]
                           + 32.0/ 9.0*k3[i]);
}

// Stage 5:  m_s = m0 + h*(19372/6561 k1 - 25360/2187 k2 + 64448/6561 k3 - 212/729 k4)
__global__ static void dopri5_stage5_kernel(
    double* __restrict__ m_s, const double* __restrict__ m0,
    const double* __restrict__ k1, const double* __restrict__ k2,
    const double* __restrict__ k3, const double* __restrict__ k4,
    double h, int N3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m_s[i] = m0[i] + h * ( 19372.0/6561.0*k1[i]
                           - 25360.0/2187.0*k2[i]
                           + 64448.0/6561.0*k3[i]
                           -   212.0/ 729.0*k4[i]);
}

// Stage 6:  m_s = m0 + h*(9017/3168 k1 - 355/33 k2 + 46732/5247 k3 + 49/176 k4 - 5103/18656 k5)
__global__ static void dopri5_stage6_kernel(
    double* __restrict__ m_s, const double* __restrict__ m0,
    const double* __restrict__ k1, const double* __restrict__ k2,
    const double* __restrict__ k3, const double* __restrict__ k4,
    const double* __restrict__ k5, double h, int N3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m_s[i] = m0[i] + h * (  9017.0/ 3168.0*k1[i]
                            -  355.0/   33.0*k2[i]
                            + 46732.0/ 5247.0*k3[i]
                            +    49.0/  176.0*k4[i]
                            -  5103.0/18656.0*k5[i]);
}

// 5th-order solution:  m5 = m0 + h*(35/384 k1 + 500/1113 k3 + 125/192 k4 - 2187/6784 k5 + 11/84 k6)
__global__ static void dopri5_m5_kernel(
    double* __restrict__ m5, const double* __restrict__ m0,
    const double* __restrict__ k1, const double* __restrict__ k3,
    const double* __restrict__ k4, const double* __restrict__ k5,
    const double* __restrict__ k6, double h, int N3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    m5[i] = m0[i] + h * (  35.0/  384.0*k1[i]
                           + 500.0/ 1113.0*k3[i]
                           + 125.0/  192.0*k4[i]
                           - 2187.0/ 6784.0*k5[i]
                           +   11.0/   84.0*k6[i]);
}

// Error estimate:  err = h*(71/57600 k1 - 71/16695 k3 + 71/1920 k4 - 17253/339200 k5 + 22/525 k6 - 1/40 k7)
__global__ static void dopri5_err_kernel(
    double* __restrict__ err,
    const double* __restrict__ k1, const double* __restrict__ k3,
    const double* __restrict__ k4, const double* __restrict__ k5,
    const double* __restrict__ k6, const double* __restrict__ k7,
    double h, int N3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    err[i] = h * (    71.0/  57600.0*k1[i]
                  -   71.0/  16695.0*k3[i]
                  +   71.0/   1920.0*k4[i]
                  - 17253.0/ 339200.0*k5[i]
                  +   22.0/    525.0*k6[i]
                  -    1.0/     40.0*k7[i]);
}

// --------------------------------------------------------------------------
// Error norm reduction:  RMS of (err[i] / (atol + rtol*max(|m[i]|,|m5[i]|)))
// Uses parallel block reduction + atomicAdd into d_sum (must be pre-zeroed).
// --------------------------------------------------------------------------
__global__ static void dopri5_err_norm_sq_kernel(
    double* __restrict__ d_sum,
    const double* __restrict__ err,
    const double* __restrict__ m,
    const double* __restrict__ m5,
    double rtol, double atol, int N3)
{
    __shared__ double smem[256];
    const int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    double local = 0.0;
    while (i < N3) {
        double sc = atol + rtol * fmax(fabs(m[i]), fabs(m5[i]));
        double v  = err[i] / sc;
        local += v * v;
        i += blockDim.x * gridDim.x;
    }

    smem[tid] = local;
    __syncthreads();
    for (int s = 128; s >= 1; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) ATOMIC_ADD_DOUBLE(d_sum, smem[0]);
}

// --------------------------------------------------------------------------
// Launcher functions
// --------------------------------------------------------------------------
#define LAUNCH3N(kernel, N, stream, ...) do { \
    const int N3_ = 3*(N); const int blk_ = 256; \
    kernel<<<(N3_+blk_-1)/blk_, blk_, 0, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__, N3_); \
    CUDA_CHECK(cudaGetLastError()); } while(0)

void launch_dopri5_stage3(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, int N, void* stream)
{ LAUNCH3N(dopri5_stage3_kernel, N, stream, m_s, m0, k1, k2, h); }

void launch_dopri5_stage4(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, const double* k3,
                           int N, void* stream)
{ LAUNCH3N(dopri5_stage4_kernel, N, stream, m_s, m0, k1, k2, k3, h); }

void launch_dopri5_stage5(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, const double* k3,
                           const double* k4, int N, void* stream)
{ LAUNCH3N(dopri5_stage5_kernel, N, stream, m_s, m0, k1, k2, k3, k4, h); }

void launch_dopri5_stage6(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, const double* k3,
                           const double* k4, const double* k5, int N, void* stream)
{ LAUNCH3N(dopri5_stage6_kernel, N, stream, m_s, m0, k1, k2, k3, k4, k5, h); }

void launch_dopri5_m5(double* m5, const double* m0, double h,
                       const double* k1, const double* k3, const double* k4,
                       const double* k5, const double* k6, int N, void* stream)
{ LAUNCH3N(dopri5_m5_kernel, N, stream, m5, m0, k1, k3, k4, k5, k6, h); }

void launch_dopri5_err(double* err, double h,
                        const double* k1, const double* k3, const double* k4,
                        const double* k5, const double* k6, const double* k7,
                        int N, void* stream)
{ LAUNCH3N(dopri5_err_kernel, N, stream, err, k1, k3, k4, k5, k6, k7, h); }

double launch_dopri5_err_norm(double* d_sum,
                               const double* d_err, const double* d_m,
                               const double* d_m5,
                               double rtol, double atol, int N, void* stream)
{
    const int N3  = 3 * N;
    const int blk = 256;
    const int grd = std::min(256, (N3 + blk - 1) / blk);
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(double),
                                static_cast<cudaStream_t>(stream)));
    dopri5_err_norm_sq_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(
        d_sum, d_err, d_m, d_m5, rtol, atol, N3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    double h_sum;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    return std::sqrt(h_sum / static_cast<double>(N3));
}

}  // namespace micromag

#endif // MICROMAG_CUDA
