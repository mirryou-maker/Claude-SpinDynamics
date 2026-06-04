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
    const int blk = 256;
    const int grd = (N + blk - 1) / blk;
    normalize_kernel<<<grd, blk, 0, static_cast<cudaStream_t>(stream)>>>(m, N);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace micromag

#endif // MICROMAG_CUDA
