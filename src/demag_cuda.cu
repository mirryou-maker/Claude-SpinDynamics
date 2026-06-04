// demag_cuda.cu — Phase 3 Step 6c: CUDA stream (async pipeline)
//
// Key changes over Step 5:
//   - cufftPlanMany (batch=3) replaces 3 separate forward / 3 separate inverse plans.
//   - pointwise_mac_all3 kernel: one launch computes all 3 H components simultaneously.
//   - extract_all3 kernel: one launch extracts all 3 unpadded H components.
//   - Single cudaMemcpy H2D (3×real_sz) + single D2H (3×unpad_sz) per step.
//
// cuFFT exec count:  Step5 = 6  →  Step6a = 2   (1 forward + 1 inverse)
// CUDA kernel count: Step5 = 6  →  Step6a = 2   (mac_all3 + extract_all3)
// PCIe downloads:    Step5 = 3  →  Step6a = 1

#ifdef MICROMAG_CUDA

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "micromag/demag_gpu.hpp"

// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA: ")             \
                                   + cudaGetErrorString(_e));           \
    } while (0)
#define CUFFT_CHECK(call)                                               \
    do {                                                                \
        cufftResult _r = (call);                                        \
        if (_r != CUFFT_SUCCESS)                                        \
            throw std::runtime_error("cuFFT error " +                  \
                                     std::to_string((int)_r));         \
    } while (0)

static inline cufftDoubleComplex* as_cx(void* p) {
    return reinterpret_cast<cufftDoubleComplex*>(p);
}
static inline const cufftDoubleComplex* as_cxc(const void* p) {
    return reinterpret_cast<const cufftDoubleComplex*>(p);
}

// ===========================================================================
// GPU Newell tensor — device functions
//
// Implement the same 64-term alternating double-cell sums as demag.cpp,
// but entirely on the GPU so precompute_kernel() needs no CPU loops.
// Each thread handles one (kx,ky,kz) lattice position; all 2.5M threads
// run in parallel, reducing 500×500×10 precompute from ~35s → <1s.
// ===========================================================================

__device__ static double gpu_newell_f(double x, double y, double z) {
    x = fabs(x); y = fabs(y); z = fabs(z);
    const double x2 = x*x, y2 = y*y, z2 = z*z;
    const double r = sqrt(x2+y2+z2);
    if (r == 0.0) return 0.0;
    double val = 0.0;
    const double d_xz = sqrt(x2+z2);
    if (d_xz > 0.0) val += y*(z2-x2)*0.5*asinh(y/d_xz);
    const double d_xy = sqrt(x2+y2);
    if (d_xy > 0.0) val += z*(y2-x2)*0.5*asinh(z/d_xy);
    if (x > 0.0) val -= x*y*z*atan(y*z/(x*r));
    val += (2.0*x2-y2-z2)*r/6.0;
    return val;
}

__device__ static double gpu_newell_g(double x, double y, double z) {
    z = fabs(z);   // abs(z) only — x,y keep their sign
    const double x2 = x*x, y2 = y*y, z2 = z*z;
    const double r = sqrt(x2+y2+z2);
    if (r == 0.0) return 0.0;
    double val = 0.0;
    const double d_xy = sqrt(x2+y2);
    if (d_xy > 0.0) val += x*y*z*asinh(z/d_xy);
    const double d_yz = sqrt(y2+z2);
    if (d_yz > 0.0) val += y*(3.0*z2-y2)/6.0*asinh(x/d_yz);
    const double d_xz = sqrt(x2+z2);
    if (d_xz > 0.0) val += x*(3.0*z2-x2)/6.0*asinh(y/d_xz);
    if (z > 0.0) val -= z*z2/6.0*atan(x*y/(z*r));
    if (fabs(y) > 0.0) val -= z*y2*0.5*atan(x*z/(y*r));
    if (fabs(x) > 0.0) val -= z*x2*0.5*atan(y*z/(x*r));
    val -= x*y*r/3.0;
    return val;
}

// 64-term double-cell sum (Newell 1993) for diagonal component.
// Arguments are INTEGER displacement indices, not physical coordinates.
__device__ static double gpu_nxx(int kx, int ky, int kz,
                                   double dx, double dy, double dz) {
    constexpr double k4pi = 4.0 * 3.14159265358979323846;
    double sum = 0.0;
    for (int ia = 0; ia <= 1; ++ia)
    for (int ib = 0; ib <= 1; ++ib)
    for (int ic = 0; ic <= 1; ++ic)
    for (int id = 0; id <= 1; ++id)
    for (int ie = 0; ie <= 1; ++ie)
    for (int ig = 0; ig <= 1; ++ig) {
        const int s = ((ia+ib+ic+id+ie+ig) % 2 == 0) ? 1 : -1;
        sum += s * gpu_newell_f((kx+ia-id)*dx, (ky+ib-ie)*dy, (kz+ic-ig)*dz);
    }
    return sum / (k4pi * dx * dy * dz);
}

// 64-term double-cell sum for off-diagonal component.
__device__ static double gpu_nxy(int kx, int ky, int kz,
                                   double dx, double dy, double dz) {
    constexpr double k4pi = 4.0 * 3.14159265358979323846;
    double sum = 0.0;
    for (int ia = 0; ia <= 1; ++ia)
    for (int ib = 0; ib <= 1; ++ib)
    for (int ic = 0; ic <= 1; ++ic)
    for (int id = 0; id <= 1; ++id)
    for (int ie = 0; ie <= 1; ++ie)
    for (int ig = 0; ig <= 1; ++ig) {
        const int s = ((ia+ib+ic+id+ie+ig) % 2 == 0) ? 1 : -1;
        sum += s * gpu_newell_g((kx+ia-id)*dx, (ky+ib-ie)*dy, (kz+ic-ig)*dz);
    }
    return sum / (k4pi * dx * dy * dz);
}

// Inline write to padded buffer with negative-index wrap.
// Each (kx,ky,kz) thread writes to disjoint positions — no atomics needed.
#define GPU_PUT(r, px, py, pz, padX, padY, padZ, v) \
    (r)[((px)<0?(px)+(padX):(px)) + (padX)*(((py)<0?(py)+(padY):(py)) + (padY)*((pz)<0?(pz)+(padZ):(pz)))] = (v)

// ---------------------------------------------------------------------------
// GPU kernel: fill padded buffer with DIAGONAL Newell component (even symmetry).
// perm 0 → N_xx, 1 → N_yy (swap x↔y), 2 → N_zz (swap x↔z).
// ---------------------------------------------------------------------------
__global__ static void fill_diag_gpu(
    double* __restrict__ r_buf,
    int nx, int ny, int nz,
    int pad_nx, int pad_ny, int pad_nz,
    double dx, double dy, double dz,
    int perm)
{
    const int kx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ky = blockIdx.y * blockDim.y + threadIdx.y;
    const int kz = blockIdx.z;
    if (kx >= nx || ky >= ny || kz >= nz) return;

    double val;
    if      (perm == 0) val = gpu_nxx(kx, ky, kz, dx, dy, dz);
    else if (perm == 1) val = gpu_nxx(ky, kx, kz, dy, dx, dz);
    else                val = gpu_nxx(kz, ky, kx, dz, dy, dx);

    GPU_PUT(r_buf,  kx,  ky,  kz, pad_nx, pad_ny, pad_nz, val);
    if (kx>0) GPU_PUT(r_buf, -kx,  ky,  kz, pad_nx, pad_ny, pad_nz, val);
    if (ky>0) GPU_PUT(r_buf,  kx, -ky,  kz, pad_nx, pad_ny, pad_nz, val);
    if (kz>0) GPU_PUT(r_buf,  kx,  ky, -kz, pad_nx, pad_ny, pad_nz, val);
    if (kx>0 && ky>0) GPU_PUT(r_buf, -kx, -ky,  kz, pad_nx, pad_ny, pad_nz, val);
    if (kx>0 && kz>0) GPU_PUT(r_buf, -kx,  ky, -kz, pad_nx, pad_ny, pad_nz, val);
    if (ky>0 && kz>0) GPU_PUT(r_buf,  kx, -ky, -kz, pad_nx, pad_ny, pad_nz, val);
    if (kx>0 && ky>0 && kz>0) GPU_PUT(r_buf, -kx, -ky, -kz, pad_nx, pad_ny, pad_nz, val);
}

// ---------------------------------------------------------------------------
// GPU kernel: fill padded buffer with OFF-DIAGONAL Newell component (mixed parity).
// perm 0 → N_xy, 1 → N_xz (swap y↔z), 2 → N_yz (rotate).
// sx/sy/sz are the parity signs applied to negative-index copies.
// ---------------------------------------------------------------------------
__global__ static void fill_offdiag_gpu(
    double* __restrict__ r_buf,
    int nx, int ny, int nz,
    int pad_nx, int pad_ny, int pad_nz,
    double dx, double dy, double dz,
    int sx, int sy, int sz,
    int perm)
{
    const int kx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ky = blockIdx.y * blockDim.y + threadIdx.y;
    const int kz = blockIdx.z;
    if (kx >= nx || ky >= ny || kz >= nz) return;

    double val;
    if      (perm == 0) val = gpu_nxy(kx, ky, kz, dx, dy, dz);
    else if (perm == 1) val = gpu_nxy(kx, kz, ky, dx, dz, dy);
    else                val = gpu_nxy(ky, kz, kx, dy, dz, dx);

    for (int ix = 0; ix <= 1; ++ix)
    for (int iy = 0; iy <= 1; ++iy)
    for (int iz = 0; iz <= 1; ++iz) {
        if (ix && kx==0) continue;
        if (iy && ky==0) continue;
        if (iz && kz==0) continue;
        const double sign = (ix?(double)sx:1.0)*(iy?(double)sy:1.0)*(iz?(double)sz:1.0);
        const int px = ix ? -kx : kx;
        const int py = iy ? -ky : ky;
        const int pz = iz ? -kz : kz;
        GPU_PUT(r_buf, px, py, pz, pad_nx, pad_ny, pad_nz, sign*val);
    }
}

#undef GPU_PUT

// ===========================================================================
// CUDA kernel: combined H = Ka*Ma + Kb*Mb + Kc*Mc for all 3 output components
//
// MF_all layout: [Mx_f (N bins) | My_f (N bins) | Mz_f (N bins)]
// HF_all layout: [Hx_f (N bins) | Hy_f (N bins) | Hz_f (N bins)]
// ===========================================================================
__global__ static void pointwise_mac_all3(
    cufftDoubleComplex* __restrict__ HF_all,
    const cufftDoubleComplex* __restrict__ Kxx,
    const cufftDoubleComplex* __restrict__ Kxy,
    const cufftDoubleComplex* __restrict__ Kxz,
    const cufftDoubleComplex* __restrict__ Kyy,
    const cufftDoubleComplex* __restrict__ Kyz,
    const cufftDoubleComplex* __restrict__ Kzz,
    const cufftDoubleComplex* __restrict__ MF_all,
    size_t N)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Load M components (one load per component from contiguous regions)
    const cufftDoubleComplex Mx = MF_all[i];
    const cufftDoubleComplex My = MF_all[N + i];
    const cufftDoubleComplex Mz = MF_all[2*N + i];

    auto re = [](const cufftDoubleComplex& a, const cufftDoubleComplex& b)
        { return a.x*b.x - a.y*b.y; };
    auto im = [](const cufftDoubleComplex& a, const cufftDoubleComplex& b)
        { return a.x*b.y + a.y*b.x; };

    // Hx = Kxx*Mx + Kxy*My + Kxz*Mz
    HF_all[i].x = re(Kxx[i],Mx) + re(Kxy[i],My) + re(Kxz[i],Mz);
    HF_all[i].y = im(Kxx[i],Mx) + im(Kxy[i],My) + im(Kxz[i],Mz);

    // Hy = Kxy*Mx + Kyy*My + Kyz*Mz
    HF_all[N+i].x = re(Kxy[i],Mx) + re(Kyy[i],My) + re(Kyz[i],Mz);
    HF_all[N+i].y = im(Kxy[i],Mx) + im(Kyy[i],My) + im(Kyz[i],Mz);

    // Hz = Kxz*Mx + Kyz*My + Kzz*Mz
    HF_all[2*N+i].x = re(Kxz[i],Mx) + re(Kyz[i],My) + re(Kzz[i],Mz);
    HF_all[2*N+i].y = im(Kxz[i],Mx) + im(Kyz[i],My) + im(Kzz[i],Mz);
}

// ===========================================================================
// CUDA kernel: extract + normalize all 3 unpadded H components at once
//
// H_all layout:      [Hx_padded (real_sz) | Hy_padded | Hz_padded]
// H_unpad_all layout:[Hx_unpad (unpad_sz) | Hy_unpad  | Hz_unpad ]
// ===========================================================================
__global__ static void extract_all3(
    double* __restrict__       H_unpad_all,
    const double* __restrict__ H_all,
    size_t nx, size_t ny, size_t nz,
    size_t pad_nx, size_t pad_ny,
    size_t real_sz, size_t unpad_sz,
    double norm)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t iz = blockIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    const size_t src = ix + pad_nx * (iy + pad_ny * iz);
    const size_t dst = ix + nx    * (iy + ny    * iz);

    H_unpad_all[dst]             = H_all[src]             * norm;
    H_unpad_all[unpad_sz + dst]  = H_all[real_sz  + src]  * norm;
    H_unpad_all[2*unpad_sz + dst] = H_all[2*real_sz + src] * norm;
}

// ===========================================================================
// CUDA kernel: scatter compact M values into padded buffer (Step 6b)
//
// d_M_all_ is pre-zeroed (cudaMemset); this kernel writes the non-zero region.
// M_compact layout: [Mx_compact (unpad_sz) | My_compact | Mz_compact]
// M_all layout:     [Mx_padded  (real_sz)  | My_padded  | Mz_padded ]
// ===========================================================================
__global__ static void scatter_m_all3(
    double* __restrict__       M_all,      // output: [3 × real_sz] (pre-zeroed)
    const double* __restrict__ M_compact,  // input:  [3 × unpad_sz]
    size_t nx, size_t ny, size_t nz,
    size_t pad_nx, size_t pad_ny,
    size_t real_sz, size_t unpad_sz)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t iz = blockIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    const size_t src = ix + nx    * (iy + ny    * iz);
    const size_t dst = ix + pad_nx * (iy + pad_ny * iz);

    M_all[dst]             = M_compact[src];              // Mx
    M_all[real_sz  + dst]  = M_compact[unpad_sz  + src];  // My
    M_all[2*real_sz + dst] = M_compact[2*unpad_sz + src]; // Mz
}

// ===========================================================================
// Constructor
// ===========================================================================
namespace micromag {

DemagFieldGPU::DemagFieldGPU(const StructuredGrid& grid)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz())
{
    pad_nx_ = 2 * nx_;
    pad_ny_ = 2 * ny_;
    pad_nz_ = 2 * nz_;
    fft_nx_ = pad_nx_ / 2 + 1;

    unpad_sz_ = static_cast<size_t>(nx_ * ny_ * nz_);
    real_sz_  = static_cast<size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    cplx_sz_  = static_cast<size_t>(fft_nx_ * pad_ny_ * pad_nz_);

    // Single-use forward D2Z — for kernel precomputation only
    CUDA_CHECK(cudaMalloc(&d_r_buf_, real_sz_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c_buf_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUFFT_CHECK(cufftPlan3d(&plan_fwd_,
                             (int)pad_nz_, (int)pad_ny_, (int)pad_nx_,
                             CUFFT_D2Z));

    // Kernel frequency-domain storage (6 components)
    CUDA_CHECK(cudaMalloc(&d_K_xx_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_yy_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_zz_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xy_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xz_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_yz_, cplx_sz_ * sizeof(cufftDoubleComplex)));

    // Step 6a: batch buffers — all 3 components contiguous
    CUDA_CHECK(cudaMalloc(&d_M_all_,      3 * real_sz_  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_MF_all_,     3 * cplx_sz_  * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_HF_all_,     3 * cplx_sz_  * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_H_all_,      3 * real_sz_  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hunpad_all_, 3 * unpad_sz_ * sizeof(double)));

    // Step 6b: compact GPU buffer + pinned host (8× smaller than full padded)
    CUDA_CHECK(cudaMalloc(&d_M_compact_, 3 * unpad_sz_ * sizeof(double)));

    // Pinned host: compact upload (3×80KB) + H download (3×80KB)
    CUDA_CHECK(cudaMallocHost(&h_M_compact_pinned_,  3 * unpad_sz_ * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_Hunpad_all_pinned_, 3 * unpad_sz_ * sizeof(double)));

    // Batch cuFFT plans (cufftPlanMany with batch=3)
    int n[3] = {(int)pad_nz_, (int)pad_ny_, (int)pad_nx_};
    CUFFT_CHECK(cufftPlanMany(
        &plan_fwd_batch_, 3, n,
        nullptr, 1, (int)real_sz_,   // input: stride=1, dist=real_sz per batch
        nullptr, 1, (int)cplx_sz_,   // output: stride=1, dist=cplx_sz per batch
        CUFFT_D2Z, 3));
    CUFFT_CHECK(cufftPlanMany(
        &plan_inv_batch_, 3, n,
        nullptr, 1, (int)cplx_sz_,   // input
        nullptr, 1, (int)real_sz_,   // output
        CUFFT_Z2D, 3));

    // Step 6c: create dedicated stream; associate plans with it
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);

    CUFFT_CHECK(cufftSetStream(plan_fwd_,       s));
    CUFFT_CHECK(cufftSetStream(plan_fwd_batch_, s));
    CUFFT_CHECK(cufftSetStream(plan_inv_batch_, s));

    precompute_kernel();
    CUDA_CHECK(cudaStreamSynchronize(s));   // ensure precompute is done
}

// ===========================================================================
// Destructor
// ===========================================================================
DemagFieldGPU::~DemagFieldGPU() {
    cufftDestroy(plan_fwd_);
    cufftDestroy(plan_fwd_batch_);
    cufftDestroy(plan_inv_batch_);

    cudaFree(d_r_buf_); cudaFree(d_c_buf_);
    cudaFree(d_K_xx_); cudaFree(d_K_yy_); cudaFree(d_K_zz_);
    cudaFree(d_K_xy_); cudaFree(d_K_xz_); cudaFree(d_K_yz_);
    cudaFree(d_M_all_);    cudaFree(d_MF_all_);
    cudaFree(d_HF_all_);   cudaFree(d_H_all_);
    cudaFree(d_Hunpad_all_);
    cudaFree(d_M_compact_);

    cudaFreeHost(h_M_compact_pinned_);
    cudaFreeHost(h_Hunpad_all_pinned_);

    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ===========================================================================
// precompute_kernel — GPU version (replaces CPU loops)
//
// Before: CPU computed 6D Newell sums in 3 nested loops → ~35s for 2.5M cells
// After:  fill_diag_gpu / fill_offdiag_gpu kernels launch 2.5M GPU threads
//         that compute their cell in parallel → <1s for 2.5M cells
//
// Per-component pipeline (6 iterations):
//   1. cudaMemsetAsync d_r_buf_ = 0          (GPU, async)
//   2. fill_diag_gpu or fill_offdiag_gpu      (GPU, parallel — all on stream_)
//   3. cufftExecD2Z                           (GPU, uses stream_ via SetStream)
//   4. cudaMemcpyAsync D2D → d_K_xxx         (GPU, async D2D)
// ===========================================================================
void DemagFieldGPU::precompute_kernel() {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);

    // Thread block: 16×16×1 = 256 threads.  Grid covers (nx, ny, nz) cells.
    const dim3 blk(16, 16, 1);
    const dim3 grd(
        static_cast<unsigned>((nx_ + 15) / 16),
        static_cast<unsigned>((ny_ + 15) / 16),
        static_cast<unsigned>(nz_));

    // Helper: zero padded buffer, run fill kernel, FFT, copy to destination.
    // 'perm' selects which axis permutation to use (see kernel comments).
    auto fill_fft_diag = [&](void* d_K_dest, int perm) {
        CUDA_CHECK(cudaMemsetAsync(d_r_buf_, 0, real_sz_ * sizeof(double), s));
        fill_diag_gpu<<<grd, blk, 0, s>>>(
            reinterpret_cast<double*>(d_r_buf_),
            (int)nx_, (int)ny_, (int)nz_,
            (int)pad_nx_, (int)pad_ny_, (int)pad_nz_,
            dx_, dy_, dz_, perm);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
            reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
            reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpyAsync(d_K_dest, d_c_buf_,
            cplx_sz_ * sizeof(cufftDoubleComplex),
            cudaMemcpyDeviceToDevice, s));
    };

    auto fill_fft_offdiag = [&](void* d_K_dest, int sx, int sy, int sz, int perm) {
        CUDA_CHECK(cudaMemsetAsync(d_r_buf_, 0, real_sz_ * sizeof(double), s));
        fill_offdiag_gpu<<<grd, blk, 0, s>>>(
            reinterpret_cast<double*>(d_r_buf_),
            (int)nx_, (int)ny_, (int)nz_,
            (int)pad_nx_, (int)pad_ny_, (int)pad_nz_,
            dx_, dy_, dz_, sx, sy, sz, perm);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
            reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
            reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpyAsync(d_K_dest, d_c_buf_,
            cplx_sz_ * sizeof(cufftDoubleComplex),
            cudaMemcpyDeviceToDevice, s));
    };

    // Diagonal: K_xx (perm=0), K_yy (perm=1: swap x↔y), K_zz (perm=2: swap x↔z)
    fill_fft_diag(d_K_xx_, 0);
    fill_fft_diag(d_K_yy_, 1);
    fill_fft_diag(d_K_zz_, 2);

    // Off-diagonal: parity (sx,sy,sz) as in CPU code, perm selects axis mapping
    fill_fft_offdiag(d_K_xy_, -1, -1, +1, 0);  // N_xy(x,y,z)
    fill_fft_offdiag(d_K_xz_, -1, +1, -1, 1);  // N_xz ≡ N_xy(x,z,y)
    fill_fft_offdiag(d_K_yz_, +1, -1, -1, 2);  // N_yz ≡ N_xy(y,z,x)
    // (cudaStreamSynchronize called by constructor after precompute_kernel)
}

// ===========================================================================
// accumulate — batch FFT pipeline (Step 6a)
//
// Pipeline (2 cuFFT calls, 2 CUDA kernels, 1 upload, 1 download):
//   1. Fill h_M_all_pinned_ [3 × real_sz] with padded Mx,My,Mz (CPU)
//   2. cudaMemcpy H2D: h_M_all → d_M_all (1 call, 3×640KB)
//   3. cufftExecD2Z BATCH=3: d_M_all → d_MF_all (1 call)
//   4. pointwise_mac_all3: d_MF_all → d_HF_all (1 kernel)
//   5. cufftExecZ2D BATCH=3: d_HF_all → d_H_all (1 call)
//   6. extract_all3: d_H_all → d_Hunpad_all (1 kernel)
//   7. cudaMemcpy D2H: d_Hunpad_all → h_Hunpad_all (1 call, 3×80KB)
//   8. Accumulate h_Hunpad_all → H_out (1 CPU loop)
// ===========================================================================
void DemagFieldGPU::accumulate(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    const double Ms   = mat.Ms;
    const double norm = -1.0 / static_cast<double>(pad_nx_ * pad_ny_ * pad_nz_);

    // Step 6c: all GPU ops submitted to stream_ without blocking the CPU.
    // CPU blocks exactly once (cudaStreamSynchronize) after the D2H download.
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);

    // ------------------------------------------------------------------
    // Step 6b+6c: sparse upload pipeline (async version)
    //   1a. Fill compact host buffer                     [CPU, ~0.01ms]
    //   1b. cudaMemcpyAsync H2D compact 0.24MB           [async, PCIe]
    //   1c. cudaMemsetAsync d_M_all_ zero (1.92MB)       [async, GPU]
    //       → Overlaps with 1b: PCIe and GPU memset run in parallel
    //       (possible because they access different buffers in same stream;
    //        the stream serializes them, but the HW can overlap DMA + GPU)
    //   1d. scatter_m_all3 kernel                        [GPU]
    // ------------------------------------------------------------------

    // 1a. Compact fill (CPU — free to run while GPU processes previous step)
    double* hMx = h_M_compact_pinned_;
    double* hMy = hMx + unpad_sz_;
    double* hMz = hMy + unpad_sz_;
    for (Index i = 0; i < static_cast<Index>(unpad_sz_); ++i) {
        hMx[i] = Ms * m[i].x;
        hMy[i] = Ms * m[i].y;
        hMz[i] = Ms * m[i].z;
    }

    // 1b. Async upload (pinned → GPU, returns immediately)
    CUDA_CHECK(cudaMemcpyAsync(d_M_compact_, h_M_compact_pinned_,
                               3 * unpad_sz_ * sizeof(double),
                               cudaMemcpyHostToDevice, s));

    // 1c. Async zero padded buffer (runs on GPU after memcpy in stream order)
    CUDA_CHECK(cudaMemsetAsync(d_M_all_, 0, 3 * real_sz_ * sizeof(double), s));

    // 1d. Scatter (depends on both 1b and 1c — stream ensures ordering)
    dim3 blk_sc(16, 16, 1);
    dim3 grd_sc(
        static_cast<unsigned>((nx_ + blk_sc.x - 1) / blk_sc.x),
        static_cast<unsigned>((ny_ + blk_sc.y - 1) / blk_sc.y),
        static_cast<unsigned>(nz_));
    scatter_m_all3<<<grd_sc, blk_sc, 0, s>>>(
        reinterpret_cast<double*>(d_M_all_),
        reinterpret_cast<const double*>(d_M_compact_),
        static_cast<size_t>(nx_), static_cast<size_t>(ny_),
        static_cast<size_t>(nz_),
        static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
        real_sz_, unpad_sz_);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 3. Batch forward FFT — uses stream_ (set via cufftSetStream in ctor)
    // ------------------------------------------------------------------
    CUFFT_CHECK(cufftExecD2Z(plan_fwd_batch_,
                              reinterpret_cast<cufftDoubleReal*>(d_M_all_),
                              reinterpret_cast<cufftDoubleComplex*>(d_MF_all_)));

    // ------------------------------------------------------------------
    // 4. Combined pointwise MAC (stream_)
    // ------------------------------------------------------------------
    constexpr int BLK = 256;
    const int gcx = static_cast<int>((cplx_sz_ + BLK - 1) / BLK);

    pointwise_mac_all3<<<gcx, BLK, 0, s>>>(
        as_cx(d_HF_all_),
        as_cxc(d_K_xx_), as_cxc(d_K_xy_), as_cxc(d_K_xz_),
        as_cxc(d_K_yy_), as_cxc(d_K_yz_), as_cxc(d_K_zz_),
        as_cxc(d_MF_all_),
        cplx_sz_);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 5. Batch inverse FFT — uses stream_
    // ------------------------------------------------------------------
    CUFFT_CHECK(cufftExecZ2D(plan_inv_batch_,
                              reinterpret_cast<cufftDoubleComplex*>(d_HF_all_),
                              reinterpret_cast<cufftDoubleReal*>(d_H_all_)));

    // ------------------------------------------------------------------
    // 6. Extract all 3 unpadded H components (stream_)
    // ------------------------------------------------------------------
    dim3 blk_ext(16, 16, 1);
    dim3 grd_ext(
        static_cast<unsigned>((nx_ + blk_ext.x - 1) / blk_ext.x),
        static_cast<unsigned>((ny_ + blk_ext.y - 1) / blk_ext.y),
        static_cast<unsigned>(nz_));

    extract_all3<<<grd_ext, blk_ext, 0, s>>>(
        reinterpret_cast<double*>(d_Hunpad_all_),
        reinterpret_cast<const double*>(d_H_all_),
        static_cast<size_t>(nx_), static_cast<size_t>(ny_),
        static_cast<size_t>(nz_),
        static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
        real_sz_, unpad_sz_, norm);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 7. Async download (pinned buffer — returns immediately)
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(h_Hunpad_all_pinned_, d_Hunpad_all_,
                               3 * unpad_sz_ * sizeof(double),
                               cudaMemcpyDeviceToHost, s));

    // ------------------------------------------------------------------
    // Step 6c: ONE sync point — CPU waits here for the full pipeline
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaStreamSynchronize(s));

    // ------------------------------------------------------------------
    // 8. Accumulate into H_out (single loop, all 3 components)
    // ------------------------------------------------------------------
    const double* hx = h_Hunpad_all_pinned_;
    const double* hy = hx + unpad_sz_;
    const double* hz = hy + unpad_sz_;
    for (Index i = 0; i < static_cast<Index>(unpad_sz_); ++i) {
        H_out[i].x += hx[i];
        H_out[i].y += hy[i];
        H_out[i].z += hz[i];
    }
}

// ===========================================================================
// G6 helper kernels
// ===========================================================================

// Scale src by 'scale' and write to dst  (flat 3N op, for Ms-scaling d_m → d_M_compact)
__global__ static void scale_copy_kernel(
    double* __restrict__       dst,
    const double* __restrict__ src,
    double scale, int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    dst[i] = scale * src[i];
}

// dst[i] += src[i]  (flat 3N op, used to add d_Hunpad_all_ to d_H_out)
__global__ static void add_3N_kernel(
    double* __restrict__       dst,
    const double* __restrict__ src,
    int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    dst[i] += src[i];
}

// ===========================================================================
// accumulate_gpu_ptr — GPU-pointer path for G6 full-LLG pipeline
//
// d_m:    [3×N] component-major (unit magnetization from GPUMagState::d_m_)
// d_H_out:[3×N] component-major (adds H_demag in-place, same layout)
//
// Pipeline (entirely on stream_; no PCIe):
//   scale_copy:    d_M_compact_ = Ms * d_m   (D2D, no PCIe)
//   memset:        d_M_all_ = 0
//   scatter:       d_M_compact_ → d_M_all_ (padded)
//   FFT batch:     d_M_all_ → d_MF_all_
//   pointwise MAC: d_MF_all_ → d_HF_all_
//   IFFT batch:    d_HF_all_ → d_H_all_
//   extract:       d_H_all_ → d_Hunpad_all_ (with IFFT normalisation)
//   add:           d_H_out += d_Hunpad_all_
// ===========================================================================
void DemagFieldGPU::accumulate_gpu_ptr(const double* d_m,
                                         const Material& mat,
                                         double* d_H_out) const {
    const cudaStream_t s    = static_cast<cudaStream_t>(stream_);
    const double Ms         = mat.Ms;
    const double norm       = -1.0 / static_cast<double>(pad_nx_ * pad_ny_ * pad_nz_);
    constexpr int BLK       = 256;

    // 1. Ms-scale d_m → d_M_compact_  (D2D, no PCIe)
    {
        const int N3  = static_cast<int>(3 * unpad_sz_);
        const int grd = (N3 + BLK - 1) / BLK;
        scale_copy_kernel<<<grd, BLK, 0, s>>>(
            reinterpret_cast<double*>(d_M_compact_), d_m, Ms, N3);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. Zero padded M buffer
    CUDA_CHECK(cudaMemsetAsync(d_M_all_, 0, 3 * real_sz_ * sizeof(double), s));

    // 3. Scatter compact → padded
    {
        dim3 blk_sc(16, 16, 1);
        dim3 grd_sc((unsigned)((nx_+15)/16), (unsigned)((ny_+15)/16), (unsigned)nz_);
        scatter_m_all3<<<grd_sc, blk_sc, 0, s>>>(
            reinterpret_cast<double*>(d_M_all_),
            reinterpret_cast<const double*>(d_M_compact_),
            (size_t)nx_, (size_t)ny_, (size_t)nz_,
            (size_t)pad_nx_, (size_t)pad_ny_, real_sz_, unpad_sz_);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. Batch forward FFT
    CUFFT_CHECK(cufftExecD2Z(plan_fwd_batch_,
        reinterpret_cast<cufftDoubleReal*>(d_M_all_),
        reinterpret_cast<cufftDoubleComplex*>(d_MF_all_)));

    // 5. Pointwise MAC
    {
        const int gcx = static_cast<int>((cplx_sz_ + BLK - 1) / BLK);
        pointwise_mac_all3<<<gcx, BLK, 0, s>>>(
            as_cx(d_HF_all_),
            as_cxc(d_K_xx_), as_cxc(d_K_xy_), as_cxc(d_K_xz_),
            as_cxc(d_K_yy_), as_cxc(d_K_yz_), as_cxc(d_K_zz_),
            as_cxc(d_MF_all_), cplx_sz_);
        CUDA_CHECK(cudaGetLastError());
    }

    // 6. Batch inverse FFT
    CUFFT_CHECK(cufftExecZ2D(plan_inv_batch_,
        reinterpret_cast<cufftDoubleComplex*>(d_HF_all_),
        reinterpret_cast<cufftDoubleReal*>(d_H_all_)));

    // 7. Extract unpadded + normalise → d_Hunpad_all_
    {
        dim3 blk_ext(16, 16, 1);
        dim3 grd_ext((unsigned)((nx_+15)/16), (unsigned)((ny_+15)/16), (unsigned)nz_);
        extract_all3<<<grd_ext, blk_ext, 0, s>>>(
            reinterpret_cast<double*>(d_Hunpad_all_),
            reinterpret_cast<const double*>(d_H_all_),
            (size_t)nx_, (size_t)ny_, (size_t)nz_,
            (size_t)pad_nx_, (size_t)pad_ny_,
            real_sz_, unpad_sz_, norm);
        CUDA_CHECK(cudaGetLastError());
    }

    // 8. Add d_Hunpad_all_ to d_H_out
    {
        const int N3  = static_cast<int>(3 * unpad_sz_);
        const int grd = (N3 + BLK - 1) / BLK;
        add_3N_kernel<<<grd, BLK, 0, s>>>(
            d_H_out,
            reinterpret_cast<const double*>(d_Hunpad_all_),
            N3);
        CUDA_CHECK(cudaGetLastError());
    }
    // Sync before returning so caller on a different stream sees d_H_out updates.
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ===========================================================================
// energy
// ===========================================================================
Real DemagFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    const StructuredGrid& g = m.grid();
    VectorField3D H(g);
    for (Index i=0;i<H.size();++i) H[i]={0,0,0};
    accumulate(m,mat,H);
    Real E=0.0;
    const Real dV=g.cell_volume();
    for (Index i=0;i<m.size();++i)
        E -= constants::mu_0 * mat.Ms * m[i].dot(H[i]) * dV;
    return 0.5*E;
}

}  // namespace micromag

#endif // MICROMAG_CUDA
