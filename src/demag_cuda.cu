// demag_cuda.cu ??Phase 3 Step 6c: CUDA stream (async pipeline)
//
// Key changes over Step 5:
//   - cufftPlanMany (batch=3) replaces 3 separate forward / 3 separate inverse plans.
//   - pointwise_mac_all3 kernel: one launch computes all 3 H components simultaneously.
//   - extract_all3 kernel: one launch extracts all 3 unpadded H components.
//   - Single cudaMemcpy H2D (3횞real_sz) + single D2H (3횞unpad_sz) per step.
//
// cuFFT exec count:  Step5 = 6  ?? Step6a = 2   (1 forward + 1 inverse)
// CUDA kernel count: Step5 = 6  ?? Step6a = 2   (mac_all3 + extract_all3)
// PCIe downloads:    Step5 = 3  ?? Step6a = 1

#ifdef MICROMAG_CUDA

#include <cufft.h>
#include "micromag/gpu_real.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "micromag/demag_gpu.hpp"

// P3: VkFFT optional batch FFT backend
#ifdef MICROMAG_VKFFT
#include <cuda.h>
#define VKFFT_BACKEND 1
#include "VkFFT/vkFFT.h"
#endif

// Bring GReal into global scope: the kernels below are defined before the
// `namespace micromag {` block opens (line ~315), so they cannot see the
// namespace-scoped alias otherwise.
using micromag::GReal;

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

static inline GREAL_CUFFT_COMPLEX* as_cx(void* p) {
    return reinterpret_cast<GREAL_CUFFT_COMPLEX*>(p);
}
static inline const GREAL_CUFFT_COMPLEX* as_cxc(const void* p) {
    return reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(p);
}
static inline const GREAL_CUFFT_REAL* as_realc(const void* p) {
    return reinterpret_cast<const GREAL_CUFFT_REAL*>(p);
}

// ===========================================================================
// GPU Newell tensor ??device functions
//
// Implement the same 64-term alternating double-cell sums as demag.cpp,
// but entirely on the GPU so precompute_kernel() needs no CPU loops.
// Each thread handles one (kx,ky,kz) lattice position; all 2.5M threads
// run in parallel, reducing 500횞500횞10 precompute from ~35s ??<1s.
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
    z = fabs(z);   // abs(z) only ??x,y keep their sign
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
// Each (kx,ky,kz) thread writes to disjoint positions ??no atomics needed.
#define GPU_PUT(r, px, py, pz, padX, padY, padZ, v) \
    (r)[((px)<0?(px)+(padX):(px)) + (padX)*(((py)<0?(py)+(padY):(py)) + (padY)*((pz)<0?(pz)+(padZ):(pz)))] = (v)

// ---------------------------------------------------------------------------
// GPU kernel: fill padded buffer with DIAGONAL Newell component (even symmetry).
// perm 0 ??N_xx, 1 ??N_yy (swap x?봸), 2 ??N_zz (swap x?봹).
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
// perm 0 ??N_xy, 1 ??N_xz (swap y?봹), 2 ??N_yz (rotate).
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
// Extract the REAL part of the (double-complex) kernel FFT into the GReal-typed
// REAL kernel storage (d_K_xx_ etc.).  The demag kernel FFTs are purely real:
// the diagonal components are even in all axes (cosine transform -> real) and
// the off-diagonals are odd-odd-even (i*i -> real), so the imaginary parts are
// zero and dropping them is exact.  Storing real (not complex) kernels halves
// both the kernel VRAM and the pointwise-MAC read bandwidth.
// KEPT for reference — superseded by dc_to_real_quadrant below.
__global__ static void dc_to_real(GREAL_CUFFT_REAL* __restrict__ dst,
                                  const cufftDoubleComplex* __restrict__ src,
                                  size_t N) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = static_cast<GREAL_CUFFT_REAL>(src[i].x);
}

// ===========================================================================
// Extract first quadrant of complex FFT output into real compressed kernel.
// Exploits Y+Z mirror symmetry: only iy∈[0..symm_ny-1], iz∈[0..symm_nz-1]
// are stored; the MAC kernel reconstructs the remaining quadrants at runtime.
//
// src: full fft_nx × pad_ny × pad_nz complex array (d_c_buf_), x-fastest
// dst: compressed fft_nx × symm_ny × symm_nz real array (d_K_xxx_), x-fastest
// Layout: linear index = ix + fft_nx*(iy2 + symm_ny*iz2)
__global__ static void dc_to_real_quadrant(
    GREAL_CUFFT_REAL* __restrict__        dst,
    const cufftDoubleComplex* __restrict__ src,
    size_t fft_nx, size_t symm_ny, size_t symm_nz,
    size_t pad_ny)
{
    const size_t flat  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = fft_nx * symm_ny * symm_nz;
    if (flat >= total) return;

    // Decode flat → (ix, iy2, iz2) in compressed layout (x-fastest)
    const size_t ix  = flat % fft_nx;
    const size_t iy2 = (flat / fft_nx) % symm_ny;
    const size_t iz2 = flat / (fft_nx * symm_ny);

    // Source index in full FFT output — iy2 and iz2 are already in first quadrant
    const size_t src_i = ix + fft_nx * (iy2 + pad_ny * iz2);
    dst[flat] = static_cast<GREAL_CUFFT_REAL>(src[src_i].x);
}

// ===========================================================================
// Symmetric MAC for 3D grids: compressed kernel lookup with Y+Z sign reconstruction.
// Kernel stored at (ix, iy2, iz2) with iy2∈[0..symm_ny-1], iz2∈[0..symm_nz-1].
// Sign conventions (from Newell symmetry of the demag tensor):
//   Kxx,Kyy,Kzz: even in ky AND kz → no sign flip
//   Kxy: odd in ky → sign flips when iy is mirrored
//   Kxz: odd in kz → sign flips when iz is mirrored
//   Kyz: odd in ky AND kz → sign flips when exactly one of iy,iz is mirrored (XOR)
__global__ static void mac_symm_3d(
    GREAL_CUFFT_COMPLEX* __restrict__       HF_all,
    const GREAL_CUFFT_REAL* __restrict__    Kxx,
    const GREAL_CUFFT_REAL* __restrict__    Kxy,
    const GREAL_CUFFT_REAL* __restrict__    Kxz,
    const GREAL_CUFFT_REAL* __restrict__    Kyy,
    const GREAL_CUFFT_REAL* __restrict__    Kyz,
    const GREAL_CUFFT_REAL* __restrict__    Kzz,
    const GREAL_CUFFT_COMPLEX* __restrict__ MF_all,
    size_t fft_nx, size_t pad_ny, size_t pad_nz,
    size_t symm_ny, size_t symm_nz,
    size_t N)   // cplx_sz_ = fft_nx * pad_ny * pad_nz
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const size_t ix = i % fft_nx;
    const size_t iy = (i / fft_nx) % pad_ny;
    const size_t iz = i / (fft_nx * pad_ny);

    // Determine if this (iy, iz) bin is in the mirrored half
    const bool iy_m = (iy > pad_ny / 2);
    const bool iz_m = (iz > pad_nz / 2);
    // Map to first-quadrant index
    const size_t iy2 = iy_m ? (pad_ny - iy) : iy;
    const size_t iz2 = iz_m ? (pad_nz - iz) : iz;

    // Off-diagonal sign reconstruction
    const GREAL_CUFFT_REAL sxy = iy_m ? GREAL_CUFFT_REAL(-1) : GREAL_CUFFT_REAL(1);
    const GREAL_CUFFT_REAL sxz = iz_m ? GREAL_CUFFT_REAL(-1) : GREAL_CUFFT_REAL(1);
    const GREAL_CUFFT_REAL syz = (iy_m != iz_m) ? GREAL_CUFFT_REAL(-1) : GREAL_CUFFT_REAL(1);

    const size_t Ki = ix + fft_nx * (iy2 + symm_ny * iz2);

    const GREAL_CUFFT_REAL kxx = Kxx[Ki];
    const GREAL_CUFFT_REAL kxy = Kxy[Ki] * sxy;
    const GREAL_CUFFT_REAL kxz = Kxz[Ki] * sxz;
    const GREAL_CUFFT_REAL kyy = Kyy[Ki];
    const GREAL_CUFFT_REAL kyz = Kyz[Ki] * syz;
    const GREAL_CUFFT_REAL kzz = Kzz[Ki];

    const GREAL_CUFFT_COMPLEX Mx = MF_all[i];
    const GREAL_CUFFT_COMPLEX My = MF_all[N + i];
    const GREAL_CUFFT_COMPLEX Mz = MF_all[2*N + i];

    // Hx = Kxx*Mx + Kxy*My + Kxz*Mz
    HF_all[i].x     = kxx*Mx.x + kxy*My.x + kxz*Mz.x;
    HF_all[i].y     = kxx*Mx.y + kxy*My.y + kxz*Mz.y;
    // Hy = Kxy*Mx + Kyy*My + Kyz*Mz
    HF_all[N+i].x   = kxy*Mx.x + kyy*My.x + kyz*Mz.x;
    HF_all[N+i].y   = kxy*Mx.y + kyy*My.y + kyz*Mz.y;
    // Hz = Kxz*Mx + Kyz*My + Kzz*Mz
    HF_all[2*N+i].x = kxz*Mx.x + kyz*My.x + kzz*Mz.x;
    HF_all[2*N+i].y = kxz*Mx.y + kyz*My.y + kzz*Mz.y;
}

// ===========================================================================
// Symmetric MAC for 2D thin-film (pad_nz==1): Kxz=Kyz=0, half Y threads.
// Launch grid: (fft_nx/16+1) × (symm_ny/16+1) × 1 with 16×16 blocks.
// Each thread handles its own row (iy) AND its Y-mirror row (pad_ny - iy).
// This halves the number of launched threads compared to the 3D kernel.
__global__ static void mac_symm_2d(
    GREAL_CUFFT_COMPLEX* __restrict__       HF_all,
    const GREAL_CUFFT_REAL* __restrict__    Kxx,
    const GREAL_CUFFT_REAL* __restrict__    Kxy,
    const GREAL_CUFFT_REAL* __restrict__    Kyy,
    const GREAL_CUFFT_REAL* __restrict__    Kzz,
    const GREAL_CUFFT_COMPLEX* __restrict__ MF_all,
    size_t fft_nx, size_t pad_ny, size_t symm_ny,
    size_t N)   // cplx_sz_ = fft_nx * pad_ny (pad_nz==1)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= fft_nx || iy >= symm_ny) return;

    const size_t Ki  = ix + fft_nx * iy;
    const GREAL_CUFFT_REAL kxx = Kxx[Ki];
    const GREAL_CUFFT_REAL kxy = Kxy[Ki];
    const GREAL_CUFFT_REAL kyy = Kyy[Ki];
    const GREAL_CUFFT_REAL kzz = Kzz[Ki];

    // Write row `row` with off-diagonal sign kxy_s (Kxz=Kyz=0 for single layer)
    auto do_row = [&](size_t row, GREAL_CUFFT_REAL kxy_s) {
        const size_t i = ix + fft_nx * row;
        const GREAL_CUFFT_COMPLEX Mx = MF_all[i];
        const GREAL_CUFFT_COMPLEX My = MF_all[N + i];
        const GREAL_CUFFT_COMPLEX Mz = MF_all[2*N + i];
        // Hx = Kxx*Mx + Kxy*My  (Kxz=0)
        HF_all[i].x     = kxx*Mx.x + kxy_s*My.x;
        HF_all[i].y     = kxx*Mx.y + kxy_s*My.y;
        // Hy = Kxy*Mx + Kyy*My  (Kyz=0)
        HF_all[N+i].x   = kxy_s*Mx.x + kyy*My.x;
        HF_all[N+i].y   = kxy_s*Mx.y + kyy*My.y;
        // Hz = Kzz*Mz  (Kxz=Kyz=0)
        HF_all[2*N+i].x = kzz*Mz.x;
        HF_all[2*N+i].y = kzz*Mz.y;
    };

    do_row(iy, kxy);
    // Mirror row: Kxy is odd in y → negate sign
    if (iy > 0 && 2 * iy != pad_ny)
        do_row(pad_ny - iy, -kxy);
}

// ===========================================================================
// CUDA kernel: combined H = Ka*Ma + Kb*Mb + Kc*Mc for all 3 output components
//
// MF_all layout: [Mx_f (N bins) | My_f (N bins) | Mz_f (N bins)]
// HF_all layout: [Hx_f (N bins) | Hy_f (N bins) | Hz_f (N bins)]
// ===========================================================================
// Real (symmetry-reduced) kernels: each K is a real scalar, so K*M just scales
// the complex M bin (no cross terms) — half the kernel read bandwidth and
// fewer flops than the old complex*complex multiply.
__global__ static void pointwise_mac_all3(
    GREAL_CUFFT_COMPLEX* __restrict__ HF_all,
    const GREAL_CUFFT_REAL* __restrict__ Kxx,
    const GREAL_CUFFT_REAL* __restrict__ Kxy,
    const GREAL_CUFFT_REAL* __restrict__ Kxz,
    const GREAL_CUFFT_REAL* __restrict__ Kyy,
    const GREAL_CUFFT_REAL* __restrict__ Kyz,
    const GREAL_CUFFT_REAL* __restrict__ Kzz,
    const GREAL_CUFFT_COMPLEX* __restrict__ MF_all,
    size_t N)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const GREAL_CUFFT_COMPLEX Mx = MF_all[i];
    const GREAL_CUFFT_COMPLEX My = MF_all[N + i];
    const GREAL_CUFFT_COMPLEX Mz = MF_all[2*N + i];

    const GREAL_CUFFT_REAL kxx = Kxx[i], kxy = Kxy[i], kxz = Kxz[i];
    const GREAL_CUFFT_REAL kyy = Kyy[i], kyz = Kyz[i], kzz = Kzz[i];

    // Hx = Kxx*Mx + Kxy*My + Kxz*Mz
    HF_all[i].x       = kxx*Mx.x + kxy*My.x + kxz*Mz.x;
    HF_all[i].y       = kxx*Mx.y + kxy*My.y + kxz*Mz.y;
    // Hy = Kxy*Mx + Kyy*My + Kyz*Mz
    HF_all[N+i].x     = kxy*Mx.x + kyy*My.x + kyz*Mz.x;
    HF_all[N+i].y     = kxy*Mx.y + kyy*My.y + kyz*Mz.y;
    // Hz = Kxz*Mx + Kyz*My + Kzz*Mz
    HF_all[2*N+i].x   = kxz*Mx.x + kyz*My.x + kzz*Mz.x;
    HF_all[2*N+i].y   = kxz*Mx.y + kyz*My.y + kzz*Mz.y;
}

// ===========================================================================
// CUDA kernel: extract + normalize all 3 unpadded H components at once
//
// H_all layout:      [Hx_padded (real_sz) | Hy_padded | Hz_padded]
// H_unpad_all layout:[Hx_unpad (unpad_sz) | Hy_unpad  | Hz_unpad ]
// ===========================================================================
__global__ static void extract_all3(
    GReal* __restrict__        H_unpad_all,
    const GReal* __restrict__  H_all,
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

    H_unpad_all[dst]              = static_cast<GReal>(static_cast<double>(H_all[src])             * norm);
    H_unpad_all[unpad_sz + dst]   = static_cast<GReal>(static_cast<double>(H_all[real_sz  + src])  * norm);
    H_unpad_all[2*unpad_sz + dst] = static_cast<GReal>(static_cast<double>(H_all[2*real_sz + src]) * norm);
}

// ===========================================================================
// CUDA kernel: extract + normalize + accumulate directly into d_H_out.
// Fuses the old extract_all3 + add_3N_kernel for the on-GPU (gpu_ptr) path,
// removing one kernel launch and the d_Hunpad_all_ round-trip.
//   d_H_out layout: [Hx_unpad (unpad_sz) | Hy_unpad | Hz_unpad]  (+=)
// ===========================================================================
__global__ static void extract_add_all3(
    GReal* __restrict__        d_H_out,
    const GReal* __restrict__  H_all,
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

    d_H_out[dst]              = static_cast<GReal>(static_cast<double>(d_H_out[dst])              + static_cast<double>(H_all[src])             * norm);
    d_H_out[unpad_sz + dst]   = static_cast<GReal>(static_cast<double>(d_H_out[unpad_sz + dst])   + static_cast<double>(H_all[real_sz  + src])  * norm);
    d_H_out[2*unpad_sz + dst] = static_cast<GReal>(static_cast<double>(d_H_out[2*unpad_sz + dst]) + static_cast<double>(H_all[2*real_sz + src]) * norm);
}

// ===========================================================================
// CUDA kernel: scatter compact M values into padded buffer (Step 6b)
//
// d_M_all_ is pre-zeroed (cudaMemset); this kernel writes the non-zero region.
// M_compact layout: [Mx_compact (unpad_sz) | My_compact | Mz_compact]
// M_all layout:     [Mx_padded  (real_sz)  | My_padded  | Mz_padded ]
// ===========================================================================
__global__ static void scatter_m_all3(
    GReal* __restrict__        M_all,      // output: [3 x real_sz] (pre-zeroed)
    const GReal* __restrict__  M_compact,  // input:  [3 x unpad_sz]
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
    // No wrap-around in a dimension of extent 1, so a single-layer film (nz=1)
    // needs NO z-padding: the demag is then a pure 2-D (x,y) convolution and the
    // FFT collapses from 3-D-with-z=2 to 2-D, ~2x less transform work.  The
    // kernel-fill z-mirror writes are all guarded by (kz>0) so they no-op here.
    pad_nz_ = (nz_ == 1) ? 1 : 2 * nz_;
    fft_nx_ = pad_nx_ / 2 + 1;

    unpad_sz_ = static_cast<size_t>(nx_ * ny_ * nz_);
    real_sz_  = static_cast<size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    cplx_sz_  = static_cast<size_t>(fft_nx_ * pad_ny_ * pad_nz_);

    // Y+Z mirror symmetry: kernels stored only for first quadrant
    symm_ny_ = pad_ny_ / 2 + 1;
    symm_nz_ = (pad_nz_ == 1) ? 1 : pad_nz_ / 2 + 1;
    symm_sz_ = fft_nx_ * symm_ny_ * symm_nz_;

    // Precompute-only buffers: always double (plan_fwd_ is always CUFFT_D2Z)
    CUDA_CHECK(cudaMalloc(&d_r_buf_, real_sz_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c_buf_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUFFT_CHECK(cufftPlan3d(&plan_fwd_,
                             (int)pad_nz_, (int)pad_ny_, (int)pad_nx_,
                             CUFFT_D2Z));

    // Kernel frequency-domain storage (6 components) — stored REAL in first quadrant
    // only (Y+Z mirror symmetry), so ~4x less VRAM than full cplx_sz_ complex storage.
    CUDA_CHECK(cudaMalloc(&d_K_xx_, symm_sz_ * sizeof(GREAL_CUFFT_REAL)));
    CUDA_CHECK(cudaMalloc(&d_K_yy_, symm_sz_ * sizeof(GREAL_CUFFT_REAL)));
    CUDA_CHECK(cudaMalloc(&d_K_zz_, symm_sz_ * sizeof(GREAL_CUFFT_REAL)));
    CUDA_CHECK(cudaMalloc(&d_K_xy_, symm_sz_ * sizeof(GREAL_CUFFT_REAL)));
    CUDA_CHECK(cudaMalloc(&d_K_xz_, symm_sz_ * sizeof(GREAL_CUFFT_REAL)));
    CUDA_CHECK(cudaMalloc(&d_K_yz_, symm_sz_ * sizeof(GREAL_CUFFT_REAL)));

    // Step 6a: batch buffers ??all 3 components contiguous
    CUDA_CHECK(cudaMalloc(&d_M_all_,      3 * real_sz_ * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_MF_all_,     3 * cplx_sz_  * sizeof(GREAL_CUFFT_COMPLEX)));
    // d_HF_all_ removed: the pointwise MAC now writes H_f in-place into d_MF_all_.
    // d_H_all_ removed: the inverse FFT writes its real output back into
    // d_M_all_ (unused after the forward FFT) — same size, saves VRAM.
    CUDA_CHECK(cudaMalloc(&d_Hunpad_all_, 3 * unpad_sz_ * sizeof(GReal)));

    // Step 6b: compact GPU buffer + pinned host (8횞 smaller than full padded)
    CUDA_CHECK(cudaMalloc(&d_M_compact_, 3 * unpad_sz_ * sizeof(GReal)));

    // Pinned host: compact upload (3횞80KB) + H download (3횞80KB)
    CUDA_CHECK(cudaMallocHost(&h_M_compact_pinned_,  3 * unpad_sz_ * sizeof(GReal)));
    CUDA_CHECK(cudaMallocHost(&h_Hunpad_all_pinned_, 3 * unpad_sz_ * sizeof(GReal)));

    // Batch FFT plans (cuFFT or VkFFT, selected at compile time).
    int n[3] = {(int)pad_nz_, (int)pad_ny_, (int)pad_nx_};
#ifndef MICROMAG_VKFFT
    // (A rank-2 plan for pad_nz==1 was tried and gave no speedup — cuFFT
    // already collapses n[0]=1, so the rank-3 form is kept for simplicity.)
    CUFFT_CHECK(cufftPlanMany(
        &plan_fwd_batch_, 3, n,
        nullptr, 1, (int)real_sz_,   // input: stride=1, dist=real_sz per batch
        nullptr, 1, (int)cplx_sz_,   // output: stride=1, dist=cplx_sz per batch
        GREAL_CUFFT_TYPE, 3));
    CUFFT_CHECK(cufftPlanMany(
        &plan_inv_batch_, 3, n,
        nullptr, 1, (int)cplx_sz_,   // input
        nullptr, 1, (int)real_sz_,   // output
        GREAL_CUFFT_ITYPE, 3));
#endif // !MICROMAG_VKFFT

    // Step 6c: create dedicated stream; associate plans with it
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);

    CUFFT_CHECK(cufftSetStream(plan_fwd_, s));
#ifndef MICROMAG_VKFFT
    CUFFT_CHECK(cufftSetStream(plan_fwd_batch_, s));
    CUFFT_CHECK(cufftSetStream(plan_inv_batch_, s));
#else
    // P3: initialise VkFFT applications for the batch forward/inverse.
    // VkFFT uses out-of-place R2C: inputBuffer(real) → buffer(complex) and
    // C2R inverse: buffer(complex) → outputBuffer(real).
    {
        int dev_idx; cudaGetDevice(&dev_idx);
        CUdevice cu_dev = static_cast<CUdevice>(dev_idx);

        struct VkFFT_State {
            VkFFTApplication fwd{};
            VkFFTApplication inv{};
            cudaStream_t stream = nullptr; // dereferenced at each VkFFTAppend call
        };
        auto* st = new VkFFT_State;
        st->stream = static_cast<cudaStream_t>(stream_);
        vkfft_state_ = st;

        const int fft_dim = (pad_nz_ == 1) ? 2 : 3;
        uint64_t real_bytes = static_cast<uint64_t>(3 * real_sz_ * sizeof(GReal));
        uint64_t cplx_bytes = static_cast<uint64_t>(3 * cplx_sz_ * sizeof(GREAL_CUFFT_COMPLEX));

        const pfUINT use_double = (sizeof(GReal) == 8) ? 1u : 0u;

        // Forward: real d_M_all_ → complex d_MF_all_
        {
            VkFFTConfiguration cfg = {};
            cfg.FFTdim           = static_cast<uint64_t>(fft_dim);
            cfg.size[0]          = static_cast<uint64_t>(pad_nx_);
            cfg.size[1]          = static_cast<uint64_t>(pad_ny_);
            if (fft_dim == 3) cfg.size[2] = static_cast<uint64_t>(pad_nz_);
            cfg.numberBatches    = 3;
            cfg.performR2C       = 1;
            cfg.doublePrecision  = use_double;   // must match GReal (default=float32)
            cfg.device           = &cu_dev;
            cfg.stream           = &st->stream;  // VkFFT dereferences at launch time
            cfg.isInputFormatted = 1;            // separate real inputBuffer
            cfg.inputBuffer      = &d_M_all_;
            cfg.inputBufferSize  = &real_bytes;
            cfg.buffer           = &d_MF_all_;   // complex output
            cfg.bufferSize       = &cplx_bytes;
            VkFFTResult vr = initializeVkFFT(&st->fwd, cfg);
            if (vr != VKFFT_SUCCESS)
                throw std::runtime_error("VkFFT fwd init failed: " + std::to_string((int)vr));
        }

        // Inverse: complex d_MF_all_ → real d_M_all_
        {
            VkFFTConfiguration cfg = {};
            cfg.FFTdim            = static_cast<uint64_t>(fft_dim);
            cfg.size[0]           = static_cast<uint64_t>(pad_nx_);
            cfg.size[1]           = static_cast<uint64_t>(pad_ny_);
            if (fft_dim == 3) cfg.size[2] = static_cast<uint64_t>(pad_nz_);
            cfg.numberBatches     = 3;
            cfg.performR2C        = 1;
            cfg.doublePrecision   = use_double;
            cfg.device            = &cu_dev;
            cfg.stream            = &st->stream;  // shared pointer → same variable
            cfg.isOutputFormatted = 1;           // separate real outputBuffer
            cfg.buffer            = &d_MF_all_;  // complex input
            cfg.bufferSize        = &cplx_bytes;
            cfg.outputBuffer      = &d_M_all_;   // real output
            cfg.outputBufferSize  = &real_bytes;
            VkFFTResult vr = initializeVkFFT(&st->inv, cfg);
            if (vr != VKFFT_SUCCESS)
                throw std::runtime_error("VkFFT inv init failed: " + std::to_string((int)vr));
        }
    }
#endif // MICROMAG_VKFFT

    precompute_kernel();
    // No CPU sync needed: precompute and all subsequent accumulate_gpu_ptr()
    // calls run on the same stream_, so ordering is guaranteed by CUDA.
}

// ===========================================================================
// set_stream (P2: single-stream refactor)
// ===========================================================================
void DemagFieldGPU::set_stream(void* s) {
    if (stream_ == s) return;  // already on this stream
    if (stream_owned_ && stream_) {
        // Sync first: precompute may still be in-flight on the private stream.
        // Without this, the MAC kernel on the new stream can race with precompute.
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    }
    stream_       = s;
    stream_owned_ = false;
    // plan_fwd_ is precompute-only (already done) — no need to update its stream.
#ifndef MICROMAG_VKFFT
    cufftSetStream(plan_fwd_batch_, static_cast<cudaStream_t>(s));
    cufftSetStream(plan_inv_batch_, static_cast<cudaStream_t>(s));
#else
    if (vkfft_state_) {
        // VkFFT dereferences cfg.stream at each VkFFTAppend call.
        // Both fwd and inv share &st->stream, so updating it here suffices.
        struct VkFFT_State { VkFFTApplication fwd; VkFFTApplication inv; cudaStream_t stream; };
        static_cast<VkFFT_State*>(vkfft_state_)->stream = static_cast<cudaStream_t>(s);
    }
#endif
}

// ===========================================================================
// Destructor
// ===========================================================================
DemagFieldGPU::~DemagFieldGPU() {
    cufftDestroy(plan_fwd_);
#ifndef MICROMAG_VKFFT
    cufftDestroy(plan_fwd_batch_);
    cufftDestroy(plan_inv_batch_);
#else
    if (vkfft_state_) {
        struct VkFFT_State { VkFFTApplication fwd; VkFFTApplication inv; };
        auto* st = static_cast<VkFFT_State*>(vkfft_state_);
        deleteVkFFT(&st->fwd);
        deleteVkFFT(&st->inv);
        delete st;
        vkfft_state_ = nullptr;
    }
#endif

    cudaFree(d_r_buf_); cudaFree(d_c_buf_);
    cudaFree(d_K_xx_); cudaFree(d_K_yy_); cudaFree(d_K_zz_);
    cudaFree(d_K_xy_); cudaFree(d_K_xz_); cudaFree(d_K_yz_);
    cudaFree(d_M_all_);    cudaFree(d_MF_all_);
    cudaFree(d_Hunpad_all_);
    cudaFree(d_M_compact_);

    cudaFreeHost(h_M_compact_pinned_);
    cudaFreeHost(h_Hunpad_all_pinned_);

    if (stream_ && stream_owned_)
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ===========================================================================
// precompute_kernel ??GPU version (replaces CPU loops)
//
// Before: CPU computed 6D Newell sums in 3 nested loops ??~35s for 2.5M cells
// After:  fill_diag_gpu / fill_offdiag_gpu kernels launch 2.5M GPU threads
//         that compute their cell in parallel ??<1s for 2.5M cells
//
// Per-component pipeline (6 iterations):
//   1. cudaMemsetAsync d_r_buf_ = 0          (GPU, async)
//   2. fill_diag_gpu or fill_offdiag_gpu      (GPU, parallel ??all on stream_)
//   3. cufftExecD2Z                           (GPU, uses stream_ via SetStream)
//   4. cudaMemcpyAsync D2D ??d_K_xxx         (GPU, async D2D)
// ===========================================================================
void DemagFieldGPU::precompute_kernel() {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);

    // Thread block: 16횞16횞1 = 256 threads.  Grid covers (nx, ny, nz) cells.
    const dim3 blk(16, 16, 1);
    const dim3 grd(
        static_cast<unsigned>((nx_ + 15) / 16),
        static_cast<unsigned>((ny_ + 15) / 16),
        static_cast<unsigned>(nz_));

    // Helper: zero padded buffer, run fill kernel, FFT (always D2Z = double),
    // then extract the REAL part of the first quadrant (iy∈[0..symm_ny-1],
    // iz∈[0..symm_nz-1]) into the compressed GReal-typed kernel storage d_K_dest.
    // Y+Z mirror symmetry means only this quadrant is needed; the MAC kernel
    // reconstructs remaining bins with appropriate sign flips at runtime.
    constexpr int BLK_PRE = 256;
    auto copy_or_convert = [&](void* d_K_dest) {
        const int gcq = static_cast<int>((symm_sz_ + BLK_PRE - 1) / BLK_PRE);
        dc_to_real_quadrant<<<gcq, BLK_PRE, 0, s>>>(
            reinterpret_cast<GREAL_CUFFT_REAL*>(d_K_dest),
            reinterpret_cast<const cufftDoubleComplex*>(d_c_buf_),
            fft_nx_, symm_ny_, symm_nz_, pad_ny_);
        CUDA_CHECK(cudaGetLastError());
    };

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
        copy_or_convert(d_K_dest);
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
        copy_or_convert(d_K_dest);
    };

    // Diagonal: K_xx (perm=0), K_yy (perm=1: swap x?봸), K_zz (perm=2: swap x?봹)
    fill_fft_diag(d_K_xx_, 0);
    fill_fft_diag(d_K_yy_, 1);
    fill_fft_diag(d_K_zz_, 2);

    // Off-diagonal: parity (sx,sy,sz) as in CPU code, perm selects axis mapping
    fill_fft_offdiag(d_K_xy_, -1, -1, +1, 0);  // N_xy(x,y,z)
    fill_fft_offdiag(d_K_xz_, -1, +1, -1, 1);  // N_xz ??N_xy(x,z,y)
    fill_fft_offdiag(d_K_yz_, +1, -1, -1, 2);  // N_yz ??N_xy(y,z,x)
    // (cudaStreamSynchronize called by constructor after precompute_kernel)
}

// ===========================================================================
// accumulate ??batch FFT pipeline (Step 6a)
//
// Pipeline (2 cuFFT calls, 2 CUDA kernels, 1 upload, 1 download):
//   1. Fill h_M_all_pinned_ [3 횞 real_sz] with padded Mx,My,Mz (CPU)
//   2. cudaMemcpy H2D: h_M_all ??d_M_all (1 call, 3횞640KB)
//   3. cufftExecD2Z BATCH=3: d_M_all ??d_MF_all (1 call)
//   4. pointwise_mac_all3: d_MF_all ??d_HF_all (1 kernel)
//   5. cufftExecZ2D BATCH=3: d_HF_all ??d_H_all (1 call)
//   6. extract_all3: d_H_all ??d_Hunpad_all (1 kernel)
//   7. cudaMemcpy D2H: d_Hunpad_all ??h_Hunpad_all (1 call, 3횞80KB)
//   8. Accumulate h_Hunpad_all ??H_out (1 CPU loop)
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
    //       ??Overlaps with 1b: PCIe and GPU memset run in parallel
    //       (possible because they access different buffers in same stream;
    //        the stream serializes them, but the HW can overlap DMA + GPU)
    //   1d. scatter_m_all3 kernel                        [GPU]
    // ------------------------------------------------------------------

    // 1a. Compact fill (CPU ??free to run while GPU processes previous step)
    GReal* hMx = h_M_compact_pinned_;
    GReal* hMy = hMx + unpad_sz_;
    GReal* hMz = hMy + unpad_sz_;
    for (Index i = 0; i < static_cast<Index>(unpad_sz_); ++i) {
        hMx[i] = Ms * m[i].x;
        hMy[i] = Ms * m[i].y;
        hMz[i] = Ms * m[i].z;
    }

    // 1b. Async upload (pinned ??GPU, returns immediately)
    CUDA_CHECK(cudaMemcpyAsync(d_M_compact_, h_M_compact_pinned_,
                               3 * unpad_sz_ * sizeof(GReal),
                               cudaMemcpyHostToDevice, s));

    // 1c. Async zero padded buffer (runs on GPU after memcpy in stream order)
    CUDA_CHECK(cudaMemsetAsync(d_M_all_, 0, 3 * real_sz_ * sizeof(GReal), s));

    // 1d. Scatter (depends on both 1b and 1c ??stream ensures ordering)
    dim3 blk_sc(16, 16, 1);
    dim3 grd_sc(
        static_cast<unsigned>((nx_ + blk_sc.x - 1) / blk_sc.x),
        static_cast<unsigned>((ny_ + blk_sc.y - 1) / blk_sc.y),
        static_cast<unsigned>(nz_));
    scatter_m_all3<<<grd_sc, blk_sc, 0, s>>>(
        reinterpret_cast<GReal*>(d_M_all_),
        reinterpret_cast<const GReal*>(d_M_compact_),
        static_cast<size_t>(nx_), static_cast<size_t>(ny_),
        static_cast<size_t>(nz_),
        static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
        real_sz_, unpad_sz_);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 3. Batch forward FFT
    // ------------------------------------------------------------------
#ifndef MICROMAG_VKFFT
    CUFFT_CHECK(GREAL_CUFFT_EXEC_FWD(plan_fwd_batch_,
                              reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_),
                              reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_)));
#else
    {
        struct VkFFT_State { VkFFTApplication fwd; VkFFTApplication inv; };
        VkFFTLaunchParams lp = {};  // buffers and stream already set in app configuration
        VkFFTResult vr = VkFFTAppend(
            &static_cast<VkFFT_State*>(vkfft_state_)->fwd, -1, &lp);
        if (vr != VKFFT_SUCCESS)
            throw std::runtime_error("VkFFT fwd exec failed: " + std::to_string((int)vr));
    }
#endif

    // ------------------------------------------------------------------
    // 4. Combined pointwise MAC — dispatch to symmetric kernel (stream_)
    // In-place: H_f overwrites M_f in d_MF_all_ (each thread reads its 3 input
    // bins into registers before writing) → no separate d_HF_all_ buffer needed.
    // ------------------------------------------------------------------
    constexpr int BLK = 256;
    {
        if (pad_nz_ == 1) {
            // Thin-film 2D: half Y threads, Kxz=Kyz=0
            dim3 blk_m(16, 16, 1);
            dim3 grd_m(static_cast<unsigned>((fft_nx_ + 15) / 16),
                       static_cast<unsigned>((symm_ny_ + 15) / 16), 1);
            mac_symm_2d<<<grd_m, blk_m, 0, s>>>(
                as_cx(d_MF_all_),
                as_realc(d_K_xx_), as_realc(d_K_xy_),
                as_realc(d_K_yy_), as_realc(d_K_zz_),
                as_cxc(d_MF_all_),
                fft_nx_, pad_ny_, symm_ny_, cplx_sz_);
        } else {
            // General 3D: full bins, compressed kernel lookup with sign reconstruction
            const int gcx = static_cast<int>((cplx_sz_ + BLK - 1) / BLK);
            mac_symm_3d<<<gcx, BLK, 0, s>>>(
                as_cx(d_MF_all_),
                as_realc(d_K_xx_), as_realc(d_K_xy_), as_realc(d_K_xz_),
                as_realc(d_K_yy_), as_realc(d_K_yz_), as_realc(d_K_zz_),
                as_cxc(d_MF_all_),
                fft_nx_, pad_ny_, pad_nz_, symm_ny_, symm_nz_, cplx_sz_);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // ------------------------------------------------------------------
    // 5. Batch inverse FFT
    // ------------------------------------------------------------------
    // Inverse FFT output reuses d_M_all_ (free after the forward FFT) instead
    // of a dedicated d_H_all_ buffer — same size (3×real_sz), saves VRAM.
#ifndef MICROMAG_VKFFT
    CUFFT_CHECK(GREAL_CUFFT_EXEC_INV(plan_inv_batch_,
                              reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_),
                              reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_)));
#else
    {
        struct VkFFT_State { VkFFTApplication fwd; VkFFTApplication inv; };
        VkFFTLaunchParams lp = {};  // buffers and stream already set in app configuration
        VkFFTResult vr = VkFFTAppend(
            &static_cast<VkFFT_State*>(vkfft_state_)->inv, 1, &lp);
        if (vr != VKFFT_SUCCESS)
            throw std::runtime_error("VkFFT inv exec failed: " + std::to_string((int)vr));
    }
#endif

    // ------------------------------------------------------------------
    // 6. Extract all 3 unpadded H components (stream_)
    // ------------------------------------------------------------------
    dim3 blk_ext(16, 16, 1);
    dim3 grd_ext(
        static_cast<unsigned>((nx_ + blk_ext.x - 1) / blk_ext.x),
        static_cast<unsigned>((ny_ + blk_ext.y - 1) / blk_ext.y),
        static_cast<unsigned>(nz_));

    extract_all3<<<grd_ext, blk_ext, 0, s>>>(
        reinterpret_cast<GReal*>(d_Hunpad_all_),
        reinterpret_cast<const GReal*>(d_M_all_),
        static_cast<size_t>(nx_), static_cast<size_t>(ny_),
        static_cast<size_t>(nz_),
        static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
        real_sz_, unpad_sz_, norm);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 7. Async download (pinned buffer ??returns immediately)
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(h_Hunpad_all_pinned_, d_Hunpad_all_,
                               3 * unpad_sz_ * sizeof(GReal),
                               cudaMemcpyDeviceToHost, s));

    // ------------------------------------------------------------------
    // Step 6c: ONE sync point ??CPU waits here for the full pipeline
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaStreamSynchronize(s));

    // ------------------------------------------------------------------
    // 8. Accumulate into H_out (single loop, all 3 components)
    // ------------------------------------------------------------------
    const GReal* hx = h_Hunpad_all_pinned_;
    const GReal* hy = hx + unpad_sz_;
    const GReal* hz = hy + unpad_sz_;
    for (Index i = 0; i < static_cast<Index>(unpad_sz_); ++i) {
        H_out[i].x += hx[i];
        H_out[i].y += hy[i];
        H_out[i].z += hz[i];
    }
}

// ===========================================================================
// G6 helper kernels
// ===========================================================================

// Scale src by 'scale' and write to dst  (flat 3N op, for Ms-scaling d_m ??d_M_compact)
__global__ static void scale_copy_kernel(
    GReal* __restrict__        dst,
    const GReal* __restrict__  src,
    double scale, int N3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N3) return;
    dst[i] = static_cast<GReal>(scale * static_cast<double>(src[i]));
}

// ===========================================================================
// accumulate_gpu_ptr ??GPU-pointer path for G6 full-LLG pipeline
//
// d_m:    [3횞N] component-major (unit magnetization from GPUMagState::d_m_)
// d_H_out:[3횞N] component-major (adds H_demag in-place, same layout)
//
// Pipeline (entirely on stream_; no PCIe):
//   scale_copy:    d_M_compact_ = Ms * d_m   (D2D, no PCIe)
//   memset:        d_M_all_ = 0
//   scatter:       d_M_compact_ ??d_M_all_ (padded)
//   FFT batch:     d_M_all_ ??d_MF_all_
//   pointwise MAC: d_MF_all_ ??d_HF_all_
//   IFFT batch:    d_HF_all_ ??d_H_all_
//   extract:       d_H_all_ ??d_Hunpad_all_ (with IFFT normalisation)
//   add:           d_H_out += d_Hunpad_all_
// ===========================================================================
void DemagFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                         const Material& mat,
                                         GReal* d_H_out) const {
    const cudaStream_t s    = static_cast<cudaStream_t>(stream_);
    const double Ms         = mat.Ms;
    const double norm       = -1.0 / static_cast<double>(pad_nx_ * pad_ny_ * pad_nz_);
    constexpr int BLK       = 256;

    // --- temporary phase profiler (MICROMAG_DEMAG_PROFILE=1) ----------------
    static const bool prof = (std::getenv("MICROMAG_DEMAG_PROFILE") != nullptr);
    static int prof_call = 0;
    cudaEvent_t ev[7];
    if (prof) { for (auto& e : ev) cudaEventCreate(&e); cudaEventRecord(ev[0], s); }
    auto mark = [&](int i){ if (prof) cudaEventRecord(ev[i], s); };
    // -----------------------------------------------------------------------

    // 1. Ms-scale d_m ??d_M_compact_  (D2D, no PCIe)
    {
        const int N3  = static_cast<int>(3 * unpad_sz_);
        const int grd = (N3 + BLK - 1) / BLK;
        scale_copy_kernel<<<grd, BLK, 0, s>>>(
            reinterpret_cast<GReal*>(d_M_compact_), d_m, Ms, N3);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. Zero padded M buffer
    CUDA_CHECK(cudaMemsetAsync(d_M_all_, 0, 3 * real_sz_ * sizeof(GReal), s));

    // 3. Scatter compact ??padded
    {
        dim3 blk_sc(16, 16, 1);
        dim3 grd_sc((unsigned)((nx_+15)/16), (unsigned)((ny_+15)/16), (unsigned)nz_);
        scatter_m_all3<<<grd_sc, blk_sc, 0, s>>>(
            reinterpret_cast<GReal*>(d_M_all_),
            reinterpret_cast<const GReal*>(d_M_compact_),
            (size_t)nx_, (size_t)ny_, (size_t)nz_,
            (size_t)pad_nx_, (size_t)pad_ny_, real_sz_, unpad_sz_);
        CUDA_CHECK(cudaGetLastError());
    }
    mark(1);  // prep done (scale+memset+scatter)

    // 4. Batch forward FFT
#ifndef MICROMAG_VKFFT
    CUFFT_CHECK(GREAL_CUFFT_EXEC_FWD(plan_fwd_batch_,
        reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_),
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_)));
#else
    {
        struct VkFFT_State { VkFFTApplication fwd; VkFFTApplication inv; cudaStream_t stream; };
        // Stream is bound at init via cfg.stream = &st->stream; updated by set_stream().
        VkFFTResult vr = VkFFTAppend(&static_cast<VkFFT_State*>(vkfft_state_)->fwd, -1, nullptr);
        if (vr != VKFFT_SUCCESS) throw std::runtime_error("VkFFT fwd exec: " + std::to_string((int)vr));
    }
#endif
    mark(2);  // fwd FFT done

    // 5. Pointwise MAC — dispatch to appropriate symmetric kernel
    // In-place: H_f overwrites M_f in d_MF_all_ (no separate d_HF_all_).
    {
        if (pad_nz_ == 1) {
            // Thin-film 2D: half Y threads, Kxz=Kyz=0
            dim3 blk_m(16, 16, 1);
            dim3 grd_m(static_cast<unsigned>((fft_nx_ + 15) / 16),
                       static_cast<unsigned>((symm_ny_ + 15) / 16), 1);
            mac_symm_2d<<<grd_m, blk_m, 0, s>>>(
                as_cx(d_MF_all_),
                as_realc(d_K_xx_), as_realc(d_K_xy_),
                as_realc(d_K_yy_), as_realc(d_K_zz_),
                as_cxc(d_MF_all_),
                fft_nx_, pad_ny_, symm_ny_, cplx_sz_);
        } else {
            // General 3D: full bins, compressed kernel lookup with sign reconstruction
            const int gcx = static_cast<int>((cplx_sz_ + BLK - 1) / BLK);
            mac_symm_3d<<<gcx, BLK, 0, s>>>(
                as_cx(d_MF_all_),
                as_realc(d_K_xx_), as_realc(d_K_xy_), as_realc(d_K_xz_),
                as_realc(d_K_yy_), as_realc(d_K_yz_), as_realc(d_K_zz_),
                as_cxc(d_MF_all_),
                fft_nx_, pad_ny_, pad_nz_, symm_ny_, symm_nz_, cplx_sz_);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    mark(3);  // MAC done (mac_symm_2d or mac_symm_3d)

    // 6. Batch inverse FFT — output reuses d_M_all_ (free after forward FFT).
#ifndef MICROMAG_VKFFT
    CUFFT_CHECK(GREAL_CUFFT_EXEC_INV(plan_inv_batch_,
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_),
        reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_)));
#else
    {
        struct VkFFT_State { VkFFTApplication fwd; VkFFTApplication inv; cudaStream_t stream; };
        VkFFTResult vr = VkFFTAppend(&static_cast<VkFFT_State*>(vkfft_state_)->inv, 1, nullptr);
        if (vr != VKFFT_SUCCESS) throw std::runtime_error("VkFFT inv exec: " + std::to_string((int)vr));
    }
#endif
    mark(4);  // inv FFT done

    // 7. Fused extract + normalise + accumulate straight into d_H_out
    //    (replaces extract_all3 + add_3N_kernel; no d_Hunpad_all_ round-trip).
    {
        dim3 blk_ext(16, 16, 1);
        dim3 grd_ext((unsigned)((nx_+15)/16), (unsigned)((ny_+15)/16), (unsigned)nz_);
        extract_add_all3<<<grd_ext, blk_ext, 0, s>>>(
            d_H_out,
            reinterpret_cast<const GReal*>(d_M_all_),
            (size_t)nx_, (size_t)ny_, (size_t)nz_,
            (size_t)pad_nx_, (size_t)pad_ny_,
            real_sz_, unpad_sz_, norm);
        CUDA_CHECK(cudaGetLastError());
    }
    mark(5);  // extract done
    // Sync only in standalone mode (caller may be on a different stream).
    // In shared-stream mode (stream_owned_==false) the caller's stream ordering
    // guarantees all our writes to d_H_out are visible before the next kernel.
    if (stream_owned_) {
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    if (prof) {
        cudaStreamSynchronize(s);   // ensure all events are completed before readback
        float t[5];
        cudaEventElapsedTime(&t[0], ev[0], ev[1]);   // prep
        cudaEventElapsedTime(&t[1], ev[1], ev[2]);   // fwd FFT
        cudaEventElapsedTime(&t[2], ev[2], ev[3]);   // MAC
        cudaEventElapsedTime(&t[3], ev[3], ev[4]);   // inv FFT
        cudaEventElapsedTime(&t[4], ev[4], ev[5]);   // extract
        if (++prof_call % 20 == 0)
            std::fprintf(stderr,
                "[demag %dx%dx%d] prep=%.3f fwdFFT=%.3f MAC=%.3f invFFT=%.3f extract=%.3f ms "
                "(FFT=%.0f%%)\n", (int)nx_, (int)ny_, (int)nz_,
                t[0], t[1], t[2], t[3], t[4],
                100.0 * (t[1]+t[3]) / (t[0]+t[1]+t[2]+t[3]+t[4]));
        for (auto& e : ev) cudaEventDestroy(e);
    }
}

// ===========================================================================
// energy
// ===========================================================================
Real DemagFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    const StructuredGrid& g = m.grid();
    VectorField3D H(g);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0,0,0};
    accumulate(m, mat, H);
    Real E = 0.0;
    const Real dV=g.cell_volume();
    for (Index i = 0; i < m.size(); ++i)
        E -= constants::mu_0 * mat.Ms * m[i].dot(H[i]) * dV;
    return 0.5 * E;
}

ScalarField3D DemagFieldGPU::energy_density(const VectorField3D& m,
                                              const Material& mat) const {
    const StructuredGrid& g = m.grid();
    VectorField3D H(g);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0,0,0};
    accumulate(m, mat, H);  // GPU compute ??downloads to CPU H
    ScalarField3D edens(g);
    const Real prefac = -0.5 * constants::mu_0 * mat.Ms;
    for (Index i = 0; i < m.size(); ++i)
        edens[i] = prefac * m[i].dot(H[i]);
    return edens;
}

}  // namespace micromag

#endif // MICROMAG_CUDA




