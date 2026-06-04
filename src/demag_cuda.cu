// demag_cuda.cu — Phase 3 Step 6b: sparse upload + cuFFT batch
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
#include <cstring>
#include <stdexcept>
#include <vector>

#include "micromag/demag_gpu.hpp"
#include "micromag/demag.hpp"

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

    precompute_kernel();
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
}

// ===========================================================================
// precompute_kernel — unchanged from Step 5 (uses single plan_fwd_)
// ===========================================================================
void DemagFieldGPU::precompute_kernel() {
    std::vector<double> h_r(real_sz_, 0.0);

    auto put = [&](Index px, Index py, Index pz, double v) {
        if (px < 0) px += pad_nx_;
        if (py < 0) py += pad_ny_;
        if (pz < 0) pz += pad_nz_;
        h_r[static_cast<size_t>(px + pad_nx_ * (py + pad_ny_ * pz))] = v;
    };

    auto fill_and_fft = [&](void* d_K_dest, auto kernel_fn) {
        std::fill(h_r.begin(), h_r.end(), 0.0);
        for (Index kz=0;kz<nz_;++kz) for (Index ky=0;ky<ny_;++ky) for (Index kx=0;kx<nx_;++kx) {
            const double val = kernel_fn(
                static_cast<double>(kx)*dx_, static_cast<double>(ky)*dy_,
                static_cast<double>(kz)*dz_);
            put( kx, ky, kz, val);
            if (kx>0) put(-kx, ky, kz, val);
            if (ky>0) put( kx,-ky, kz, val);
            if (kz>0) put( kx, ky,-kz, val);
            if (kx>0&&ky>0) put(-kx,-ky, kz, val);
            if (kx>0&&kz>0) put(-kx, ky,-kz, val);
            if (ky>0&&kz>0) put( kx,-ky,-kz, val);
            if (kx>0&&ky>0&&kz>0) put(-kx,-ky,-kz, val);
        }
        CUDA_CHECK(cudaMemcpy(d_r_buf_,h_r.data(),real_sz_*sizeof(double),cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
            reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
            reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpy(d_K_dest,d_c_buf_,cplx_sz_*sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToDevice));
    };

    auto fill_and_fft_offdiag = [&](void* d_K_dest, int sx, int sy, int sz, auto kernel_fn) {
        std::fill(h_r.begin(), h_r.end(), 0.0);
        for (Index kz=0;kz<nz_;++kz) for (Index ky=0;ky<ny_;++ky) for (Index kx=0;kx<nx_;++kx) {
            const double val = kernel_fn(
                static_cast<double>(kx)*dx_, static_cast<double>(ky)*dy_,
                static_cast<double>(kz)*dz_);
            for (int ix:{0,1}) for (int iy:{0,1}) for (int iz:{0,1}) {
                if (ix&&kx==0) continue; if (iy&&ky==0) continue; if (iz&&kz==0) continue;
                const double s=(ix?(double)sx:1.)*(iy?(double)sy:1.)*(iz?(double)sz:1.);
                put(ix?-kx:kx, iy?-ky:ky, iz?-kz:kz, s*val);
            }
        }
        CUDA_CHECK(cudaMemcpy(d_r_buf_,h_r.data(),real_sz_*sizeof(double),cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
            reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
            reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpy(d_K_dest,d_c_buf_,cplx_sz_*sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToDevice));
    };

    const double dx=dx_,dy=dy_,dz=dz_;
    fill_and_fft(d_K_xx_,[&](double x,double y,double z){return DemagField::nxx(x,y,z,dx,dy,dz);});
    fill_and_fft(d_K_yy_,[&](double x,double y,double z){return DemagField::nxx(y,x,z,dy,dx,dz);});
    fill_and_fft(d_K_zz_,[&](double x,double y,double z){return DemagField::nxx(z,y,x,dz,dy,dx);});
    fill_and_fft_offdiag(d_K_xy_,-1,-1,+1,[&](double x,double y,double z){return DemagField::nxy(x,y,z,dx,dy,dz);});
    fill_and_fft_offdiag(d_K_xz_,-1,+1,-1,[&](double x,double y,double z){return DemagField::nxy(x,z,y,dx,dz,dy);});
    fill_and_fft_offdiag(d_K_yz_,+1,-1,-1,[&](double x,double y,double z){return DemagField::nxy(y,z,x,dy,dz,dx);});
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

    // ------------------------------------------------------------------
    // Step 6b: sparse upload pipeline
    //   1a. Fill compact host buffer (3 × unpad_sz, NO zero-fill needed)
    //   1b. Upload compact 0.24MB  (vs 1.92MB before — 8× smaller)
    //   1c. GPU: zero full padded buffer (fast GPU memset ~5μs)
    //   1d. GPU: scatter compact → padded (one kernel)
    // ------------------------------------------------------------------

    // 1a. Compact fill: interleaved loop for better cache behaviour
    double* hMx = h_M_compact_pinned_;
    double* hMy = hMx + unpad_sz_;
    double* hMz = hMy + unpad_sz_;
    for (Index i = 0; i < static_cast<Index>(unpad_sz_); ++i) {
        hMx[i] = Ms * m[i].x;
        hMy[i] = Ms * m[i].y;
        hMz[i] = Ms * m[i].z;
    }

    // 1b. Upload compact buffer (0.24MB pinned → GPU)
    CUDA_CHECK(cudaMemcpy(d_M_compact_, h_M_compact_pinned_,
                          3 * unpad_sz_ * sizeof(double), cudaMemcpyHostToDevice));

    // 1c. Zero full padded buffer on GPU (1.92MB @ 400GB/s ≈ 5μs)
    CUDA_CHECK(cudaMemset(d_M_all_, 0, 3 * real_sz_ * sizeof(double)));

    // 1d. Scatter compact values into correct padded positions
    dim3 blk_sc(16, 16, 1);
    dim3 grd_sc(
        static_cast<unsigned>((nx_ + blk_sc.x - 1) / blk_sc.x),
        static_cast<unsigned>((ny_ + blk_sc.y - 1) / blk_sc.y),
        static_cast<unsigned>(nz_));
    scatter_m_all3<<<grd_sc, blk_sc>>>(
        reinterpret_cast<double*>(d_M_all_),
        reinterpret_cast<const double*>(d_M_compact_),
        static_cast<size_t>(nx_), static_cast<size_t>(ny_),
        static_cast<size_t>(nz_),
        static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
        real_sz_, unpad_sz_);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 3. Batch forward FFT: d_M_all → d_MF_all  (1 exec instead of 3)
    // ------------------------------------------------------------------
    CUFFT_CHECK(cufftExecD2Z(plan_fwd_batch_,
                              reinterpret_cast<cufftDoubleReal*>(d_M_all_),
                              reinterpret_cast<cufftDoubleComplex*>(d_MF_all_)));

    // ------------------------------------------------------------------
    // 4. Combined pointwise MAC: computes Hx,Hy,Hz frequency-domain  (1 kernel)
    // ------------------------------------------------------------------
    constexpr int BLK = 256;
    const int gcx = static_cast<int>((cplx_sz_ + BLK - 1) / BLK);

    pointwise_mac_all3<<<gcx, BLK>>>(
        as_cx(d_HF_all_),
        as_cxc(d_K_xx_), as_cxc(d_K_xy_), as_cxc(d_K_xz_),
        as_cxc(d_K_yy_), as_cxc(d_K_yz_), as_cxc(d_K_zz_),
        as_cxc(d_MF_all_),
        cplx_sz_);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 5. Batch inverse FFT: d_HF_all → d_H_all  (1 exec instead of 3)
    // ------------------------------------------------------------------
    CUFFT_CHECK(cufftExecZ2D(plan_inv_batch_,
                              reinterpret_cast<cufftDoubleComplex*>(d_HF_all_),
                              reinterpret_cast<cufftDoubleReal*>(d_H_all_)));

    // ------------------------------------------------------------------
    // 6. Extract all 3 unpadded H components in one kernel
    // ------------------------------------------------------------------
    dim3 blk_ext(16, 16, 1);
    dim3 grd_ext(
        static_cast<unsigned>((nx_ + blk_ext.x - 1) / blk_ext.x),
        static_cast<unsigned>((ny_ + blk_ext.y - 1) / blk_ext.y),
        static_cast<unsigned>(nz_));

    extract_all3<<<grd_ext, blk_ext>>>(
        reinterpret_cast<double*>(d_Hunpad_all_),
        reinterpret_cast<const double*>(d_H_all_),
        static_cast<size_t>(nx_), static_cast<size_t>(ny_),
        static_cast<size_t>(nz_),
        static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
        real_sz_, unpad_sz_, norm);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 7. Download all 3 H components in one DMA transfer
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_Hunpad_all_pinned_, d_Hunpad_all_,
                          3 * unpad_sz_ * sizeof(double), cudaMemcpyDeviceToHost));

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
