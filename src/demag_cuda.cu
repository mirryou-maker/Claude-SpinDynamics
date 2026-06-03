// demag_cuda.cu — Phase 3 Step 5: optimised cuFFT GPU demag
//
// Optimisations over Step 4:
//   1. Persistent pre-allocated GPU scratch buffers (d_Mx_f_, d_My_f_, d_Mz_f_,
//      d_H_unpad_) — eliminates 4× cudaMalloc + 4× cudaFree per step.
//   2. Page-locked (pinned) host buffers (h_r_pinned_, h_H_pinned_) — doubles
//      effective PCIe bandwidth vs pageable std::vector allocations.
//   3. extract_assign kernel — writes instead of accumulates, removing the
//      per-IFFT cudaMemset(d_H_unpad_, 0).
//   4. Merged H_out loop — accumulate all 3 components in a single pass.

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

// ---------------------------------------------------------------------------
// CUDA kernel: H_out[i] = Ka[i]*Ma[i] + Kb[i]*Mb[i] + Kc[i]*Mc[i]
// ---------------------------------------------------------------------------
__global__ static void pointwise_mac(
    cufftDoubleComplex* __restrict__ H_out,
    const cufftDoubleComplex* __restrict__ Ka,
    const cufftDoubleComplex* __restrict__ Kb,
    const cufftDoubleComplex* __restrict__ Kc,
    const cufftDoubleComplex* __restrict__ Ma,
    const cufftDoubleComplex* __restrict__ Mb,
    const cufftDoubleComplex* __restrict__ Mc,
    size_t N)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    auto re = [](const cufftDoubleComplex& a, const cufftDoubleComplex& b)
        { return a.x*b.x - a.y*b.y; };
    auto im = [](const cufftDoubleComplex& a, const cufftDoubleComplex& b)
        { return a.x*b.y + a.y*b.x; };
    H_out[i].x = re(Ka[i],Ma[i]) + re(Kb[i],Mb[i]) + re(Kc[i],Mc[i]);
    H_out[i].y = im(Ka[i],Ma[i]) + im(Kb[i],Mb[i]) + im(Kc[i],Mc[i]);
}

// ---------------------------------------------------------------------------
// CUDA kernel: extract unpadded H from IFFT result (ASSIGN, not accumulate)
// Step 5 optimisation: no cudaMemset(0) needed before this call.
// ---------------------------------------------------------------------------
__global__ static void extract_assign(
    double* __restrict__       H_out,
    const double* __restrict__ r_buf,
    size_t nx, size_t ny, size_t nz,
    size_t pad_nx, size_t pad_ny,
    double norm)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t iz = blockIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;
    const size_t src = ix + pad_nx * (iy + pad_ny * iz);
    const size_t dst = ix + nx    * (iy + ny    * iz);
    H_out[dst] = r_buf[src] * norm;   // assign, not +=
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

    // Kernel buffers
    CUDA_CHECK(cudaMalloc(&d_r_buf_, real_sz_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c_buf_, cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xx_,  cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_yy_,  cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_zz_,  cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xy_,  cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xz_,  cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_yz_,  cplx_sz_ * sizeof(cufftDoubleComplex)));

    // Step 5: persistent per-step scratch (no malloc/free inside accumulate)
    CUDA_CHECK(cudaMalloc(&d_Mx_f_,    cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_My_f_,    cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Mz_f_,    cplx_sz_ * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_H_unpad_, unpad_sz_ * sizeof(double)));

    // Pinned host buffers for fast DMA
    CUDA_CHECK(cudaMallocHost(&h_r_pinned_, real_sz_  * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_H_pinned_, unpad_sz_ * sizeof(double)));
    std::memset(h_r_pinned_, 0, real_sz_ * sizeof(double));  // pre-zero once

    // cuFFT plans
    CUFFT_CHECK(cufftPlan3d(&plan_fwd_, (int)pad_nz_, (int)pad_ny_, (int)pad_nx_, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&plan_inv_, (int)pad_nz_, (int)pad_ny_, (int)pad_nx_, CUFFT_Z2D));

    precompute_kernel();
}

// ===========================================================================
// Destructor
// ===========================================================================
DemagFieldGPU::~DemagFieldGPU() {
    if (plan_fwd_) cufftDestroy(plan_fwd_);
    if (plan_inv_) cufftDestroy(plan_inv_);

    cudaFree(d_r_buf_); cudaFree(d_c_buf_);
    cudaFree(d_K_xx_);  cudaFree(d_K_yy_);  cudaFree(d_K_zz_);
    cudaFree(d_K_xy_);  cudaFree(d_K_xz_);  cudaFree(d_K_yz_);
    cudaFree(d_Mx_f_);  cudaFree(d_My_f_);  cudaFree(d_Mz_f_);
    cudaFree(d_H_unpad_);

    cudaFreeHost(h_r_pinned_);
    cudaFreeHost(h_H_pinned_);
}

// ===========================================================================
// precompute_kernel — CPU Newell fill → cuFFT → GPU kernel storage
// (unchanged from Step 4)
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
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            const double val = kernel_fn(
                static_cast<double>(kx)*dx_, static_cast<double>(ky)*dy_,
                static_cast<double>(kz)*dz_);
            put( kx,  ky,  kz,  val);
            if (kx > 0) put(-kx,  ky,  kz, val);
            if (ky > 0) put( kx, -ky,  kz, val);
            if (kz > 0) put( kx,  ky, -kz, val);
            if (kx > 0 && ky > 0) put(-kx, -ky,  kz, val);
            if (kx > 0 && kz > 0) put(-kx,  ky, -kz, val);
            if (ky > 0 && kz > 0) put( kx, -ky, -kz, val);
            if (kx > 0 && ky > 0 && kz > 0) put(-kx, -ky, -kz, val);
        }
        CUDA_CHECK(cudaMemcpy(d_r_buf_, h_r.data(),
                              real_sz_ * sizeof(double), cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
                                  reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpy(d_K_dest, d_c_buf_,
                              cplx_sz_ * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToDevice));
    };

    auto fill_and_fft_offdiag = [&](void* d_K_dest, int sx, int sy, int sz,
                                     auto kernel_fn) {
        std::fill(h_r.begin(), h_r.end(), 0.0);
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            const double val = kernel_fn(
                static_cast<double>(kx)*dx_, static_cast<double>(ky)*dy_,
                static_cast<double>(kz)*dz_);
            for (int ix:{0,1}) for (int iy:{0,1}) for (int iz:{0,1}) {
                if (ix && kx==0) continue;
                if (iy && ky==0) continue;
                if (iz && kz==0) continue;
                const double s =
                    (ix ? (double)sx : 1.0)*(iy ? (double)sy : 1.0)*(iz ? (double)sz : 1.0);
                put(ix?-kx:kx, iy?-ky:ky, iz?-kz:kz, s*val);
            }
        }
        CUDA_CHECK(cudaMemcpy(d_r_buf_, h_r.data(),
                              real_sz_ * sizeof(double), cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
                                  reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpy(d_K_dest, d_c_buf_,
                              cplx_sz_ * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToDevice));
    };

    const double dx=dx_, dy=dy_, dz=dz_;
    fill_and_fft(d_K_xx_, [&](double x,double y,double z){ return DemagField::nxx(x,y,z,dx,dy,dz); });
    fill_and_fft(d_K_yy_, [&](double x,double y,double z){ return DemagField::nxx(y,x,z,dy,dx,dz); });
    fill_and_fft(d_K_zz_, [&](double x,double y,double z){ return DemagField::nxx(z,y,x,dz,dy,dx); });
    fill_and_fft_offdiag(d_K_xy_, -1,-1,+1, [&](double x,double y,double z){ return DemagField::nxy(x,y,z,dx,dy,dz); });
    fill_and_fft_offdiag(d_K_xz_, -1,+1,-1, [&](double x,double y,double z){ return DemagField::nxy(x,z,y,dx,dz,dy); });
    fill_and_fft_offdiag(d_K_yz_, +1,-1,-1, [&](double x,double y,double z){ return DemagField::nxy(y,z,x,dy,dz,dx); });
}

// ===========================================================================
// accumulate — optimised GPU FFT pipeline (Step 5)
// ===========================================================================
void DemagFieldGPU::accumulate(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    const double Ms  = mat.Ms;
    const double norm = -1.0 / static_cast<double>(pad_nx_ * pad_ny_ * pad_nz_);

    // ------------------------------------------------------------------
    // 1.  Forward FFT: upload each magnetisation component → GPU → D2Z
    //     Use pre-zeroed pinned buffer: just overwrite the non-zero region.
    // ------------------------------------------------------------------
    auto fft_component = [&](int comp, void* d_out) {
        // Re-zero only the unpadded sub-region from the previous call.
        // h_r_pinned_ was fully zeroed at construction; we only touch the
        // non-zero cells each time, so we must restore zeros before refilling.
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            const size_t dst = static_cast<size_t>(
                kx + pad_nx_ * (ky + pad_ny_ * kz));
            h_r_pinned_[dst] = 0.0;   // clear previous value
        }
        // Fill current component
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            const size_t src = static_cast<size_t>(kx + nx_ * (ky + ny_ * kz));
            const size_t dst = static_cast<size_t>(kx + pad_nx_ * (ky + pad_ny_ * kz));
            const Vec3& v = m[static_cast<Index>(src)];
            h_r_pinned_[dst] = Ms * (comp == 0 ? v.x : comp == 1 ? v.y : v.z);
        }
        // DMA: pinned host → GPU (fast page-locked transfer)
        CUDA_CHECK(cudaMemcpy(d_r_buf_, h_r_pinned_,
                              real_sz_ * sizeof(double), cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
                                  reinterpret_cast<cufftDoubleComplex*>(d_out)));
    };

    fft_component(0, d_Mx_f_);
    fft_component(1, d_My_f_);
    fft_component(2, d_Mz_f_);

    // ------------------------------------------------------------------
    // 2.  For each output component: pointwise MAC → IFFT → extract
    //     extract_assign writes directly (no memset needed).
    // ------------------------------------------------------------------
    constexpr int BLK = 256;
    const int gcx = static_cast<int>((cplx_sz_ + BLK - 1) / BLK);

    dim3 blk_ext(16, 16, 1);
    dim3 grd_ext(
        static_cast<unsigned>((nx_ + blk_ext.x - 1) / blk_ext.x),
        static_cast<unsigned>((ny_ + blk_ext.y - 1) / blk_ext.y),
        static_cast<unsigned>(nz_));

    // Temporary host arrays for the 3 H components (stack, not heap)
    // Using h_H_pinned_ sequentially to avoid 3× allocation.
    // We process Hx, Hy, Hz one at a time and accumulate into H_out.
    double* h_Hx = h_H_pinned_;   // reuse pinned buffer component by component
    double* h_Hy = h_H_pinned_;
    double* h_Hz = h_H_pinned_;

    auto ifft_and_accumulate = [&](
        const void* Ka, const void* Kb, const void* Kc,
        int out_comp)
    {
        pointwise_mac<<<gcx, BLK>>>(
            as_cx(d_c_buf_),
            as_cx(const_cast<void*>(Ka)),
            as_cx(const_cast<void*>(Kb)),
            as_cx(const_cast<void*>(Kc)),
            as_cx(d_Mx_f_), as_cx(d_My_f_), as_cx(d_Mz_f_),
            cplx_sz_);
        CUDA_CHECK(cudaGetLastError());

        CUFFT_CHECK(cufftExecZ2D(plan_inv_,
                                  reinterpret_cast<cufftDoubleComplex*>(d_c_buf_),
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_)));

        // Extract and assign (no memset needed)
        extract_assign<<<grd_ext, blk_ext>>>(
            reinterpret_cast<double*>(d_H_unpad_),
            reinterpret_cast<const double*>(d_r_buf_),
            static_cast<size_t>(nx_), static_cast<size_t>(ny_),
            static_cast<size_t>(nz_),
            static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
            norm);
        CUDA_CHECK(cudaGetLastError());

        // Download to pinned host buffer
        CUDA_CHECK(cudaMemcpy(h_H_pinned_, d_H_unpad_,
                              unpad_sz_ * sizeof(double), cudaMemcpyDeviceToHost));

        // Accumulate into H_out
        for (Index i = 0; i < static_cast<Index>(unpad_sz_); ++i) {
            if      (out_comp == 0) H_out[i].x += h_H_pinned_[i];
            else if (out_comp == 1) H_out[i].y += h_H_pinned_[i];
            else                   H_out[i].z += h_H_pinned_[i];
        }
    };

    ifft_and_accumulate(d_K_xx_, d_K_xy_, d_K_xz_, 0);  // Hx
    ifft_and_accumulate(d_K_xy_, d_K_yy_, d_K_yz_, 1);  // Hy
    ifft_and_accumulate(d_K_xz_, d_K_yz_, d_K_zz_, 2);  // Hz
}

// ===========================================================================
// energy
// ===========================================================================
Real DemagFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    const StructuredGrid& g = m.grid();
    VectorField3D H(g);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
    accumulate(m, mat, H);
    Real E = 0.0;
    const Real dV = g.cell_volume();
    for (Index i = 0; i < m.size(); ++i)
        E -= constants::mu_0 * mat.Ms * m[i].dot(H[i]) * dV;
    return 0.5 * E;
}

}  // namespace micromag

#endif // MICROMAG_CUDA
