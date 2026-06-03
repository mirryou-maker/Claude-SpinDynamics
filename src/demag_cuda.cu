// demag_cuda.cu — Phase 3 Step 4: cuFFT GPU demag implementation
//
// Architecture:
//   CPU: precompute 6 Newell kernel arrays → upload to GPU (once at construction)
//   GPU: per-step FFT convolution pipeline
//        3× D2Z forward FFT  (Mx, My, Mz)
//        3× pointwise kernel multiply
//        3× Z2D inverse FFT  (Hx, Hy, Hz)
//
// Memory layout: x-fastest (matches StructuredGrid linear_index).
//   r_buf: double[pad_nz × pad_ny × pad_nx]
//   c_buf: cufftDoubleComplex[pad_nz × pad_ny × (pad_nx/2+1)]

#ifdef MICROMAG_CUDA

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

#include "micromag/demag_gpu.hpp"
#include "micromag/demag.hpp"   // DemagField::nxx / nxy

// ===========================================================================
// Error-checking helpers (inline, no separate header needed)
// ===========================================================================
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA error: ")       \
                                   + cudaGetErrorString(_e));           \
    } while (0)

#define CUFFT_CHECK(call)                                               \
    do {                                                                \
        cufftResult _r = (call);                                        \
        if (_r != CUFFT_SUCCESS)                                        \
            throw std::runtime_error("cuFFT error code " +             \
                                     std::to_string((int)_r));         \
    } while (0)

// Convenience cast from void* member to cufftDoubleComplex*
static inline cufftDoubleComplex* as_cx(void* p) {
    return reinterpret_cast<cufftDoubleComplex*>(p);
}

// ===========================================================================
// CUDA kernel: pointwise complex multiply-accumulate
//   H_out[i] = Ka[i]*Ma[i] + Kb[i]*Mb[i] + Kc[i]*Mc[i]
// ===========================================================================
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

    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    auto cmul_re = [](const cufftDoubleComplex& a, const cufftDoubleComplex& b) {
        return a.x * b.x - a.y * b.y;
    };
    auto cmul_im = [](const cufftDoubleComplex& a, const cufftDoubleComplex& b) {
        return a.x * b.y + a.y * b.x;
    };

    H_out[i].x = cmul_re(Ka[i], Ma[i]) + cmul_re(Kb[i], Mb[i]) + cmul_re(Kc[i], Mc[i]);
    H_out[i].y = cmul_im(Ka[i], Ma[i]) + cmul_im(Kb[i], Mb[i]) + cmul_im(Kc[i], Mc[i]);
}

// CUDA kernel: copy unpadded H back to H_out, applying -1/pad_total sign
__global__ static void extract_and_add(
    double* __restrict__       H_out_x,   // unpadded, size nx*ny*nz
    const double* __restrict__ r_buf,     // padded IFFT result
    size_t nx, size_t ny, size_t nz,
    size_t pad_nx, size_t pad_ny,
    double norm)                           // = -1 / pad_total
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t iz = blockIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    const size_t src = ix + pad_nx * (iy + pad_ny * iz);
    const size_t dst = ix + nx    * (iy + ny    * iz);
    H_out_x[dst] += r_buf[src] * norm;
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

    const size_t real_size    = static_cast<size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    const size_t complex_size = static_cast<size_t>(fft_nx_ * pad_ny_ * pad_nz_);

    // Allocate GPU buffers
    CUDA_CHECK(cudaMalloc(&d_r_buf_, real_size    * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c_buf_, complex_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xx_,  complex_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_yy_,  complex_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_zz_,  complex_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xy_,  complex_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_xz_,  complex_size * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_K_yz_,  complex_size * sizeof(cufftDoubleComplex)));

    // cuFFT plans — argument order: (nz, ny, nx) = slowest to fastest axis
    CUFFT_CHECK(cufftPlan3d(&plan_fwd_, (int)pad_nz_, (int)pad_ny_, (int)pad_nx_,
                             CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&plan_inv_, (int)pad_nz_, (int)pad_ny_, (int)pad_nx_,
                             CUFFT_Z2D));

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
}

// ===========================================================================
// precompute_kernel — CPU computes 6D Newell kernel, FFTs on GPU, stores
// ===========================================================================
void DemagFieldGPU::precompute_kernel() {
    const size_t real_sz    = static_cast<size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    const size_t complex_sz = static_cast<size_t>(fft_nx_ * pad_ny_ * pad_nz_);

    // Host buffers for kernel filling
    std::vector<double>               h_r(real_sz, 0.0);
    std::vector<std::complex<double>> h_c(complex_sz);

    // Helper: place a value at (px,py,pz) in padded real buffer
    auto put = [&](Index px, Index py, Index pz, double v) {
        if (px < 0) px += pad_nx_;
        if (py < 0) py += pad_ny_;
        if (pz < 0) pz += pad_nz_;
        h_r[static_cast<size_t>(px + pad_nx_ * (py + pad_ny_ * pz))] = v;
    };

    // --- One kernel component at a time ---
    // fill_and_fft: fills h_r, uploads to GPU, FFTs, stores in d_K_dest
    auto fill_and_fft = [&](void* d_K_dest, auto kernel_fn) {
        std::fill(h_r.begin(), h_r.end(), 0.0);

        // 6D double-cell sum (same as CPU DemagField::nxx/nxy)
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            const double val = kernel_fn(
                static_cast<double>(kx) * dx_,
                static_cast<double>(ky) * dy_,
                static_cast<double>(kz) * dz_);
            put( kx,  ky,  kz,  val);
            if (kx > 0) put(-kx,  ky,  kz, val);
            if (ky > 0) put( kx, -ky,  kz, val);
            if (kz > 0) put( kx,  ky, -kz, val);
            if (kx > 0 && ky > 0) put(-kx, -ky,  kz, val);
            if (kx > 0 && kz > 0) put(-kx,  ky, -kz, val);
            if (ky > 0 && kz > 0) put( kx, -ky, -kz, val);
            if (kx > 0 && ky > 0 && kz > 0) put(-kx, -ky, -kz, val);
        }

        // Upload real buffer → GPU, FFT → GPU complex buffer
        CUDA_CHECK(cudaMemcpy(d_r_buf_, h_r.data(),
                              real_sz * sizeof(double), cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
                                  reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpy(d_K_dest, d_c_buf_,
                              complex_sz * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToDevice));
    };

    // Same off-diagonal helper as CPU (sx,sy,sz = parity signs)
    auto fill_and_fft_offdiag = [&](void* d_K_dest, int sx, int sy, int sz,
                                     auto kernel_fn) {
        std::fill(h_r.begin(), h_r.end(), 0.0);

        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            const double val = kernel_fn(
                static_cast<double>(kx) * dx_,
                static_cast<double>(ky) * dy_,
                static_cast<double>(kz) * dz_);
            for (int ix : {0, 1})
            for (int iy : {0, 1})
            for (int iz : {0, 1}) {
                if (ix && kx == 0) continue;
                if (iy && ky == 0) continue;
                if (iz && kz == 0) continue;
                const double sign =
                    (ix ? static_cast<double>(sx) : 1.0) *
                    (iy ? static_cast<double>(sy) : 1.0) *
                    (iz ? static_cast<double>(sz) : 1.0);
                put(ix ? -kx : kx, iy ? -ky : ky, iz ? -kz : kz, sign * val);
            }
        }

        CUDA_CHECK(cudaMemcpy(d_r_buf_, h_r.data(),
                              real_sz * sizeof(double), cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
                                  reinterpret_cast<cufftDoubleComplex*>(d_c_buf_)));
        CUDA_CHECK(cudaMemcpy(d_K_dest, d_c_buf_,
                              complex_sz * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToDevice));
    };

    const double dx = dx_, dy = dy_, dz = dz_;

    // Diagonal components
    fill_and_fft(d_K_xx_, [&](double x, double y, double z) {
        return DemagField::nxx(x, y, z, dx, dy, dz);
    });
    fill_and_fft(d_K_yy_, [&](double x, double y, double z) {
        return DemagField::nxx(y, x, z, dy, dx, dz);
    });
    fill_and_fft(d_K_zz_, [&](double x, double y, double z) {
        return DemagField::nxx(z, y, x, dz, dy, dx);
    });

    // Off-diagonal: N_xy (odd in x,y even in z)
    fill_and_fft_offdiag(d_K_xy_, -1, -1, +1,
        [&](double x, double y, double z) {
            return DemagField::nxy(x, y, z, dx, dy, dz);
        });
    // N_xz: odd in x,z even in y → swap y↔z args
    fill_and_fft_offdiag(d_K_xz_, -1, +1, -1,
        [&](double x, double y, double z) {
            return DemagField::nxy(x, z, y, dx, dz, dy);
        });
    // N_yz: odd in y,z even in x → swap x↔y↔z args
    fill_and_fft_offdiag(d_K_yz_, +1, -1, -1,
        [&](double x, double y, double z) {
            return DemagField::nxy(y, z, x, dy, dz, dx);
        });
}

// ===========================================================================
// accumulate — full GPU FFT convolution pipeline
// ===========================================================================
void DemagFieldGPU::accumulate(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    const double Ms = mat.Ms;
    const size_t real_sz    = static_cast<size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    const size_t complex_sz = static_cast<size_t>(fft_nx_ * pad_ny_ * pad_nz_);
    const double norm = -1.0 / static_cast<double>(pad_nx_ * pad_ny_ * pad_nz_);

    // Host unpadded H arrays (to accumulate IFFT results)
    const size_t unpad_sz = static_cast<size_t>(nx_ * ny_ * nz_);
    std::vector<double> h_Hx(unpad_sz, 0.0), h_Hy(unpad_sz, 0.0), h_Hz(unpad_sz, 0.0);

    // GPU scratch for the three magnetisation component FFTs
    void* d_Mx_f = nullptr;
    void* d_My_f = nullptr;
    void* d_Mz_f = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Mx_f, complex_sz * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_My_f, complex_sz * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_Mz_f, complex_sz * sizeof(cufftDoubleComplex)));

    // GPU scratch for unpadded H (one component at a time)
    void* d_H_unpad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H_unpad, unpad_sz * sizeof(double)));

    // Helper: upload one magnetisation component to padded GPU buffer and FFT
    auto fft_component = [&](int comp, void* d_out) {
        // Zero padded buffer on GPU
        CUDA_CHECK(cudaMemset(d_r_buf_, 0, real_sz * sizeof(double)));

        // Fill unpadded region on host, then upload rows
        // We upload only the non-zero (unpadded) slice to avoid a full host buffer.
        // Build a host padded array for the component, then copy.
        std::vector<double> h_r(real_sz, 0.0);
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            const size_t src = static_cast<size_t>(kx + nx_ * (ky + ny_ * kz));
            const size_t dst = static_cast<size_t>(kx + pad_nx_ * (ky + pad_ny_ * kz));
            const Vec3& v = m[static_cast<Index>(src)];
            h_r[dst] = Ms * (comp == 0 ? v.x : comp == 1 ? v.y : v.z);
        }
        CUDA_CHECK(cudaMemcpy(d_r_buf_, h_r.data(),
                              real_sz * sizeof(double), cudaMemcpyHostToDevice));
        CUFFT_CHECK(cufftExecD2Z(plan_fwd_,
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_),
                                  reinterpret_cast<cufftDoubleComplex*>(d_out)));
    };

    // Forward FFT for each component
    fft_component(0, d_Mx_f);
    fft_component(1, d_My_f);
    fft_component(2, d_Mz_f);

    // Block/grid for pointwise kernel
    constexpr int BLOCK = 256;
    const int grid_cx = static_cast<int>((complex_sz + BLOCK - 1) / BLOCK);

    // Block/grid for extract kernel
    dim3 block_ext(16, 16, 1);
    dim3 grid_ext(
        static_cast<unsigned>((nx_ + block_ext.x - 1) / block_ext.x),
        static_cast<unsigned>((ny_ + block_ext.y - 1) / block_ext.y),
        static_cast<unsigned>(nz_));

    // Helper: compute H component = -(Ka*Ma + Kb*Mb + Kc*Mc), extract unpadded
    auto ifft_and_add = [&](
        const void* Ka, const void* Kb, const void* Kc,
        double* h_H_comp)
    {
        // Pointwise multiply
        pointwise_mac<<<grid_cx, BLOCK>>>(
            as_cx(d_c_buf_),
            as_cx(const_cast<void*>(Ka)),
            as_cx(const_cast<void*>(Kb)),
            as_cx(const_cast<void*>(Kc)),
            as_cx(d_Mx_f), as_cx(d_My_f), as_cx(d_Mz_f),
            complex_sz);
        CUDA_CHECK(cudaGetLastError());

        // Inverse FFT → r_buf
        CUFFT_CHECK(cufftExecZ2D(plan_inv_,
                                  reinterpret_cast<cufftDoubleComplex*>(d_c_buf_),
                                  reinterpret_cast<cufftDoubleReal*>(d_r_buf_)));

        // Extract unpadded region into device buffer
        CUDA_CHECK(cudaMemset(d_H_unpad, 0, unpad_sz * sizeof(double)));
        extract_and_add<<<grid_ext, block_ext>>>(
            reinterpret_cast<double*>(d_H_unpad),
            reinterpret_cast<const double*>(d_r_buf_),
            static_cast<size_t>(nx_), static_cast<size_t>(ny_),
            static_cast<size_t>(nz_),
            static_cast<size_t>(pad_nx_), static_cast<size_t>(pad_ny_),
            norm);
        CUDA_CHECK(cudaGetLastError());

        // Copy result to host
        CUDA_CHECK(cudaMemcpy(h_H_comp, d_H_unpad,
                              unpad_sz * sizeof(double), cudaMemcpyDeviceToHost));
    };

    // H_x = -(K_xx*Mx + K_xy*My + K_xz*Mz)
    ifft_and_add(d_K_xx_, d_K_xy_, d_K_xz_, h_Hx.data());
    // H_y = -(K_xy*Mx + K_yy*My + K_yz*Mz)
    ifft_and_add(d_K_xy_, d_K_yy_, d_K_yz_, h_Hy.data());
    // H_z = -(K_xz*Mx + K_yz*My + K_zz*Mz)
    ifft_and_add(d_K_xz_, d_K_yz_, d_K_zz_, h_Hz.data());

    // Accumulate into H_out
    for (Index i = 0; i < static_cast<Index>(unpad_sz); ++i) {
        H_out[i].x += h_Hx[i];
        H_out[i].y += h_Hy[i];
        H_out[i].z += h_Hz[i];
    }

    // Cleanup temporary GPU memory
    cudaFree(d_Mx_f); cudaFree(d_My_f); cudaFree(d_Mz_f);
    cudaFree(d_H_unpad);
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
