#pragma once

// DemagFieldGPU: cuFFT-based GPU demag (Phase 3).
// Requires MICROMAG_CUDA=1 (cmake -DMICROMAG_USE_CUDA=ON).
// Same IEffectiveField interface as DemagField — drop-in replacement.

#ifdef MICROMAG_CUDA

#include "effective_field.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

// cufft.h included only in the .cu translation unit to avoid polluting
// C++ headers with CUDA-only types.
using cufftHandle = int;  // forward alias — actual type defined in cufft.h

namespace micromag {

// ---------------------------------------------------------------------------
// GPU demag using cuFFT zero-padded convolution.
// Precomputes the 6 Newell tensor components on the CPU and uploads them
// to the GPU once at construction; runtime cost is 6 forward FFTs +
// 6 pointwise products + 3 inverse FFTs per step, all on the GPU.
// ---------------------------------------------------------------------------
class DemagFieldGPU : public IEffectiveField {
public:
    explicit DemagFieldGPU(const StructuredGrid& grid);
    ~DemagFieldGPU();

    // Non-copyable
    DemagFieldGPU(const DemagFieldGPU&)            = delete;
    DemagFieldGPU& operator=(const DemagFieldGPU&) = delete;

    // IEffectiveField
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    const char* name() const override { return "DemagFieldGPU"; }

private:
    // Grid geometry
    Index nx_, ny_, nz_;
    Real  dx_, dy_, dz_;
    Index pad_nx_, pad_ny_, pad_nz_, fft_nx_;

    // Derived sizes (set in constructor)
    size_t unpad_sz_ = 0;   // nx_ * ny_ * nz_
    size_t real_sz_  = 0;   // pad_nx_ * pad_ny_ * pad_nz_
    size_t cplx_sz_  = 0;   // fft_nx_ * pad_ny_ * pad_nz_

    // GPU device pointers (void* here; cast to proper types in .cu)
    void* d_r_buf_  = nullptr;   // double[real_sz_]            real scratch
    void* d_c_buf_  = nullptr;   // cufftDoubleComplex[cplx_sz_] complex scratch
    void* d_K_xx_   = nullptr;   // cufftDoubleComplex[cplx_sz_] kernel components
    void* d_K_yy_   = nullptr;
    void* d_K_zz_   = nullptr;
    void* d_K_xy_   = nullptr;
    void* d_K_xz_   = nullptr;
    void* d_K_yz_   = nullptr;

    // Step 5: persistent per-step scratch (pre-allocated, no malloc per call)
    void* d_Mx_f_     = nullptr; // cufftDoubleComplex[cplx_sz_] — Mx FFT
    void* d_My_f_     = nullptr; // My FFT
    void* d_Mz_f_     = nullptr; // Mz FFT
    void* d_H_unpad_  = nullptr; // double[unpad_sz_] — unpadded IFFT result

    // Pinned (page-locked) host buffers for fast DMA transfers
    double* h_r_pinned_ = nullptr; // double[real_sz_]  — magnetisation upload
    double* h_H_pinned_ = nullptr; // double[unpad_sz_] — H field download

    // cuFFT plans (int handles; real type is cufftHandle = int)
    cufftHandle plan_fwd_ = 0;   // D2Z  real → complex
    cufftHandle plan_inv_ = 0;   // Z2D  complex → real

    void precompute_kernel();
};

}  // namespace micromag

#endif // MICROMAG_CUDA
