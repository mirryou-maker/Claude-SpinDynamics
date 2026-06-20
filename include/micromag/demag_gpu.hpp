#pragma once

// DemagFieldGPU: cuFFT-based GPU demag (Phase 3).
// Requires MICROMAG_CUDA=1 (cmake -DMICROMAG_USE_CUDA=ON).
// Same IEffectiveField interface as DemagField — drop-in replacement.

#ifdef MICROMAG_CUDA

#include "demag_gpu_iface.hpp"
#include "effective_field.hpp"
#include "field.hpp"
#include "gpu_real.hpp"
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
class DemagFieldGPU : public IEffectiveField, public IDemagGPU {
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
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;
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

    // Step 6a: batch buffers — all 3 components contiguous for batch FFT
    // Layout: [comp0 | comp1 | comp2], each slice = real_sz_ or cplx_sz_
    void* d_M_all_      = nullptr; // double[3 × real_sz_]              padded M upload
    void* d_MF_all_     = nullptr; // cufftDoubleComplex[3 × cplx_sz_]  FFT(M) output
    void* d_HF_all_     = nullptr; // cufftDoubleComplex[3 × cplx_sz_]  kernel product
    void* d_H_all_      = nullptr; // double[3 × real_sz_]              IFFT(HF) output
    void* d_Hunpad_all_ = nullptr; // double[3 × unpad_sz_]             extracted H

    // Step 6b: compact GPU buffer for sparse upload (only unpadded region)
    void*   d_M_compact_ = nullptr;        // double[3 × unpad_sz_]  — GPU scatter src

    // Pinned host buffers (fast DMA transfers) — GReal for P11 float32 compat
    GReal* h_M_compact_pinned_      = nullptr; // GReal[3 × unpad_sz_] — compact upload
    GReal* h_Hunpad_all_pinned_     = nullptr; // GReal[3 × unpad_sz_] — H download

    // cuFFT plans
    cufftHandle plan_fwd_ = 0;        // single D2Z — used by precompute_kernel
    cufftHandle plan_fwd_batch_ = 0;  // batch=3 D2Z — used by accumulate
    cufftHandle plan_inv_batch_ = 0;  // batch=3 Z2D — used by accumulate

    // Step 6c: dedicated CUDA stream — all GPU ops submitted to this stream.
    void* stream_ = nullptr;

    void precompute_kernel();

public:
    // G6: GPU-pointer path (d_m [3×N] component-major → adds H_demag to d_H_out).
    // Runs on internal stream_ and syncs before returning, so caller can safely
    // continue on a different stream after this call.
    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const override;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
