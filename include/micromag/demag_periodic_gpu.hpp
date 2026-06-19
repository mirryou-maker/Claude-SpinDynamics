#pragma once

// DemagFieldPeriodicGPU: cuFFT-based GPU periodic-BC demag.
// No zero-padding (8x smaller FFT than open-BC DemagFieldGPU).
// Kernel precomputed on CPU via periodic Newell image sum, then uploaded once.
// Same IEffectiveField interface — drop-in replacement for DemagFieldPeriodic.

#ifdef MICROMAG_CUDA

#include "demag_gpu_iface.hpp"
#include "effective_field.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

using cufftHandle = int;

namespace micromag {

class DemagFieldPeriodicGPU : public IEffectiveField, public IDemagGPU {
public:
    // n_rep: number of image-cell repetitions per dimension in Newell sum
    // (default 2 → 5^3 = 125 images, same as CPU DemagFieldPeriodic).
    explicit DemagFieldPeriodicGPU(const StructuredGrid& grid, int n_rep = 2);
    ~DemagFieldPeriodicGPU();

    DemagFieldPeriodicGPU(const DemagFieldPeriodicGPU&)            = delete;
    DemagFieldPeriodicGPU& operator=(const DemagFieldPeriodicGPU&) = delete;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;
    const char* name() const override { return "DemagFieldPeriodicGPU"; }

    // G-path: add H_demag to d_H_out [3×N component-major, on-GPU].
    // d_m must already be on GPU; syncs before returning.
    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;

private:
    Index nx_, ny_, nz_;
    Real  dx_, dy_, dz_;
    Index fft_nx_;        // nx_/2 + 1

    size_t real_sz_;      // nx_ * ny_ * nz_
    size_t cplx_sz_;      // fft_nx_ * ny_ * nz_

    int n_rep_;

    // GPU buffers — component-major, 3 slices each
    void* d_M_all_  = nullptr;  // double[3 × real_sz_]                M upload
    void* d_MF_all_ = nullptr;  // cufftDoubleComplex[3 × cplx_sz_]   FFT(M)
    void* d_HF_all_ = nullptr;  // cufftDoubleComplex[3 × cplx_sz_]   kernel product
    void* d_H_all_  = nullptr;  // double[3 × real_sz_]                IFFT(HF) = H

    // GPU kernel (frequency-domain, each 6 components)
    void* d_K_xx_ = nullptr;  // cufftDoubleComplex[cplx_sz_]
    void* d_K_yy_ = nullptr;
    void* d_K_zz_ = nullptr;
    void* d_K_xy_ = nullptr;
    void* d_K_xz_ = nullptr;
    void* d_K_yz_ = nullptr;

    // Pinned host staging
    double* h_M_pinned_ = nullptr;   // double[3 × real_sz_]
    double* h_H_pinned_ = nullptr;   // double[3 × real_sz_]

    // cuFFT plans
    cufftHandle plan_fwd_single_  = 0;  // single D2Z (kernel upload + FFT)
    cufftHandle plan_fwd_batch_   = 0;  // batch=3 D2Z
    cufftHandle plan_inv_batch_   = 0;  // batch=3 Z2D

    // CUDA stream
    void* stream_ = nullptr;

    void precompute_kernel();
};

}  // namespace micromag

#endif // MICROMAG_CUDA
