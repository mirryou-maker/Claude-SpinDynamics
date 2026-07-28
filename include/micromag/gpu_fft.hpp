// gpu_fft.hpp — Task 2 multi-vendor Phase 0 (G0-3).
//
// Batched real<->complex FFT seam for the demag convolution. The replica-
// batched demag needs a batched forward (D2Z) and inverse (Z2D) transform of
// 3R fields; wrapping cufftPlanMany here lets a HIP backend swap to hipFFT/
// rocFFT and a SYCL backend to oneMKL DFT (or VkFFT) by replacing ONE class.
// Under CUDA this is a 1:1 inline over cuFFT — bitwise identical.
//
// Requires MICROMAG_CUDA=1.
#pragma once

#ifdef MICROMAG_CUDA

#include <stdexcept>
#include <string>

#include "micromag/gpu_backend.hpp"   // stream_t, backend macros

#if defined(MICROMAG_GPU_BACKEND_CUDA)
#  include <cufft.h>
#endif

namespace micromag {
namespace gpu {

#if defined(MICROMAG_GPU_BACKEND_CUDA)
using fft_complex_t = cufftDoubleComplex;
using fft_real_t    = double;
#endif

// Paired batched forward (real->complex) and inverse (complex->real) plans over
// `batch` contiguous transforms of the given rank/dims. idist/odist are the
// per-transform strides (real_sz for real side, cplx_sz for complex side).
class GpuFftManyRC {
public:
    // n: dims in FFT order {nz,ny,nx} (len rank). real_sz/cplx_sz per transform.
    GpuFftManyRC(int rank, const int* n, int real_sz, int cplx_sz, int batch)
        : real_sz_(real_sz), cplx_sz_(cplx_sz)
    {
#if defined(MICROMAG_GPU_BACKEND_CUDA)
        check(cufftPlanMany(&fwd_, rank, const_cast<int*>(n), nullptr, 1, real_sz,
                            nullptr, 1, cplx_sz, CUFFT_D2Z, batch), "plan fwd");
        check(cufftPlanMany(&inv_, rank, const_cast<int*>(n), nullptr, 1, cplx_sz,
                            nullptr, 1, real_sz, CUFFT_Z2D, batch), "plan inv");
#endif
    }
    ~GpuFftManyRC() {
#if defined(MICROMAG_GPU_BACKEND_CUDA)
        if (fwd_) cufftDestroy(fwd_);
        if (inv_) cufftDestroy(inv_);
#endif
    }
    GpuFftManyRC(const GpuFftManyRC&) = delete;
    GpuFftManyRC& operator=(const GpuFftManyRC&) = delete;

    void set_stream(stream_t s) {
#if defined(MICROMAG_GPU_BACKEND_CUDA)
        check(cufftSetStream(fwd_, s), "setstream fwd");
        check(cufftSetStream(inv_, s), "setstream inv");
#endif
    }
    void exec_fwd(fft_real_t* in, fft_complex_t* out) {
#if defined(MICROMAG_GPU_BACKEND_CUDA)
        check(cufftExecD2Z(fwd_, in, out), "exec fwd");
#endif
    }
    void exec_inv(fft_complex_t* in, fft_real_t* out) {
#if defined(MICROMAG_GPU_BACKEND_CUDA)
        check(cufftExecZ2D(inv_, in, out), "exec inv");
#endif
    }

private:
    int real_sz_, cplx_sz_;
#if defined(MICROMAG_GPU_BACKEND_CUDA)
    cufftHandle fwd_ = 0, inv_ = 0;
    static void check(cufftResult r, const char* what) {
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error(std::string("GPU FFT error [") + what + "] " +
                                     std::to_string((int)r));
    }
#endif
};

}  // namespace gpu
}  // namespace micromag

#endif // MICROMAG_CUDA
