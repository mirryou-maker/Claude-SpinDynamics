// gpu_fft.hpp — Task 2 multi-vendor Phase 0 (G0-3).
//
// Batched real<->complex FFT seam for the demag convolution. cuFFT and hipFFT
// (rocFFT) share a name-for-name API (cufft* <-> hipfft*), so one GpuFftManyRC
// body serves both arms; a SYCL backend swaps to oneMKL DFT (or VkFFT). The
// batched demag (batch=3R) is its first consumer. Under CUDA this is a 1:1
// inline over cuFFT — bitwise identical.
//
// Requires MICROMAG_CUDA=1.
#pragma once

#ifdef MICROMAG_CUDA

#include <stdexcept>
#include <string>

#include "micromag/gpu_backend.hpp"   // stream_t, backend macros

#if defined(MICROMAG_GPU_BACKEND_CUDA)
#  include <cufft.h>
#  define GPU_FFT_FN(name)  cufft##name       // cufft* <-> hipfft* are 1:1
#  define GPU_FFT_PFX(name) CUFFT_##name
#elif defined(MICROMAG_GPU_BACKEND_HIP)
#  include <hipfft/hipfft.h>
#  define GPU_FFT_FN(name)  hipfft##name
#  define GPU_FFT_PFX(name) HIPFFT_##name
#endif

namespace micromag {
namespace gpu {

using fft_complex_t = GPU_FFT_FN(DoubleComplex);
using fft_real_t    = double;

// Paired batched forward (real->complex, D2Z) and inverse (complex->real, Z2D)
// plans over `batch` contiguous transforms. idist/odist are per-transform
// strides (real_sz on the real side, cplx_sz on the complex side).
class GpuFftManyRC {
public:
    // n: dims in FFT order {nz,ny,nx} (len rank). real_sz/cplx_sz per transform.
    GpuFftManyRC(int rank, const int* n, int real_sz, int cplx_sz, int batch) {
        check(GPU_FFT_FN(PlanMany)(&fwd_, rank, const_cast<int*>(n), nullptr, 1, real_sz,
                                   nullptr, 1, cplx_sz, GPU_FFT_PFX(D2Z), batch), "plan fwd");
        check(GPU_FFT_FN(PlanMany)(&inv_, rank, const_cast<int*>(n), nullptr, 1, cplx_sz,
                                   nullptr, 1, real_sz, GPU_FFT_PFX(Z2D), batch), "plan inv");
    }
    ~GpuFftManyRC() {
        if (fwd_) GPU_FFT_FN(Destroy)(fwd_);
        if (inv_) GPU_FFT_FN(Destroy)(inv_);
    }
    GpuFftManyRC(const GpuFftManyRC&) = delete;
    GpuFftManyRC& operator=(const GpuFftManyRC&) = delete;

    void set_stream(stream_t s) {
        check(GPU_FFT_FN(SetStream)(fwd_, s), "setstream fwd");
        check(GPU_FFT_FN(SetStream)(inv_, s), "setstream inv");
    }
    void exec_fwd(fft_real_t* in, fft_complex_t* out) {
        check(GPU_FFT_FN(ExecD2Z)(fwd_, in, out), "exec fwd");
    }
    void exec_inv(fft_complex_t* in, fft_real_t* out) {
        check(GPU_FFT_FN(ExecZ2D)(inv_, in, out), "exec inv");
    }

private:
    GPU_FFT_FN(Handle) fwd_ = 0, inv_ = 0;
    static void check(GPU_FFT_FN(Result) r, const char* what) {
        if (r != GPU_FFT_PFX(SUCCESS))
            throw std::runtime_error(std::string("GPU FFT error [") + what + "] " +
                                     std::to_string((int)r));
    }
};

}  // namespace gpu
}  // namespace micromag

#endif // MICROMAG_CUDA
