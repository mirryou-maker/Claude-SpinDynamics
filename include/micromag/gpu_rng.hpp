// gpu_rng.hpp — Task 2 multi-vendor Phase 0 (G0-4).
//
// Device-side counter-based RNG seam. The batched engine draws 3 standard
// normals per (replica,cell) per step from a Philox stream keyed by
// (seed, subsequence, offset). cuRAND and rocRAND expose the same device
// Philox API (curand_* <-> hiprand_*), so one body generates both arms; a SYCL
// backend swaps to oneMKL RNG. Under CUDA this is a 1:1 inline over cuRAND —
// bitwise identical to the pre-seam kernels.
//
// Requires MICROMAG_CUDA=1.
#pragma once

#ifdef MICROMAG_CUDA

#include "micromag/gpu_backend.hpp"   // backend-selection macros

#if defined(MICROMAG_GPU_BACKEND_CUDA)
#  include <curand_kernel.h>
#  define GPU_RNG_STATE   curandStatePhilox4_32_10_t
#  define GPU_RNG_INIT    curand_init
#  define GPU_RNG_NORMAL  curand_normal_double
#elif defined(MICROMAG_GPU_BACKEND_HIP)
#  include <hiprand/hiprand_kernel.h>
#  define GPU_RNG_STATE   hiprandStatePhilox4_32_10_t
#  define GPU_RNG_INIT    hiprand_init
#  define GPU_RNG_NORMAL  hiprand_normal_double
#endif

namespace micromag {
namespace gpu {

// Draw 3 N(0,1) samples from the counter-based stream identified by
// (seed, subsequence, offset). Same realisation for the same arguments →
// reusable across a predictor/corrector pair (Stratonovich).
__device__ inline void philox_normal3(unsigned seed,
                                      unsigned long long subsequence,
                                      unsigned long long offset,
                                      double& e0, double& e1, double& e2) {
#if defined(MICROMAG_GPU_BACKEND_CUDA) || defined(MICROMAG_GPU_BACKEND_HIP)
    GPU_RNG_STATE st;
    GPU_RNG_INIT((unsigned long long)seed, subsequence, offset, &st);
    e0 = GPU_RNG_NORMAL(&st);
    e1 = GPU_RNG_NORMAL(&st);
    e2 = GPU_RNG_NORMAL(&st);
#endif
}

}  // namespace gpu
}  // namespace micromag

#undef GPU_RNG_STATE
#undef GPU_RNG_INIT
#undef GPU_RNG_NORMAL

#endif // MICROMAG_CUDA
