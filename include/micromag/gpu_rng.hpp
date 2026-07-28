// gpu_rng.hpp — Task 2 multi-vendor Phase 0 (G0-4).
//
// Device-side counter-based RNG seam. The batched engine draws 3 standard
// normals per (replica,cell) per step from a Philox stream keyed by
// (seed, subsequence, offset); wrapping it here lets a HIP backend swap to
// rocRAND and a SYCL backend to oneMKL RNG without touching kernel bodies.
// Under CUDA this is a 1:1 inline over cuRAND's device Philox — bitwise
// identical to the pre-seam kernels.
//
// Requires MICROMAG_CUDA=1.
#pragma once

#ifdef MICROMAG_CUDA

#include "micromag/gpu_backend.hpp"   // backend-selection macros

#if defined(MICROMAG_GPU_BACKEND_CUDA)
#  include <curand_kernel.h>
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
#if defined(MICROMAG_GPU_BACKEND_CUDA)
    curandStatePhilox4_32_10_t st;
    curand_init((unsigned long long)seed, subsequence, offset, &st);
    e0 = curand_normal_double(&st);
    e1 = curand_normal_double(&st);
    e2 = curand_normal_double(&st);
#endif
}

}  // namespace gpu
}  // namespace micromag

#endif // MICROMAG_CUDA
