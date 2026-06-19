#pragma once

// IDemagGPU: abstract interface for GPU demag fields.
// Both DemagFieldGPU (open BC) and DemagFieldPeriodicGPU (periodic BC) implement this.
// GPU integrators accept IDemagGPU& so either demag can be used without code changes.

#ifdef MICROMAG_CUDA

#include "material.hpp"
#include "types.hpp"

namespace micromag {

class IDemagGPU {
public:
    virtual ~IDemagGPU() = default;

    // Add H_demag to d_H_out (both [3×N] component-major on GPU).
    // Syncs before returning; caller may safely continue on a different stream.
    virtual void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                                     double* d_H_out) const = 0;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
