#pragma once

// IDemagGPU: abstract interface for GPU demag fields.
// Both DemagFieldGPU (open BC) and DemagFieldPeriodicGPU (periodic BC) implement this.
// GPU integrators accept IDemagGPU& so either demag can be used without code changes.

#ifdef MICROMAG_CUDA

#include "gpu_real.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

class IDemagGPU {
public:
    virtual ~IDemagGPU() = default;

    // Add H_demag to d_H_out (both [3×N] component-major on GPU).
    // In standalone mode (stream_owned_=true): syncs before returning so caller
    // on a different stream sees d_H_out updates.
    // In shared-stream mode (after set_stream()): no internal sync; caller's
    // stream ordering guarantees correctness.
    virtual void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                                     GReal* d_H_out) const = 0;

    // Redirect GPU work to an external stream (P2: single-stream refactor).
    // Also updates cufftSetStream so FFT batches run on the shared stream.
    // Default no-op; DemagFieldGPU and DemagFieldPeriodicGPU override this.
    virtual void set_stream(void* /*s*/) {}
};

}  // namespace micromag

#endif // MICROMAG_CUDA
