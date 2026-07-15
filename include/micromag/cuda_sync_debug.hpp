#pragma once

// Runtime-selectable post-kernel-launch checking.
//
// Normally a launch is followed by CUDA_CHECK(cudaGetLastError()) — cheap and
// asynchronous, but an async fault or a cross-stream race surfaces far from the
// kernel that caused it. Setting the environment variable
//
//     MICROMAG_SYNC_DEBUG=1
//
// makes MICROMAG_KERNEL_CHECK() also cudaDeviceSynchronize() after every
// checked launch, so the failure (or the racing result) is pinned to the exact
// launch site. This is for debugging only — it serialises the GPU and defeats
// CUDA-graph/stream overlap. Motivation: the FieldSumGPU DMI stream race took
// days to localise without this.
//
// Usage (inside a .cu that defines its own CUDA_CHECK):
//     kernel<<<...>>>(...);
//     MICROMAG_KERNEL_CHECK();
//
// The macro expands at the use site, so it picks up that TU's CUDA_CHECK.

#include <cstdlib>

#include <cuda_runtime.h>

namespace micromag::detail {

inline bool sync_debug_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("MICROMAG_SYNC_DEBUG");
        return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return on;
}

}  // namespace micromag::detail

#define MICROMAG_KERNEL_CHECK()                                              \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        if (::micromag::detail::sync_debug_enabled())                        \
            CUDA_CHECK(cudaDeviceSynchronize());                             \
    } while (0)
