// gpu_backend.hpp — Task 2 multi-vendor Phase 0 (G0-1/G0-2/G0-5).
//
// Thin compile-time seam over the GPU runtime + kernel launch so a later HIP or
// SYCL backend replaces ONE header instead of touching every .cu. Under CUDA the
// wrappers are inline 1:1 maps (zero overhead) — the batched engine's results
// stay bitwise-identical. Mirrors the GReal type-seam pattern (gpu_real.hpp).
//
// Backend selection (G0-5): MICROMAG_GPU_BACKEND_{CUDA,HIP,SYCL} is set by CMake
// from -DMICROMAG_GPU_BACKEND=... (default cuda). Only CUDA is implemented now;
// the HIP/SYCL arms are declared so the extension points are visible.
//
// Requires MICROMAG_CUDA=1 (the header is only pulled into GPU translation units).
#pragma once

#ifdef MICROMAG_CUDA

// default the backend macro to CUDA when CMake did not set one
#if !defined(MICROMAG_GPU_BACKEND_CUDA) && !defined(MICROMAG_GPU_BACKEND_HIP) \
    && !defined(MICROMAG_GPU_BACKEND_SYCL)
#  define MICROMAG_GPU_BACKEND_CUDA 1
#endif

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(MICROMAG_GPU_BACKEND_CUDA)
#  include <cuda_runtime.h>
#else
#  error "gpu_backend.hpp: only the CUDA backend is implemented (Phase 0)."
#endif

namespace micromag {
namespace gpu {

// ---- backend-specific opaque types --------------------------------------
#if defined(MICROMAG_GPU_BACKEND_CUDA)
using stream_t = cudaStream_t;
enum class MemcpyKind { H2D, D2H, D2D };
namespace detail {
inline cudaMemcpyKind to_native(MemcpyKind k) {
    switch (k) {
        case MemcpyKind::H2D: return cudaMemcpyHostToDevice;
        case MemcpyKind::D2H: return cudaMemcpyDeviceToHost;
        default:              return cudaMemcpyDeviceToDevice;
    }
}
inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("GPU error [") + what + "]: " +
                                 cudaGetErrorString(e));
}
}  // namespace detail
#endif

// ---- G0-1 runtime wrappers (throw std::runtime_error on failure) ---------
inline void* malloc(std::size_t bytes) {
    void* p = nullptr;
    detail::check(cudaMalloc(&p, bytes), "malloc");
    return p;
}
inline void free(void* p) { if (p) cudaFree(p); }

inline void memcpy(void* dst, const void* src, std::size_t bytes, MemcpyKind k) {
    detail::check(cudaMemcpy(dst, src, bytes, detail::to_native(k)), "memcpy");
}
inline void memcpy_async(void* dst, const void* src, std::size_t bytes,
                         MemcpyKind k, stream_t s) {
    detail::check(cudaMemcpyAsync(dst, src, bytes, detail::to_native(k), s),
                  "memcpy_async");
}
inline void memset(void* p, int v, std::size_t bytes) {
    detail::check(cudaMemset(p, v, bytes), "memset");
}
inline void memset_async(void* p, int v, std::size_t bytes, stream_t s) {
    detail::check(cudaMemsetAsync(p, v, bytes, s), "memset_async");
}

inline stream_t stream_create() {
    stream_t s; detail::check(cudaStreamCreate(&s), "stream_create"); return s;
}
inline void stream_destroy(stream_t s) { if (s) cudaStreamDestroy(s); }
inline void stream_sync(stream_t s) {
    detail::check(cudaStreamSynchronize(s), "stream_sync");
}
inline void device_sync() { detail::check(cudaDeviceSynchronize(), "device_sync"); }

// post-launch error check (cudaGetLastError); call after GPU_LAUNCH.
inline void check_last(const char* ctx) {
    detail::check(cudaGetLastError(), ctx);
}

}  // namespace gpu
}  // namespace micromag

// ---- G0-2 kernel-launch macro -------------------------------------------
// Under CUDA expands to <<<...>>>. A HIP backend maps this to
// hipLaunchKernelGGL; a SYCL backend replaces the kernel bodies (Phase 2).
#if defined(MICROMAG_GPU_BACKEND_CUDA)
#  define GPU_LAUNCH(kernel, grid, block, shmem, stream, ...) \
      kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__)
#endif

#endif // MICROMAG_CUDA
