// gpu_backend.hpp — Task 2 multi-vendor Phase 0 (G0-1/G0-2/G0-5).
//
// Thin compile-time seam over the GPU runtime + kernel launch so a HIP or SYCL
// backend replaces ONE header instead of touching every .cu. The CUDA and HIP
// runtimes are name-for-name 1:1 (cuda*<->hip*), so both are generated from a
// single body via the GPU_FN(prefix) macro — the HIP arm is correct by
// construction (compiles under ROCm; the CUDA arm is what this repo builds/tests
// on the available NVIDIA hardware). Mirrors the GReal type-seam (gpu_real.hpp).
//
// Backend selection (G0-5): MICROMAG_GPU_BACKEND_{CUDA,HIP,SYCL} is set by CMake
// from -DMICROMAG_GPU_BACKEND=... (default cuda). SYCL is a reserved arm.
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
#  define GPU_FN(name) cuda##name          // cuda* <-> hip* are 1:1
#elif defined(MICROMAG_GPU_BACKEND_HIP)
#  include <hip/hip_runtime.h>
#  define GPU_FN(name) hip##name
#else
#  error "gpu_backend.hpp: SYCL backend not implemented (Phase 0 = cuda|hip)."
#endif

namespace micromag {
namespace gpu {

// ---- backend-neutral runtime aliases ------------------------------------
using error_t  = GPU_FN(Error_t);
using stream_t = GPU_FN(Stream_t);
static constexpr error_t success = GPU_FN(Success);

enum class MemcpyKind { H2D, D2H, D2D };
namespace detail {
inline int to_native(MemcpyKind k) {
    switch (k) {
        case MemcpyKind::H2D: return GPU_FN(MemcpyHostToDevice);
        case MemcpyKind::D2H: return GPU_FN(MemcpyDeviceToHost);
        default:              return GPU_FN(MemcpyDeviceToDevice);
    }
}
inline void check(error_t e, const char* what) {
    if (e != success)
        throw std::runtime_error(std::string("GPU error [") + what + "]: " +
                                 GPU_FN(GetErrorString)(e));
}
}  // namespace detail

// ---- G0-1 runtime wrappers (throw std::runtime_error on failure) ---------
inline void* malloc(std::size_t bytes) {
    void* p = nullptr;
    detail::check(GPU_FN(Malloc)(&p, bytes), "malloc");
    return p;
}
inline void free(void* p) { if (p) GPU_FN(Free)(p); }

inline void memcpy(void* dst, const void* src, std::size_t bytes, MemcpyKind k) {
    detail::check(GPU_FN(Memcpy)(dst, src, bytes,
                  (GPU_FN(MemcpyKind))detail::to_native(k)), "memcpy");
}
inline void memcpy_async(void* dst, const void* src, std::size_t bytes,
                         MemcpyKind k, stream_t s) {
    detail::check(GPU_FN(MemcpyAsync)(dst, src, bytes,
                  (GPU_FN(MemcpyKind))detail::to_native(k), s), "memcpy_async");
}
inline void memset(void* p, int v, std::size_t bytes) {
    detail::check(GPU_FN(Memset)(p, v, bytes), "memset");
}
inline void memset_async(void* p, int v, std::size_t bytes, stream_t s) {
    detail::check(GPU_FN(MemsetAsync)(p, v, bytes, s), "memset_async");
}

inline stream_t stream_create() {
    stream_t s; detail::check(GPU_FN(StreamCreate)(&s), "stream_create"); return s;
}
inline void stream_destroy(stream_t s) { if (s) GPU_FN(StreamDestroy)(s); }
inline void stream_sync(stream_t s) {
    detail::check(GPU_FN(StreamSynchronize)(s), "stream_sync");
}
inline void device_sync() { detail::check(GPU_FN(DeviceSynchronize)(), "device_sync"); }

// post-launch error check; call after GPU_LAUNCH.
inline void check_last(const char* ctx) { detail::check(GPU_FN(GetLastError)(), ctx); }

}  // namespace gpu
}  // namespace micromag

// ---- G0-2 kernel-launch macro -------------------------------------------
// Both nvcc and hipcc accept the triple-chevron launch syntax, so one macro
// serves CUDA and HIP; a SYCL backend replaces the kernel bodies (Phase 2).
#if defined(MICROMAG_GPU_BACKEND_CUDA) || defined(MICROMAG_GPU_BACKEND_HIP)
#  define GPU_LAUNCH(kernel, grid, block, shmem, stream, ...) \
      kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__)
#endif

#endif // MICROMAG_CUDA
