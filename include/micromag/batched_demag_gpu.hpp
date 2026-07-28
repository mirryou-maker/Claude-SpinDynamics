// batched_demag_gpu.hpp — Task 2 Phase 2.2: replica-batched demag (cuFFT batch=R).
//
// The demag kernel tensor is identical across replicas (shared geometry), so it
// is precomputed ONCE; each step runs a batched forward FFT of M over all R
// replicas (batch = 3R), a pointwise tensor contraction that broadcasts the
// shared kernel, and a batched inverse FFT. Physics matches DemagField /
// DemagFieldGPU exactly (Newell tensor, H_demag = −N·M, 1/real_sz norm) — a
// single replica reproduces the reference field. See roadmap Task 2, Phase 2.2.
//
// Standalone/full-complex kernel form (not the production symmetry-compressed
// one): simpler and correct; kernel memory is O(6·cplx_sz), negligible for the
// small grids (10²–10³ cells) that finite-T replica batching targets.
//
// Requires MICROMAG_CUDA=1.
#pragma once

#ifdef MICROMAG_CUDA

#include "micromag/gpu_real.hpp"
#include "micromag/gpu_fft.hpp"   // G0-3 batched FFT seam
#include "micromag/grid.hpp"

namespace micromag {

class BatchedDemagGPU {
public:
    BatchedDemagGPU(const StructuredGrid& grid, int R);
    ~BatchedDemagGPU();

    BatchedDemagGPU(const BatchedDemagGPU&) = delete;
    BatchedDemagGPU& operator=(const BatchedDemagGPU&) = delete;

    // H += H_demag(m) for all R replicas. d_m, d_H are [R·3N] component-major
    // within a replica (m[r·3N + c·N + idx]) — the BatchedLLGGPU layout. Runs on
    // `stream` (the integrator's stream) with no internal sync.
    void accumulate_add(const GReal* d_m, GReal* d_H, double Ms, void* stream);

    // Bind the batched cuFFT plans to an external stream (call once).
    void set_stream(void* stream);

private:
    int    R_, nx_, ny_, nz_;
    long   N_;
    int    pad_nx_, pad_ny_, pad_nz_, fft_nx_;
    size_t real_sz_, cplx_sz_;   // per replica-component
    double norm_;

    void precompute_kernel_(double dx, double dy, double dz);

    // device kernel (shared): 6 real components [cplx_sz_]
    void* d_Kxx_ = nullptr; void* d_Kyy_ = nullptr; void* d_Kzz_ = nullptr;
    void* d_Kxy_ = nullptr; void* d_Kxz_ = nullptr; void* d_Kyz_ = nullptr;
    // batched scratch
    void* d_M_  = nullptr;   // double [R·3·real_sz]  (padded M; reused for inverse output)
    void* d_MF_ = nullptr;   // fft_complex_t [R·3·cplx_sz]

    gpu::GpuFftManyRC* fft_  = nullptr;   // batched fwd(D2Z)+inv(Z2D), batch = 3R
    gpu::GpuFftManyRC* fft1_ = nullptr;   // single-transform fwd, kernel precompute
};

}  // namespace micromag

#endif // MICROMAG_CUDA
