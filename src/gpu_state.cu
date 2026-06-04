// gpu_state.cu — G3: GPU-resident magnetization state
//
// GPUMagState: 5 × [3×N] double buffers on GPU + pinned host staging.
// Provides upload/download (CPU↔GPU) and async GPU bookkeeping ops.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "micromag/gpu_state.hpp"

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess)                                         \
            throw std::runtime_error(std::string("CUDA(state): ")     \
                                   + cudaGetErrorString(_e));          \
    } while (0)

namespace micromag {

// ===========================================================================
// Constructor / Destructor
// ===========================================================================
GPUMagState::GPUMagState(const StructuredGrid& grid)
    : N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz()))
{
    const size_t bytes = 3 * N_ * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_m_,     bytes));
    CUDA_CHECK(cudaMalloc(&d_H_,     bytes));
    CUDA_CHECK(cudaMalloc(&d_m0_,    bytes));
    CUDA_CHECK(cudaMalloc(&d_ki_,    bytes));
    CUDA_CHECK(cudaMalloc(&d_k_acc_, bytes));

    // Pinned host buffer: fast DMA for H2D and D2H transfers
    CUDA_CHECK(cudaMallocHost(&h_staging_, bytes));

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

GPUMagState::~GPUMagState() {
    cudaFree(d_m_);
    cudaFree(d_H_);
    cudaFree(d_m0_);
    cudaFree(d_ki_);
    cudaFree(d_k_acc_);
    cudaFreeHost(h_staging_);
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ===========================================================================
// upload — VectorField3D → d_m_
// ===========================================================================
void GPUMagState::upload(const VectorField3D& m) {
    // Pack into [Mx|My|Mz] component-major layout
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_staging_[i]           = m[i].x;
        h_staging_[N_  + i]     = m[i].y;
        h_staging_[2*N_ + i]    = m[i].z;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_m_, h_staging_,
                               3*N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));   // caller sees completed upload
}

// ===========================================================================
// download — d_m_ → VectorField3D
// ===========================================================================
void GPUMagState::download(VectorField3D& m) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_m_,
                               3*N_*sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        m[i].x = h_staging_[i];
        m[i].y = h_staging_[N_  + i];
        m[i].z = h_staging_[2*N_ + i];
    }
}

// ===========================================================================
// download_H — d_H_ → VectorField3D  (monitoring / testing)
// ===========================================================================
void GPUMagState::download_H(VectorField3D& H) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_H_,
                               3*N_*sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        H[i].x = h_staging_[i];
        H[i].y = h_staging_[N_  + i];
        H[i].z = h_staging_[2*N_ + i];
    }
}

// ===========================================================================
// download_m0 / download_k_acc — snapshot buffers (testing / debugging)
// ===========================================================================
void GPUMagState::download_m0(VectorField3D& m) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_m0_,
                               3*N_*sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        m[i].x = h_staging_[i];
        m[i].y = h_staging_[N_  + i];
        m[i].z = h_staging_[2*N_ + i];
    }
}

void GPUMagState::download_k_acc(VectorField3D& k) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_k_acc_,
                               3*N_*sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        k[i].x = h_staging_[i];
        k[i].y = h_staging_[N_  + i];
        k[i].z = h_staging_[2*N_ + i];
    }
}

// ===========================================================================
// Async GPU bookkeeping — all enqueued on stream_; no CPU barrier
// ===========================================================================

void GPUMagState::zero_H() {
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(double),
                               static_cast<cudaStream_t>(stream_)));
}

void GPUMagState::zero_k_acc() {
    CUDA_CHECK(cudaMemsetAsync(d_k_acc_, 0, 3*N_*sizeof(double),
                               static_cast<cudaStream_t>(stream_)));
}

void GPUMagState::save_m0() {
    CUDA_CHECK(cudaMemcpyAsync(d_m0_, d_m_, 3*N_*sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               static_cast<cudaStream_t>(stream_)));
}

void GPUMagState::sync() const {
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

}  // namespace micromag

#endif // MICROMAG_CUDA
