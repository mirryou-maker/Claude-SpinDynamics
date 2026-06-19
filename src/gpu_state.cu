// gpu_state.cu — G3: GPU-resident magnetization state
//
// GPUMagState: 5 × [3×N] double buffers on GPU + pinned host staging.
// Provides upload/download (CPU↔GPU) and async GPU bookkeeping ops.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

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
// max_angle_gpu kernels — two-stage reduction without external libraries.
//
// Stage 1 (min_dot_neighbors_kernel): per-cell min dot product with 6 neighbors.
// Stage 2 (block_min_kernel):         reduce stage-1 output to one value/block.
// Stage 3 (CPU):                      find global min across block results.
//
// Only n_blocks doubles (≈N/256) are transferred D2H instead of 3N doubles.
// ===========================================================================
namespace {

#define ANGLE_BLK 256

__global__ static void min_dot_neighbors_kernel(
    const double* __restrict__ m,
    double* __restrict__ out,
    int nx, int ny, int nz)
{
    const int N   = nx * ny * nz;
    const int idx = static_cast<int>(blockIdx.x) * ANGLE_BLK
                  + static_cast<int>(threadIdx.x);
    if (idx >= N) return;

    const int ix  = idx % nx;
    const int iy  = (idx / nx) % ny;
    const int iz  = idx / (nx * ny);

    const double mx = m[idx],
                 my = m[N   + idx],
                 mz = m[2*N + idx];

    double min_d = 1.0;

#define CHK_DOT(j) do { \
    const double _d = mx*m[j] + my*m[N+(j)] + mz*m[2*N+(j)]; \
    if (_d < min_d) min_d = _d; \
} while (0)

    if (ix > 0)      CHK_DOT(idx - 1);
    if (ix < nx - 1) CHK_DOT(idx + 1);
    if (iy > 0)      CHK_DOT(idx - nx);
    if (iy < ny - 1) CHK_DOT(idx + nx);
    if (iz > 0)      CHK_DOT(idx - nx * ny);
    if (iz < nz - 1) CHK_DOT(idx + nx * ny);

#undef CHK_DOT

    out[idx] = min_d;
}

// Block-level minimum reduction; one result per block written to block_out.
__global__ static void block_min_kernel(
    const double* __restrict__ in,
    double* __restrict__ block_out,
    int N)
{
    __shared__ double smem[ANGLE_BLK];
    const int tid = static_cast<int>(threadIdx.x);
    const int idx = static_cast<int>(blockIdx.x) * ANGLE_BLK + tid;

    smem[tid] = (idx < N) ? in[idx] : 1.0;
    __syncthreads();

    for (int s = ANGLE_BLK / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = min(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        block_out[blockIdx.x] = smem[0];
}

#undef ANGLE_BLK

} // anonymous namespace

// ===========================================================================
// Constructor / Destructor
// ===========================================================================
GPUMagState::GPUMagState(const StructuredGrid& grid)
    : N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz())),
      nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz())
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
    // max_angle_gpu scratch (lazily allocated, may be nullptr)
    if (d_angle_buf_) cudaFree(d_angle_buf_);
    if (d_block_min_) cudaFree(d_block_min_);
    if (h_block_min_) cudaFreeHost(h_block_min_);
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

void GPUMagState::download_ki(VectorField3D& k) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_ki_,
                               3*N_*sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        k[i].x = h_staging_[i];
        k[i].y = h_staging_[N_  + i];
        k[i].z = h_staging_[2*N_ + i];
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
// upload_H — VectorField3D → d_H_ (same stream; used in G4/G5 tests)
// ===========================================================================
void GPUMagState::upload_H(const VectorField3D& H) {
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_staging_[i]           = H[i].x;
        h_staging_[N_  + i]     = H[i].y;
        h_staging_[2*N_ + i]    = H[i].z;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_H_, h_staging_,
                               3*N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
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

// ===========================================================================
// max_angle_gpu — returns max misalignment angle between adjacent spins (°).
//
// Two-stage GPU reduction: per-cell → block minimums → CPU final reduce.
// Transfers only n_blocks doubles D2H (≈N/256) instead of the full 3N field.
// Lazily allocates scratch buffers on the first call.
// ===========================================================================
double GPUMagState::max_angle_gpu() const {
    const cudaStream_t s  = static_cast<cudaStream_t>(stream_);
    const int          Ni = static_cast<int>(N_);
    constexpr int      BLK = 256;
    const int n_blocks = (Ni + BLK - 1) / BLK;

    // Lazy init: allocate scratch buffers sized for this grid.
    if (!d_angle_buf_) {
        n_angle_blocks_ = static_cast<size_t>(n_blocks);
        CUDA_CHECK(cudaMalloc(&d_angle_buf_, N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_min_, n_angle_blocks_ * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_block_min_, n_angle_blocks_ * sizeof(double)));
    }

    // Stage 1: per-cell min dot product with all 6 neighbors → d_angle_buf_.
    min_dot_neighbors_kernel<<<n_blocks, BLK, 0, s>>>(
        reinterpret_cast<const double*>(d_m_),
        static_cast<double*>(d_angle_buf_),
        nx_, ny_, nz_);

    // Stage 2: block-level reduction → d_block_min_ (one double per block).
    block_min_kernel<<<n_blocks, BLK, 0, s>>>(
        static_cast<const double*>(d_angle_buf_),
        static_cast<double*>(d_block_min_),
        Ni);

    // Stage 3: D2H the n_blocks doubles and finish on CPU.
    CUDA_CHECK(cudaMemcpyAsync(h_block_min_,
                               d_block_min_, n_angle_blocks_ * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    double min_dot = 1.0;
    for (size_t i = 0; i < n_angle_blocks_; ++i)
        if (h_block_min_[i] < min_dot) min_dot = h_block_min_[i];

    min_dot = std::max(-1.0, std::min(1.0, min_dot));
    constexpr double kPi = 3.14159265358979323846;
    return std::acos(min_dot) * (180.0 / kPi);
}

}  // namespace micromag

#endif // MICROMAG_CUDA
