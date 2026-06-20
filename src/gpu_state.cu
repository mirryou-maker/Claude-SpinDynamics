// gpu_state.cu — G3: GPU-resident magnetization state
//
// GPUMagState: 5 × [3×N] double buffers on GPU + pinned host staging.
// Provides upload/download (CPU↔GPU) and async GPU bookkeeping ops.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>   // P12: CUB one-call GPU reduction (needs /Zc:preprocessor)
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
// max_angle_gpu kernels — P12: CUB-accelerated min-dot reduction.
//
// Stage 1 (min_dot_neighbors_kernel): per-cell min dot product with 6 neighbors.
// Stage 2 (CUB DeviceReduce::Min):    one-call global min over stage-1 output.
//   Only 1 double transferred D2H vs n_blocks in the old manual 2-stage approach.
// ===========================================================================
namespace {

#define ANGLE_BLK 256

__global__ static void min_dot_neighbors_kernel(
    const GReal* __restrict__ m,   // P11: GReal (float or double per build flag)
    double* __restrict__ out,      // always double for reduction precision
    int nx, int ny, int nz)
{
    const int N   = nx * ny * nz;
    const int idx = static_cast<int>(blockIdx.x) * ANGLE_BLK
                  + static_cast<int>(threadIdx.x);
    if (idx >= N) return;

    const int ix  = idx % nx;
    const int iy  = (idx / nx) % ny;
    const int iz  = idx / (nx * ny);

    const double mx = static_cast<double>(m[idx]),
                 my = static_cast<double>(m[N   + idx]),
                 mz = static_cast<double>(m[2*N + idx]);

    double min_d = 1.0;

#define CHK_DOT(j) do { \
    const double _d = mx*static_cast<double>(m[j]) \
                    + my*static_cast<double>(m[N+(j)]) \
                    + mz*static_cast<double>(m[2*N+(j)]); \
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
    // P11: buffer size uses GReal (float when MICROMAG_FLOAT32=ON, else double).
    const size_t bytes = 3 * N_ * sizeof(GReal);

    CUDA_CHECK(cudaMalloc(&d_m_,     bytes));
    CUDA_CHECK(cudaMalloc(&d_H_,     bytes));
    CUDA_CHECK(cudaMalloc(&d_m0_,    bytes));
    CUDA_CHECK(cudaMalloc(&d_ki_,    bytes));
    CUDA_CHECK(cudaMalloc(&d_k_acc_, bytes));

    // Pinned host buffer: fast DMA for H2D and D2H transfers.
    // Same element type as GPU buffers; upload/download converts CPU double↔GReal.
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
    if (d_angle_buf_)    cudaFree(d_angle_buf_);
    if (d_angle_result_) cudaFree(d_angle_result_);
    if (d_cub_tmp_)      cudaFree(d_cub_tmp_);
}

// ===========================================================================
// upload — VectorField3D → d_m_
// ===========================================================================
void GPUMagState::upload(const VectorField3D& m) {
    // Pack into [Mx|My|Mz] component-major layout.
    // P11: static_cast<GReal> is a no-op in double mode; converts to float in float32 mode.
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_staging_[i]        = static_cast<GReal>(m[i].x);
        h_staging_[N_  + i]  = static_cast<GReal>(m[i].y);
        h_staging_[2*N_ + i] = static_cast<GReal>(m[i].z);
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_m_, h_staging_,
                               3*N_*sizeof(GReal), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));   // caller sees completed upload
}

// ===========================================================================
// download — d_m_ → VectorField3D
// ===========================================================================
void GPUMagState::download(VectorField3D& m) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_m_,
                               3*N_*sizeof(GReal), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    // P11: cast GReal→double (no-op in double mode; promotes float→double in float32 mode).
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        m[i].x = static_cast<double>(h_staging_[i]);
        m[i].y = static_cast<double>(h_staging_[N_  + i]);
        m[i].z = static_cast<double>(h_staging_[2*N_ + i]);
    }
}

// ===========================================================================
// download_H — d_H_ → VectorField3D  (monitoring / testing)
// ===========================================================================
void GPUMagState::download_H(VectorField3D& H) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_H_,
                               3*N_*sizeof(GReal), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        H[i].x = static_cast<double>(h_staging_[i]);
        H[i].y = static_cast<double>(h_staging_[N_  + i]);
        H[i].z = static_cast<double>(h_staging_[2*N_ + i]);
    }
}

// ===========================================================================
// download_m0 / download_k_acc — snapshot buffers (testing / debugging)
// ===========================================================================
void GPUMagState::download_m0(VectorField3D& m) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_m0_,
                               3*N_*sizeof(GReal), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        m[i].x = static_cast<double>(h_staging_[i]);
        m[i].y = static_cast<double>(h_staging_[N_  + i]);
        m[i].z = static_cast<double>(h_staging_[2*N_ + i]);
    }
}

void GPUMagState::download_ki(VectorField3D& k) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_ki_,
                               3*N_*sizeof(GReal), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        k[i].x = static_cast<double>(h_staging_[i]);
        k[i].y = static_cast<double>(h_staging_[N_  + i]);
        k[i].z = static_cast<double>(h_staging_[2*N_ + i]);
    }
}

void GPUMagState::download_k_acc(VectorField3D& k) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_staging_, d_k_acc_,
                               3*N_*sizeof(GReal), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        k[i].x = static_cast<double>(h_staging_[i]);
        k[i].y = static_cast<double>(h_staging_[N_  + i]);
        k[i].z = static_cast<double>(h_staging_[2*N_ + i]);
    }
}

// ===========================================================================
// upload_H — VectorField3D → d_H_ (same stream; used in G4/G5 tests)
// ===========================================================================
void GPUMagState::upload_H(const VectorField3D& H) {
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_staging_[i]        = static_cast<GReal>(H[i].x);
        h_staging_[N_  + i]  = static_cast<GReal>(H[i].y);
        h_staging_[2*N_ + i] = static_cast<GReal>(H[i].z);
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_H_, h_staging_,
                               3*N_*sizeof(GReal), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ===========================================================================
// Async GPU bookkeeping — all enqueued on stream_; no CPU barrier
// ===========================================================================

void GPUMagState::zero_H() {
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal),
                               static_cast<cudaStream_t>(stream_)));
}

void GPUMagState::zero_k_acc() {
    CUDA_CHECK(cudaMemsetAsync(d_k_acc_, 0, 3*N_*sizeof(GReal),
                               static_cast<cudaStream_t>(stream_)));
}

void GPUMagState::save_m0() {
    CUDA_CHECK(cudaMemcpyAsync(d_m0_, d_m_, 3*N_*sizeof(GReal),
                               cudaMemcpyDeviceToDevice,
                               static_cast<cudaStream_t>(stream_)));
}

void GPUMagState::sync() const {
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

// ===========================================================================
// max_angle_gpu — returns max misalignment angle between adjacent spins (°).
//
// P12 CUB-accelerated implementation:
//   Stage 1: per-cell min-dot kernel → d_angle_buf_  (double[N], entirely GPU)
//   Stage 2: cub::DeviceReduce::Min → d_angle_result_ (one double, D2H 8 bytes)
// Lazily allocates scratch buffers on the first call.
// ===========================================================================
double GPUMagState::max_angle_gpu() const {
    const cudaStream_t s  = static_cast<cudaStream_t>(stream_);
    const int          Ni = static_cast<int>(N_);
    constexpr int      BLK = 256;
    const int n_blocks = (Ni + BLK - 1) / BLK;

    // Lazy init: allocate per-cell scratch + CUB temp storage on first call.
    if (!d_angle_buf_) {
        CUDA_CHECK(cudaMalloc(&d_angle_buf_,    N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_angle_result_,      sizeof(double)));

        // Query CUB for required temp storage size (null output ptr → size query).
        cub::DeviceReduce::Min(nullptr, n_cub_tmp_bytes_,
                               static_cast<const double*>(d_angle_buf_),
                               static_cast<double*>(d_angle_result_),
                               Ni, s);
        CUDA_CHECK(cudaMalloc(&d_cub_tmp_, n_cub_tmp_bytes_));
    }

    // Stage 1: per-cell min dot product with all 6 neighbors → d_angle_buf_.
    // Kernel accepts GReal* (m values) and writes double (min dot) to d_angle_buf_.
    min_dot_neighbors_kernel<<<n_blocks, BLK, 0, s>>>(
        reinterpret_cast<const GReal*>(d_m_),
        static_cast<double*>(d_angle_buf_),
        nx_, ny_, nz_);

    // Stage 2: CUB global minimum → d_angle_result_ (single double on device).
    cub::DeviceReduce::Min(d_cub_tmp_, n_cub_tmp_bytes_,
                           static_cast<const double*>(d_angle_buf_),
                           static_cast<double*>(d_angle_result_),
                           Ni, s);

    // Stage 3: D2H 1 double (was n_blocks doubles in the manual version).
    double min_dot = 1.0;
    CUDA_CHECK(cudaMemcpyAsync(&min_dot, d_angle_result_, sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    min_dot = std::max(-1.0, std::min(1.0, min_dot));
    constexpr double kPi = 3.14159265358979323846;
    return std::acos(min_dot) * (180.0 / kPi);
}

}  // namespace micromag

#endif // MICROMAG_CUDA
