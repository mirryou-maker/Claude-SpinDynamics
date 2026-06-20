// zeeman_spatial_gpu.cu ??ZeemanFieldSpatialGPU: per-cell Zeeman on GPU.
//
// Kernel: H_out[c*N+idx] += H_field[c*N+idx]  (direct device-buffer add)
// Memory layout: [3횞N] component-major (same as all other GPU fields).

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/gpu_real.hpp"
#include "micromag/zeeman_spatial_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess)                                              \
            throw std::runtime_error(std::string("CUDA(zsGPU): ")          \
                                   + cudaGetErrorString(_e));               \
    } while (0)

namespace micromag {

// ---------------------------------------------------------------------------
// CUDA kernel ??per-cell add (d_H_field not function of m)
// ---------------------------------------------------------------------------
__global__ static void zeeman_spatial_kernel(
    GReal* __restrict__       H_out,
    const double* __restrict__ H_field,   // spatial field kept in double (shared with CPU fallback)
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    H_out[0*N + idx] += H_field[0*N + idx];
    H_out[1*N + idx] += H_field[1*N + idx];
    H_out[2*N + idx] += H_field[2*N + idx];
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
ZeemanFieldSpatialGPU::ZeemanFieldSpatialGPU(const StructuredGrid& grid)
    : N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz())),
      H_host_(grid)
{
    CUDA_CHECK(cudaMalloc(&d_H_field_, 3 * N_ * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_H_field_, 0, 3 * N_ * sizeof(double)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

ZeemanFieldSpatialGPU::~ZeemanFieldSpatialGPU() {
    cudaFree(d_H_field_);
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ---------------------------------------------------------------------------
// set_field ??host->device upload (interleaved VectorField3D -> component-major)
// ---------------------------------------------------------------------------
void ZeemanFieldSpatialGPU::set_field(const VectorField3D& H_field) {
    // Cache a CPU copy for the fallback accumulate() path.
    for (Index i = 0; i < static_cast<Index>(N_); ++i)
        H_host_[i] = H_field[i];

    // Pack into temporary component-major buffer then H2D copy.
    std::vector<double> buf(3 * N_);
    for (size_t i = 0; i < N_; ++i) {
        const Vec3& v = H_field[static_cast<Index>(i)];
        buf[0*N_ + i] = v.x;
        buf[1*N_ + i] = v.y;
        buf[2*N_ + i] = v.z;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_H_field_, buf.data(),
                               3 * N_ * sizeof(double),
                               cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream_)));
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

// ---------------------------------------------------------------------------
// CPU fallback path (IEffectiveField)
// ---------------------------------------------------------------------------
void ZeemanFieldSpatialGPU::accumulate(const VectorField3D& /*m*/,
                                        const Material& /*mat*/,
                                        VectorField3D& H_out) const {
    for (Index i = 0; i < static_cast<Index>(N_); ++i)
        H_out[i] += H_host_[i];
}

Real ZeemanFieldSpatialGPU::energy(const VectorField3D& m,
                                    const Material& mat) const {
    const double mu0 = 4e-7 * 3.14159265358979323846;
    double E = 0.0;
    const auto& g = m.grid();
    const double dV = g.dx() * g.dy() * g.dz();
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        const Vec3& mi = m[i];
        const Vec3& Hi = H_host_[i];
        E -= mu0 * mat.Ms * (mi.x*Hi.x + mi.y*Hi.y + mi.z*Hi.z) * dV;
    }
    return E;
}

// ---------------------------------------------------------------------------
// Full-GPU path (IEffectiveFieldGPU)
// ---------------------------------------------------------------------------
void ZeemanFieldSpatialGPU::accumulate_gpu_ptr(const GReal* /*d_m*/,
                                                const Material& /*mat*/,
                                                GReal* d_H_out) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    zeeman_spatial_kernel<<<grd, blk, 0, s>>>(
        d_H_out,
        d_H_field_,
        static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace micromag

#endif // MICROMAG_CUDA

