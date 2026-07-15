// rkky_gpu.cu ??RKKYFieldGPU: interlayer RKKY coupling on GPU.
//
// Kernel: H_out[idx] += factor * d_ref[idx]
// factor = -J / (mu_0 * Ms * d_spacer)
// Memory layout: [3횞N] component-major (same as all other GPU fields).

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include "micromag/cuda_sync_debug.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/gpu_real.hpp"
#include "micromag/rkky_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess)                                            \
            throw std::runtime_error(std::string("CUDA(rkky): ")         \
                                   + cudaGetErrorString(_e));             \
    } while (0)

namespace micromag {

// ---------------------------------------------------------------------------
// Kernel: H_out += factor * m_ref
// ---------------------------------------------------------------------------
__global__ static void rkky_kernel(
    GReal* __restrict__       H_out,
    const double* __restrict__ d_ref,   // ref layer kept in double (shared with CPU fallback D2H)
    double factor,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    H_out[0*N + idx] += factor * d_ref[0*N + idx];
    H_out[1*N + idx] += factor * d_ref[1*N + idx];
    H_out[2*N + idx] += factor * d_ref[2*N + idx];
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
RKKYFieldGPU::RKKYFieldGPU(const StructuredGrid& grid, Real J_RKKY, Real d_spacer)
    : N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz())),
      J_RKKY_(J_RKKY), d_spacer_(d_spacer)
{
    CUDA_CHECK(cudaMalloc(&d_ref_, 3 * N_ * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_ref_, 0, 3 * N_ * sizeof(double)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

RKKYFieldGPU::~RKKYFieldGPU() {
    cudaFree(d_ref_);
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ---------------------------------------------------------------------------
// set_ref ??upload reference layer CPU -> device (interleaved -> component-major)
// ---------------------------------------------------------------------------
void RKKYFieldGPU::set_ref(const VectorField3D& ref_m) {
    std::vector<double> buf(3 * N_);
    for (size_t i = 0; i < N_; ++i) {
        const Vec3& v = ref_m[static_cast<Index>(i)];
        buf[0*N_ + i] = v.x;
        buf[1*N_ + i] = v.y;
        buf[2*N_ + i] = v.z;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_ref_, buf.data(), 3*N_*sizeof(double),
                               cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream_)));
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

// ---------------------------------------------------------------------------
// CPU fallback (IEffectiveField)
// ---------------------------------------------------------------------------
void RKKYFieldGPU::accumulate(const VectorField3D& /*m*/,
                                const Material& mat,
                                VectorField3D& H_out) const {
    if (d_spacer_ == 0.0 || mat.Ms == 0.0) return;
    const double mu0 = 4e-7 * 3.14159265358979323846;
    // coeff = +J/(mu0*Ms*d)  (same sign as CPU rkky.cpp)
    const double factor = J_RKKY_ / (mu0 * mat.Ms * d_spacer_);

    // Download ref from device
    std::vector<double> h_ref(3 * N_);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref_, 3*N_*sizeof(double),
                          cudaMemcpyDeviceToHost));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        H_out[i].x += factor * h_ref[0*N_ + i];
        H_out[i].y += factor * h_ref[1*N_ + i];
        H_out[i].z += factor * h_ref[2*N_ + i];
    }
}

Real RKKYFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    if (d_spacer_ == 0.0 || mat.Ms == 0.0) return 0.0;
    const double mu0 = 4e-7 * 3.14159265358979323846;
    const auto& g = m.grid();
    const double dV = g.dx() * g.dy() * g.dz();

    std::vector<double> h_ref(3 * N_);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref_, 3*N_*sizeof(double),
                          cudaMemcpyDeviceToHost));
    // E = -mu_0 * Ms * sum(m 쨌 H_RKKY) * dV
    //   = mu_0 * Ms * (J/(mu_0*Ms*d)) * sum(m 쨌 m_ref) * dV
    //   = (J/d) * sum(m 쨌 m_ref) * dV
    double dot_sum = 0.0;
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        dot_sum += m[i].x * h_ref[0*N_ + i]
                 + m[i].y * h_ref[1*N_ + i]
                 + m[i].z * h_ref[2*N_ + i];
    }
    // E = -J/(2*d) * sum(m.m_ref) * dV  (matches CPU rkky.cpp)
    return -(J_RKKY_ / (2.0 * d_spacer_)) * dot_sum * dV;
}

// ---------------------------------------------------------------------------
// Full-GPU path (IEffectiveFieldGPU)
// ---------------------------------------------------------------------------
void RKKYFieldGPU::accumulate_gpu_ptr(const GReal* /*d_m*/,
                                       const Material& mat,
                                       GReal* d_H_out) const {
    if (d_spacer_ == 0.0 || mat.Ms == 0.0) return;
    const double mu0 = 4e-7 * 3.14159265358979323846;
    // coeff = +J/(mu0*Ms*d)  (same sign as CPU rkky.cpp)
    const double factor = J_RKKY_ / (mu0 * mat.Ms * d_spacer_);

    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    rkky_kernel<<<grd, blk, 0, s>>>(
        d_H_out,
        d_ref_,
        factor, static_cast<int>(N_));
    MICROMAG_KERNEL_CHECK();
}

}  // namespace micromag

#endif // MICROMAG_CUDA

