// field_kernels_gpu.cu — G2 + Phase E: GPU Zeeman, Uniaxial, and Cubic-Anisotropy fields
//
// Zeeman:     H_out[idx] += H_ext                        (uniform, m-independent)
// Anisotropy: H_out[idx] += (2K/μ₀Ms)(m[idx]·û) û      (per-cell dot product)
//
// Memory layout: [3×N] component-major, buf[c*N + idx]  (x-fastest)
// Same convention as ExchangeFieldGPU / DemagFieldGPU.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/anisotropy.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/types.hpp"
#include "micromag/zeeman.hpp"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA(fields): ")     \
                                   + cudaGetErrorString(_e));           \
    } while (0)

namespace micromag {

// ===========================================================================
// CUDA kernels
// ===========================================================================

// Zeeman: H_out[c*N+idx] += H_c  (same value for every cell)
__global__ static void zeeman_kernel(
    double* __restrict__ H_out,
    double Hx, double Hy, double Hz,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    H_out[0*N + idx] += Hx;
    H_out[1*N + idx] += Hy;
    H_out[2*N + idx] += Hz;
}

// Uniaxial anisotropy: H_out[idx] += factor × (m[idx]·û) × û
// û = (ux,uy,uz) is assumed pre-normalised by the caller.
__global__ static void anisotropy_kernel(
    double* __restrict__       H_out,
    const double* __restrict__ m,
    double factor,              // 2K / (μ₀ Ms)
    double ux, double uy, double uz,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double dot = m[0*N + idx]*ux + m[1*N + idx]*uy + m[2*N + idx]*uz;
    const double h   = factor * dot;
    H_out[0*N + idx] += h * ux;
    H_out[1*N + idx] += h * uy;
    H_out[2*N + idx] += h * uz;
}

// ===========================================================================
// ZeemanFieldGPU
// ===========================================================================

ZeemanFieldGPU::ZeemanFieldGPU(const StructuredGrid& grid, const Vec3& H_ext)
    : N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz())),
      H_ext_(H_ext)
{
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

ZeemanFieldGPU::~ZeemanFieldGPU() {
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// Standalone path: H_ext is uniform → direct CPU add, no PCIe needed
void ZeemanFieldGPU::accumulate(const VectorField3D& m,
                                  const Material& /*mat*/,
                                  VectorField3D& H_out) const {
    for (Index i = 0; i < m.size(); ++i)
        H_out[i] += H_ext_;
}

Real ZeemanFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    ZeemanField cpu(H_ext_);
    return cpu.energy(m, mat);
}

ScalarField3D ZeemanFieldGPU::energy_density(const VectorField3D& m,
                                               const Material& mat) const {
    ZeemanField cpu(H_ext_);
    return cpu.energy_density(m, mat);
}

// GPU-pointer path: used by G6 full-LLG pipeline (d_m is ignored — Zeeman ≠ f(m))
void ZeemanFieldGPU::accumulate_gpu_ptr(const double* /*d_m*/,
                                          const Material& /*mat*/,
                                          double* d_H_out) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    zeeman_kernel<<<grd, blk, 0, s>>>(
        d_H_out, H_ext_.x, H_ext_.y, H_ext_.z, static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// UniaxialAnisotropyFieldGPU
// ===========================================================================

UniaxialAnisotropyFieldGPU::UniaxialAnisotropyFieldGPU(const StructuredGrid& grid)
    : N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz()))
{
    CUDA_CHECK(cudaMalloc(&d_m_scratch_, 3 * N_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_, 3 * N_ * sizeof(double)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

UniaxialAnisotropyFieldGPU::~UniaxialAnisotropyFieldGPU() {
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void UniaxialAnisotropyFieldGPU::accumulate(const VectorField3D& m,
                                              const Material& mat,
                                              VectorField3D& H_out) const {
    if (mat.K_uniaxial == 0.0) return;

    Vec3 u = mat.easy_axis;
    const double unorm = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    if (unorm < 1e-30) return;
    u.x /= unorm; u.y /= unorm; u.z /= unorm;
    const double factor = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);

    // Pack m into [Mx|My|Mz] host buffer
    std::vector<double> h_m(3 * N_);
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_m[i]           = m[i].x;
        h_m[N_  + i]     = m[i].y;
        h_m[2*N_ + i]    = m[i].z;
    }

    auto* dm = static_cast<double*>(d_m_scratch_);
    auto* dH = static_cast<double*>(d_H_scratch_);
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);

    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(double)));

    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    anisotropy_kernel<<<grd, blk, 0, s>>>(
        dH, dm, factor, u.x, u.y, u.z, static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_H(3 * N_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(h_H.data(), dH, 3*N_*sizeof(double),
                          cudaMemcpyDeviceToHost));

    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        H_out[i].x += h_H[i];
        H_out[i].y += h_H[N_  + i];
        H_out[i].z += h_H[2*N_ + i];
    }
}

Real UniaxialAnisotropyFieldGPU::energy(const VectorField3D& m,
                                          const Material& mat) const {
    UniaxialAnisotropyField cpu;
    return cpu.energy(m, mat);
}

ScalarField3D UniaxialAnisotropyFieldGPU::energy_density(const VectorField3D& m,
                                                           const Material& mat) const {
    UniaxialAnisotropyField cpu;
    return cpu.energy_density(m, mat);
}

// GPU-pointer path (no PCIe): used by G6 pipeline
void UniaxialAnisotropyFieldGPU::accumulate_gpu_ptr(const double* d_m,
                                                      const Material& mat,
                                                      double* d_H_out) const {
    if (mat.K_uniaxial == 0.0) return;

    Vec3 u = mat.easy_axis;
    const double unorm = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    if (unorm < 1e-30) return;
    u.x /= unorm; u.y /= unorm; u.z /= unorm;
    const double factor = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);

    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    anisotropy_kernel<<<grd, blk, 0, s>>>(
        d_H_out, d_m, factor, u.x, u.y, u.z, static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// CubicAnisotropyFieldGPU
// ===========================================================================

// e = Kc1*(a1²a2² + a2²a3² + a3²a1²) + Kc2*(a1²a2²a3²),  ai = m·ci
// H = pre1*(a1(a2²+a3²)c1 + a2(a1²+a3²)c2 + a3(a1²+a2²)c3)
//   + pre2*(a1a2²a3²c1 + a1²a2a3²c2 + a1²a2²a3c3)
// pre1 = -2Kc1/(mu0*Ms),  pre2 = -2Kc2/(mu0*Ms)
__global__ static void cubic_anisotropy_kernel(
    double* __restrict__       H_out,
    const double* __restrict__ m,
    double pre1, double pre2,
    double c1x, double c1y, double c1z,
    double c2x, double c2y, double c2z,
    double c3x, double c3y, double c3z,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
    const double a1 = mx*c1x + my*c1y + mz*c1z;
    const double a2 = mx*c2x + my*c2y + mz*c2z;
    const double a3 = mx*c3x + my*c3y + mz*c3z;
    const double a11 = a1*a1, a22 = a2*a2, a33 = a3*a3;
    const double f1 = pre1*a1*(a22+a33) + pre2*a1*a22*a33;
    const double f2 = pre1*a2*(a11+a33) + pre2*a11*a2*a33;
    const double f3 = pre1*a3*(a11+a22) + pre2*a11*a22*a3;
    H_out[0*N+idx] += f1*c1x + f2*c2x + f3*c3x;
    H_out[1*N+idx] += f1*c1y + f2*c2y + f3*c3y;
    H_out[2*N+idx] += f1*c1z + f2*c2z + f3*c3z;
}

CubicAnisotropyFieldGPU::CubicAnisotropyFieldGPU(const StructuredGrid& grid,
                                                   Real Kc1, Real Kc2,
                                                   Vec3 c1, Vec3 c2)
    : N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz())),
      Kc1_(Kc1), Kc2_(Kc2), c1_(c1), c2_(c2), c3_(c1.cross(c2))
{
    CUDA_CHECK(cudaMalloc(&d_m_scratch_, 3 * N_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_, 3 * N_ * sizeof(double)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

CubicAnisotropyFieldGPU::~CubicAnisotropyFieldGPU() {
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void CubicAnisotropyFieldGPU::accumulate(const VectorField3D& m,
                                          const Material& mat,
                                          VectorField3D& H_out) const {
    if (Kc1_ == 0.0 && Kc2_ == 0.0) return;
    const double mu0Ms = constants::mu_0 * mat.Ms;
    if (mu0Ms < 1e-30) return;

    std::vector<double> h_m(3 * N_);
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_m[i]       = m[i].x;
        h_m[N_+i]    = m[i].y;
        h_m[2*N_+i]  = m[i].z;
    }
    auto* dm = static_cast<double*>(d_m_scratch_);
    auto* dH = static_cast<double*>(d_H_scratch_);
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(double)));
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    cubic_anisotropy_kernel<<<grd, blk, 0, s>>>(
        dH, dm,
        -2.0*Kc1_/mu0Ms, -2.0*Kc2_/mu0Ms,
        c1_.x, c1_.y, c1_.z,
        c2_.x, c2_.y, c2_.z,
        c3_.x, c3_.y, c3_.z,
        static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> h_H(3 * N_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(h_H.data(), dH, 3*N_*sizeof(double), cudaMemcpyDeviceToHost));
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        H_out[i].x += h_H[i];
        H_out[i].y += h_H[N_+i];
        H_out[i].z += h_H[2*N_+i];
    }
}

Real CubicAnisotropyFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    CubicAnisotropyField cpu(Kc1_, Kc2_, c1_, c2_);
    return cpu.energy(m, mat);
}

ScalarField3D CubicAnisotropyFieldGPU::energy_density(const VectorField3D& m,
                                                        const Material& mat) const {
    CubicAnisotropyField cpu(Kc1_, Kc2_, c1_, c2_);
    return cpu.energy_density(m, mat);
}

void CubicAnisotropyFieldGPU::accumulate_gpu_ptr(const double* d_m,
                                                   const Material& mat,
                                                   double* d_H_out) const {
    if (Kc1_ == 0.0 && Kc2_ == 0.0) return;
    const double mu0Ms = constants::mu_0 * mat.Ms;
    if (mu0Ms < 1e-30) return;
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    cubic_anisotropy_kernel<<<grd, blk, 0, s>>>(
        d_H_out, d_m,
        -2.0*Kc1_/mu0Ms, -2.0*Kc2_/mu0Ms,
        c1_.x, c1_.y, c1_.z,
        c2_.x, c2_.y, c2_.z,
        c3_.x, c3_.y, c3_.z,
        static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace micromag

#endif // MICROMAG_CUDA
