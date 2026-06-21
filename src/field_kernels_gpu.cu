// field_kernels_gpu.cu ??G2 + Phase E: GPU Zeeman, Uniaxial, and Cubic-Anisotropy fields
//
// Zeeman:     H_out[idx] += H_ext                        (uniform, m-independent)
// Anisotropy: H_out[idx] += (2K/關?Ms)(m[idx]쨌청) 청      (per-cell dot product)
//
// Memory layout: [3횞N] component-major, buf[c*N + idx]  (x-fastest)
// Same convention as ExchangeFieldGPU / DemagFieldGPU.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/anisotropy.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/gpu_real.hpp"
#include "micromag/material_field.hpp"
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
    GReal* __restrict__ H_out,
    double Hx, double Hy, double Hz,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    H_out[0*N + idx] += Hx;
    H_out[1*N + idx] += Hy;
    H_out[2*N + idx] += Hz;
}

// Uniaxial anisotropy: H_out[idx] += factor 횞 (m[idx]쨌청) 횞 청
// 청 = (ux,uy,uz) is assumed pre-normalised by the caller.
__global__ static void anisotropy_kernel(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    double factor,              // 2K / (關? Ms)
    double k2_factor,           // 4*Ku2 / (mu0 Ms), 2nd-order uniaxial term
    double ux, double uy, double uz,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double dot = m[0*N + idx]*ux + m[1*N + idx]*uy + m[2*N + idx]*uz;
    const double h   = factor * dot + k2_factor * dot*dot*dot;
    H_out[0*N + idx] += h * ux;
    H_out[1*N + idx] += h * uy;
    H_out[2*N + idx] += h * uz;
}

// ===========================================================================
// Fused Exchange + Zeeman + Uniaxial Anisotropy kernel (uniform-material only)
//
// Replaces three separate kernel launches with one pass over m and H_out,
// cutting global-memory reads/writes by ~36% for the non-demag fields:
//   Separate: (7×3 m-reads + 3 H-reads + 3 H-writes) × 3 kernels  = 42 ops
//   Fused:     7×3 m-reads + 3 H-reads + 3 H-writes               = 27 ops
//
// Parameters:
//   fx, fy, fz       — exchange factors: 2A/(μ₀ Ms d²) per axis
//   Hx, Hy, Hz       — uniform Zeeman field (A/m)
//   aniso_factor     — 2K/(μ₀ Ms); set to 0.0 to skip anisotropy
//   ux, uy, uz       — easy axis (pre-normalised)
//   bc_periodic      — true: periodic BC on all axes; false: Neumann BC
// ===========================================================================
__global__ static void exch_zeeman_aniso_fused_kernel(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    int nx, int ny, int nz,
    double fx, double fy, double fz,
    double Hx, double Hy, double Hz,
    double aniso_factor,
    double aniso_k2_factor,
    double ux, double uy, double uz,
    bool bc_periodic)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    // Neighbour indices (Neumann BC: clamp to self → zero contribution)
    int xm, xp, ym, yp, zm, zp;
    if (bc_periodic) {
        xm = (ix > 0)    ? idx - 1      : idx - 1 + nx;
        xp = (ix < nx-1) ? idx + 1      : idx + 1 - nx;
        ym = (iy > 0)    ? idx - nx      : idx - nx + nx*ny;
        yp = (iy < ny-1) ? idx + nx      : idx + nx - nx*ny;
        zm = (iz > 0)    ? idx - nx*ny   : idx - nx*ny + N;
        zp = (iz < nz-1) ? idx + nx*ny   : idx + nx*ny - N;
    } else {
        xm = (ix > 0)    ? idx - 1      : idx;
        xp = (ix < nx-1) ? idx + 1      : idx;
        ym = (iy > 0)    ? idx - nx      : idx;
        yp = (iy < ny-1) ? idx + nx      : idx;
        zm = (iz > 0)    ? idx - nx*ny   : idx;
        zp = (iz < nz-1) ? idx + nx*ny   : idx;
    }

    // Load centre magnetization into registers (reused for anisotropy)
    const double mx = m[0*N + idx];
    const double my = m[1*N + idx];
    const double mz = m[2*N + idx];

    // Accumulate into registers (single H_out read per component)
    double hx = H_out[0*N + idx];
    double hy = H_out[1*N + idx];
    double hz = H_out[2*N + idx];

    // --- Exchange (6-point Laplacian) ---
    hx += (m[0*N+xm]-mx)*fx + (m[0*N+xp]-mx)*fx
        + (m[0*N+ym]-mx)*fy + (m[0*N+yp]-mx)*fy
        + (m[0*N+zm]-mx)*fz + (m[0*N+zp]-mx)*fz;
    hy += (m[1*N+xm]-my)*fx + (m[1*N+xp]-my)*fx
        + (m[1*N+ym]-my)*fy + (m[1*N+yp]-my)*fy
        + (m[1*N+zm]-my)*fz + (m[1*N+zp]-my)*fz;
    hz += (m[2*N+xm]-mz)*fx + (m[2*N+xp]-mz)*fx
        + (m[2*N+ym]-mz)*fy + (m[2*N+yp]-mz)*fy
        + (m[2*N+zm]-mz)*fz + (m[2*N+zp]-mz)*fz;

    // --- Uniaxial Anisotropy (1st + 2nd order; skipped when K1=Ku2=0) ---
    if (aniso_factor != 0.0 || aniso_k2_factor != 0.0) {
        const double dot = mx*ux + my*uy + mz*uz;
        const double h   = aniso_factor * dot + aniso_k2_factor * dot*dot*dot;
        hx += h * ux;
        hy += h * uy;
        hz += h * uz;
    }

    // --- Zeeman (uniform) ---
    hx += Hx;
    hy += Hy;
    hz += Hz;

    // Single H_out write per component
    H_out[0*N + idx] = static_cast<GReal>(hx);
    H_out[1*N + idx] = static_cast<GReal>(hy);
    H_out[2*N + idx] = static_cast<GReal>(hz);
}

// Free function called from rk4_integrator_gpu.cu
void launch_fused_local_fields(
    const GReal* d_m,
    GReal*       d_H_out,
    int nx, int ny, int nz,
    double fx, double fy, double fz,
    double Hx, double Hy, double Hz,
    double aniso_factor,
    double aniso_k2_factor,
    double ux, double uy, double uz,
    bool bc_periodic,
    void* stream)
{
    const int N   = nx * ny * nz;
    const int blk = 256;
    const int grd = (N + blk - 1) / blk;
    exch_zeeman_aniso_fused_kernel<<<grd, blk, 0,
        static_cast<cudaStream_t>(stream)>>>(
        d_H_out, d_m, nx, ny, nz,
        fx, fy, fz, Hx, Hy, Hz,
        aniso_factor, aniso_k2_factor, ux, uy, uz,
        bc_periodic);
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
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// Standalone path: H_ext is uniform ??direct CPU add, no PCIe needed
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

// GPU-pointer path: used by G6 full-LLG pipeline (d_m is ignored ??Zeeman ??f(m))
void ZeemanFieldGPU::accumulate_gpu_ptr(const GReal* /*d_m*/,
                                          const Material& /*mat*/,
                                          GReal* d_H_out) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    zeeman_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(d_H_out),
        H_ext_.x, H_ext_.y, H_ext_.z, static_cast<int>(N_));
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
    CUDA_CHECK(cudaMalloc(&d_m_scratch_, 3 * N_ * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_, 3 * N_ * sizeof(GReal)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

UniaxialAnisotropyFieldGPU::~UniaxialAnisotropyFieldGPU() {
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (d_K_field_)    cudaFree(d_K_field_);
    if (d_axis_field_) cudaFree(d_axis_field_);
    if (d_Ms_field_)   cudaFree(d_Ms_field_);
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void UniaxialAnisotropyFieldGPU::accumulate(const VectorField3D& m,
                                              const Material& mat,
                                              VectorField3D& H_out) const {
    if (mat.K_uniaxial == 0.0 && mat.Ku2 == 0.0) return;

    Vec3 u = mat.easy_axis;
    const double unorm = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
    if (unorm < 1e-30) return;
    u.x /= unorm; u.y /= unorm; u.z /= unorm;
    const double inv_mu0Ms = 1.0 / (constants::mu_0 * mat.Ms);
    const double factor    = 2.0 * mat.K_uniaxial * inv_mu0Ms;
    const double k2_factor = 4.0 * mat.Ku2 * inv_mu0Ms;

    // Pack m into [Mx|My|Mz] host buffer
    std::vector<GReal> h_m(3 * N_);
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_m[i]           = static_cast<GReal>(m[i].x);
        h_m[N_  + i]     = static_cast<GReal>(m[i].y);
        h_m[2*N_ + i]    = static_cast<GReal>(m[i].z);
    }

    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);

    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));

    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    anisotropy_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        factor, k2_factor, u.x, u.y, u.z, static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());

    std::vector<GReal> h_H(3 * N_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(h_H.data(), dH, 3*N_*sizeof(GReal),
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

// Per-cell uniaxial anisotropy kernel
// d_K:    [N]   ??K_uniaxial per cell
// d_axis: [3N]  ??easy_axis per cell (component-major: [ux0..uxN-1 | uy | uz])
// d_Ms:   [N]   ??Ms per cell
// Per-cell K1/easy_axis/Ms; Ku2 is uniform (from mat) matching the CPU
// UniaxialAnisotropyField per-cell convention.
__global__ static void anisotropy_kernel_percell(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    const double* __restrict__ d_K,
    const double* __restrict__ d_axis,
    const double* __restrict__ d_Ms,
    double mu0_inv2,   // 2.0 / mu_0
    double ku2_uniform, // uniform Ku2 [J/m^3]
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double K_i  = d_K[idx];
    const double Ms_i = d_Ms[idx];
    if (Ms_i <= 0.0 || (K_i == 0.0 && ku2_uniform == 0.0)) return;

    // Normalize easy axis
    double ux = d_axis[0*N + idx];
    double uy = d_axis[1*N + idx];
    double uz = d_axis[2*N + idx];
    const double unorm = sqrt(ux*ux + uy*uy + uz*uz);
    if (unorm < 1e-30) return;
    ux /= unorm; uy /= unorm; uz /= unorm;

    const double inv_mu0Ms = mu0_inv2 / (2.0 * Ms_i);   // 1/(mu0 Ms)
    const double factor    = 2.0 * K_i * inv_mu0Ms;
    const double k2_factor = 4.0 * ku2_uniform * inv_mu0Ms;
    const double dot = m[0*N + idx]*ux + m[1*N + idx]*uy + m[2*N + idx]*uz;
    const double h   = factor * dot + k2_factor * dot*dot*dot;
    H_out[0*N + idx] += h * ux;
    H_out[1*N + idx] += h * uy;
    H_out[2*N + idx] += h * uz;
}

// GPU-pointer path (no PCIe): used by G6 pipeline
void UniaxialAnisotropyFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                                      const Material& mat,
                                                      GReal* d_H_out) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);

    auto* gm = reinterpret_cast<const GReal*>(d_m);
    auto* gH = reinterpret_cast<GReal*>(d_H_out);

    if (d_K_field_) {
        // Per-cell mode (K1/axis/Ms per cell; Ku2 uniform from mat)
        const double mu0_inv2 = 2.0 / constants::mu_0;
        anisotropy_kernel_percell<<<grd, blk, 0, s>>>(
            gH, gm, d_K_field_, d_axis_field_, d_Ms_field_,
            mu0_inv2, mat.Ku2, static_cast<int>(N_));
    } else {
        // Uniform mode
        if (mat.K_uniaxial == 0.0 && mat.Ku2 == 0.0) return;
        Vec3 u = mat.easy_axis;
        const double unorm = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
        if (unorm < 1e-30) return;
        u.x /= unorm; u.y /= unorm; u.z /= unorm;
        const double inv_mu0Ms = 1.0 / (constants::mu_0 * mat.Ms);
        const double factor    = 2.0 * mat.K_uniaxial * inv_mu0Ms;
        const double k2_factor = 4.0 * mat.Ku2 * inv_mu0Ms;
        anisotropy_kernel<<<grd, blk, 0, s>>>(
            gH, gm, factor, k2_factor, u.x, u.y, u.z, static_cast<int>(N_));
    }
    CUDA_CHECK(cudaGetLastError());
}

void UniaxialAnisotropyFieldGPU::set_material_field(const MaterialField3D& matf) {
    if (!d_K_field_) {
        CUDA_CHECK(cudaMalloc(&d_K_field_,    N_   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_axis_field_, 3*N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ms_field_,   N_   * sizeof(double)));
    }
    std::vector<double> h_K(N_), h_Ms(N_), h_axis(3 * N_);
    for (size_t i = 0; i < N_; ++i) {
        h_K[i]  = matf.K_uniaxial(static_cast<Index>(i));
        h_Ms[i] = matf.Ms(static_cast<Index>(i));
        const Vec3& u = matf.easy_axis(static_cast<Index>(i));
        h_axis[0*N_ + i] = u.x;
        h_axis[1*N_ + i] = u.y;
        h_axis[2*N_ + i] = u.z;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_K_field_,    h_K.data(),    N_  *sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_axis_field_, h_axis.data(), 3*N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_Ms_field_,   h_Ms.data(),   N_  *sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void UniaxialAnisotropyFieldGPU::clear_material_field() {
    if (d_K_field_)    { cudaFree(d_K_field_);    d_K_field_    = nullptr; }
    if (d_axis_field_) { cudaFree(d_axis_field_); d_axis_field_ = nullptr; }
    if (d_Ms_field_)   { cudaFree(d_Ms_field_);   d_Ms_field_   = nullptr; }
}

// ===========================================================================
// CubicAnisotropyFieldGPU
// ===========================================================================

// e = Kc1*(a1짼a2짼 + a2짼a3짼 + a3짼a1짼) + Kc2*(a1짼a2짼a3짼),  ai = m쨌ci
// H = pre1*(a1(a2짼+a3짼)c1 + a2(a1짼+a3짼)c2 + a3(a1짼+a2짼)c3)
//   + pre2*(a1a2짼a3짼c1 + a1짼a2a3짼c2 + a1짼a2짼a3c3)
// pre1 = -2Kc1/(mu0*Ms),  pre2 = -2Kc2/(mu0*Ms)
__global__ static void cubic_anisotropy_kernel(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
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
    CUDA_CHECK(cudaMalloc(&d_m_scratch_, 3 * N_ * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_, 3 * N_ * sizeof(GReal)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

// Per-cell cubic anisotropy kernel
// d_Kc1, d_Kc2: [N] ??per-cell coupling constants
// d_c1, d_c2, d_c3: [3N] component-major ??per-cell cubic axes
// d_Ms: [N] ??per-cell Ms
__global__ static void cubic_anisotropy_kernel_percell(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    const double* __restrict__ d_Kc1,
    const double* __restrict__ d_Kc2,
    const double* __restrict__ d_c1,
    const double* __restrict__ d_c2,
    const double* __restrict__ d_c3,
    const double* __restrict__ d_Ms,
    double mu0,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double Ms_i  = d_Ms[idx];
    const double Kc1_i = d_Kc1[idx];
    const double Kc2_i = d_Kc2[idx];
    if (Ms_i <= 0.0 || (Kc1_i == 0.0 && Kc2_i == 0.0)) return;
    const double mu0Ms = mu0 * Ms_i;
    const double pre1  = -2.0 * Kc1_i / mu0Ms;
    const double pre2  = -2.0 * Kc2_i / mu0Ms;

    const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
    const double c1x = d_c1[0*N+idx], c1y = d_c1[1*N+idx], c1z = d_c1[2*N+idx];
    const double c2x = d_c2[0*N+idx], c2y = d_c2[1*N+idx], c2z = d_c2[2*N+idx];
    const double c3x = d_c3[0*N+idx], c3y = d_c3[1*N+idx], c3z = d_c3[2*N+idx];

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

CubicAnisotropyFieldGPU::~CubicAnisotropyFieldGPU() {
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (d_Kc1_field_) cudaFree(d_Kc1_field_);
    if (d_Kc2_field_) cudaFree(d_Kc2_field_);
    if (d_c1_field_)  cudaFree(d_c1_field_);
    if (d_c2_field_)  cudaFree(d_c2_field_);
    if (d_c3_field_)  cudaFree(d_c3_field_);
    if (d_Ms_field_)  cudaFree(d_Ms_field_);
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void CubicAnisotropyFieldGPU::accumulate(const VectorField3D& m,
                                          const Material& mat,
                                          VectorField3D& H_out) const {
    if (Kc1_ == 0.0 && Kc2_ == 0.0) return;
    const double mu0Ms = constants::mu_0 * mat.Ms;
    if (mu0Ms < 1e-30) return;

    std::vector<GReal> h_m(3 * N_);
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_m[i]       = static_cast<GReal>(m[i].x);
        h_m[N_+i]    = static_cast<GReal>(m[i].y);
        h_m[2*N_+i]  = static_cast<GReal>(m[i].z);
    }
    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    cubic_anisotropy_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        -2.0*Kc1_/mu0Ms, -2.0*Kc2_/mu0Ms,
        c1_.x, c1_.y, c1_.z,
        c2_.x, c2_.y, c2_.z,
        c3_.x, c3_.y, c3_.z,
        static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
    std::vector<GReal> h_H(3 * N_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(h_H.data(), dH, 3*N_*sizeof(GReal), cudaMemcpyDeviceToHost));
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

void CubicAnisotropyFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                                   const Material& mat,
                                                   GReal* d_H_out) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);

    auto* gm = reinterpret_cast<const GReal*>(d_m);
    auto* gH = reinterpret_cast<GReal*>(d_H_out);

    if (d_Kc1_field_) {
        cubic_anisotropy_kernel_percell<<<grd, blk, 0, s>>>(
            gH, gm,
            d_Kc1_field_, d_Kc2_field_,
            d_c1_field_, d_c2_field_, d_c3_field_,
            d_Ms_field_,
            constants::mu_0, static_cast<int>(N_));
    } else {
        if (Kc1_ == 0.0 && Kc2_ == 0.0) return;
        const double mu0Ms = constants::mu_0 * mat.Ms;
        if (mu0Ms < 1e-30) return;
        cubic_anisotropy_kernel<<<grd, blk, 0, s>>>(
            gH, gm,
            -2.0*Kc1_/mu0Ms, -2.0*Kc2_/mu0Ms,
            c1_.x, c1_.y, c1_.z,
            c2_.x, c2_.y, c2_.z,
            c3_.x, c3_.y, c3_.z,
            static_cast<int>(N_));
    }
    CUDA_CHECK(cudaGetLastError());
}

void CubicAnisotropyFieldGPU::set_Kc_field(
    const ScalarField3D& Kc1_f, const ScalarField3D& Kc2_f,
    const VectorField3D& c1_f,  const VectorField3D& c2_f,
    const ScalarField3D& Ms_f)
{
    if (!d_Kc1_field_) {
        CUDA_CHECK(cudaMalloc(&d_Kc1_field_, N_   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Kc2_field_, N_   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_c1_field_,  3*N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_c2_field_,  3*N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_c3_field_,  3*N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ms_field_,  N_   * sizeof(double)));
    }
    std::vector<double> h_k1(N_), h_k2(N_), h_Ms(N_);
    std::vector<double> h_c1(3*N_), h_c2(3*N_), h_c3(3*N_);
    for (size_t i = 0; i < N_; ++i) {
        h_k1[i] = Kc1_f[static_cast<Index>(i)];
        h_k2[i] = Kc2_f[static_cast<Index>(i)];
        h_Ms[i] = Ms_f[static_cast<Index>(i)];
        Vec3 e1 = c1_f[static_cast<Index>(i)];
        Vec3 e2 = c2_f[static_cast<Index>(i)];
        Vec3 e3 = e1.cross(e2);
        h_c1[0*N_+i]=e1.x; h_c1[1*N_+i]=e1.y; h_c1[2*N_+i]=e1.z;
        h_c2[0*N_+i]=e2.x; h_c2[1*N_+i]=e2.y; h_c2[2*N_+i]=e2.z;
        h_c3[0*N_+i]=e3.x; h_c3[1*N_+i]=e3.y; h_c3[2*N_+i]=e3.z;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_Kc1_field_, h_k1.data(), N_  *sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_Kc2_field_, h_k2.data(), N_  *sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_c1_field_,  h_c1.data(), 3*N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_c2_field_,  h_c2.data(), 3*N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_c3_field_,  h_c3.data(), 3*N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_Ms_field_,  h_Ms.data(), N_  *sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void CubicAnisotropyFieldGPU::clear_Kc_field() {
    if (d_Kc1_field_) { cudaFree(d_Kc1_field_); d_Kc1_field_ = nullptr; }
    if (d_Kc2_field_) { cudaFree(d_Kc2_field_); d_Kc2_field_ = nullptr; }
    if (d_c1_field_)  { cudaFree(d_c1_field_);  d_c1_field_  = nullptr; }
    if (d_c2_field_)  { cudaFree(d_c2_field_);  d_c2_field_  = nullptr; }
    if (d_c3_field_)  { cudaFree(d_c3_field_);  d_c3_field_  = nullptr; }
    if (d_Ms_field_)  { cudaFree(d_Ms_field_);  d_Ms_field_  = nullptr; }
}

}  // namespace micromag

#endif // MICROMAG_CUDA

