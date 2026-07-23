// dmi_gpu.cu ??GPU-accelerated BulkDMI and InterfacialDMI fields.
//
// Memory layout: [3횞N] component-major, x-fastest.
//   m[c*N + ix + nx*(iy + ny*iz)]
//
// Gradient: central diff (2d) at interior, one-sided (d) at boundary.
// This matches the CPU detail::grad_* helpers in grad_helpers.hpp.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include "micromag/cuda_sync_debug.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/dmi.hpp"
#include "micromag/dmi_gpu.hpp"
#include "micromag/gpu_real.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess)                                        \
            throw std::runtime_error(std::string("CUDA(dmi): ")      \
                                   + cudaGetErrorString(_e));         \
    } while (0)

namespace micromag {

// ---------------------------------------------------------------------------
// Device helpers: gradient in one direction for component c
// ---------------------------------------------------------------------------
__device__ __forceinline__ static double grad1(
    const GReal* m, int c, int N,
    int idx, int idx_m1, int idx_p1,   // neighbor linear indices
    bool at_min, bool at_max,          // boundary flags
    double inv_2d, double inv_d)       // 1/(2d) and 1/d
{
    if (at_min && at_max) return 0.0;  // size-1 dimension
    const GReal* mc = m + c * N;
    if (at_min) return (static_cast<double>(mc[idx_p1]) - static_cast<double>(mc[idx])) * inv_d;
    if (at_max) return (static_cast<double>(mc[idx]) - static_cast<double>(mc[idx_m1])) * inv_d;
    return (static_cast<double>(mc[idx_p1]) - static_cast<double>(mc[idx_m1])) * inv_2d;
}

// Unified axis gradient with the free-boundary DMI condition applied at
// MISSING neighbours: outside the grid OR vacuum (GeomMask, |m| ~ 0) cells,
// matching mumax3's zero-Msat-neighbour treatment (see BulkDMIField comments).
// Slots A/B are the two m-components whose gradients this axis needs; gamA/gamB
// are the matching gamma components; bc_over_d = D/(mu0 Ms d) exchange-ghost
// factor added into the matching dH slots. Both-faces-free: g = 2 qDA gamma.
__device__ __forceinline__ static void axis_grad_bc(
    const GReal* __restrict__ m, int N, int idx, int stride,
    bool in_lo, bool in_hi, double inv_d,
    int cA, int cB, double gamA, double gamB,
    double qDA, double bc_over_d,
    double& gA, double& gB, double& dHA, double& dHB)
{
    bool lo = in_lo, hi = in_hi;
    if (lo) { const int j = idx - stride;
              const double vx=m[j], vy=m[N+j], vz=m[2*N+j];
              lo = (vx*vx + vy*vy + vz*vz) > 0.25; }
    if (hi) { const int j = idx + stride;
              const double vx=m[j], vy=m[N+j], vz=m[2*N+j];
              hi = (vx*vx + vy*vy + vz*vz) > 0.25; }
    const double aC = m[cA*N + idx], bC = m[cB*N + idx];
    if (lo && hi) {
        gA = (static_cast<double>(m[cA*N + idx + stride]) - static_cast<double>(m[cA*N + idx - stride])) * (0.5 * inv_d);
        gB = (static_cast<double>(m[cB*N + idx + stride]) - static_cast<double>(m[cB*N + idx - stride])) * (0.5 * inv_d);
    } else if (hi) {
        gA = (static_cast<double>(m[cA*N + idx + stride]) - aC) * (0.5 * inv_d) + qDA * gamA;
        gB = (static_cast<double>(m[cB*N + idx + stride]) - bC) * (0.5 * inv_d) + qDA * gamB;
        dHA -= bc_over_d * gamA;  dHB -= bc_over_d * gamB;
    } else if (lo) {
        gA = (aC - static_cast<double>(m[cA*N + idx - stride])) * (0.5 * inv_d) + qDA * gamA;
        gB = (bC - static_cast<double>(m[cB*N + idx - stride])) * (0.5 * inv_d) + qDA * gamB;
        dHA += bc_over_d * gamA;  dHB += bc_over_d * gamB;
    } else {
        gA = 2.0 * qDA * gamA;
        gB = 2.0 * qDA * gamB;
    }
}

// ---------------------------------------------------------------------------
// Bulk DMI kernel: H = prefac * curl(m)
// curl_m = (?굆z/?굖 - ?굆y/?굗, ?굆x/?굗 - ?굆z/?굕, ?굆y/?굕 - ?굆x/?굖)
// ---------------------------------------------------------------------------
__global__ static void bulk_dmi_kernel(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    int nx, int ny, int nz,
    double inv_2dx, double inv_dx,
    double inv_2dy, double inv_dy,
    double inv_2dz, double inv_dz,
    double prefac,
    int use_bc, double qDA, double bcx, double bcy, double bcz)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    const int idx_xm = idx - 1,     idx_xp = idx + 1;
    const int idx_ym = idx - nx,    idx_yp = idx + nx;
    const int idx_zm = idx - nx*ny, idx_zp = idx + nx*ny;

    const bool xmin=(ix==0), xmax=(ix==nx-1);
    const bool ymin=(iy==0), ymax=(iy==ny-1);
    const bool zmin=(iz==0), zmax=(iz==nz-1);

    // gx[c] = ?굆c/?굕,  gy[c] = ?굆c/?굖,  gz[c] = ?굆c/?굗
    // Needed: gx[1](?굆y/?굕), gx[2](?굆z/?굕), gy[0](?굆x/?굖), gy[2](?굆z/?굖),
    //         gz[0](?굆x/?굗), gz[1](?굆y/?굗)
    double dmydx, dmzdx, dmxdy, dmzdy, dmxdz, dmydz;
    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (!use_bc) {
        dmydx = grad1(m,1,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmxdy = grad1(m,0,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
        dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
        dmxdz = grad1(m,0,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
        dmydz = grad1(m,1,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
    } else {
        // Free-boundary DMI condition dm/dn = (D/2A)(n_hat x m) at missing
        // (out-of-grid or vacuum) neighbours; see BulkDMIField::accumulate.
        const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
        if (mx*mx + my*my + mz*mz < 0.25) return;       // vacuum cell itself
        // gamma_x = -(x_hat x m) = (0, mz, -my), etc.
        axis_grad_bc(m, N, idx, 1,     !xmin, !xmax, inv_dx, 1, 2,  mz, -my, qDA, bcx, dmydx, dmzdx, dHy, dHz);
        axis_grad_bc(m, N, idx, nx,    !ymin, !ymax, inv_dy, 0, 2, -mz,  mx, qDA, bcy, dmxdy, dmzdy, dHx, dHz);
        axis_grad_bc(m, N, idx, nx*ny, !zmin, !zmax, inv_dz, 0, 1,  my, -mx, qDA, bcz, dmxdz, dmydz, dHx, dHy);
    }

    H_out[0*N + idx] += prefac * (dmzdy - dmydz) + dHx;
    H_out[1*N + idx] += prefac * (dmxdz - dmzdx) + dHy;
    H_out[2*N + idx] += prefac * (dmydx - dmxdy) + dHz;
}

// ---------------------------------------------------------------------------
// Interfacial DMI kernel: H = prefac * (?굆z/?굕, ?굆z/?굖, -(?굆x/?굕 + ?굆y/?굖))
// ---------------------------------------------------------------------------
__global__ static void interfacial_dmi_kernel(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    int nx, int ny, int nz,
    double inv_2dx, double inv_dx,
    double inv_2dy, double inv_dy,
    double prefac,
    int use_bc, double qDA, double bcx, double bcy)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;

    const int idx_xm = idx - 1, idx_xp = idx + 1;
    const int idx_ym = idx - nx, idx_yp = idx + nx;

    const bool xmin=(ix==0), xmax=(ix==nx-1);
    const bool ymin=(iy==0), ymax=(iy==ny-1);

    double dmxdx, dmzdx, dmydy, dmzdy;
    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (!use_bc) {
        dmxdx = grad1(m,0,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmydy = grad1(m,1,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
        dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    } else {
        // Free-boundary DMI condition dm/dn = -(D/2A)(z_hat x n_hat) x m at
        // missing (out-of-grid or vacuum) neighbours; verified against mumax3
        // relax() edge canting; see InterfacialDMIField::accumulate.
        const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
        if (mx*mx + my*my + mz*mz < 0.25) return;       // vacuum cell itself
        // gamma_x = (-mz, 0, mx), gamma_y = (0, -mz, my)
        axis_grad_bc(m, N, idx, 1,  !xmin, !xmax, inv_dx, 0, 2, -mz, mx, qDA, bcx, dmxdx, dmzdx, dHx, dHz);
        axis_grad_bc(m, N, idx, nx, !ymin, !ymax, inv_dy, 1, 2, -mz, my, qDA, bcy, dmydy, dmzdy, dHy, dHz);
    }

    H_out[0*N + idx] += prefac * dmzdx + dHx;
    H_out[1*N + idx] += prefac * dmzdy + dHy;
    H_out[2*N + idx] += prefac * (-(dmxdx + dmydy)) + dHz;
}

// ---------------------------------------------------------------------------
// Per-cell Bulk DMI kernel: prefac_i = 2*D_i/(mu0*Ms_i)
// ---------------------------------------------------------------------------
__global__ static void bulk_dmi_kernel_percell(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    const double* __restrict__ d_D,
    const double* __restrict__ d_Ms,
    int nx, int ny, int nz,
    double inv_2dx, double inv_dx,
    double inv_2dy, double inv_dy,
    double inv_2dz, double inv_dz,
    double mu0_inv2,
    int use_bc, double inv4A, double dxv, double dyv, double dzv)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double D_i  = d_D[idx];
    const double Ms_i = d_Ms[idx];
    if (Ms_i <= 0.0 || D_i == 0.0) return;
    const double prefac = D_i * mu0_inv2 / Ms_i;
    const double qDA = use_bc ? D_i * inv4A : 0.0;                   // D_i/4A
    const double bcf = use_bc ? D_i * (mu0_inv2 * 0.5) / Ms_i : 0.0; // D_i/(mu0 Ms_i)

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    const int idx_xm = idx - 1,     idx_xp = idx + 1;
    const int idx_ym = idx - nx,    idx_yp = idx + nx;
    const int idx_zm = idx - nx*ny, idx_zp = idx + nx*ny;

    const bool xmin=(ix==0), xmax=(ix==nx-1);
    const bool ymin=(iy==0), ymax=(iy==ny-1);
    const bool zmin=(iz==0), zmax=(iz==nz-1);

    double dmydx, dmzdx, dmxdy, dmzdy, dmxdz, dmydz;
    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (!use_bc) {
        dmydx = grad1(m,1,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmxdy = grad1(m,0,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
        dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
        dmxdz = grad1(m,0,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
        dmydz = grad1(m,1,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
    } else {   // dm/dn = (D/2A)(n_hat x m); see uniform-D kernel comments
        const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
        if (mx*mx + my*my + mz*mz < 0.25) return;       // vacuum cell itself
        axis_grad_bc(m, N, idx, 1,     !xmin, !xmax, inv_dx, 1, 2,  mz, -my, qDA, bcf/dxv, dmydx, dmzdx, dHy, dHz);
        axis_grad_bc(m, N, idx, nx,    !ymin, !ymax, inv_dy, 0, 2, -mz,  mx, qDA, bcf/dyv, dmxdy, dmzdy, dHx, dHz);
        axis_grad_bc(m, N, idx, nx*ny, !zmin, !zmax, inv_dz, 0, 1,  my, -mx, qDA, bcf/dzv, dmxdz, dmydz, dHx, dHy);
    }

    H_out[0*N + idx] += prefac * (dmzdy - dmydz) + dHx;
    H_out[1*N + idx] += prefac * (dmxdz - dmzdx) + dHy;
    H_out[2*N + idx] += prefac * (dmydx - dmxdy) + dHz;
}

// ---------------------------------------------------------------------------
// Per-cell Interfacial DMI kernel
// ---------------------------------------------------------------------------
__global__ static void interfacial_dmi_kernel_percell(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    const double* __restrict__ d_D,
    const double* __restrict__ d_Ms,
    int nx, int ny, int nz,
    double inv_2dx, double inv_dx,
    double inv_2dy, double inv_dy,
    double mu0_inv2,
    int use_bc, double inv4A, double dxv, double dyv)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double D_i  = d_D[idx];
    const double Ms_i = d_Ms[idx];
    if (Ms_i <= 0.0 || D_i == 0.0) return;
    const double prefac = D_i * mu0_inv2 / Ms_i;
    const double qDA = use_bc ? D_i * inv4A : 0.0;
    const double bcf = use_bc ? D_i * (mu0_inv2 * 0.5) / Ms_i : 0.0;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;

    const int idx_xm = idx - 1, idx_xp = idx + 1;
    const int idx_ym = idx - nx, idx_yp = idx + nx;

    const bool xmin=(ix==0), xmax=(ix==nx-1);
    const bool ymin=(iy==0), ymax=(iy==ny-1);

    double dmxdx, dmzdx, dmydy, dmzdy;
    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (!use_bc) {
        dmxdx = grad1(m,0,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
        dmydy = grad1(m,1,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
        dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    } else {   // dm/dn = -(D/2A)(z_hat x n_hat) x m; see uniform-D kernel
        const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
        if (mx*mx + my*my + mz*mz < 0.25) return;       // vacuum cell itself
        axis_grad_bc(m, N, idx, 1,  !xmin, !xmax, inv_dx, 0, 2, -mz, mx, qDA, bcf/dxv, dmxdx, dmzdx, dHx, dHz);
        axis_grad_bc(m, N, idx, nx, !ymin, !ymax, inv_dy, 1, 2, -mz, my, qDA, bcf/dyv, dmydy, dmzdy, dHy, dHz);
    }

    H_out[0*N + idx] += prefac * dmzdx + dHx;
    H_out[1*N + idx] += prefac * dmzdy + dHy;
    H_out[2*N + idx] += prefac * (-(dmxdx + dmydy)) + dHz;
}

// ---------------------------------------------------------------------------
// Shared constructor / destructor logic
// ---------------------------------------------------------------------------
static void dmi_gpu_alloc(size_t N, void*& d_m, void*& d_H, void*& stream)
{
    CUDA_CHECK(cudaMalloc(&d_m, 3 * N * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H, 3 * N * sizeof(GReal)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream = static_cast<void*>(s);
}

static void dmi_gpu_free(void* d_m, void* d_H, void* stream, bool owned)
{
    cudaFree(d_m);
    cudaFree(d_H);
    if (stream && owned) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

static void dmi_pack(const VectorField3D& m, size_t N, std::vector<GReal>& h_m)
{
    h_m.resize(3 * N);
    for (Index i = 0; i < static_cast<Index>(N); ++i) {
        h_m[i]       = static_cast<GReal>(m[i].x);
        h_m[N   + i] = static_cast<GReal>(m[i].y);
        h_m[2*N + i] = static_cast<GReal>(m[i].z);
    }
}

static void dmi_unpack_add(const std::vector<GReal>& h_H, size_t N, VectorField3D& H_out)
{
    for (Index i = 0; i < static_cast<Index>(N); ++i) {
        H_out[i].x += h_H[i];
        H_out[i].y += h_H[N   + i];
        H_out[i].z += h_H[2*N + i];
    }
}

// ===========================================================================
// BulkDMIFieldGPU
// ===========================================================================
BulkDMIFieldGPU::BulkDMIFieldGPU(const StructuredGrid& grid, Real D)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      N_(static_cast<size_t>(grid.size())),
      D_(D)
{
    dmi_gpu_alloc(N_, d_m_scratch_, d_H_scratch_, stream_);
}

BulkDMIFieldGPU::~BulkDMIFieldGPU() {
    dmi_gpu_free(d_m_scratch_, d_H_scratch_, stream_, stream_owned_);
    if (d_D_field_)  cudaFree(d_D_field_);
    if (d_Ms_field_) cudaFree(d_Ms_field_);
}

void BulkDMIFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                           const Material& mat,
                                           GReal* d_H_out) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    const double i2dx = 1.0/(2.0*dx_), idx = 1.0/dx_;
    const double i2dy = 1.0/(2.0*dy_), idy = 1.0/dy_;
    const double i2dz = 1.0/(2.0*dz_), idz = 1.0/dz_;

    auto* gm = reinterpret_cast<const GReal*>(d_m);
    auto* gH = reinterpret_cast<GReal*>(d_H_out);

    const int use_bc = (!open_bc_ && mat.A_exchange > 0.0) ? 1 : 0;
    const double inv4A = use_bc ? 1.0 / (4.0 * mat.A_exchange) : 0.0;
    if (d_D_field_ != nullptr) {
        const double mu0_inv2 = 2.0 / constants::mu_0;
        bulk_dmi_kernel_percell<<<grd, blk, 0, s>>>(
            gH, gm, d_D_field_, d_Ms_field_,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, i2dz, idz, mu0_inv2,
            use_bc, inv4A, dx_, dy_, dz_);
    } else {
        if (D_ == 0.0) return;
        const double prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);
        const double bcf = D_ / (constants::mu_0 * mat.Ms);
        bulk_dmi_kernel<<<grd, blk, 0, s>>>(
            gH, gm,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, i2dz, idz, prefac,
            use_bc, D_ * inv4A, bcf / dx_, bcf / dy_, bcf / dz_);
    }
    MICROMAG_KERNEL_CHECK();
}

void BulkDMIFieldGPU::set_D_field(const ScalarField3D& D_field,
                                    const ScalarField3D& Ms_field) {
    if (!d_D_field_) {
        CUDA_CHECK(cudaMalloc(&d_D_field_,  N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ms_field_, N_ * sizeof(double)));
    }
    std::vector<double> h_D(N_), h_Ms(N_);
    for (size_t i = 0; i < N_; ++i) {
        h_D[i]  = D_field[static_cast<Index>(i)];
        h_Ms[i] = Ms_field[static_cast<Index>(i)];
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_D_field_,  h_D.data(),  N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_Ms_field_, h_Ms.data(), N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void BulkDMIFieldGPU::clear_D_field() {
    if (d_D_field_)  { cudaFree(d_D_field_);  d_D_field_  = nullptr; }
    if (d_Ms_field_) { cudaFree(d_Ms_field_); d_Ms_field_ = nullptr; }
}

void BulkDMIFieldGPU::accumulate(const VectorField3D& m,
                                   const Material& mat,
                                   VectorField3D& H_out) const {
    // Guard covers per-cell D mode; kernel dispatch (uniform vs per-cell)
    // lives in accumulate_gpu_ptr so host and integrator paths agree.
    if (!d_D_field_ && D_ == 0.0) return;
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);

    std::vector<GReal> h_m;
    dmi_pack(m, N_, h_m);
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));

    accumulate_gpu_ptr(dm, mat, dH);

    std::vector<GReal> h_H(3*N_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(h_H.data(), dH, 3*N_*sizeof(GReal), cudaMemcpyDeviceToHost));
    dmi_unpack_add(h_H, N_, H_out);
}

Real BulkDMIFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    BulkDMIField cpu(D_);
    return cpu.energy(m, mat);
}

ScalarField3D BulkDMIFieldGPU::energy_density(const VectorField3D& m,
                                                const Material& mat) const {
    BulkDMIField cpu(D_);
    return cpu.energy_density(m, mat);
}

// ===========================================================================
// InterfacialDMIFieldGPU
// ===========================================================================
InterfacialDMIFieldGPU::InterfacialDMIFieldGPU(const StructuredGrid& grid, Real D)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      N_(static_cast<size_t>(grid.size())),
      D_(D)
{
    dmi_gpu_alloc(N_, d_m_scratch_, d_H_scratch_, stream_);
}

InterfacialDMIFieldGPU::~InterfacialDMIFieldGPU() {
    dmi_gpu_free(d_m_scratch_, d_H_scratch_, stream_, stream_owned_);
    if (d_D_field_)  cudaFree(d_D_field_);
    if (d_Ms_field_) cudaFree(d_Ms_field_);
}

void InterfacialDMIFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                                  const Material& mat,
                                                  GReal* d_H_out) const {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256, grd = static_cast<int>((N_+blk-1)/blk);
    const double i2dx = 1.0/(2.0*dx_), idx = 1.0/dx_;
    const double i2dy = 1.0/(2.0*dy_), idy = 1.0/dy_;

    auto* gm = reinterpret_cast<const GReal*>(d_m);
    auto* gH = reinterpret_cast<GReal*>(d_H_out);

    const int use_bc = (!open_bc_ && mat.A_exchange > 0.0) ? 1 : 0;
    const double inv4A = use_bc ? 1.0 / (4.0 * mat.A_exchange) : 0.0;
    if (d_D_field_) {
        const double mu0_inv2 = 2.0 / constants::mu_0;
        interfacial_dmi_kernel_percell<<<grd, blk, 0, s>>>(
            gH, gm, d_D_field_, d_Ms_field_,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, mu0_inv2,
            use_bc, inv4A, dx_, dy_);
    } else {
        if (D_ == 0.0) return;
        const double prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);
        const double bcf = D_ / (constants::mu_0 * mat.Ms);
        interfacial_dmi_kernel<<<grd, blk, 0, s>>>(
            gH, gm,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, prefac,
            use_bc, D_ * inv4A, bcf / dx_, bcf / dy_);
    }
    MICROMAG_KERNEL_CHECK();
}

void InterfacialDMIFieldGPU::set_D_field(const ScalarField3D& D_field,
                                           const ScalarField3D& Ms_field) {
    if (!d_D_field_) {
        CUDA_CHECK(cudaMalloc(&d_D_field_,  N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ms_field_, N_ * sizeof(double)));
    }
    std::vector<double> h_D(N_), h_Ms(N_);
    for (size_t i = 0; i < N_; ++i) {
        h_D[i]  = D_field[static_cast<Index>(i)];
        h_Ms[i] = Ms_field[static_cast<Index>(i)];
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_D_field_,  h_D.data(),  N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_Ms_field_, h_Ms.data(), N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void InterfacialDMIFieldGPU::clear_D_field() {
    if (d_D_field_)  { cudaFree(d_D_field_);  d_D_field_  = nullptr; }
    if (d_Ms_field_) { cudaFree(d_Ms_field_); d_Ms_field_ = nullptr; }
}

void InterfacialDMIFieldGPU::accumulate(const VectorField3D& m,
                                          const Material& mat,
                                          VectorField3D& H_out) const {
    // Guard covers per-cell D mode; kernel dispatch (uniform vs per-cell)
    // lives in accumulate_gpu_ptr so host and integrator paths agree.
    if (!d_D_field_ && D_ == 0.0) return;
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);

    std::vector<GReal> h_m;
    dmi_pack(m, N_, h_m);
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));

    accumulate_gpu_ptr(dm, mat, dH);

    std::vector<GReal> h_H(3*N_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(h_H.data(), dH, 3*N_*sizeof(GReal), cudaMemcpyDeviceToHost));
    dmi_unpack_add(h_H, N_, H_out);
}

Real InterfacialDMIFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    InterfacialDMIField cpu(D_);
    return cpu.energy(m, mat);
}

ScalarField3D InterfacialDMIFieldGPU::energy_density(const VectorField3D& m,
                                                       const Material& mat) const {
    InterfacialDMIField cpu(D_);
    return cpu.energy_density(m, mat);
}

}  // namespace micromag

#endif // MICROMAG_CUDA

