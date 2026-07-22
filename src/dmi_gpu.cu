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
    double dmydx = grad1(m,1,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmxdy = grad1(m,0,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    double dmxdz = grad1(m,0,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
    double dmydz = grad1(m,1,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);

    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (use_bc) {
        // Free-boundary DMI condition dm/dn = (D/2A)(n_hat x m): boundary
        // gradient g_BC = g_onesided/2 + (D/4A) gamma, plus the exchange-ghost
        // correction s (D / mu0 Ms d) gamma (see BulkDMIField::accumulate).
        const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
        if (xmin || xmax) {                             // gamma_x = x_hat x m
            const double gX = mz, gY = -my;             // -(x_hat x m) = (0, mz, -my)
            if (nx == 1) { dmydx = 2.0*qDA*gX; dmzdx = 2.0*qDA*gY; }
            else {
                const double sgn = xmin ? -1.0 : 1.0;
                dmydx = 0.5*dmydx + qDA*gX;
                dmzdx = 0.5*dmzdx + qDA*gY;
                dHy += sgn * bcx * gX;  dHz += sgn * bcx * gY;
            }
        }
        if (ymin || ymax) {                             // gamma_y = y_hat x m
            const double gX = -mz, gZ = mx;             // -(y_hat x m) = (-mz, 0, mx)
            if (ny == 1) { dmxdy = 2.0*qDA*gX; dmzdy = 2.0*qDA*gZ; }
            else {
                const double sgn = ymin ? -1.0 : 1.0;
                dmxdy = 0.5*dmxdy + qDA*gX;
                dmzdy = 0.5*dmzdy + qDA*gZ;
                dHx += sgn * bcy * gX;  dHz += sgn * bcy * gZ;
            }
        }
        if (zmin || zmax) {                             // gamma_z = z_hat x m
            const double gX = my, gY = -mx;             // -(z_hat x m) = (my, -mx, 0)
            if (nz == 1) { dmxdz = 2.0*qDA*gX; dmydz = 2.0*qDA*gY; }
            else {
                const double sgn = zmin ? -1.0 : 1.0;
                dmxdz = 0.5*dmxdz + qDA*gX;
                dmydz = 0.5*dmydz + qDA*gY;
                dHx += sgn * bcz * gX;  dHy += sgn * bcz * gY;
            }
        }
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

    double dmxdx = grad1(m,0,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmydy = grad1(m,1,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);

    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (use_bc) {
        // Free-boundary DMI condition dm/dn = -(D/2A)(z_hat x n_hat) x m
        // (sign follows this code's energy convention; verified against mumax3
        //  relax() edge canting; see InterfacialDMIField::accumulate).
        const double mx = m[0*N+idx], mz = m[2*N+idx];
        const double my = m[1*N+idx];
        if (xmin || xmax) {                             // gamma_x = (-mz, 0, mx)
            const double gX = -mz, gZ = mx;
            if (nx == 1) { dmxdx = 2.0*qDA*gX; dmzdx = 2.0*qDA*gZ; }
            else {
                const double sgn = xmin ? -1.0 : 1.0;
                dmxdx = 0.5*dmxdx + qDA*gX;
                dmzdx = 0.5*dmzdx + qDA*gZ;
                dHx += sgn * bcx * gX;  dHz += sgn * bcx * gZ;
            }
        }
        if (ymin || ymax) {                             // gamma_y = (0, -mz, my)
            const double gY = -mz, gZ = my;
            if (ny == 1) { dmydy = 2.0*qDA*gY; dmzdy = 2.0*qDA*gZ; }
            else {
                const double sgn = ymin ? -1.0 : 1.0;
                dmydy = 0.5*dmydy + qDA*gY;
                dmzdy = 0.5*dmzdy + qDA*gZ;
                dHy += sgn * bcy * gY;  dHz += sgn * bcy * gZ;
            }
        }
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

    double dmydx = grad1(m,1,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmxdy = grad1(m,0,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    double dmxdz = grad1(m,0,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
    double dmydz = grad1(m,1,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);

    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (use_bc) {   // dm/dn = (D/2A)(n_hat x m); see uniform-D kernel comments
        const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
        if (xmin || xmax) {
            const double gX = mz, gY = -my;
            if (nx == 1) { dmydx = 2.0*qDA*gX; dmzdx = 2.0*qDA*gY; }
            else {
                const double sgn = xmin ? -1.0 : 1.0;
                dmydx = 0.5*dmydx + qDA*gX;
                dmzdx = 0.5*dmzdx + qDA*gY;
                dHy += sgn * (bcf/dxv) * gX;  dHz += sgn * (bcf/dxv) * gY;
            }
        }
        if (ymin || ymax) {
            const double gX = -mz, gZ = mx;
            if (ny == 1) { dmxdy = 2.0*qDA*gX; dmzdy = 2.0*qDA*gZ; }
            else {
                const double sgn = ymin ? -1.0 : 1.0;
                dmxdy = 0.5*dmxdy + qDA*gX;
                dmzdy = 0.5*dmzdy + qDA*gZ;
                dHx += sgn * (bcf/dyv) * gX;  dHz += sgn * (bcf/dyv) * gZ;
            }
        }
        if (zmin || zmax) {
            const double gX = my, gY = -mx;
            if (nz == 1) { dmxdz = 2.0*qDA*gX; dmydz = 2.0*qDA*gY; }
            else {
                const double sgn = zmin ? -1.0 : 1.0;
                dmxdz = 0.5*dmxdz + qDA*gX;
                dmydz = 0.5*dmydz + qDA*gY;
                dHx += sgn * (bcf/dzv) * gX;  dHy += sgn * (bcf/dzv) * gY;
            }
        }
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

    double dmxdx = grad1(m,0,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    double dmydy = grad1(m,1,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);

    double dHx = 0.0, dHy = 0.0, dHz = 0.0;
    if (use_bc) {   // dm/dn = -(D/2A)(z_hat x n_hat) x m; see uniform-D kernel
        const double mx = m[0*N+idx], my = m[1*N+idx], mz = m[2*N+idx];
        if (xmin || xmax) {
            const double gX = -mz, gZ = mx;
            if (nx == 1) { dmxdx = 2.0*qDA*gX; dmzdx = 2.0*qDA*gZ; }
            else {
                const double sgn = xmin ? -1.0 : 1.0;
                dmxdx = 0.5*dmxdx + qDA*gX;
                dmzdx = 0.5*dmzdx + qDA*gZ;
                dHx += sgn * (bcf/dxv) * gX;  dHz += sgn * (bcf/dxv) * gZ;
            }
        }
        if (ymin || ymax) {
            const double gY = -mz, gZ = my;
            if (ny == 1) { dmydy = 2.0*qDA*gY; dmzdy = 2.0*qDA*gZ; }
            else {
                const double sgn = ymin ? -1.0 : 1.0;
                dmydy = 0.5*dmydy + qDA*gY;
                dmzdy = 0.5*dmzdy + qDA*gZ;
                dHy += sgn * (bcf/dyv) * gY;  dHz += sgn * (bcf/dyv) * gZ;
            }
        }
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
    if (D_ == 0.0) return;
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);

    std::vector<GReal> h_m;
    dmi_pack(m, N_, h_m);
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));

    const double prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);
    const int blk = 256, grd = static_cast<int>((N_+blk-1)/blk);
    const int use_bc = (!open_bc_ && mat.A_exchange > 0.0) ? 1 : 0;
    const double qDA = use_bc ? D_ / (4.0 * mat.A_exchange) : 0.0;
    const double bcf = D_ / (constants::mu_0 * mat.Ms);
    bulk_dmi_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        (int)nx_, (int)ny_, (int)nz_,
        1.0/(2.0*dx_), 1.0/dx_, 1.0/(2.0*dy_), 1.0/dy_,
        1.0/(2.0*dz_), 1.0/dz_, prefac,
        use_bc, qDA, bcf/dx_, bcf/dy_, bcf/dz_);
    MICROMAG_KERNEL_CHECK();

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
    if (D_ == 0.0) return;
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);

    std::vector<GReal> h_m;
    dmi_pack(m, N_, h_m);
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));

    const double prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);
    const int blk = 256, grd = static_cast<int>((N_+blk-1)/blk);
    const int use_bc = (!open_bc_ && mat.A_exchange > 0.0) ? 1 : 0;
    const double qDA = use_bc ? D_ / (4.0 * mat.A_exchange) : 0.0;
    const double bcf = D_ / (constants::mu_0 * mat.Ms);
    interfacial_dmi_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        (int)nx_, (int)ny_, (int)nz_,
        1.0/(2.0*dx_), 1.0/dx_, 1.0/(2.0*dy_), 1.0/dy_, prefac,
        use_bc, qDA, bcf/dx_, bcf/dy_);
    MICROMAG_KERNEL_CHECK();

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

