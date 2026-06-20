// dmi_gpu.cu ??GPU-accelerated BulkDMI and InterfacialDMI fields.
//
// Memory layout: [3횞N] component-major, x-fastest.
//   m[c*N + ix + nx*(iy + ny*iz)]
//
// Gradient: central diff (2d) at interior, one-sided (d) at boundary.
// This matches the CPU detail::grad_* helpers in grad_helpers.hpp.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
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
    double prefac)
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
    const double dmydx = grad1(m,1,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmxdy = grad1(m,0,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    const double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    const double dmxdz = grad1(m,0,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
    const double dmydz = grad1(m,1,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);

    H_out[0*N + idx] += prefac * (dmzdy - dmydz);
    H_out[1*N + idx] += prefac * (dmxdz - dmzdx);
    H_out[2*N + idx] += prefac * (dmydx - dmxdy);
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
    double prefac)
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

    const double dmxdx = grad1(m,0,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmydy = grad1(m,1,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    const double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);

    H_out[0*N + idx] += prefac * dmzdx;
    H_out[1*N + idx] += prefac * dmzdy;
    H_out[2*N + idx] += prefac * (-(dmxdx + dmydy));
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
    double mu0_inv2)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double D_i  = d_D[idx];
    const double Ms_i = d_Ms[idx];
    if (Ms_i <= 0.0 || D_i == 0.0) return;
    const double prefac = D_i * mu0_inv2 / Ms_i;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    const int idx_xm = idx - 1,     idx_xp = idx + 1;
    const int idx_ym = idx - nx,    idx_yp = idx + nx;
    const int idx_zm = idx - nx*ny, idx_zp = idx + nx*ny;

    const bool xmin=(ix==0), xmax=(ix==nx-1);
    const bool ymin=(iy==0), ymax=(iy==ny-1);
    const bool zmin=(iz==0), zmax=(iz==nz-1);

    const double dmydx = grad1(m,1,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmxdy = grad1(m,0,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    const double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    const double dmxdz = grad1(m,0,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);
    const double dmydz = grad1(m,1,N, idx, idx_zm,idx_zp, zmin,zmax, inv_2dz,inv_dz);

    H_out[0*N + idx] += prefac * (dmzdy - dmydz);
    H_out[1*N + idx] += prefac * (dmxdz - dmzdx);
    H_out[2*N + idx] += prefac * (dmydx - dmxdy);
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
    double mu0_inv2)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double D_i  = d_D[idx];
    const double Ms_i = d_Ms[idx];
    if (Ms_i <= 0.0 || D_i == 0.0) return;
    const double prefac = D_i * mu0_inv2 / Ms_i;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;

    const int idx_xm = idx - 1, idx_xp = idx + 1;
    const int idx_ym = idx - nx, idx_yp = idx + nx;

    const bool xmin=(ix==0), xmax=(ix==nx-1);
    const bool ymin=(iy==0), ymax=(iy==ny-1);

    const double dmxdx = grad1(m,0,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmzdx = grad1(m,2,N, idx, idx_xm,idx_xp, xmin,xmax, inv_2dx,inv_dx);
    const double dmydy = grad1(m,1,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);
    const double dmzdy = grad1(m,2,N, idx, idx_ym,idx_yp, ymin,ymax, inv_2dy,inv_dy);

    H_out[0*N + idx] += prefac * dmzdx;
    H_out[1*N + idx] += prefac * dmzdy;
    H_out[2*N + idx] += prefac * (-(dmxdx + dmydy));
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

static void dmi_gpu_free(void* d_m, void* d_H, void* stream)
{
    cudaFree(d_m);
    cudaFree(d_H);
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
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
    dmi_gpu_free(d_m_scratch_, d_H_scratch_, stream_);
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

    if (d_D_field_ != nullptr) {
        const double mu0_inv2 = 2.0 / constants::mu_0;
        bulk_dmi_kernel_percell<<<grd, blk, 0, s>>>(
            gH, gm, d_D_field_, d_Ms_field_,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, i2dz, idz, mu0_inv2);
    } else {
        if (D_ == 0.0) return;
        const double prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);
        bulk_dmi_kernel<<<grd, blk, 0, s>>>(
            gH, gm,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, i2dz, idz, prefac);
    }
    CUDA_CHECK(cudaGetLastError());
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
    bulk_dmi_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        (int)nx_, (int)ny_, (int)nz_,
        1.0/(2.0*dx_), 1.0/dx_, 1.0/(2.0*dy_), 1.0/dy_,
        1.0/(2.0*dz_), 1.0/dz_, prefac);
    CUDA_CHECK(cudaGetLastError());

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
    dmi_gpu_free(d_m_scratch_, d_H_scratch_, stream_);
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

    if (d_D_field_) {
        const double mu0_inv2 = 2.0 / constants::mu_0;
        interfacial_dmi_kernel_percell<<<grd, blk, 0, s>>>(
            gH, gm, d_D_field_, d_Ms_field_,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, mu0_inv2);
    } else {
        if (D_ == 0.0) return;
        const double prefac = 2.0 * D_ / (constants::mu_0 * mat.Ms);
        interfacial_dmi_kernel<<<grd, blk, 0, s>>>(
            gH, gm,
            (int)nx_, (int)ny_, (int)nz_,
            i2dx, idx, i2dy, idy, prefac);
    }
    CUDA_CHECK(cudaGetLastError());
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
    interfacial_dmi_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        (int)nx_, (int)ny_, (int)nz_,
        1.0/(2.0*dx_), 1.0/dx_, 1.0/(2.0*dy_), 1.0/dy_, prefac);
    CUDA_CHECK(cudaGetLastError());

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

