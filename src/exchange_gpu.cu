// exchange_gpu.cu — GPU exchange field (Phase G, Step G1)
//
// Implements the same 6-point Laplacian as ExchangeField (CPU):
//   H_exch[cell] += (2A / μ₀Ms) × Σ_±x,±y,±z  (m_neigh − m_self) / d²
//
// Memory layout: [3 × N] component-major
//   buf[c*N + idx],  idx = ix + nx*(iy + ny*iz)  (x-fastest)
//   c = 0 → Mx,  1 → My,  2 → Mz
//
// Neumann BC: out-of-bounds neighbor → use self → contribution = 0.
// This exactly matches CPU ExchangeField with BoundaryCondition::Neumann.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/exchange.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/material_field.hpp"
#include "micromag/types.hpp"

// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t _e = (call);                                     \
        if (_e != cudaSuccess)                                       \
            throw std::runtime_error(std::string("CUDA(exch): ")    \
                                   + cudaGetErrorString(_e));        \
    } while (0)

namespace micromag {

// ===========================================================================
// CUDA kernel: 6-point Laplacian exchange field
//
// One thread per cell (1-D launch, idx ∈ [0, N)).
// Neumann BC: when idx_neigh == idx (boundary cell), m_neigh − m_self = 0.
// ===========================================================================
__global__ static void exchange_kernel(
    double* __restrict__       H_out,   // [3×N] add into (never zeros)
    const double* __restrict__ m,       // [3×N] magnetization
    int nx, int ny, int nz,
    double fx,                          // 2A/(μ₀ Ms dx²)
    double fy,                          // 2A/(μ₀ Ms dy²)
    double fz)                          // 2A/(μ₀ Ms dz²)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Decompose linear index → (ix, iy, iz)
    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    // Neighbor linear indices; Neumann BC: clamp to self (→ zero contribution)
    const int idx_xm = (ix > 0)    ? idx - 1      : idx;
    const int idx_xp = (ix < nx-1) ? idx + 1      : idx;
    const int idx_ym = (iy > 0)    ? idx - nx      : idx;
    const int idx_yp = (iy < ny-1) ? idx + nx      : idx;
    const int idx_zm = (iz > 0)    ? idx - nx*ny   : idx;
    const int idx_zp = (iz < nz-1) ? idx + nx*ny   : idx;

    // Process all 3 components in one thread (coalesced within component slice)
    for (int c = 0; c < 3; ++c) {
        const int base = c * N;
        const double mc = m[base + idx];
        const double lap =
            (m[base + idx_xm] - mc) * fx +
            (m[base + idx_xp] - mc) * fx +
            (m[base + idx_ym] - mc) * fy +
            (m[base + idx_yp] - mc) * fy +
            (m[base + idx_zm] - mc) * fz +
            (m[base + idx_zp] - mc) * fz;
        H_out[base + idx] += lap;
    }
}

// ===========================================================================
// CUDA kernel: 6-point Laplacian — Neumann BC, 2D shared-memory tile.
//
// Block: EXCH_BX×EXCH_BY = 32×8 = 256 threads.
// Shared tile: (BX+2)×(BY+2) per component = 34×10 = 340 doubles per comp.
// Total shared mem: 3 × 340 × 8 = 8160 bytes (well within 48KB limit).
//
// x,y neighbors are served from the tile; z-neighbors from global memory.
// Neumann BC: boundary clamping happens via block-level halo indices
// (halo_ixL/R, halo_iyB/T) that clamp to [0, n-1].  OOB threads load the
// clamped Neumann extension into their tile slot; in-bounds threads read
// adjacent slots → zero contribution at the boundary.  ✓
// ===========================================================================
#define EXCH_BX 32
#define EXCH_BY 8

__global__ static void exchange_kernel_smem(
    double* __restrict__       H_out,
    const double* __restrict__ m,
    int nx, int ny, int nz,
    double fx, double fy, double fz)
{
    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int ix = (int)blockIdx.x * EXCH_BX + tx;
    const int iy = (int)blockIdx.y * EXCH_BY + ty;
    const int iz = (int)blockIdx.z;
    const int N  = nx * ny * nz;

    // Neumann-clamped cell index for OOB threads (Neumann extension)
    const int cix = min(max(ix, 0), nx - 1);
    const int ciy = min(max(iy, 0), ny - 1);

    // Block-level halo positions, clamped to [0, n-1] (Neumann BC)
    const int halo_ixL = max((int)blockIdx.x * EXCH_BX - 1, 0);
    const int halo_ixR = min((int)blockIdx.x * EXCH_BX + EXCH_BX, nx - 1);
    const int halo_iyB = max((int)blockIdx.y * EXCH_BY - 1, 0);
    const int halo_iyT = min((int)blockIdx.y * EXCH_BY + EXCH_BY, ny - 1);

    __shared__ double tile[3][EXCH_BY + 2][EXCH_BX + 2];

    const int iz_off = ny * iz;

    for (int c = 0; c < 3; ++c) {
        const int base = c * N;

        // Center: clamped index → OOB threads load Neumann extension
        tile[c][ty + 1][tx + 1] = m[base + cix + nx * (ciy + iz_off)];

        // Left halo (tx == 0 loads for entire column)
        if (tx == 0)
            tile[c][ty + 1][0] = m[base + halo_ixL + nx * (ciy + iz_off)];

        // Right halo (tx == EXCH_BX-1 loads for entire column)
        if (tx == EXCH_BX - 1)
            tile[c][ty + 1][EXCH_BX + 1] = m[base + halo_ixR + nx * (ciy + iz_off)];

        // Bottom halo (ty == 0 loads for entire row)
        if (ty == 0)
            tile[c][0][tx + 1] = m[base + cix + nx * (halo_iyB + iz_off)];

        // Top halo (ty == EXCH_BY-1 loads for entire row)
        if (ty == EXCH_BY - 1)
            tile[c][EXCH_BY + 1][tx + 1] = m[base + cix + nx * (halo_iyT + iz_off)];
    }
    __syncthreads();

    if (ix >= nx || iy >= ny) return;

    const int idx    = ix + nx * (iy + ny * iz);
    const int idx_zm = (iz > 0)     ? idx - nx * ny : idx;
    const int idx_zp = (iz < nz - 1)? idx + nx * ny : idx;

    for (int c = 0; c < 3; ++c) {
        const int    base = c * N;
        const double mc   = tile[c][ty + 1][tx + 1];
        const double lap  =
            (tile[c][ty + 1][tx    ] - mc) * fx +
            (tile[c][ty + 1][tx + 2] - mc) * fx +
            (tile[c][ty    ][tx + 1] - mc) * fy +
            (tile[c][ty + 2][tx + 1] - mc) * fy +
            (m[base + idx_zm]         - mc) * fz +
            (m[base + idx_zp]         - mc) * fz;
        H_out[base + idx] += lap;
    }
}
#undef EXCH_BX
#undef EXCH_BY

// ===========================================================================
// CUDA kernel: 6-point Laplacian — periodic BC (wrap-around indices)
// ===========================================================================
__global__ static void exchange_kernel_periodic(
    double* __restrict__       H_out,
    const double* __restrict__ m,
    int nx, int ny, int nz,
    double fx, double fy, double fz)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    // Periodic wrap
    const int idx_xm = ((ix - 1 + nx) % nx) + nx * (iy + ny * iz);
    const int idx_xp = ((ix + 1)      % nx) + nx * (iy + ny * iz);
    const int idx_ym = ix + nx * (((iy - 1 + ny) % ny) + ny * iz);
    const int idx_yp = ix + nx * (((iy + 1)      % ny) + ny * iz);
    const int idx_zm = ix + nx * (iy + ny * ((iz - 1 + nz) % nz));
    const int idx_zp = ix + nx * (iy + ny * ((iz + 1)      % nz));

    for (int c = 0; c < 3; ++c) {
        const int base = c * N;
        const double mc = m[base + idx];
        const double lap =
            (m[base + idx_xm] - mc) * fx +
            (m[base + idx_xp] - mc) * fx +
            (m[base + idx_ym] - mc) * fy +
            (m[base + idx_yp] - mc) * fy +
            (m[base + idx_zm] - mc) * fz +
            (m[base + idx_zp] - mc) * fz;
        H_out[base + idx] += lap;
    }
}

// ===========================================================================
// CUDA kernel: 6-point Laplacian — per-cell A and Ms (Neumann BC)
// ===========================================================================
__global__ static void exchange_kernel_percell(
    double* __restrict__       H_out,
    const double* __restrict__ m,
    const double* __restrict__ d_A,    // [N] A_exchange per cell
    const double* __restrict__ d_Ms,   // [N] Ms per cell
    int nx, int ny, int nz,
    double mu0_inv2,   // 2.0 / mu_0
    double idx2, double idy2, double idz2)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double A_c  = d_A[idx];
    const double Ms_c = d_Ms[idx];
    if (Ms_c <= 0.0 || A_c <= 0.0) return;
    const double pre_c = mu0_inv2 / Ms_c;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    // Neighbor indices (Neumann: clamp to self → zero contribution)
    const int idx_xm = (ix > 0)    ? idx - 1      : idx;
    const int idx_xp = (ix < nx-1) ? idx + 1      : idx;
    const int idx_ym = (iy > 0)    ? idx - nx      : idx;
    const int idx_yp = (iy < ny-1) ? idx + nx      : idx;
    const int idx_zm = (iz > 0)    ? idx - nx*ny   : idx;
    const int idx_zp = (iz < nz-1) ? idx + nx*ny   : idx;

    // Harmonic mean A at each bond
    auto harm = [](double a, double b) -> double {
        const double s = a + b;
        return (s > 0.0) ? (2.0 * a * b / s) : 0.0;
    };
    const double A_xm = harm(A_c, d_A[idx_xm]);
    const double A_xp = harm(A_c, d_A[idx_xp]);
    const double A_ym = harm(A_c, d_A[idx_ym]);
    const double A_yp = harm(A_c, d_A[idx_yp]);
    const double A_zm = harm(A_c, d_A[idx_zm]);
    const double A_zp = harm(A_c, d_A[idx_zp]);

    for (int c = 0; c < 3; ++c) {
        const int base = c * N;
        const double mc = m[base + idx];
        const double acc =
            (m[base + idx_xm] - mc) * A_xm * idx2 +
            (m[base + idx_xp] - mc) * A_xp * idx2 +
            (m[base + idx_ym] - mc) * A_ym * idy2 +
            (m[base + idx_yp] - mc) * A_yp * idy2 +
            (m[base + idx_zm] - mc) * A_zm * idz2 +
            (m[base + idx_zp] - mc) * A_zp * idz2;
        H_out[base + idx] += acc * pre_c;
    }
}

// Per-cell periodic BC version
__global__ static void exchange_kernel_percell_periodic(
    double* __restrict__       H_out,
    const double* __restrict__ m,
    const double* __restrict__ d_A,
    const double* __restrict__ d_Ms,
    int nx, int ny, int nz,
    double mu0_inv2,
    double idx2, double idy2, double idz2)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double A_c  = d_A[idx];
    const double Ms_c = d_Ms[idx];
    if (Ms_c <= 0.0 || A_c <= 0.0) return;
    const double pre_c = mu0_inv2 / Ms_c;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    const int idx_xm = ((ix - 1 + nx) % nx) + nx * (iy + ny * iz);
    const int idx_xp = ((ix + 1)      % nx) + nx * (iy + ny * iz);
    const int idx_ym = ix + nx * (((iy - 1 + ny) % ny) + ny * iz);
    const int idx_yp = ix + nx * (((iy + 1)      % ny) + ny * iz);
    const int idx_zm = ix + nx * (iy + ny * ((iz - 1 + nz) % nz));
    const int idx_zp = ix + nx * (iy + ny * ((iz + 1)      % nz));

    auto harm = [](double a, double b) -> double {
        const double s = a + b;
        return (s > 0.0) ? (2.0 * a * b / s) : 0.0;
    };
    const double A_xm = harm(A_c, d_A[idx_xm]);
    const double A_xp = harm(A_c, d_A[idx_xp]);
    const double A_ym = harm(A_c, d_A[idx_ym]);
    const double A_yp = harm(A_c, d_A[idx_yp]);
    const double A_zm = harm(A_c, d_A[idx_zm]);
    const double A_zp = harm(A_c, d_A[idx_zp]);

    for (int c = 0; c < 3; ++c) {
        const int base = c * N;
        const double mc = m[base + idx];
        const double acc =
            (m[base + idx_xm] - mc) * A_xm * idx2 +
            (m[base + idx_xp] - mc) * A_xp * idx2 +
            (m[base + idx_ym] - mc) * A_ym * idy2 +
            (m[base + idx_yp] - mc) * A_yp * idy2 +
            (m[base + idx_zm] - mc) * A_zm * idz2 +
            (m[base + idx_zp] - mc) * A_zp * idz2;
        H_out[base + idx] += acc * pre_c;
    }
}

// ===========================================================================
// Constructor / Destructor
// ===========================================================================
ExchangeFieldGPU::ExchangeFieldGPU(const StructuredGrid& grid, BoundaryCondition bc)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz())),
      bc_(bc)
{
    CUDA_CHECK(cudaMalloc(&d_m_scratch_, 3 * N_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_, 3 * N_ * sizeof(double)));

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

ExchangeFieldGPU::~ExchangeFieldGPU() {
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (d_A_field_)  cudaFree(d_A_field_);
    if (d_Ms_field_) cudaFree(d_Ms_field_);
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ===========================================================================
// accumulate — upload m, run kernel, download H and add to H_out
// ===========================================================================
void ExchangeFieldGPU::accumulate(const VectorField3D& m,
                                   const Material& mat,
                                   VectorField3D& H_out) const {
    if (mat.A_exchange == 0.0) return;

    const cudaStream_t s   = static_cast<cudaStream_t>(stream_);
    const double pre = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);
    const double fx  = pre / (dx_ * dx_);
    const double fy  = pre / (dy_ * dy_);
    const double fz  = pre / (dz_ * dz_);

    // ------------------------------------------------------------------
    // 1. Pack VectorField3D → compact host buffer [Mx|My|Mz]
    // ------------------------------------------------------------------
    std::vector<double> h_m(3 * N_);
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_m[i]           = m[i].x;
        h_m[N_  + i]     = m[i].y;
        h_m[2*N_ + i]    = m[i].z;
    }

    auto* dm = static_cast<double*>(d_m_scratch_);
    auto* dH = static_cast<double*>(d_H_scratch_);

    // ------------------------------------------------------------------
    // 2. Upload m, zero H scratch
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(double)));

    // ------------------------------------------------------------------
    // 3. Launch exchange kernel (Neumann or Periodic BC)
    // ------------------------------------------------------------------
    if (bc_ == BoundaryCondition::Periodic) {
        const int blk = 256;
        const int grd = static_cast<int>((N_ + blk - 1) / blk);
        exchange_kernel_periodic<<<grd, blk, 0, s>>>(
            dH, dm, (int)nx_, (int)ny_, (int)nz_, fx, fy, fz);
    } else {
        // 2D shared-memory kernel: better cache locality for x/y neighbors
        const dim3 blk2(32, 8, 1);
        const dim3 grd2(((int)nx_ + 31) / 32, ((int)ny_ + 7) / 8, (int)nz_);
        exchange_kernel_smem<<<grd2, blk2, 0, s>>>(
            dH, dm, (int)nx_, (int)ny_, (int)nz_, fx, fy, fz);
    }
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 4. Download H result and accumulate into H_out
    // ------------------------------------------------------------------
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

// ===========================================================================
// accumulate_gpu_ptr — direct GPU-pointer path (no PCIe; used in G6 pipeline)
//
// Caller is responsible for zeroing d_H_out before this call (or relying on
// prior accumulation order). Uses the same stream_ as accumulate().
// ===========================================================================
void ExchangeFieldGPU::accumulate_gpu_ptr(const double* d_m,
                                            const Material& mat,
                                            double* d_H_out) const {
    const cudaStream_t s   = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    const double idx2 = 1.0 / (dx_ * dx_);
    const double idy2 = 1.0 / (dy_ * dy_);
    const double idz2 = 1.0 / (dz_ * dz_);

    if (d_A_field_) {
        // Per-cell mode
        const double mu0_inv2 = 2.0 / constants::mu_0;
        if (bc_ == BoundaryCondition::Periodic)
            exchange_kernel_percell_periodic<<<grd, blk, 0, s>>>(
                d_H_out, d_m, d_A_field_, d_Ms_field_,
                (int)nx_, (int)ny_, (int)nz_, mu0_inv2, idx2, idy2, idz2);
        else
            exchange_kernel_percell<<<grd, blk, 0, s>>>(
                d_H_out, d_m, d_A_field_, d_Ms_field_,
                (int)nx_, (int)ny_, (int)nz_, mu0_inv2, idx2, idy2, idz2);
    } else {
        // Uniform mode
        if (mat.A_exchange == 0.0) return;
        const double pre = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);
        if (bc_ == BoundaryCondition::Periodic)
            exchange_kernel_periodic<<<grd, blk, 0, s>>>(
                d_H_out, d_m, (int)nx_, (int)ny_, (int)nz_,
                pre * idx2, pre * idy2, pre * idz2);
        else {
            const dim3 blk2(32, 8, 1);
            const dim3 grd2(((int)nx_ + 31) / 32, ((int)ny_ + 7) / 8, (int)nz_);
            exchange_kernel_smem<<<grd2, blk2, 0, s>>>(
                d_H_out, d_m, (int)nx_, (int)ny_, (int)nz_,
                pre * idx2, pre * idy2, pre * idz2);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void ExchangeFieldGPU::set_material_field(const MaterialField3D& matf) {
    // Allocate per-cell buffers if not yet done
    if (!d_A_field_) {
        CUDA_CHECK(cudaMalloc(&d_A_field_,  N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ms_field_, N_ * sizeof(double)));
    }
    // Upload A and Ms from CPU MaterialField3D
    std::vector<double> h_A(N_), h_Ms(N_);
    for (size_t i = 0; i < N_; ++i) {
        h_A[i]  = matf.A_exchange(static_cast<Index>(i));
        h_Ms[i] = matf.Ms(static_cast<Index>(i));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_A_field_,  h_A.data(),  N_*sizeof(double),
                               cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream_)));
    CUDA_CHECK(cudaMemcpyAsync(d_Ms_field_, h_Ms.data(), N_*sizeof(double),
                               cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream_)));
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

void ExchangeFieldGPU::clear_material_field() {
    if (d_A_field_)  { cudaFree(d_A_field_);  d_A_field_  = nullptr; }
    if (d_Ms_field_) { cudaFree(d_Ms_field_); d_Ms_field_ = nullptr; }
}

// ===========================================================================
// energy — delegates to CPU ExchangeField (G3+ will add GPU reduction)
// ===========================================================================
Real ExchangeFieldGPU::energy(const VectorField3D& m, const Material& mat) const {
    ExchangeField cpu;
    return cpu.energy(m, mat);
}

ScalarField3D ExchangeFieldGPU::energy_density(const VectorField3D& m,
                                                 const Material& mat) const {
    ExchangeField cpu;
    return cpu.energy_density(m, mat);
}

}  // namespace micromag

#endif // MICROMAG_CUDA
