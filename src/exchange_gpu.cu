// exchange_gpu.cu ??GPU exchange field (Phase G, Step G1)
//
// Implements the same 6-point Laplacian as ExchangeField (CPU):
//   H_exch[cell] += (2A / 關?Ms) 횞 誇_짹x,짹y,짹z  (m_neigh ??m_self) / d짼
//
// Memory layout: [3 횞 N] component-major
//   buf[c*N + idx],  idx = ix + nx*(iy + ny*iz)  (x-fastest)
//   c = 0 ??Mx,  1 ??My,  2 ??Mz
//
// Neumann BC: out-of-bounds neighbor ??use self ??contribution = 0.
// This exactly matches CPU ExchangeField with BoundaryCondition::Neumann.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/exchange.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/geom_mask.hpp"
#include "micromag/gpu_real.hpp"
#include "micromag/material_field.hpp"
#include "micromag/region_map.hpp"
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
// One thread per cell (1-D launch, idx ??[0, N)).
// Neumann BC: when idx_neigh == idx (boundary cell), m_neigh ??m_self = 0.
// ===========================================================================
__global__ static void exchange_kernel(
    GReal* __restrict__       H_out,   // [3횞N] add into (never zeros)
    const GReal* __restrict__ m,       // [3횞N] magnetization
    int nx, int ny, int nz,
    double fx,                          // 2A/(關? Ms dx짼)
    double fy,                          // 2A/(關? Ms dy짼)
    double fz)                          // 2A/(關? Ms dz짼)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Decompose linear index ??(ix, iy, iz)
    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    // Neighbor linear indices; Neumann BC: clamp to self (??zero contribution)
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
// CUDA kernel: 6-point Laplacian ??Neumann BC, 2D shared-memory tile.
//
// Block: EXCH_BX횞EXCH_BY = 32횞8 = 256 threads.
// Shared tile: (BX+2)횞(BY+2) per component = 34횞10 = 340 doubles per comp.
// Total shared mem: 3 횞 340 횞 8 = 8160 bytes (well within 48KB limit).
//
// x,y neighbors are served from the tile; z-neighbors from global memory.
// Neumann BC: boundary clamping happens via block-level halo indices
// (halo_ixL/R, halo_iyB/T) that clamp to [0, n-1].  OOB threads load the
// clamped Neumann extension into their tile slot; in-bounds threads read
// adjacent slots ??zero contribution at the boundary.  ??// ===========================================================================
#define EXCH_BX 32
#define EXCH_BY 8

__global__ static void exchange_kernel_smem(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
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

        // Center: clamped index ??OOB threads load Neumann extension
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
// CUDA kernel: 6-point Laplacian ??periodic BC (wrap-around indices)
// ===========================================================================
__global__ static void exchange_kernel_periodic(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
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
// CUDA kernel: 6-point Laplacian ??per-cell A and Ms (Neumann BC)
// ===========================================================================
__global__ static void exchange_kernel_percell(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
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

    // Neighbor indices (Neumann: clamp to self ??zero contribution)
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
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
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
// General kernel: geometry mask + region-pair inter-exchange + optional
// per-cell A/Ms. Used only when a mask or region map is attached (the fast
// kernels above handle the common no-geometry case). Mirrors CPU
// ExchangeField::accumulate per-bond logic exactly.
// ===========================================================================

// Neighbour index with Neumann/periodic BC and mask-interface clamp.
// Returns `self` when the bond should carry zero flux (boundary or vacuum).
__device__ __forceinline__ int exch_neighbor(
    int ix, int iy, int iz, int dix, int diy, int diz,
    int nx, int ny, int nz, int self, bool periodic,
    const double* __restrict__ mask)
{
    int ni = ix + dix, nj = iy + diy, nk = iz + diz;
    const bool out = (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz);
    if (!out) {
        const int nidx = ni + nx * (nj + ny * nk);
        if (mask && mask[nidx] < 0.5) return self;   // vacuum neighbour → no flux
        return nidx;
    }
    if (!periodic) return self;                       // Neumann edge → no flux
    ni = (ni % nx + nx) % nx;
    nj = (nj % ny + ny) % ny;
    nk = (nk % nz + nz) % nz;
    return ni + nx * (nj + ny * nk);
}

__global__ static void exchange_kernel_general(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    const double* __restrict__ d_A,      // null → uniform A
    const double* __restrict__ d_Ms,     // null → uniform Ms
    double A_unif, double Ms_unif,
    const double* __restrict__  d_mask,  // null → no geometry
    const uint8_t* __restrict__ d_region,// null → no regions
    const double* __restrict__  d_inter, // null → no inter-exchange table
    int nx, int ny, int nz,
    double mu0, double idx2, double idy2, double idz2,
    bool periodic)
{
    const int N   = nx * ny * nz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (d_mask && d_mask[idx] < 0.5) return;          // vacuum cell → skip

    const double Ms_c = d_Ms ? d_Ms[idx] : Ms_unif;
    if (Ms_c <= 0.0) return;
    const double A_c   = d_A ? d_A[idx] : A_unif;
    const double pre_c = 2.0 / (mu0 * Ms_c);
    const int    rid_c = d_region ? (int)d_region[idx] : 0;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    int nb[6];
    nb[0] = exch_neighbor(ix,iy,iz, +1,0,0, nx,ny,nz, idx, periodic, d_mask);
    nb[1] = exch_neighbor(ix,iy,iz, -1,0,0, nx,ny,nz, idx, periodic, d_mask);
    nb[2] = exch_neighbor(ix,iy,iz, 0,+1,0, nx,ny,nz, idx, periodic, d_mask);
    nb[3] = exch_neighbor(ix,iy,iz, 0,-1,0, nx,ny,nz, idx, periodic, d_mask);
    nb[4] = exch_neighbor(ix,iy,iz, 0,0,+1, nx,ny,nz, idx, periodic, d_mask);
    nb[5] = exch_neighbor(ix,iy,iz, 0,0,-1, nx,ny,nz, idx, periodic, d_mask);
    const double ih2[6] = {idx2, idx2, idy2, idy2, idz2, idz2};

    double bondA[6];
    for (int b = 0; b < 6; ++b) {
        const int in = nb[b];
        double A_b;
        bool used_iec = false;
        if (d_region) {
            const int rid_n = (int)d_region[in];
            if (rid_n != rid_c && d_inter) {
                const double iec = d_inter[rid_c * 256 + rid_n];
                if (!isnan(iec)) { A_b = iec; used_iec = true; }
            }
        }
        if (!used_iec) {
            const double A_n = d_A ? d_A[in] : A_unif;
            const double s   = A_c + A_n;
            A_b = (s > 0.0) ? (2.0 * A_c * A_n / s) : 0.0;  // harmonic mean
        }
        bondA[b] = A_b;
    }

    for (int c = 0; c < 3; ++c) {
        const int base = c * N;
        const double mc = m[base + idx];
        double acc = 0.0;
        for (int b = 0; b < 6; ++b)
            acc += (m[base + nb[b]] - mc) * bondA[b] * ih2[b];
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
    CUDA_CHECK(cudaMalloc(&d_m_scratch_, 3 * N_ * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_, 3 * N_ * sizeof(GReal)));

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

ExchangeFieldGPU::~ExchangeFieldGPU() {
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (d_A_field_)  cudaFree(d_A_field_);
    if (d_Ms_field_) cudaFree(d_Ms_field_);
    if (d_mask_)     cudaFree(d_mask_);
    if (d_region_)   cudaFree(d_region_);
    if (d_inter_)    cudaFree(d_inter_);
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

// ===========================================================================
// accumulate ??upload m, run kernel, download H and add to H_out
// ===========================================================================
void ExchangeFieldGPU::accumulate(const VectorField3D& m,
                                   const Material& mat,
                                   VectorField3D& H_out) const {
    // Per-cell A or geometry can contribute even when uniform A_exchange is 0.
    if (mat.A_exchange == 0.0 && !d_A_field_ && !has_geometry()) return;

    const cudaStream_t s   = static_cast<cudaStream_t>(stream_);
    const double pre = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);
    const double fx  = pre / (dx_ * dx_);
    const double fy  = pre / (dy_ * dy_);
    const double fz  = pre / (dz_ * dz_);

    // ------------------------------------------------------------------
    // 1. Pack VectorField3D ??compact host buffer [Mx|My|Mz]
    // ------------------------------------------------------------------
    std::vector<GReal> h_m(3 * N_);
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_m[i]           = static_cast<GReal>(m[i].x);
        h_m[N_  + i]     = static_cast<GReal>(m[i].y);
        h_m[2*N_ + i]    = static_cast<GReal>(m[i].z);
    }

    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);

    // ------------------------------------------------------------------
    // 2. Upload m, zero H scratch
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));

    // ------------------------------------------------------------------
    // 3. Launch exchange kernel (Neumann or Periodic BC)
    // ------------------------------------------------------------------
    if (has_geometry() || d_A_field_) {
        // Geometry/region/per-cell path → general kernel
        const int blk = 256;
        const int grd = static_cast<int>((N_ + blk - 1) / blk);
        exchange_kernel_general<<<grd, blk, 0, s>>>(
            reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
            d_A_field_, d_Ms_field_, mat.A_exchange, mat.Ms,
            d_mask_, d_region_, d_inter_,
            (int)nx_, (int)ny_, (int)nz_,
            constants::mu_0, 1.0/(dx_*dx_), 1.0/(dy_*dy_), 1.0/(dz_*dz_),
            bc_ == BoundaryCondition::Periodic);
    } else if (bc_ == BoundaryCondition::Periodic) {
        const int blk = 256;
        const int grd = static_cast<int>((N_ + blk - 1) / blk);
        exchange_kernel_periodic<<<grd, blk, 0, s>>>(
            reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
            (int)nx_, (int)ny_, (int)nz_, fx, fy, fz);
    } else {
        // 2D shared-memory kernel: better cache locality for x/y neighbors
        const dim3 blk2(32, 8, 1);
        const dim3 grd2(((int)nx_ + 31) / 32, ((int)ny_ + 7) / 8, (int)nz_);
        exchange_kernel_smem<<<grd2, blk2, 0, s>>>(
            reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
            (int)nx_, (int)ny_, (int)nz_, fx, fy, fz);
    }
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------------
    // 4. Download H result and accumulate into H_out
    // ------------------------------------------------------------------
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

// ===========================================================================
// accumulate_gpu_ptr ??direct GPU-pointer path (no PCIe; used in G6 pipeline)
//
// Caller is responsible for zeroing d_H_out before this call (or relying on
// prior accumulation order). Uses the same stream_ as accumulate().
// ===========================================================================
void ExchangeFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                            const Material& mat,
                                            GReal* d_H_out) const {
    const cudaStream_t s   = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    const double idx2 = 1.0 / (dx_ * dx_);
    const double idy2 = 1.0 / (dy_ * dy_);
    const double idz2 = 1.0 / (dz_ * dz_);

    auto* gm  = reinterpret_cast<const GReal*>(d_m);
    auto* gH  = reinterpret_cast<GReal*>(d_H_out);

    // Geometry (mask) or region-pair coupling → general kernel (handles
    // per-cell A/Ms too). Checked first so mask+per-cell combine correctly.
    if (has_geometry()) {
        exchange_kernel_general<<<grd, blk, 0, s>>>(
            gH, gm, d_A_field_, d_Ms_field_,
            mat.A_exchange, mat.Ms,
            d_mask_, d_region_, d_inter_,
            (int)nx_, (int)ny_, (int)nz_,
            constants::mu_0, idx2, idy2, idz2,
            bc_ == BoundaryCondition::Periodic);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (d_A_field_) {
        // Per-cell mode
        const double mu0_inv2 = 2.0 / constants::mu_0;
        if (bc_ == BoundaryCondition::Periodic)
            exchange_kernel_percell_periodic<<<grd, blk, 0, s>>>(
                gH, gm, d_A_field_, d_Ms_field_,
                (int)nx_, (int)ny_, (int)nz_, mu0_inv2, idx2, idy2, idz2);
        else
            exchange_kernel_percell<<<grd, blk, 0, s>>>(
                gH, gm, d_A_field_, d_Ms_field_,
                (int)nx_, (int)ny_, (int)nz_, mu0_inv2, idx2, idy2, idz2);
    } else {
        // Uniform mode
        if (mat.A_exchange == 0.0) return;
        const double pre = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);
        if (bc_ == BoundaryCondition::Periodic)
            exchange_kernel_periodic<<<grd, blk, 0, s>>>(
                gH, gm, (int)nx_, (int)ny_, (int)nz_,
                pre * idx2, pre * idy2, pre * idz2);
        else {
            const dim3 blk2(32, 8, 1);
            const dim3 grd2(((int)nx_ + 31) / 32, ((int)ny_ + 7) / 8, (int)nz_);
            exchange_kernel_smem<<<grd2, blk2, 0, s>>>(
                gH, gm, (int)nx_, (int)ny_, (int)nz_,
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
// Geometry mask
// ===========================================================================
void ExchangeFieldGPU::set_mask(const GeomMask& mask) {
    if (!d_mask_)
        CUDA_CHECK(cudaMalloc(&d_mask_, N_ * sizeof(double)));
    std::vector<double> h_mask(N_);
    for (size_t i = 0; i < N_; ++i)
        h_mask[i] = static_cast<double>(mask[static_cast<Index>(i)]);
    CUDA_CHECK(cudaMemcpyAsync(d_mask_, h_mask.data(), N_ * sizeof(double),
                               cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream_)));
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

void ExchangeFieldGPU::clear_mask() {
    if (d_mask_) { cudaFree(d_mask_); d_mask_ = nullptr; }
}

// ===========================================================================
// Region map + inter-region exchange
// ===========================================================================
void ExchangeFieldGPU::set_region_map(const RegionMap& rm) {
    if (!d_region_)
        CUDA_CHECK(cudaMalloc(&d_region_, N_ * sizeof(uint8_t)));
    std::vector<uint8_t> h_r(N_);
    for (size_t i = 0; i < N_; ++i)
        h_r[i] = rm[static_cast<Index>(i)];
    CUDA_CHECK(cudaMemcpyAsync(d_region_, h_r.data(), N_ * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream_)));
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

void ExchangeFieldGPU::clear_region_map() {
    if (d_region_) { cudaFree(d_region_); d_region_ = nullptr; }
}

void ExchangeFieldGPU::upload_inter_table() {
    if (inter_A_.empty()) {
        if (d_inter_) { cudaFree(d_inter_); d_inter_ = nullptr; }
        return;
    }
    if (!d_inter_)
        CUDA_CHECK(cudaMalloc(&d_inter_, 256 * 256 * sizeof(double)));
    // NaN = "unset" (matches CPU lookup_inter convention); fill set pairs both ways.
    std::vector<double> h(256 * 256, std::numeric_limits<double>::quiet_NaN());
    for (const auto& kv : inter_A_) {
        const uint8_t lo = static_cast<uint8_t>(kv.first / 256u);
        const uint8_t hi = static_cast<uint8_t>(kv.first % 256u);
        h[lo * 256 + hi] = kv.second;
        h[hi * 256 + lo] = kv.second;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_inter_, h.data(), 256 * 256 * sizeof(double),
                               cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(stream_)));
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

void ExchangeFieldGPU::set_inter_exchange(uint8_t ri, uint8_t rj, Real A_IEC) {
    const uint8_t lo = ri < rj ? ri : rj;
    const uint8_t hi = ri < rj ? rj : ri;
    inter_A_[static_cast<uint32_t>(lo) * 256u + hi] = A_IEC;
    upload_inter_table();
}

Real ExchangeFieldGPU::inter_exchange(uint8_t ri, uint8_t rj) const {
    const uint8_t lo = ri < rj ? ri : rj;
    const uint8_t hi = ri < rj ? rj : ri;
    auto it = inter_A_.find(static_cast<uint32_t>(lo) * 256u + hi);
    return (it != inter_A_.end()) ? it->second : Real{-1};
}

void ExchangeFieldGPU::clear_inter_exchange() {
    inter_A_.clear();
    if (d_inter_) { cudaFree(d_inter_); d_inter_ = nullptr; }
}

// ===========================================================================
// energy ??delegates to CPU ExchangeField (G3+ will add GPU reduction)
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

