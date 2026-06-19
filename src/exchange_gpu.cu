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
// Constructor / Destructor
// ===========================================================================
ExchangeFieldGPU::ExchangeFieldGPU(const StructuredGrid& grid)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz()))
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
    // 3. Launch exchange kernel
    // ------------------------------------------------------------------
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    exchange_kernel<<<grd, blk, 0, s>>>(
        dH, dm, (int)nx_, (int)ny_, (int)nz_, fx, fy, fz);
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
    if (mat.A_exchange == 0.0) return;

    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const double pre = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);
    const double fx  = pre / (dx_ * dx_);
    const double fy  = pre / (dy_ * dy_);
    const double fz  = pre / (dz_ * dz_);

    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    exchange_kernel<<<grd, blk, 0, s>>>(
        d_H_out, d_m, (int)nx_, (int)ny_, (int)nz_, fx, fy, fz);
    CUDA_CHECK(cudaGetLastError());
    // No sync — caller owns the stream synchronisation
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
