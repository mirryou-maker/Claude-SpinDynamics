// surface_anisotropy_gpu.cu ??GPU surface / interface anisotropy field
//
// CUDA kernel applies H_s = (2Ks/쨉?Ms*t)(m쨌n)n only to surface cells
// identified by the precomputed d_is_surface mask.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include "micromag/cuda_sync_debug.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/field.hpp"
#include "micromag/gpu_real.hpp"
#include "micromag/surface_anisotropy.hpp"
#include "micromag/surface_anisotropy_gpu.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess)                                             \
            throw std::runtime_error(std::string("CUDA(sa_gpu): ")        \
                                   + cudaGetErrorString(_e));              \
    } while (0)

namespace micromag {

// ===========================================================================
// CUDA kernel
// ===========================================================================
__global__ static void surface_anisotropy_kernel(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    const int*   __restrict__  is_surface,
    double prefac,
    double nx, double ny, double nz,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!is_surface[idx]) return;

    const double mx = m[0*N + idx];
    const double my = m[1*N + idx];
    const double mz = m[2*N + idx];
    const double mn = mx*nx + my*ny + mz*nz;
    const double h  = prefac * mn;
    H_out[0*N + idx] += h * nx;
    H_out[1*N + idx] += h * ny;
    H_out[2*N + idx] += h * nz;
}

// Per-cell Ks kernel (only fires for surface cells)
__global__ static void surface_anisotropy_kernel_percell(
    GReal* __restrict__       H_out,
    const GReal* __restrict__ m,
    const int*   __restrict__  is_surface,
    const double* __restrict__ d_Ks,
    const double* __restrict__ d_Ms,
    double mu0_inv2,
    double t_cell_inv,
    double nx, double ny, double nz,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!is_surface[idx]) return;
    const double Ms_i = d_Ms[idx];
    if (Ms_i <= 0.0) return;
    const double prefac = d_Ks[idx] * mu0_inv2 / (Ms_i * t_cell_inv);
    const double mn = m[0*N+idx]*nx + m[1*N+idx]*ny + m[2*N+idx]*nz;
    const double h  = prefac * mn;
    H_out[0*N+idx] += h * nx;
    H_out[1*N+idx] += h * ny;
    H_out[2*N+idx] += h * nz;
}

// ===========================================================================
// Helpers
// ===========================================================================
static Vec3 normalise_vec(Vec3 v)
{
    const Real len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return (len > Real{0}) ? Vec3{v.x/len, v.y/len, v.z/len} : Vec3{0,0,1};
}

Real SurfaceAnisotropyFieldGPU::compute_t_cell() const
{
    return std::abs(n_.x)*dx_ + std::abs(n_.y)*dy_ + std::abs(n_.z)*dz_;
}

bool SurfaceAnisotropyFieldGPU::is_surface_cell(Index ix, Index iy, Index iz) const
{
    const Real ax = std::abs(n_.x), ay = std::abs(n_.y), az = std::abs(n_.z);

    if (geom_mask_ != nullptr) {
        auto outside = [&](Index ii, Index jj, Index kk) -> bool {
            if (ii < 0 || ii >= nx_ || jj < 0 || jj >= ny_ ||
                kk < 0 || kk >= nz_)
                return true;
            return (*geom_mask_)[ii + nx_*(jj + ny_*kk)] < Real{0.5};
        };
        if (az >= ay && az >= ax)
            return outside(ix, iy, iz-1) || outside(ix, iy, iz+1);
        else if (ay >= ax)
            return outside(ix, iy-1, iz) || outside(ix, iy+1, iz);
        else
            return outside(ix-1, iy, iz) || outside(ix+1, iy, iz);
    } else {
        if (az >= ay && az >= ax)
            return (iz == 0) || (iz == nz_ - 1);
        else if (ay >= ax)
            return (iy == 0) || (iy == ny_ - 1);
        else
            return (ix == 0) || (ix == nx_ - 1);
    }
}

void SurfaceAnisotropyFieldGPU::rebuild_surface_mask()
{
    t_cell_ = compute_t_cell();
    std::vector<int> h_surf(N_, 0);
    for (Index iz = 0; iz < nz_; ++iz)
    for (Index iy = 0; iy < ny_; ++iy)
    for (Index ix = 0; ix < nx_; ++ix) {
        Index lin = ix + nx_*(iy + ny_*iz);
        if (geom_mask_ && (*geom_mask_)[lin] < Real{0.5}) continue;
        if (is_surface_cell(ix, iy, iz)) h_surf[lin] = 1;
    }
    CUDA_CHECK(cudaMemcpy(static_cast<int*>(d_is_surface_),
                          h_surf.data(), N_*sizeof(int),
                          cudaMemcpyHostToDevice));
}

// ===========================================================================
// Constructor / Destructor
// ===========================================================================
SurfaceAnisotropyFieldGPU::SurfaceAnisotropyFieldGPU(Real Ks,
                                                       const StructuredGrid& grid,
                                                       Vec3 n_hat)
    : Ks_(Ks),
      n_(normalise_vec(n_hat)),
      nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      t_cell_(Real{0}),
      N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz()))
{
    CUDA_CHECK(cudaMalloc(&d_is_surface_, N_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_m_scratch_,  3 * N_ * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_,  3 * N_ * sizeof(GReal)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
    rebuild_surface_mask();
}

SurfaceAnisotropyFieldGPU::~SurfaceAnisotropyFieldGPU()
{
    cudaFree(d_is_surface_);
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (d_Ks_field_) cudaFree(d_Ks_field_);
    if (d_Ms_field_) cudaFree(d_Ms_field_);
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void SurfaceAnisotropyFieldGPU::set_n_hat(Vec3 n)
{
    n_ = normalise_vec(n);
    rebuild_surface_mask();
}

// ===========================================================================
// accumulate ??CPU interface path (upload m, kernel, download H)
// ===========================================================================
void SurfaceAnisotropyFieldGPU::accumulate(const VectorField3D& m,
                                             const Material& mat,
                                             VectorField3D& H_out) const
{
    const Real Ms = mat.Ms;
    if (Ms == Real{0} || t_cell_ == Real{0}) return;

    const double prefac = 2.0 * static_cast<double>(Ks_)
                        / (constants::mu_0 * static_cast<double>(Ms)
                         * static_cast<double>(t_cell_));

    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    auto* dm  = static_cast<GReal*>(d_m_scratch_);
    auto* dH  = static_cast<GReal*>(d_H_scratch_);
    auto* dIS = static_cast<int*>(d_is_surface_);

    std::vector<GReal> h_m(3 * N_);
    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        h_m[i]        = static_cast<GReal>(m[i].x);
        h_m[N_   + i] = static_cast<GReal>(m[i].y);
        h_m[2*N_ + i] = static_cast<GReal>(m[i].z);
    }

    CUDA_CHECK(cudaMemcpy(dm, h_m.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dH, 0, 3*N_*sizeof(GReal)));

    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    surface_anisotropy_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        dIS, prefac,
        static_cast<double>(n_.x), static_cast<double>(n_.y),
        static_cast<double>(n_.z), static_cast<int>(N_));
    MICROMAG_KERNEL_CHECK();

    std::vector<GReal> h_H(3 * N_);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(h_H.data(), dH, 3*N_*sizeof(GReal), cudaMemcpyDeviceToHost));

    for (Index i = 0; i < static_cast<Index>(N_); ++i) {
        H_out[i].x += h_H[i];
        H_out[i].y += h_H[N_   + i];
        H_out[i].z += h_H[2*N_ + i];
    }
}

// ===========================================================================
// accumulate_gpu_ptr ??direct GPU pointer path
// ===========================================================================
void SurfaceAnisotropyFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                                     const Material& mat,
                                                     GReal* d_H_out) const
{
    if (t_cell_ == Real{0}) return;
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    auto* dIS = static_cast<int*>(d_is_surface_);

    auto* gm = reinterpret_cast<const GReal*>(d_m);
    auto* gH = reinterpret_cast<GReal*>(d_H_out);

    if (d_Ks_field_) {
        // Per-cell mode: prefac_i = 2*Ks_i/(mu0*Ms_i*t_cell)
        const double mu0_inv2   = 2.0 / constants::mu_0;
        const double t_cell_inv = 1.0 / static_cast<double>(t_cell_);
        surface_anisotropy_kernel_percell<<<grd, blk, 0, s>>>(
            gH, gm, dIS, d_Ks_field_, d_Ms_field_,
            mu0_inv2, t_cell_inv,
            static_cast<double>(n_.x), static_cast<double>(n_.y),
            static_cast<double>(n_.z), static_cast<int>(N_));
    } else {
        const Real Ms = mat.Ms;
        if (Ms == Real{0}) return;
        const double prefac = 2.0 * static_cast<double>(Ks_)
                            / (constants::mu_0 * static_cast<double>(Ms)
                             * static_cast<double>(t_cell_));
        surface_anisotropy_kernel<<<grd, blk, 0, s>>>(
            gH, gm, dIS, prefac,
            static_cast<double>(n_.x), static_cast<double>(n_.y),
            static_cast<double>(n_.z), static_cast<int>(N_));
    }
    MICROMAG_KERNEL_CHECK();
}

void SurfaceAnisotropyFieldGPU::set_Ks_field(const ScalarField3D& Ks_f,
                                               const ScalarField3D& Ms_f) {
    if (!d_Ks_field_) {
        CUDA_CHECK(cudaMalloc(&d_Ks_field_, N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ms_field_, N_ * sizeof(double)));
    }
    std::vector<double> h_Ks(N_), h_Ms(N_);
    for (size_t i = 0; i < N_; ++i) {
        h_Ks[i] = Ks_f[static_cast<Index>(i)];
        h_Ms[i] = Ms_f[static_cast<Index>(i)];
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_Ks_field_, h_Ks.data(), N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_Ms_field_, h_Ms.data(), N_*sizeof(double), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void SurfaceAnisotropyFieldGPU::clear_Ks_field() {
    if (d_Ks_field_) { cudaFree(d_Ks_field_); d_Ks_field_ = nullptr; }
    if (d_Ms_field_) { cudaFree(d_Ms_field_); d_Ms_field_ = nullptr; }
}

// ===========================================================================
// energy ??CPU delegate (boundary overhead is small)
// ===========================================================================
Real SurfaceAnisotropyFieldGPU::energy(const VectorField3D& m,
                                        const Material& mat) const
{
    SurfaceAnisotropyField cpu(Ks_, n_);
    if (geom_mask_) cpu.set_mask(geom_mask_);
    return cpu.energy(m, mat);
}

}  // namespace micromag

#endif // MICROMAG_CUDA

