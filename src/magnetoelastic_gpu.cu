// magnetoelastic_gpu.cu ??GPU magnetoelastic field
//
// CUDA kernel: all N cells in parallel.
// Memory layout: [3횞N] component-major, buf[c*N + idx].

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include "micromag/cuda_sync_debug.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "micromag/gpu_real.hpp"
#include "micromag/magnetoelastic.hpp"
#include "micromag/magnetoelastic_gpu.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess)                                            \
            throw std::runtime_error(std::string("CUDA(me): ")           \
                                   + cudaGetErrorString(_e));             \
    } while (0)

namespace micromag {

// ===========================================================================
// CUDA kernel
// ===========================================================================
__global__ static void magnetoelastic_kernel(
    GReal* __restrict__       H_out,  // [3횞N] accumulate
    const GReal* __restrict__ m,      // [3횞N]
    double p1, double p2,              // -2B1/(mu0Ms), -2B2/(mu0Ms)
    double exx, double eyy, double ezz,
    double exy, double exz, double eyz,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double mx = m[0*N + idx];
    const double my = m[1*N + idx];
    const double mz = m[2*N + idx];
    H_out[0*N + idx] += p1*mx*exx + p2*(my*exy + mz*exz);
    H_out[1*N + idx] += p1*my*eyy + p2*(mx*exy + mz*eyz);
    H_out[2*N + idx] += p1*mz*ezz + p2*(mx*exz + my*eyz);
}

// ===========================================================================
// Constructor / Destructor
// ===========================================================================
MagnetoelasticFieldGPU::MagnetoelasticFieldGPU(Real B1, Real B2,
                                                 const StructuredGrid& grid)
    : B1_(B1), B2_(B2),
      N_(static_cast<size_t>(grid.nx()) *
         static_cast<size_t>(grid.ny()) *
         static_cast<size_t>(grid.nz()))
{
    CUDA_CHECK(cudaMalloc(&d_m_scratch_, 3 * N_ * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H_scratch_, 3 * N_ * sizeof(GReal)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

MagnetoelasticFieldGPU::~MagnetoelasticFieldGPU() {
    cudaFree(d_m_scratch_);
    cudaFree(d_H_scratch_);
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void MagnetoelasticFieldGPU::set_strain(Real exx, Real eyy, Real ezz,
                                          Real exy, Real exz, Real eyz)
{
    exx_ = exx; eyy_ = eyy; ezz_ = ezz;
    exy_ = exy; exz_ = exz; eyz_ = eyz;
}

// ===========================================================================
// accumulate (CPU path: upload m, kernel, download H contribution)
// ===========================================================================
void MagnetoelasticFieldGPU::accumulate(const VectorField3D& m,
                                          const Material& mat,
                                          VectorField3D& H_out) const {
    const Real Ms = mat.Ms;
    if (Ms == Real{0}) return;

    const double prefac = -2.0 / (constants::mu_0 * static_cast<double>(Ms));
    const double p1 = prefac * static_cast<double>(B1_);
    const double p2 = prefac * static_cast<double>(B2_);

    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    auto* dm = static_cast<GReal*>(d_m_scratch_);
    auto* dH = static_cast<GReal*>(d_H_scratch_);

    // Pack VectorField3D ??component-major host buffer
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
    magnetoelastic_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(dH), reinterpret_cast<const GReal*>(dm),
        p1, p2,
        static_cast<double>(exx_), static_cast<double>(eyy_),
        static_cast<double>(ezz_), static_cast<double>(exy_),
        static_cast<double>(exz_), static_cast<double>(eyz_),
        static_cast<int>(N_));
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
// accumulate_gpu_ptr (direct GPU path for future integrator integration)
// ===========================================================================
void MagnetoelasticFieldGPU::accumulate_gpu_ptr(const GReal* d_m,
                                                  const Material& mat,
                                                  GReal* d_H_out) const {
    const Real Ms = mat.Ms;
    if (Ms == Real{0}) return;

    const double prefac = -2.0 / (constants::mu_0 * static_cast<double>(Ms));
    const double p1 = prefac * static_cast<double>(B1_);
    const double p2 = prefac * static_cast<double>(B2_);

    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int blk = 256;
    const int grd = static_cast<int>((N_ + blk - 1) / blk);
    magnetoelastic_kernel<<<grd, blk, 0, s>>>(
        reinterpret_cast<GReal*>(d_H_out),
        reinterpret_cast<const GReal*>(d_m),
        p1, p2,
        static_cast<double>(exx_), static_cast<double>(eyy_),
        static_cast<double>(ezz_), static_cast<double>(exy_),
        static_cast<double>(exz_), static_cast<double>(eyz_),
        static_cast<int>(N_));
}

// ===========================================================================
// energy (CPU delegate)
// ===========================================================================
Real MagnetoelasticFieldGPU::energy(const VectorField3D& m,
                                     const Material& mat) const {
    MagnetoelasticField cpu(B1_, B2_);
    cpu.set_strain(exx_, eyy_, ezz_, exy_, exz_, eyz_);
    return cpu.energy(m, mat);
}

}  // namespace micromag

#endif // MICROMAG_CUDA

