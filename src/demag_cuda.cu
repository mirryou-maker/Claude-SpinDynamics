// demag_cuda.cu — Phase 3: cuFFT GPU demag implementation
// Step 1: Stub that compiles and links. Real implementation in Step 4.
//
// Build with:  cmake --preset windows-msvc-cuda
//              cmake --build build/windows-msvc-cuda --config Release

#ifdef MICROMAG_CUDA

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "micromag/demag_gpu.hpp"
#include "micromag/demag.hpp"    // reuse nxx/nxy for kernel computation

namespace micromag {

// ---------------------------------------------------------------------------
// Constructor — allocate GPU memory, create cuFFT plans, precompute kernel
// ---------------------------------------------------------------------------
DemagFieldGPU::DemagFieldGPU(const StructuredGrid& grid)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz())
{
    pad_nx_ = 2 * nx_;
    pad_ny_ = 2 * ny_;
    pad_nz_ = 2 * nz_;
    fft_nx_ = pad_nx_ / 2 + 1;

    // TODO (Step 4): replace with actual GPU allocation and cuFFT plan creation
    // For Step 1 this is a stub — precompute_kernel() will be a no-op.
    precompute_kernel();
}

DemagFieldGPU::~DemagFieldGPU() {
    // TODO (Step 4): cudaFree all d_* pointers, cufftDestroy plans
}

// ---------------------------------------------------------------------------
// precompute_kernel — CPU computes Newell kernel, uploads to GPU
// ---------------------------------------------------------------------------
void DemagFieldGPU::precompute_kernel() {
    // TODO (Step 4): implement 6D Newell kernel → FFT → upload to GPU
}

// ---------------------------------------------------------------------------
// accumulate — GPU FFT convolution (Step 1 stub: falls back to CPU)
// ---------------------------------------------------------------------------
void DemagFieldGPU::accumulate(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    // STUB: delegate to CPU implementation until Step 4 is complete.
    // This ensures DemagFieldGPU compiles and links correctly.
    const StructuredGrid& g = m.grid();
    DemagField cpu(g);
    cpu.accumulate(m, mat, H_out);
}

Real DemagFieldGPU::energy(const VectorField3D& m,
                             const Material& mat) const {
    // STUB
    const StructuredGrid& g = m.grid();
    DemagField cpu(g);
    return cpu.energy(m, mat);
}

}  // namespace micromag

#endif // MICROMAG_CUDA
