#pragma once

// ExchangeFieldGPU — GPU-accelerated Heisenberg exchange via 6-point Laplacian.
// Same IEffectiveField interface as ExchangeField; drop-in replacement for the
// GPU build.  For Phase G (GPU full LLG), the raw kernel will also be callable
// directly without upload/download — see accumulate_gpu_ptr() in exchange_gpu.cu.
//
// Requires MICROMAG_CUDA=1 (cmake --preset windows-msvc-cuda).

#ifdef MICROMAG_CUDA

#include "effective_field.hpp"
#include "exchange.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

class ExchangeFieldGPU : public IEffectiveField {
public:
    explicit ExchangeFieldGPU(const StructuredGrid& grid);
    ~ExchangeFieldGPU();

    // Non-copyable
    ExchangeFieldGPU(const ExchangeFieldGPU&)            = delete;
    ExchangeFieldGPU& operator=(const ExchangeFieldGPU&) = delete;

    // IEffectiveField: uploads m, runs GPU kernel, accumulates result into H_out
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;

    // Energy: delegates to CPU ExchangeField (G3+ will add GPU reduction)
    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "ExchangeFieldGPU"; }

    // Direct GPU-pointer path (used in G6 full-GPU LLG pipeline).
    // Adds H_exch to d_H_out in-place; both pointers are [3×N] component-major.
    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;

private:
    Index  nx_, ny_, nz_;
    Real   dx_, dy_, dz_;
    size_t N_;               // nx * ny * nz

    // GPU scratch buffers (allocated once, reused per accumulate)
    void* d_m_scratch_ = nullptr;   // double[3 × N]
    void* d_H_scratch_ = nullptr;   // double[3 × N]

    // CUDA stream — all GPU work serialised here
    void* stream_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
