#pragma once

// ExchangeFieldGPU — GPU-accelerated Heisenberg exchange via 6-point Laplacian.
// Same IEffectiveField interface as ExchangeField; drop-in replacement for the
// GPU build.  For Phase G (GPU full LLG), the raw kernel will also be callable
// directly without upload/download — see accumulate_gpu_ptr() in exchange_gpu.cu.
//
// Requires MICROMAG_CUDA=1 (cmake --preset windows-msvc-cuda).

#ifdef MICROMAG_CUDA

#include "effective_field.hpp"
#include "effective_field_gpu_iface.hpp"
#include "exchange.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "material_field.hpp"
#include "types.hpp"

namespace micromag {

class ExchangeFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    // bc = Neumann: zero-flux boundary (open systems).
    // bc = Periodic: wrap-around boundary (periodic supercell, e.g. with DemagFieldPeriodicGPU).
    explicit ExchangeFieldGPU(const StructuredGrid& grid,
                               BoundaryCondition bc = BoundaryCondition::Neumann);
    ~ExchangeFieldGPU();

    // Non-copyable
    ExchangeFieldGPU(const ExchangeFieldGPU&)            = delete;
    ExchangeFieldGPU& operator=(const ExchangeFieldGPU&) = delete;

    // IEffectiveField: uploads m, runs GPU kernel, accumulates result into H_out
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "ExchangeFieldGPU"; }

    // Direct GPU-pointer path (G6 pipeline). Adds H_exch to d_H_out in-place.
    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const;

    // Redirect all kernels to an external stream (P2 single-stream).
    // Sets stream_owned_=false so the destructor does not destroy the external stream.
    void set_stream(void* s) {
        if (s == stream_) return;
        if (stream_owned_ && stream_)
            cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_       = s;
        stream_owned_ = false;
    }

    // Per-cell material: uploads A_exchange and Ms from MaterialField3D to device.
    // Once set, per-cell mode is active; call clear_material_field() to revert.
    void set_material_field(const MaterialField3D& matf);
    void clear_material_field();
    bool has_material_field() const { return d_A_field_ != nullptr; }

    // Cell-size and BC accessors — used by fused local-field kernel in the integrator.
    Real dx() const { return dx_; }
    Real dy() const { return dy_; }
    Real dz() const { return dz_; }
    BoundaryCondition bc() const { return bc_; }

private:
    Index  nx_, ny_, nz_;
    Real   dx_, dy_, dz_;
    size_t N_;               // nx * ny * nz

    // GPU scratch buffers (allocated once, reused per accumulate)
    void* d_m_scratch_ = nullptr;   // double[3 × N]
    void* d_H_scratch_ = nullptr;   // double[3 × N]

    // Per-cell material buffers (null = uniform mode)
    double* d_A_field_  = nullptr;  // double[N] — A_exchange per cell
    double* d_Ms_field_ = nullptr;  // double[N] — Ms per cell

    // CUDA stream — all GPU work serialised here
    void* stream_       = nullptr;
    bool  stream_owned_ = true;

    BoundaryCondition bc_;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
