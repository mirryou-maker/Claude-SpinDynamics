#pragma once

// field_kernels_gpu.hpp — G2: GPU Zeeman and Uniaxial-Anisotropy fields.
//
// Both classes implement IEffectiveField (drop-in for CPU equivalents) and
// expose an accumulate_gpu_ptr() path for the G6 full-GPU LLG pipeline.
//
// Memory layout (GPU buffers): [3×N] component-major
//   buf[c*N + idx],  idx = ix + nx*(iy + ny*iz),  c ∈ {0,1,2}
// Same convention as ExchangeFieldGPU and DemagFieldGPU::d_M_compact_.

#ifdef MICROMAG_CUDA

#include "anisotropy.hpp"
#include "effective_field.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"
#include "zeeman.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// ZeemanFieldGPU
//
// accumulate(): same result as ZeemanField (CPU) — adds H_ext to every cell.
// The standalone path does this on the CPU (no upload/download needed since
// H_ext is uniform).  accumulate_gpu_ptr() launches a GPU kernel for the
// full-GPU LLG pipeline (G6).
// ---------------------------------------------------------------------------
class ZeemanFieldGPU : public IEffectiveField {
public:
    ZeemanFieldGPU(const StructuredGrid& grid, const Vec3& H_ext = {0, 0, 0});
    ~ZeemanFieldGPU();

    ZeemanFieldGPU(const ZeemanFieldGPU&)            = delete;
    ZeemanFieldGPU& operator=(const ZeemanFieldGPU&) = delete;

    // IEffectiveField — standalone path (CPU, no PCIe overhead)
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    const char* name() const override { return "ZeemanFieldGPU"; }

    // Direct GPU-pointer path for G6 pipeline; d_m is unused (Zeeman ≠ f(m))
    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;

    Vec3 H_ext() const            { return H_ext_; }
    void set_H_ext(const Vec3& H) { H_ext_ = H; }
    void set_stream(void* s)      { stream_ = s; }

private:
    size_t N_;
    Vec3   H_ext_;
    void*  stream_ = nullptr;
};

// ---------------------------------------------------------------------------
// UniaxialAnisotropyFieldGPU
//
// H_ani = (2K / μ₀Ms)(m·û)û   where û = normalised easy_axis from Material.
// accumulate(): uploads m, runs kernel, downloads H, adds to H_out.
// accumulate_gpu_ptr(): kernel only, no PCIe (for G6 pipeline).
// ---------------------------------------------------------------------------
class UniaxialAnisotropyFieldGPU : public IEffectiveField {
public:
    explicit UniaxialAnisotropyFieldGPU(const StructuredGrid& grid);
    ~UniaxialAnisotropyFieldGPU();

    UniaxialAnisotropyFieldGPU(const UniaxialAnisotropyFieldGPU&)            = delete;
    UniaxialAnisotropyFieldGPU& operator=(const UniaxialAnisotropyFieldGPU&) = delete;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    const char* name() const override { return "UniaxialAnisotropyFieldGPU"; }

    // Direct GPU-pointer path for G6 pipeline
    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;
    void set_stream(void* s) { stream_ = s; }

private:
    size_t N_;
    void*  d_m_scratch_ = nullptr;   // double[3×N]
    void*  d_H_scratch_ = nullptr;   // double[3×N]
    void*  stream_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
