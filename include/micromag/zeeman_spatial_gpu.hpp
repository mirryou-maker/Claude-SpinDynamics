#pragma once

// zeeman_spatial_gpu.hpp — GPU drop-in for ZeemanFieldSpatial.
//
// Stores a per-cell H_ext field in a [3×N] device buffer (component-major,
// same layout as all other GPU fields). Caller uploads the field once via
// set_field() or upload_field() and can update it at any time between steps.
//
// Usage (GPU LLG pipeline):
//   ZeemanFieldSpatialGPU zsGPU(grid);
//   zsGPU.set_field(H_cpu_field);      // H2D copy
//   fsum.add(zsGPU);
//   integ.step(mat, demag, fsum);      // per-cell H added GPU-side
//
//   // Time-varying: update H field between steps
//   update_H(H_cpu_field, t);
//   zsGPU.set_field(H_cpu_field);      // H2D upload
//   integ.step(mat, demag, fsum);

#ifdef MICROMAG_CUDA

#include "effective_field.hpp"
#include "effective_field_gpu_iface.hpp"
#include "field.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

class ZeemanFieldSpatialGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    explicit ZeemanFieldSpatialGPU(const StructuredGrid& grid);
    ~ZeemanFieldSpatialGPU();

    ZeemanFieldSpatialGPU(const ZeemanFieldSpatialGPU&)            = delete;
    ZeemanFieldSpatialGPU& operator=(const ZeemanFieldSpatialGPU&) = delete;

    // Upload per-cell field from CPU VectorField3D (host -> device, interleaved->component-major).
    void set_field(const VectorField3D& H_field);

    // IEffectiveField (CPU path — runs accumulate on CPU using cached host copy).
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    const char* name() const override { return "ZeemanSpatialGPU"; }

    // IEffectiveFieldGPU (full-GPU path).
    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const override;

    void set_stream(void* s) { stream_ = s; }

private:
    size_t N_;
    double* d_H_field_ = nullptr;  // [3×N] component-major on device
    VectorField3D H_host_;         // CPU copy (for accumulate() fallback + energy())
    void* stream_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
