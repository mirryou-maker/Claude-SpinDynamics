#pragma once
// effective_field_gpu_iface.hpp — Abstract interface for GPU effective fields.
//
// IEffectiveFieldGPU: any GPU field that can run via raw device pointers.
// FieldSumGPU: ordered list of IEffectiveFieldGPU*, runs them all in sequence.
//
// This enables arbitrary field combinations in GPU integrators without
// recompiling when new GPU fields are added.
//
// Usage:
//   FieldSumGPU fields;
//   fields.add(exch_gpu);      // ExchangeFieldGPU
//   fields.add(zeeman_gpu);    // ZeemanFieldGPU
//   fields.add(dmi_gpu);       // BulkDMIFieldGPU or InterfacialDMIFieldGPU
//   integ.step(mat, demag_gpu, fields);  // new overload

#ifdef MICROMAG_CUDA

#include "gpu_real.hpp"
#include "material.hpp"
#include <vector>
#include <cuda_runtime.h>

namespace micromag {

// ---------------------------------------------------------------------------
// IEffectiveFieldGPU — pure virtual interface for all GPU effective fields.
// Implemented by: ExchangeFieldGPU, ZeemanFieldGPU, UniaxialAnisotropyFieldGPU,
//   CubicAnisotropyFieldGPU, BulkDMIFieldGPU, InterfacialDMIFieldGPU,
//   MagnetoelasticFieldGPU, SurfaceAnisotropyFieldGPU.
//
// P11: pointer type uses GReal (float when MICROMAG_FLOAT32=ON, else double).
// ---------------------------------------------------------------------------
class IEffectiveFieldGPU {
public:
    virtual ~IEffectiveFieldGPU() = default;

    // Add H_field to d_H_out in-place (both [3×N] component-major, on device).
    virtual void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                                     GReal* d_H_out) const = 0;
};

// ---------------------------------------------------------------------------
// FieldSumGPU — ordered compositor of IEffectiveFieldGPU pointers.
// ---------------------------------------------------------------------------
class FieldSumGPU {
public:
    FieldSumGPU() = default;

    // Add any GPU field (must outlive FieldSumGPU — stores raw pointer).
    void add(IEffectiveFieldGPU& f) { fields_.push_back(&f); }

    void clear() { fields_.clear(); }

    std::size_t size() const { return fields_.size(); }

    // Accumulate all fields into d_H_out (must be pre-zeroed by caller).
    //
    // Each field runs on its OWN CUDA stream and accumulates in-place
    // (d_H_out += H_field).  Without a barrier between fields the read-
    // modify-write of d_H_out races across the independent streams, so some
    // cells silently lose a field's contribution — worst at high-field
    // boundary cells, which injects spurious energy dissipation into the LLG
    // dynamics.  Serialise with a device sync after each field.
    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const {
        for (auto* f : fields_) {
            f->accumulate_gpu_ptr(d_m, mat, d_H_out);
            cudaDeviceSynchronize();
        }
    }

private:
    std::vector<IEffectiveFieldGPU*> fields_;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
