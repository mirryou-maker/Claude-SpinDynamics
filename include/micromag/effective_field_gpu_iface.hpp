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

    // Redirect GPU kernels to an external stream (P2: single-stream refactor).
    // Default no-op; concrete classes override to redirect their private stream.
    virtual void set_stream(void* /*s*/) {}
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

    // Route all registered fields (and future calls) to a shared CUDA stream.
    // When set, accumulate_gpu_ptr omits cudaDeviceSynchronize() between fields
    // because stream ordering guarantees serial execution.
    // Call from the integrator at the start of each step() to enable single-stream mode.
    void set_stream(void* s) {
        stream_ = s;
        for (auto* f : fields_) f->set_stream(s);
    }

    // Accumulate all fields into d_H_out (must be pre-zeroed by caller).
    //
    // Single-stream mode (set_stream was called): all fields run on stream_,
    // stream ordering serialises them — no cudaDeviceSynchronize needed.
    //
    // Multi-stream mode (default, stream_==nullptr): each field uses its own
    // private stream; cudaDeviceSynchronize serialises after each field to
    // prevent read-modify-write races on d_H_out.
    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const {
        for (auto* f : fields_) {
            f->accumulate_gpu_ptr(d_m, mat, d_H_out);
            if (!stream_) cudaDeviceSynchronize();
        }
    }

private:
    std::vector<IEffectiveFieldGPU*> fields_;
    void* stream_ = nullptr;  // null = each field on its own stream
};

}  // namespace micromag

#endif // MICROMAG_CUDA
