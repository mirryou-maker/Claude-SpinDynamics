#pragma once

// rkky_gpu.hpp — GPU drop-in for RKKYField.
//
// Models antiferromagnetic/ferromagnetic interlayer coupling for SAF stacks
// and bilayer structures.  Keeps the reference layer on the GPU device;
// the caller uploads it via set_ref(ref_m) before each step or whenever
// the reference layer changes.
//
// Formula:
//   H_RKKY[i] += -J / (mu_0 * Ms * d) * m_ref[i]
//
// Usage:
//   RKKYFieldGPU rkky1(grid, J, d);
//   RKKYFieldGPU rkky2(grid, J, d);
//   rkky1.set_ref(m2_cpu);        // reference = layer 2
//   rkky2.set_ref(m1_cpu);        // reference = layer 1
//   fsum1.add(rkky1); fsum2.add(rkky2);
//   // Alternate GPU steps:
//   integ1.step(mat, demag, fsum1);
//   rkky1.set_ref(m2_download); ...

#ifdef MICROMAG_CUDA

#include "effective_field.hpp"
#include "effective_field_gpu_iface.hpp"
#include "field.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

class RKKYFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    // J_RKKY  [J/m²]  — coupling constant (< 0 = antiferromagnetic)
    // d_spacer [m]    — spacer thickness
    RKKYFieldGPU(const StructuredGrid& grid, Real J_RKKY, Real d_spacer);
    ~RKKYFieldGPU();

    RKKYFieldGPU(const RKKYFieldGPU&)            = delete;
    RKKYFieldGPU& operator=(const RKKYFieldGPU&) = delete;

    // Upload reference magnetization from CPU VectorField3D to device.
    // Must be called before the first step and whenever ref layer updates.
    void set_ref(const VectorField3D& ref_m);

    // IEffectiveField (CPU fallback; reads d_ref_ via D2H)
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    const char* name() const override { return "RKKYFieldGPU"; }

    // IEffectiveFieldGPU (full-GPU path; adds to d_H_out directly)
    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const override;

    void   set_stream(void* s) { stream_ = s; }
    Real   J()        const    { return J_RKKY_; }
    Real   d()        const    { return d_spacer_; }
    void   set_J(Real J)       { J_RKKY_ = J; }

private:
    size_t N_;
    Real   J_RKKY_;
    Real   d_spacer_;
    double* d_ref_ = nullptr;   // [3×N] component-major reference layer on device
    void*   stream_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
