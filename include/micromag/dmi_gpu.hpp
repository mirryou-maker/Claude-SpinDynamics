#pragma once

// dmi_gpu.hpp — GPU-accelerated DMI fields (Bulk and Interfacial).
// Same interface as BulkDMIField / InterfacialDMIField — drop-in replacements.
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "dmi.hpp"
#include "effective_field.hpp"
#include "effective_field_gpu_iface.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// BulkDMIFieldGPU  H = (2D/μ₀Ms) ∇×m  (Bloch skyrmion, D > 0)
// ---------------------------------------------------------------------------
class BulkDMIFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    explicit BulkDMIFieldGPU(const StructuredGrid& grid, Real D = 0.0);
    ~BulkDMIFieldGPU();

    BulkDMIFieldGPU(const BulkDMIFieldGPU&)            = delete;
    BulkDMIFieldGPU& operator=(const BulkDMIFieldGPU&) = delete;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;
    const char* name() const override { return "BulkDMIGPU"; }

    // G-path: add H_DMI to d_H_out (both [3×N] component-major, on GPU).
    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const;

    Real D() const       { return D_; }
    void set_D(Real D)   { D_ = D;   }

    // Per-cell D: upload D_per_cell [J/m²] and Ms_per_cell [A/m] ScalarField3D
    // arrays to activate per-cell mode where prefac_i = 2*D_i/(mu0*Ms_i).
    // Call again to update; call clear_D_field() to revert to uniform mode.
    void set_D_field(const ScalarField3D& D_field, const ScalarField3D& Ms_field);
    void clear_D_field();
    bool has_D_field() const { return d_D_field_ != nullptr; }

private:
    Index  nx_, ny_, nz_;
    Real   dx_, dy_, dz_;
    size_t N_;
    Real   D_;

    void*   d_m_scratch_ = nullptr;
    void*   d_H_scratch_ = nullptr;
    void*   stream_      = nullptr;

    // Per-cell buffers (null = uniform mode)
    double* d_D_field_  = nullptr;  // [N] — D_bulk per cell
    double* d_Ms_field_ = nullptr;  // [N] — Ms per cell
};

// ---------------------------------------------------------------------------
// InterfacialDMIFieldGPU  H = (2D/μ₀Ms)(∂mz/∂x, ∂mz/∂y, -∇_xy·m_xy)
// ---------------------------------------------------------------------------
class InterfacialDMIFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    explicit InterfacialDMIFieldGPU(const StructuredGrid& grid, Real D = 0.0);
    ~InterfacialDMIFieldGPU();

    InterfacialDMIFieldGPU(const InterfacialDMIFieldGPU&)            = delete;
    InterfacialDMIFieldGPU& operator=(const InterfacialDMIFieldGPU&) = delete;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;
    const char* name() const override { return "InterfacialDMIGPU"; }

    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const;

    Real D() const       { return D_; }
    void set_D(Real D)   { D_ = D;   }

    void set_D_field(const ScalarField3D& D_field, const ScalarField3D& Ms_field);
    void clear_D_field();
    bool has_D_field() const { return d_D_field_ != nullptr; }

private:
    Index  nx_, ny_, nz_;
    Real   dx_, dy_, dz_;
    size_t N_;
    Real   D_;

    void*   d_m_scratch_ = nullptr;
    void*   d_H_scratch_ = nullptr;
    void*   stream_      = nullptr;

    double* d_D_field_  = nullptr;
    double* d_Ms_field_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
