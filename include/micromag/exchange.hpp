#pragma once

#include "effective_field.hpp"

namespace micromag {

// Forward declarations — include geom_mask.hpp / material_field.hpp for the
// full definitions.
class GeomMask;
class MaterialField3D;

// Heisenberg exchange via 6-point Laplacian:
//   E = A ∫ |∇m|² dV,   H = (2A / μ₀ Ms) ∇²m
//
// When a GeomMask is attached (set_mask), cells with mask < 0.5 are treated
// as Neumann boundaries: no exchange flux crosses the geometry boundary.
// Cells with mask == 0 contribute nothing to H_out (skipped entirely).
//
// When a MaterialField3D is attached (set_material_field), per-cell Ms and
// A_exchange override the uniform `mat` argument:
//   H_i  = (2 / μ₀ Ms_i) Σ_neighbours (A_ij / d²)(m_j - m_i)
//   A_ij = 2 A_i A_j / (A_i + A_j)   (harmonic mean — region-boundary stiffness)
class ExchangeField : public IEffectiveField {
public:
    explicit ExchangeField(BoundaryCondition bc = BoundaryCondition::Neumann)
        : bc_(bc) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "Exchange"; }

    BoundaryCondition boundary() const { return bc_; }
    void set_boundary(BoundaryCondition bc) { bc_ = bc; }

    // Attach a geometry mask. mask=nullptr disables masking (default).
    // Caller must keep the mask alive for the lifetime of this field.
    void set_mask(const GeomMask* mask) { mask_ = mask; }
    void clear_mask() { mask_ = nullptr; }
    const GeomMask* mask() const { return mask_; }

    // Attach per-cell material parameters. nullptr disables it (default),
    // falling back to the uniform `mat` passed to accumulate()/energy().
    // Caller must keep the field alive for the lifetime of this object.
    void set_material_field(const MaterialField3D* matf) { matf_ = matf; }
    void clear_material_field() { matf_ = nullptr; }
    const MaterialField3D* material_field() const { return matf_; }

private:
    BoundaryCondition bc_;
    const GeomMask*        mask_{nullptr};
    const MaterialField3D* matf_{nullptr};
};

}  // namespace micromag
