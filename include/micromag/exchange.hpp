#pragma once

#include "effective_field.hpp"

namespace micromag {

// Forward declaration — include geom_mask.hpp for the definition.
class GeomMask;

// Heisenberg exchange via 6-point Laplacian:
//   E = A ∫ |∇m|² dV,   H = (2A / μ₀ Ms) ∇²m
//
// When a GeomMask is attached (set_mask), cells with mask < 0.5 are treated
// as Neumann boundaries: no exchange flux crosses the geometry boundary.
// Cells with mask == 0 contribute nothing to H_out (skipped entirely).
class ExchangeField : public IEffectiveField {
public:
    explicit ExchangeField(BoundaryCondition bc = BoundaryCondition::Neumann)
        : bc_(bc) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "Exchange"; }

    BoundaryCondition boundary() const { return bc_; }
    void set_boundary(BoundaryCondition bc) { bc_ = bc; }

    // Attach a geometry mask. mask=nullptr disables masking (default).
    // Caller must keep the mask alive for the lifetime of this field.
    void set_mask(const GeomMask* mask) { mask_ = mask; }
    void clear_mask() { mask_ = nullptr; }
    const GeomMask* mask() const { return mask_; }

private:
    BoundaryCondition bc_;
    const GeomMask*   mask_{nullptr};
};

}  // namespace micromag
