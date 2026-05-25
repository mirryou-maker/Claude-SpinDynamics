#pragma once

#include "effective_field.hpp"

namespace micromag {

// Heisenberg exchange via 6-point Laplacian:
//   E = A ∫ |∇m|² dV,   H = (2A / μ₀ Ms) ∇²m
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

private:
    BoundaryCondition bc_;
};

}  // namespace micromag
