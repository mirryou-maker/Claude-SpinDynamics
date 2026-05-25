#pragma once

#include "effective_field.hpp"

namespace micromag {

// Uniaxial anisotropy (easy-axis from Material::easy_axis):
//   E = -K ∫ (m·û)² dV,   H = (2K / μ₀ Ms) (m·û) û
class UniaxialAnisotropyField : public IEffectiveField {
public:
    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "UniaxialAnisotropy"; }
};

}  // namespace micromag
