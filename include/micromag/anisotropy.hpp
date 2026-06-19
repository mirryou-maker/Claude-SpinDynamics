#pragma once

#include "effective_field.hpp"

namespace micromag {

// Forward declaration — include material_field.hpp for the full definition.
class MaterialField3D;

// Uniaxial anisotropy with first and second-order terms.
// E = -∫(K1*(m·û)² + Ku2*(m·û)⁴) dV
// H = (1/μ₀Ms)(2K1*(m·û) + 4Ku2*(m·û)³) û
// K1 = mat.K_uniaxial, Ku2 = mat.Ku2, û = mat.easy_axis (normalised).
//
// When a MaterialField3D is attached (set_material_field), per-cell
// K_uniaxial, easy_axis, Ms are used; Ku2 is always from the uniform mat.
class UniaxialAnisotropyField : public IEffectiveField {
public:
    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "UniaxialAnisotropy"; }

    // Attach per-cell K_uniaxial/easy_axis/Ms. nullptr disables it (default),
    // falling back to the uniform `mat` passed to accumulate()/energy().
    // Caller must keep the field alive for the lifetime of this object.
    void set_material_field(const MaterialField3D* matf) { matf_ = matf; }
    void clear_material_field() { matf_ = nullptr; }
    const MaterialField3D* material_field() const { return matf_; }

private:
    const MaterialField3D* matf_{nullptr};
};

}  // namespace micromag
