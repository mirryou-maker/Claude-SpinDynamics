#pragma once

#include "effective_field.hpp"

namespace micromag {

// Forward declaration — include material_field.hpp for the full definition.
class MaterialField3D;

// Uniaxial anisotropy (easy-axis from Material::easy_axis):
//   E = -K ∫ (m·û)² dV,   H = (2K / μ₀ Ms) (m·û) û
//
// When a MaterialField3D is attached (set_material_field), the per-cell
// K_uniaxial, easy_axis and Ms are used (mumax3 "Regions" style spatially-
// varying anisotropy — e.g. randomly-oriented grains from voronoi_grains).
class UniaxialAnisotropyField : public IEffectiveField {
public:
    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

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
