#pragma once

#include "effective_field.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// BulkDMIField  — Bulk Dzyaloshinskii-Moriya Interaction
//
// Corresponds to mumax3 `Dbulk` parameter.
// Arises in non-centrosymmetric bulk magnets (B20 compounds: MnSi, FeGe).
//
//   e_DMI  = D  m·(∇×m)
//   H_DMI  = (2D / μ₀Ms) ∇×m
//
//   [∇×m]_x = ∂mz/∂y − ∂my/∂z
//   [∇×m]_y = ∂mx/∂z − ∂mz/∂x
//   [∇×m]_z = ∂my/∂x − ∂mx/∂y
//
// Discretized with central differences; Neumann BC (∂m/∂n=0) at surfaces
// produces a canting of the surface spins (Bogdanov boundary condition).
//
// D > 0: right-handed helicity (Bloch skyrmion preferred)
// D < 0: left-handed helicity
// ---------------------------------------------------------------------------
class BulkDMIField : public IEffectiveField {
public:
    // D: bulk DMI constant [J/m²]
    explicit BulkDMIField(Real D = 0.0) : D_(D) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "BulkDMI"; }

    Real D() const          { return D_; }
    void set_D(Real D)      { D_ = D; }

private:
    Real D_;
};

// ---------------------------------------------------------------------------
// InterfacialDMIField  — Interfacial Dzyaloshinskii-Moriya Interaction
//
// Corresponds to mumax3 `Dind` parameter.
// Arises at heavy-metal/ferromagnet interfaces (Pt/Co, Ta/CoFeB).
// Film normal is assumed to be along ẑ; interaction acts in the xy-plane.
//
//   e_DMI  = D [mz(∇·m) − m·∇mz]  (xy-plane contribution only)
//   H_x    = (2D / μ₀Ms) ∂mz/∂x
//   H_y    = (2D / μ₀Ms) ∂mz/∂y
//   H_z    = −(2D / μ₀Ms) (∂mx/∂x + ∂my/∂y)
//
// Discretized with central differences in x,y; Neumann BC at surfaces.
// Produces Néel-type skyrmions with winding number +1 for D > 0.
//
// D > 0: inward-pointing Néel skyrmion (standard Pt/Co sign)
// D < 0: outward-pointing Néel skyrmion
// ---------------------------------------------------------------------------
class InterfacialDMIField : public IEffectiveField {
public:
    // D: interfacial DMI constant [J/m²]
    explicit InterfacialDMIField(Real D = 0.0) : D_(D) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "InterfacialDMI"; }

    Real D() const          { return D_; }
    void set_D(Real D)      { D_ = D; }

private:
    Real D_;
};

}  // namespace micromag
