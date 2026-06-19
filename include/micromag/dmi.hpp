#pragma once

#include "effective_field.hpp"

namespace micromag {

// BulkDMIField — bulk DMI (mumax3 Dbulk), B20 compounds (MnSi, FeGe).
// e_DMI = D m·(∇×m),  H = (2D/μ₀Ms) ∇×m
// D > 0: right-handed helicity (Bloch skyrmion)
class BulkDMIField : public IEffectiveField {
public:
    explicit BulkDMIField(Real D = 0.0) : D_(D) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "BulkDMI"; }

    Real D() const     { return D_; }
    void set_D(Real D) { D_ = D; }

private:
    Real D_;
};

// InterfacialDMIField — interfacial DMI (mumax3 Dind), HM/FM interfaces.
// Film normal ẑ; xy-plane interaction.
// H_x = (2D/μ₀Ms) ∂mz/∂x,  H_y = (2D/μ₀Ms) ∂mz/∂y,
// H_z = -(2D/μ₀Ms)(∂mx/∂x + ∂my/∂y)
// D > 0: inward Néel skyrmion (Pt/Co sign convention)
class InterfacialDMIField : public IEffectiveField {
public:
    explicit InterfacialDMIField(Real D = 0.0) : D_(D) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "InterfacialDMI"; }

    Real D() const     { return D_; }
    void set_D(Real D) { D_ = D; }

private:
    Real D_;
};

}  // namespace micromag
