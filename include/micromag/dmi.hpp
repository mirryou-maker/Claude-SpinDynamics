#pragma once

#include "effective_field.hpp"
#include "material_field.hpp"

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

    // mumax3-style boundary handling. Default (false) applies the free-boundary
    // DMI condition dm/dn = (D/2A)(n_hat x m) at missing neighbours (the ghost
    // enters both the DMI gradient and, as a correction added here, the exchange
    // Laplacian). true = legacy naive stencil (mumax3 OpenBC=1); a uniform film
    // then becomes a spurious equilibrium (no Rohart-Thiaville edge canting).
    bool open_bc() const      { return open_bc_; }
    void set_open_bc(bool ob) { open_bc_ = ob; }

    // Per-cell Ms from a MaterialField3D (uniform D kept), matching the GPU
    // set_material_field and ExchangeField/UniaxialAnisotropyField so a granular
    // sample drives DMI's 1/(mu0 Ms) prefactor per cell. Only the FIELD depends
    // on Ms; the DMI energy is Ms-independent.
    void set_material_field(const MaterialField3D* matf) { matf_ = matf; }
    bool has_material_field() const { return matf_ != nullptr; }

private:
    Real D_;
    bool open_bc_ = false;
    const MaterialField3D* matf_ = nullptr;
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

    // Free-boundary DMI condition dm/dn = (D/2A)(z_hat x n_hat) x m (Rohart-
    // Thiaville). Default false = condition applied (mumax3 default);
    // true = legacy naive stencil (mumax3 OpenBC=1).
    bool open_bc() const      { return open_bc_; }
    void set_open_bc(bool ob) { open_bc_ = ob; }

    // Per-cell Ms from a MaterialField3D (uniform D kept), matching the GPU
    // set_material_field and ExchangeField/UniaxialAnisotropyField so a granular
    // sample drives DMI's 1/(mu0 Ms) prefactor per cell. Only the FIELD depends
    // on Ms; the DMI energy is Ms-independent.
    void set_material_field(const MaterialField3D* matf) { matf_ = matf; }
    bool has_material_field() const { return matf_ != nullptr; }

private:
    Real D_;
    bool open_bc_ = false;
    const MaterialField3D* matf_ = nullptr;
};

}  // namespace micromag
