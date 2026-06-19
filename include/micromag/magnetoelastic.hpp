#pragma once

// magnetoelastic.hpp — Magnetoelastic (magnetostrictive) effective field
//
// Implements the cubic-symmetry magnetoelastic coupling (mumax3 B1/B2).
//
// Energy density [J/m³]:
//   e_me = B1*(mx²*exx + my²*eyy + mz²*ezz)
//        + 2*B2*(mx*my*exy + my*mz*eyz + mx*mz*exz)
//
// Effective field (H = -1/(µ₀Ms) * ∂e/∂m):
//   H_x = -(2/µ₀Ms) * [B1*mx*exx + B2*(my*exy + mz*exz)]
//   H_y = -(2/µ₀Ms) * [B1*my*eyy + B2*(mx*exy + mz*eyz)]
//   H_z = -(2/µ₀Ms) * [B1*mz*ezz + B2*(mx*exz + my*eyz)]
//
// Two strain modes:
//   (a) Uniform strain (default): scalar exx/eyy/... apply to every cell.
//   (b) Spatially varying: attach a ScalarField3D per component via
//       set_exx_field(), set_exy_field(), etc.  Per-cell values override
//       the scalar when set.  Useful for SAW mode shapes, localised stress.
//
// Typical values:
//   Ni:    B1 = -62.4e6, B2 = -27.1e6 J/m³
//   Fe:    B1 = +6.96e6, B2 = -3.43e6 J/m³
//   Galfenol (Fe81Ga19): B1 = -10.5e6 J/m³
//
// Usage:
//   MagnetoelasticField me(B1=-62.4e6, B2=-27.1e6);
//
//   // Uniform strain:
//   me.set_strain(exx=0.001);
//
//   // SAW mode shape (spatially varying exx):
//   ScalarField3D exx_saw = make_saw_profile(grid, k, A);
//   me.set_exx_field(&exx_saw);    // per-cell overrides scalar
//
//   heff.add(std::make_shared<MagnetoelasticField>(me));
//
// mumax3 counterpart: B1, B2, exx, eyy, ezz, exy, exz, eyz quantities.

#include "effective_field.hpp"
#include "field.hpp"
#include "types.hpp"

namespace micromag {

class MagnetoelasticField : public IEffectiveField {
public:
    // B1, B2 : coupling constants [J/m³]
    // Strain defaults to zero (no magnetoelastic contribution until set).
    explicit MagnetoelasticField(Real B1 = Real{0}, Real B2 = Real{0});

    // -----------------------------------------------------------------------
    // Uniform strain tensor (all cells)
    // -----------------------------------------------------------------------
    void set_strain(Real exx, Real eyy = Real{0}, Real ezz = Real{0},
                    Real exy = Real{0}, Real exz = Real{0}, Real eyz = Real{0});

    void set_exx(Real v)  { exx_ = v; }
    void set_eyy(Real v)  { eyy_ = v; }
    void set_ezz(Real v)  { ezz_ = v; }
    void set_exy(Real v)  { exy_ = v; }
    void set_exz(Real v)  { exz_ = v; }
    void set_eyz(Real v)  { eyz_ = v; }

    Real exx() const { return exx_; }
    Real eyy() const { return eyy_; }
    Real ezz() const { return ezz_; }
    Real exy() const { return exy_; }
    Real exz() const { return exz_; }
    Real eyz() const { return eyz_; }

    // -----------------------------------------------------------------------
    // Per-cell strain fields (override scalar when non-null)
    // The ScalarField3D must outlive this object (pointer ownership stays
    // with the caller — no copy is made).
    // -----------------------------------------------------------------------
    void set_exx_field(const ScalarField3D* f) { f_exx_ = f; }
    void set_eyy_field(const ScalarField3D* f) { f_eyy_ = f; }
    void set_ezz_field(const ScalarField3D* f) { f_ezz_ = f; }
    void set_exy_field(const ScalarField3D* f) { f_exy_ = f; }
    void set_exz_field(const ScalarField3D* f) { f_exz_ = f; }
    void set_eyz_field(const ScalarField3D* f) { f_eyz_ = f; }

    void clear_strain_fields() {
        f_exx_ = f_eyy_ = f_ezz_ = f_exy_ = f_exz_ = f_eyz_ = nullptr;
    }

    bool has_spatial_strain() const {
        return f_exx_ || f_eyy_ || f_ezz_ || f_exy_ || f_exz_ || f_eyz_;
    }

    // -----------------------------------------------------------------------
    // Coupling constants
    // -----------------------------------------------------------------------
    Real B1() const           { return B1_; }
    Real B2() const           { return B2_; }
    void set_B1(Real B1)      { B1_ = B1; }
    void set_B2(Real B2)      { B2_ = B2; }

    // -----------------------------------------------------------------------
    // IEffectiveField interface
    // -----------------------------------------------------------------------
    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "MagnetoelasticField"; }

private:
    Real B1_, B2_;

    // Scalar (uniform) strain
    Real exx_{0}, eyy_{0}, ezz_{0};
    Real exy_{0}, exz_{0}, eyz_{0};

    // Optional per-cell strain fields (non-owning pointers)
    const ScalarField3D* f_exx_{nullptr};
    const ScalarField3D* f_eyy_{nullptr};
    const ScalarField3D* f_ezz_{nullptr};
    const ScalarField3D* f_exy_{nullptr};
    const ScalarField3D* f_exz_{nullptr};
    const ScalarField3D* f_eyz_{nullptr};
};

}  // namespace micromag
