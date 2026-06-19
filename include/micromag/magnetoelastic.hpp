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
// Strain tensor is uniform (same for all cells).  B1, B2 [J/m³] are the
// first- and second-order magnetoelastic coupling constants.
//
// Typical values:
//   Ni:    B1 = -62.4e6, B2 = -27.1e6 J/m³
//   Fe:    B1 = +6.96e6, B2 = -3.43e6 J/m³
//   Galfenol (Fe81Ga19): B1 = -10.5e6 J/m³
//
// Usage (mumax3 analogue):
//   me = MagnetoelasticField(B1=-62.4e6, B2=-27.1e6);
//   me.set_strain(exx=0.001);           // uniaxial strain along x
//   heff.add(std::make_shared<MagnetoelasticField>(me));
//
// mumax3 counterpart: B1, B2, exx, eyy, ezz, exy, exz, eyz quantities
//   and the MagnetoElastic effective field contribution.

#include "effective_field.hpp"
#include "types.hpp"

namespace micromag {

class MagnetoelasticField : public IEffectiveField {
public:
    // B1, B2 : coupling constants [J/m³]
    // Strain defaults to zero (no magnetoelastic contribution until set).
    explicit MagnetoelasticField(Real B1 = Real{0}, Real B2 = Real{0});

    // ---------------------------------------------------------------------------
    // Strain tensor setters (uniform, all cells)
    // ---------------------------------------------------------------------------
    void set_strain(Real exx, Real eyy = Real{0}, Real ezz = Real{0},
                    Real exy = Real{0}, Real exz = Real{0}, Real eyz = Real{0});

    // Individual components (mumax3 style: me.exx = 0.001)
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

    // Coupling constants
    Real B1() const           { return B1_; }
    Real B2() const           { return B2_; }
    void set_B1(Real B1)      { B1_ = B1; }
    void set_B2(Real B2)      { B2_ = B2; }

    // ---------------------------------------------------------------------------
    // IEffectiveField interface
    // ---------------------------------------------------------------------------
    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "MagnetoelasticField"; }

private:
    Real B1_, B2_;
    Real exx_{0}, eyy_{0}, ezz_{0};
    Real exy_{0}, exz_{0}, eyz_{0};
};

}  // namespace micromag
