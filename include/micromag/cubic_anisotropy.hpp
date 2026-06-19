#pragma once

#include "effective_field.hpp"

namespace micromag {

// Cubic magnetocrystalline anisotropy (mumax3 Kc1/Kc2 parameters).
//
// Energy density: e = Kc1*(α1²α2² + α2²α3² + α3²α1²) + Kc2*(α1²α2²α3²)
// where αi = m·ci are direction cosines along cubic axes c1, c2, c3.
// c3 is computed internally as c1 × c2.
//
// Effective field (variational derivative of E w.r.t. m):
//   H = -(2Kc1/μ₀Ms)[α1(α2²+α3²)c1 + α2(α1²+α3²)c2 + α3(α1²+α2²)c3]
//     - (2Kc2/μ₀Ms)[α1α2²α3²c1 + α1²α2α3²c2 + α1²α2²α3c3]
//
// Default axes: c1={1,0,0}, c2={0,1,0}, c3={0,0,1} (cubic crystal aligned with grid).
// For Fe/Ni: Kc1 ~ +48kJ/m³ (Fe), Kc1 ~ -5kJ/m³ (Ni).
class CubicAnisotropyField : public IEffectiveField {
public:
    // c1 and c2 should be orthonormal; c3 = c1×c2 is computed.
    CubicAnisotropyField(Real Kc1 = 0, Real Kc2 = 0,
                         Vec3 c1 = {1, 0, 0}, Vec3 c2 = {0, 1, 0})
        : Kc1_(Kc1), Kc2_(Kc2), c1_(c1), c2_(c2) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "CubicAnisotropy"; }

    Real Kc1() const { return Kc1_; }
    Real Kc2() const { return Kc2_; }
    Vec3 c1()  const { return c1_; }
    Vec3 c2()  const { return c2_; }
    Vec3 c3()  const { return c1_.cross(c2_); }

    void set_Kc1(Real k) { Kc1_ = k; }
    void set_Kc2(Real k) { Kc2_ = k; }
    void set_axes(Vec3 c1, Vec3 c2) { c1_ = c1; c2_ = c2; }

private:
    Real Kc1_, Kc2_;
    Vec3 c1_, c2_;
};

}  // namespace micromag
