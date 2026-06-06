#pragma once

// rkky.hpp — RKKY inter-layer exchange coupling
//
// Models antiferromagnetic (or ferromagnetic) coupling between two magnetic
// layers separated by a non-magnetic spacer, as in mumax3 "RKKY" example.
//
// The RKKY effective field on layer m due to reference layer m_ref:
//
//   H_RKKY = -J_RKKY / (mu_0 * Ms * d_spacer) * m_ref
//
// Units:
//   J_RKKY  [J/m²]   coupling constant (negative = antiferromagnetic)
//   d_spacer [m]      spacer thickness
//
// Usage (two-layer simulation):
//   RKKYField rkky1(m2, J_RKKY, d);   // field on layer 1 due to layer 2
//   RKKYField rkky2(m1, J_RKKY, d);   // field on layer 2 due to layer 1
//   heff1.add(rkky1_ptr); heff2.add(rkky2_ptr);
//   // Alternate stepping: integ.step(m1, mat, heff1); integ.step(m2, mat, heff2);
//
// Note: J_RKKY < 0 → antiferromagnetic (field opposes m_ref).
//       J_RKKY > 0 → ferromagnetic   (field aligns with m_ref).

#include "effective_field.hpp"
#include "field.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

class RKKYField : public IEffectiveField {
public:
    // ref_m   : reference layer magnetization (must share the same grid)
    // J_RKKY  : coupling constant [J/m²]  (< 0 = antiferromagnetic)
    // d_spacer: non-magnetic spacer thickness [m]
    RKKYField(const VectorField3D& ref_m, Real J_RKKY, Real d_spacer)
        : ref_m_(&ref_m), J_RKKY_(J_RKKY), d_spacer_(d_spacer) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "RKKY"; }

    // Runtime update of coupling constant (e.g. for sweeps)
    void set_J(Real J) { J_RKKY_ = J; }
    Real J()     const { return J_RKKY_; }
    Real d()     const { return d_spacer_; }

private:
    const VectorField3D* ref_m_;
    Real J_RKKY_;
    Real d_spacer_;
};

}  // namespace micromag
