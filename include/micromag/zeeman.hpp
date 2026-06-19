#pragma once

#include "effective_field.hpp"

namespace micromag {

// Uniform, time-constant external field.
// H_eff = H_ext,  E = -μ₀ Ms ∫ (m·H_ext) dV
class ZeemanField : public IEffectiveField {
public:
    explicit ZeemanField(const Vec3& H_ext = {0, 0, 0}) : H_ext_(H_ext) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "Zeeman"; }

    Vec3 H_ext() const { return H_ext_; }
    void set_H_ext(const Vec3& H) { H_ext_ = H; }

private:
    Vec3 H_ext_;
};

}  // namespace micromag
