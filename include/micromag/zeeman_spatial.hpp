#pragma once

// zeeman_spatial.hpp — Spatially varying (and time-varying) external field
//
// Replicates mumax3's ability to define B_ext as a function of position.
// The field H_ext(r) is stored as a VectorField3D and can be updated
// each step by the caller (enabling time-varying fields, e.g. a moving
// write head in a hard-disk simulation).
//
// Usage:
//   auto hz = std::make_shared<ZeemanFieldSpatial>(H_field);
//   heff.add(hz);
//   // Each step:
//   update_write_head_field(H_field, t);   // modify H_field in-place
//   integ.step(m, mat, heff);              // heff reads updated H_field

#include "effective_field.hpp"
#include "field.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

class ZeemanFieldSpatial : public IEffectiveField {
public:
    // H_field: per-cell external field [A/m]; updated by caller at runtime.
    // The field is READ by accumulate() — caller owns and updates it.
    explicit ZeemanFieldSpatial(const VectorField3D& H_field)
        : H_field_(&H_field) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    const char* name() const override { return "ZeemanSpatial"; }

    // Replace the field reference at runtime.
    void set_field(const VectorField3D& H_field) { H_field_ = &H_field; }

private:
    const VectorField3D* H_field_;
};

}  // namespace micromag
