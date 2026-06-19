#pragma once

#include <memory>
#include <vector>

#include "field.hpp"
#include "material.hpp"

namespace micromag {

enum class BoundaryCondition {
    Neumann,   // ∂m/∂n = 0  (free surface), default
    Periodic   // wrap-around
};

class IEffectiveField {
public:
    virtual ~IEffectiveField() = default;

    // Add this term's contribution into H_out (caller zeroes H_out first).
    virtual void accumulate(const VectorField3D& m,
                            const Material& mat,
                            VectorField3D& H_out) const = 0;

    // Total energy [J] for this contribution.
    virtual Real energy(const VectorField3D& m,
                        const Material& mat) const = 0;

    // Per-cell energy density [J/m³].
    // Default implementation: uniform = energy() / (N * cell_volume).
    // Override in subclasses for spatially-resolved maps (mumax3 Edens_*).
    virtual ScalarField3D energy_density(const VectorField3D& m,
                                         const Material& mat) const;

    virtual const char* name() const = 0;
};

class EffectiveFieldSum {
public:
    void add(std::shared_ptr<IEffectiveField> term);

    // Zero H_out, then accumulate every term.
    void compute(const VectorField3D& m,
                 const Material& mat,
                 VectorField3D& H_out) const;

    Real total_energy(const VectorField3D& m, const Material& mat) const;

    // Sum per-cell energy densities from all terms.
    ScalarField3D energy_density(const VectorField3D& m, const Material& mat) const;

    std::size_t num_terms() const { return terms_.size(); }
    const std::vector<std::shared_ptr<IEffectiveField>>& terms() const { return terms_; }

private:
    std::vector<std::shared_ptr<IEffectiveField>> terms_;
};

}  // namespace micromag
