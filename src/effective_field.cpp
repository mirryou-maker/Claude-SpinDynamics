#include "micromag/effective_field.hpp"

namespace micromag {

void EffectiveFieldSum::add(std::shared_ptr<IEffectiveField> term) {
    if (term) terms_.push_back(std::move(term));
}

void EffectiveFieldSum::compute(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    for (Index idx = 0; idx < H_out.size(); ++idx)
        H_out[idx] = Vec3{0, 0, 0};
    for (const auto& term : terms_)
        term->accumulate(m, mat, H_out);
}

Real EffectiveFieldSum::total_energy(const VectorField3D& m,
                                     const Material& mat) const {
    Real total = 0;
    for (const auto& term : terms_)
        total += term->energy(m, mat);
    return total;
}

ScalarField3D EffectiveFieldSum::energy_density(const VectorField3D& m,
                                                const Material& mat) const {
    ScalarField3D edens(m.grid());
    for (const auto& term : terms_) {
        ScalarField3D ed = term->energy_density(m, mat);
        for (Index i = 0; i < edens.size(); ++i)
            edens[i] += ed[i];
    }
    return edens;
}

// Default implementation: uniform per-cell value = E_total / (N * dV)
ScalarField3D IEffectiveField::energy_density(const VectorField3D& m,
                                               const Material& mat) const {
    ScalarField3D edens(m.grid());
    const Real dV = m.grid().cell_volume();
    if (dV > 0 && m.size() > 0) {
        const Real e_avg = energy(m, mat) / (static_cast<Real>(m.size()) * dV);
        edens.set_uniform(e_avg);
    }
    return edens;
}

}  // namespace micromag
