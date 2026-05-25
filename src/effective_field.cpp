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

}  // namespace micromag
