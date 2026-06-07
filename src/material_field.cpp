#include "micromag/material_field.hpp"

namespace micromag {

MaterialField3D::MaterialField3D(const StructuredGrid& grid, const Material& uniform)
    : Ms_(grid), A_(grid), K_(grid), alpha_(grid), easy_axis_(grid) {
    set_uniform(uniform);
}

void MaterialField3D::set_uniform(const Material& mat) {
    Ms_.set_uniform(mat.Ms);
    A_.set_uniform(mat.A_exchange);
    K_.set_uniform(mat.K_uniaxial);
    alpha_.set_uniform(mat.alpha);
    easy_axis_.set_uniform(mat.easy_axis);
}

Material MaterialField3D::at(Index i, Index j, Index k) const {
    const Index idx = grid().linear_index(i, j, k);
    return (*this)[idx];
}

Material MaterialField3D::operator[](Index linear) const {
    return Material{Ms_[linear], A_[linear], K_[linear], easy_axis_[linear], alpha_[linear]};
}

}  // namespace micromag
