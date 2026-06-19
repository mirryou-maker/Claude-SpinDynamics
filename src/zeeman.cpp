#include "micromag/zeeman.hpp"

namespace micromag {

void ZeemanField::accumulate(const VectorField3D& m,
                              const Material& /*mat*/,
                              VectorField3D& H_out) const {
    for (Index idx = 0; idx < m.size(); ++idx)
        H_out[idx] += H_ext_;
}

Real ZeemanField::energy(const VectorField3D& m, const Material& mat) const {
    Real sum = 0;
    for (Index idx = 0; idx < m.size(); ++idx)
        sum += m[idx].dot(H_ext_);
    return -constants::mu_0 * mat.Ms * sum * m.grid().cell_volume();
}

ScalarField3D ZeemanField::energy_density(const VectorField3D& m,
                                           const Material& mat) const {
    ScalarField3D edens(m.grid());
    const Real prefac = -constants::mu_0 * mat.Ms;
    for (Index i = 0; i < m.size(); ++i)
        edens[i] = prefac * m[i].dot(H_ext_);
    return edens;
}

}  // namespace micromag
