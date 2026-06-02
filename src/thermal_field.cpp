#include <cmath>
#include "micromag/thermal_field.hpp"

namespace micromag {

ThermalField::ThermalField(const StructuredGrid& grid, Real T_K, Real dt,
                             unsigned seed)
    : T_K_(T_K), dt_(dt), rng_(seed),
      noise_(std::make_unique<VectorField3D>(grid))
{
    for (Index i = 0; i < noise_->size(); ++i)
        (*noise_)[i] = {0.0, 0.0, 0.0};
}

void ThermalField::set_temperature(Real T_K) { T_K_ = T_K; }
void ThermalField::set_dt(Real dt)           { dt_  = dt;  }

Real ThermalField::sigma(Real T_K, Real dt,
                          const Material& mat, const StructuredGrid& grid) {
    const Real V   = grid.dx() * grid.dy() * grid.dz();   // cell volume [m³]
    const Real num = 2.0 * mat.alpha * constants::k_B * T_K;
    const Real den = constants::mu_0 * mat.Ms * constants::gamma_0 * V * dt;
    return std::sqrt(num / den);
}

void ThermalField::resample(const Material& mat) {
    const Real sig = sigma(T_K_, dt_, mat, noise_->grid());
    for (Index i = 0; i < noise_->size(); ++i)
        (*noise_)[i] = {
            dist_(rng_) * sig,
            dist_(rng_) * sig,
            dist_(rng_) * sig
        };
}

void ThermalField::accumulate(const VectorField3D& /*m*/,
                               const Material&      /*mat*/,
                               VectorField3D&       H_out) const {
    for (Index i = 0; i < noise_->size(); ++i)
        H_out[i] = H_out[i] + (*noise_)[i];
}

Real ThermalField::energy(const VectorField3D& /*m*/,
                           const Material&      /*mat*/) const {
    return 0.0;   // stochastic field has no static potential energy
}

}  // namespace micromag
