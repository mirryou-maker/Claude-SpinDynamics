#include "micromag/zeeman_spatial.hpp"

namespace micromag {

void ZeemanFieldSpatial::accumulate(const VectorField3D& /*m*/,
                                     const Material& /*mat*/,
                                     VectorField3D& H_out) const
{
    const Index N = H_out.size();
    #pragma omp parallel for schedule(static) if(N > 4096)
    for (Index i = 0; i < N; ++i) {
        H_out[i].x += (*H_field_)[i].x;
        H_out[i].y += (*H_field_)[i].y;
        H_out[i].z += (*H_field_)[i].z;
    }
}

// E = -mu_0 * Ms * Sum(m . H_ext) * dV
Real ZeemanFieldSpatial::energy(const VectorField3D& m,
                                 const Material& mat) const
{
    const Real dV = m.grid().cell_volume();
    Real E = 0.0;
    for (Index i = 0; i < m.size(); ++i)
        E -= m[i].dot((*H_field_)[i]);
    return constants::mu_0 * mat.Ms * E * dV;
}

}  // namespace micromag
