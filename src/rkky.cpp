#include "micromag/rkky.hpp"
#include <cmath>

namespace micromag {

// H_RKKY = -J / (mu_0 * Ms * d) * m_ref
// This adds to H_out (same convention as all other IEffectiveField).
void RKKYField::accumulate(const VectorField3D& /*m*/,
                            const Material& mat,
                            VectorField3D& H_out) const
{
    // H = J/(mu_0 Ms d) * m_ref
    // J < 0 (antiferromagnetic): H opposes m_ref → drives antiparallel alignment
    const Real coeff = J_RKKY_ / (constants::mu_0 * mat.Ms * d_spacer_);
    for (Index i = 0; i < H_out.size(); ++i) {
        H_out[i].x += coeff * (*ref_m_)[i].x;
        H_out[i].y += coeff * (*ref_m_)[i].y;
        H_out[i].z += coeff * (*ref_m_)[i].z;
    }
}

// E = -mu_0/2 * Ms * Sum(m . H_RKKY) * dV
//   = (J / (2 * d)) * Sum(m . m_ref) * dV
Real RKKYField::energy(const VectorField3D& m, const Material& mat) const
{
    const Real dV    = m.grid().cell_volume();
    // E = -J/(2d) * Sum(m.m_ref) * dV
    // (derived from E = -mu_0/2 * Ms * Sum(m.H_RKKY) * dV with H = J/(mu_0*Ms*d)*m_ref)
    const Real coeff = -J_RKKY_ / (2.0 * d_spacer_);
    Real E = 0.0;
    for (Index i = 0; i < m.size(); ++i)
        E += m[i].dot((*ref_m_)[i]);
    return coeff * E * dV;
}

}  // namespace micromag
