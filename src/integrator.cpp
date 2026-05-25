#include "micromag/integrator.hpp"

namespace micromag {

RK4Integrator::RK4Integrator(Real dt) : dt_(dt) {}

void RK4Integrator::ensure_scratch(const StructuredGrid& g) {
    if (H_) return;
    H_     = std::make_unique<VectorField3D>(g);
    m_tmp_ = std::make_unique<VectorField3D>(g);
    k1_    = std::make_unique<VectorField3D>(g);
    k2_    = std::make_unique<VectorField3D>(g);
    k3_    = std::make_unique<VectorField3D>(g);
    k4_    = std::make_unique<VectorField3D>(g);
}

namespace {

void torque_field(const VectorField3D& m, const VectorField3D& H,
                  Real alpha, VectorField3D& out) {
    for (Index i = 0; i < m.size(); ++i)
        out[i] = llg_torque(m[i], H[i], alpha);
}

// out[i] = base[i] + scale * k[i]
void axpy(const VectorField3D& base, Real scale,
          const VectorField3D& k, VectorField3D& out) {
    for (Index i = 0; i < base.size(); ++i)
        out[i] = base[i] + k[i] * scale;
}

}  // namespace

void RK4Integrator::step(VectorField3D& m, const Material& mat,
                          const EffectiveFieldSum& heff) {
    ensure_scratch(m.grid());
    const Real alpha = mat.alpha;
    const Real h     = dt_;

    heff.compute(m, mat, *H_);
    torque_field(m, *H_, alpha, *k1_);

    axpy(m, h * 0.5, *k1_, *m_tmp_);
    heff.compute(*m_tmp_, mat, *H_);
    torque_field(*m_tmp_, *H_, alpha, *k2_);

    axpy(m, h * 0.5, *k2_, *m_tmp_);
    heff.compute(*m_tmp_, mat, *H_);
    torque_field(*m_tmp_, *H_, alpha, *k3_);

    axpy(m, h, *k3_, *m_tmp_);
    heff.compute(*m_tmp_, mat, *H_);
    torque_field(*m_tmp_, *H_, alpha, *k4_);

    const Real c = h / 6.0;
    for (Index i = 0; i < m.size(); ++i)
        m[i] += (*k1_)[i] * c
              + (*k2_)[i] * (2 * c)
              + (*k3_)[i] * (2 * c)
              + (*k4_)[i] * c;

    m.normalize();
}

}  // namespace micromag
