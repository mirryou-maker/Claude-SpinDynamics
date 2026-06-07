#include <cmath>
#include <stdexcept>
#include "micromag/integrator.hpp"
#include "micromag/material_field.hpp"
#include "micromag/thermal_field.hpp"

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

// Per-cell LLG torque. When `matf` is attached, alpha is looked up per cell
// (mumax3 "Regions" style spatially-varying damping); otherwise the uniform
// `mat.alpha` is used for every cell.
void torque_field(const VectorField3D& m, const VectorField3D& H,
                  const Material& mat, const MaterialField3D* matf,
                  VectorField3D& out) {
    if (matf) {
        for (Index i = 0; i < m.size(); ++i)
            out[i] = llg_torque(m[i], H[i], matf->alpha(i));
    } else {
        const Real alpha = mat.alpha;
        for (Index i = 0; i < m.size(); ++i)
            out[i] = llg_torque(m[i], H[i], alpha);
    }
}

// out[i] = base[i] + scale * k[i]
void axpy(const VectorField3D& base, Real scale,
          const VectorField3D& k, VectorField3D& out) {
    for (Index i = 0; i < base.size(); ++i)
        out[i] = base[i] + k[i] * scale;
}

}  // namespace

void RK4Integrator::step(VectorField3D& m, const Material& mat,
                          const EffectiveFieldSum& heff,
                          const SpinTorqueSum* stt) {
    ensure_scratch(m.grid());
    const Real h = dt_;

    // k1 = f(m)
    heff.compute(m, mat, *H_);
    torque_field(m, *H_, mat, matf_, *k1_);
    if (stt) stt->accumulate(m, mat, *k1_);

    // k2 = f(m + h/2 k1)
    axpy(m, h * 0.5, *k1_, *m_tmp_);
    heff.compute(*m_tmp_, mat, *H_);
    torque_field(*m_tmp_, *H_, mat, matf_, *k2_);
    if (stt) stt->accumulate(*m_tmp_, mat, *k2_);

    // k3 = f(m + h/2 k2)
    axpy(m, h * 0.5, *k2_, *m_tmp_);
    heff.compute(*m_tmp_, mat, *H_);
    torque_field(*m_tmp_, *H_, mat, matf_, *k3_);
    if (stt) stt->accumulate(*m_tmp_, mat, *k3_);

    // k4 = f(m + h k3)
    axpy(m, h, *k3_, *m_tmp_);
    heff.compute(*m_tmp_, mat, *H_);
    torque_field(*m_tmp_, *H_, mat, matf_, *k4_);
    if (stt) stt->accumulate(*m_tmp_, mat, *k4_);

    const Real c = h / 6.0;
    for (Index i = 0; i < m.size(); ++i)
        m[i] += (*k1_)[i] * c
              + (*k2_)[i] * (2 * c)
              + (*k3_)[i] * (2 * c)
              + (*k4_)[i] * c;

    m.normalize();
}

// ===========================================================================
// RK45Integrator  —  Dormand-Prince DOPRI5 with adaptive step control
// ===========================================================================

// Dormand-Prince Butcher tableau (5th-order solution b, 4th-order b*):
//
//  Stage  c     a-row
//   k1    0     —
//   k2   1/5   1/5
//   k3   3/10  3/40  9/40
//   k4   4/5   44/45 -56/15 32/9
//   k5   8/9   19372/6561 -25360/2187 64448/6561 -212/729
//   k6   1     9017/3168  -355/33  46732/5247  49/176  -5103/18656
//   k7   1     [FSAL: k7 = k1 of next step]
//
//  5th:  35/384 0 500/1113 125/192 -2187/6784 11/84 0
//  err = b - b*: 71/57600 0 -71/16695 71/1920 -17253/339200 22/525 -1/40

namespace {

// in-place: out[i] = a + scale*k
void axpy_rk45(VectorField3D& out, const VectorField3D& a,
               Real s, const VectorField3D& k) {
    for (Index i = 0; i < a.size(); ++i)
        out[i] = a[i] + k[i] * s;
}

// in-place multi-axpy helper
void multi_axpy(VectorField3D& out, const VectorField3D& base,
                Real h,
                Real c1, const VectorField3D& k1,
                Real c2, const VectorField3D& k2,
                Real c3, const VectorField3D& k3) {
    const Real hc1 = h * c1, hc2 = h * c2, hc3 = h * c3;
    for (Index i = 0; i < base.size(); ++i)
        out[i] = base[i] + k1[i]*hc1 + k2[i]*hc2 + k3[i]*hc3;
}
void multi_axpy4(VectorField3D& out, const VectorField3D& base,
                 Real h,
                 Real c1, const VectorField3D& k1,
                 Real c2, const VectorField3D& k2,
                 Real c3, const VectorField3D& k3,
                 Real c4, const VectorField3D& k4) {
    const Real hc1=h*c1, hc2=h*c2, hc3=h*c3, hc4=h*c4;
    for (Index i = 0; i < base.size(); ++i)
        out[i] = base[i] + k1[i]*hc1 + k2[i]*hc2 + k3[i]*hc3 + k4[i]*hc4;
}
void multi_axpy5(VectorField3D& out, const VectorField3D& base,
                 Real h,
                 Real c1, const VectorField3D& k1,
                 Real c2, const VectorField3D& k2,
                 Real c3, const VectorField3D& k3,
                 Real c4, const VectorField3D& k4,
                 Real c5, const VectorField3D& k5) {
    const Real hc1=h*c1, hc2=h*c2, hc3=h*c3, hc4=h*c4, hc5=h*c5;
    for (Index i = 0; i < base.size(); ++i)
        out[i] = base[i] + k1[i]*hc1 + k2[i]*hc2 + k3[i]*hc3
               + k4[i]*hc4 + k5[i]*hc5;
}

}  // namespace

RK45Integrator::RK45Integrator(Options opts)
    : opts_(opts), dt_(opts.dt_init) {}

void RK45Integrator::ensure_scratch(const StructuredGrid& g) {
    if (H_) return;
    H_     = std::make_unique<VectorField3D>(g);
    m_tmp_ = std::make_unique<VectorField3D>(g);
    k1_    = std::make_unique<VectorField3D>(g);
    k2_    = std::make_unique<VectorField3D>(g);
    k3_    = std::make_unique<VectorField3D>(g);
    k4_    = std::make_unique<VectorField3D>(g);
    k5_    = std::make_unique<VectorField3D>(g);
    k6_    = std::make_unique<VectorField3D>(g);
    k7_    = std::make_unique<VectorField3D>(g);
    m5_    = std::make_unique<VectorField3D>(g);
    err_   = std::make_unique<VectorField3D>(g);
}

Real RK45Integrator::error_norm(const VectorField3D& m,
                                 const VectorField3D& m5,
                                 const VectorField3D& e) const {
    Real sum = 0.0;
    const Index N = m.size();
    for (Index i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
            const Real mi  = (&m[i].x)[c];
            const Real m5i = (&m5[i].x)[c];
            const Real ei  = (&e[i].x)[c];
            const Real sc  = opts_.atol + opts_.rtol * std::max(std::abs(mi),
                                                                  std::abs(m5i));
            sum += (ei / sc) * (ei / sc);
        }
    }
    return std::sqrt(sum / static_cast<Real>(3 * N));
}

Real RK45Integrator::step(VectorField3D& m, const Material& mat,
                            const EffectiveFieldSum& heff,
                            const SpinTorqueSum* stt) {
    ensure_scratch(m.grid());

    // FSAL: k1 was already computed as stage-7 of the previous accepted step.
    // Recompute only on the very first call or after a rejected step.
    if (!k1_valid_) {
        heff.compute(m, mat, *H_);
        torque_field(m, *H_, mat, matf_, *k1_);
        if (stt) stt->accumulate(m, mat, *k1_);
    }

    for (;;) {  // retry loop on rejection
        const Real h = dt_;

        // --- Stage 2 ---
        axpy_rk45(*m_tmp_, m, h*(1.0/5.0), *k1_);
        heff.compute(*m_tmp_, mat, *H_);
        torque_field(*m_tmp_, *H_, mat, matf_, *k2_);
        if (stt) stt->accumulate(*m_tmp_, mat, *k2_);

        // --- Stage 3 ---
        for (Index i = 0; i < m.size(); ++i)
            (*m_tmp_)[i] = m[i] + (*k1_)[i]*(h*3.0/40.0)
                                 + (*k2_)[i]*(h*9.0/40.0);
        heff.compute(*m_tmp_, mat, *H_);
        torque_field(*m_tmp_, *H_, mat, matf_, *k3_);
        if (stt) stt->accumulate(*m_tmp_, mat, *k3_);

        // --- Stage 4 ---
        for (Index i = 0; i < m.size(); ++i)
            (*m_tmp_)[i] = m[i] + (*k1_)[i]*(h*44.0/45.0)
                                 + (*k2_)[i]*(h*(-56.0/15.0))
                                 + (*k3_)[i]*(h*32.0/9.0);
        heff.compute(*m_tmp_, mat, *H_);
        torque_field(*m_tmp_, *H_, mat, matf_, *k4_);
        if (stt) stt->accumulate(*m_tmp_, mat, *k4_);

        // --- Stage 5 ---
        for (Index i = 0; i < m.size(); ++i)
            (*m_tmp_)[i] = m[i] + (*k1_)[i]*(h*19372.0/6561.0)
                                 + (*k2_)[i]*(h*(-25360.0/2187.0))
                                 + (*k3_)[i]*(h*64448.0/6561.0)
                                 + (*k4_)[i]*(h*(-212.0/729.0));
        heff.compute(*m_tmp_, mat, *H_);
        torque_field(*m_tmp_, *H_, mat, matf_, *k5_);
        if (stt) stt->accumulate(*m_tmp_, mat, *k5_);

        // --- Stage 6 ---
        for (Index i = 0; i < m.size(); ++i)
            (*m_tmp_)[i] = m[i] + (*k1_)[i]*(h*9017.0/3168.0)
                                 + (*k2_)[i]*(h*(-355.0/33.0))
                                 + (*k3_)[i]*(h*46732.0/5247.0)
                                 + (*k4_)[i]*(h*49.0/176.0)
                                 + (*k5_)[i]*(h*(-5103.0/18656.0));
        heff.compute(*m_tmp_, mat, *H_);
        torque_field(*m_tmp_, *H_, mat, matf_, *k6_);
        if (stt) stt->accumulate(*m_tmp_, mat, *k6_);

        // --- 5th-order solution m5 ---
        for (Index i = 0; i < m.size(); ++i)
            (*m5_)[i] = m[i] + (*k1_)[i]*(h*35.0/384.0)
                              + (*k3_)[i]*(h*500.0/1113.0)
                              + (*k4_)[i]*(h*125.0/192.0)
                              + (*k5_)[i]*(h*(-2187.0/6784.0))
                              + (*k6_)[i]*(h*11.0/84.0);

        // --- Stage 7 (= k1 of next step for FSAL) ---
        heff.compute(*m5_, mat, *H_);
        torque_field(*m5_, *H_, mat, matf_, *k7_);
        if (stt) stt->accumulate(*m5_, mat, *k7_);

        // --- Error estimate: err = h*(e1*k1+e3*k3+e4*k4+e5*k5+e6*k6+e7*k7) ---
        for (Index i = 0; i < m.size(); ++i)
            (*err_)[i] = (*k1_)[i]*(h*71.0/57600.0)
                       + (*k3_)[i]*(h*(-71.0/16695.0))
                       + (*k4_)[i]*(h*71.0/1920.0)
                       + (*k5_)[i]*(h*(-17253.0/339200.0))
                       + (*k6_)[i]*(h*22.0/525.0)
                       + (*k7_)[i]*(h*(-1.0/40.0));

        const Real err_norm = error_norm(m, *m5_, *err_);

        if (err_norm <= 1.0) {
            // --- Accept step ---
            ++n_accepted_;
            m = *m5_;
            m.normalize();

            // FSAL: k7 becomes k1 for next step (saves one field evaluation)
            std::swap(k1_, k7_);
            k1_valid_ = true;

            // Adjust step size (grow, but cap at fac_max)
            const Real fac = std::min(opts_.fac_max,
                opts_.safety * std::pow(1.0 / err_norm, 1.0/5.0));
            dt_ = std::min(opts_.dt_max, dt_ * fac);
            return h;
        }

        // --- Reject step: reduce dt and retry ---
        ++n_rejected_;
        const Real fac = std::max(opts_.fac_min,
            opts_.safety * std::pow(1.0 / err_norm, 1.0/5.0));
        dt_ = std::max(opts_.dt_min, dt_ * fac);

        k1_valid_ = false;   // k1 was computed for the original m; still valid
        // (We keep k1 from the start-of-step computation; no recompute needed
        //  since m hasn't changed after a rejection.)
        k1_valid_ = true;

        if (dt_ <= opts_.dt_min)
            throw std::runtime_error(
                "RK45: step size reached minimum — solution may be stiff");
    }
}

// ===========================================================================
// HeunIntegrator  —  fixed-Δt stochastic Heun for SLLG
// ===========================================================================

HeunIntegrator::HeunIntegrator(Real dt) : dt_(dt) {}

void HeunIntegrator::ensure_scratch(const StructuredGrid& g) {
    if (H_) return;
    H_      = std::make_unique<VectorField3D>(g);
    m_pred_ = std::make_unique<VectorField3D>(g);
    k1_     = std::make_unique<VectorField3D>(g);
    k2_     = std::make_unique<VectorField3D>(g);
}

void HeunIntegrator::step(VectorField3D& m, const Material& mat,
                           const EffectiveFieldSum& heff,
                           ThermalField*            thermal,
                           const SpinTorqueSum*     stt) {
    ensure_scratch(m.grid());
    const Real h = dt_;

    // ---- Generate new thermal noise η^n (once per step) ----
    if (thermal) thermal->resample(mat);

    // ---- Predictor ----
    // H = H_eff(m) + H_th(η^n)
    heff.compute(m, mat, *H_);
    if (thermal) thermal->accumulate(m, mat, *H_);

    // k1 = f(m, H)
    torque_field(m, *H_, mat, matf_, *k1_);
    if (stt) stt->accumulate(m, mat, *k1_);

    // m_pred = normalize(m + h*k1)
    for (Index i = 0; i < m.size(); ++i)
        (*m_pred_)[i] = m[i] + (*k1_)[i] * h;
    m_pred_->normalize();

    // ---- Corrector ----
    // H = H_eff(m_pred) + H_th(η^n)  ← SAME noise (Stratonovich)
    heff.compute(*m_pred_, mat, *H_);
    if (thermal) thermal->accumulate(*m_pred_, mat, *H_);

    // k2 = f(m_pred, H)
    torque_field(*m_pred_, *H_, mat, matf_, *k2_);
    if (stt) stt->accumulate(*m_pred_, mat, *k2_);

    // m = normalize(m + h/2*(k1+k2))
    const Real hh = h * 0.5;
    for (Index i = 0; i < m.size(); ++i)
        m[i] = m[i] + ((*k1_)[i] + (*k2_)[i]) * hh;
    m.normalize();
}

}  // namespace micromag
