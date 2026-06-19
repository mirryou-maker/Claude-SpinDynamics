#include <cmath>
#include <stdexcept>
#include "micromag/solver.hpp"
#include "micromag/material_field.hpp"
#include "micromag/types.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// max_torque
// ---------------------------------------------------------------------------

Real max_torque(const VectorField3D& m,
                const Material& mat,
                const EffectiveFieldSum& heff) {
    VectorField3D H(m.grid());
    heff.compute(m, mat, H);

    Real max_t = 0.0;
    for (Index i = 0; i < m.size(); ++i) {
        Vec3 t = m[i].cross(H[i]);
        Real t2 = t.dot(t);
        if (t2 > max_t) max_t = t2;
    }
    return std::sqrt(max_t);
}

// ---------------------------------------------------------------------------
// Internal: damping-only LLG torque  dm/dt = -γ'αμ₀ m×(m×H)
// ---------------------------------------------------------------------------

namespace {

inline Vec3 damping_torque(Vec3 m, Vec3 H, Real alpha) noexcept {
    // dm/dt = -γ'αμ₀ m×(m×H)
    const Real gp = constants::gamma_0 * constants::mu_0 / (1.0 + alpha * alpha);
    Vec3 mxH  = m.cross(H);
    Vec3 mxmxH = m.cross(mxH);
    return mxmxH * (-gp * alpha);
}

// Single Euler step in the damping-only direction
void relax_step(VectorField3D& m, const VectorField3D& H,
                Real alpha, Real dt,
                const MaterialField3D* matf) {
    for (Index i = 0; i < m.size(); ++i) {
        Real a = matf ? matf->alpha(i) : alpha;
        m[i] += damping_torque(m[i], H[i], a) * dt;
    }
    m.normalize();
}

}  // namespace

// ---------------------------------------------------------------------------
// relax()
// ---------------------------------------------------------------------------

int relax(VectorField3D& m,
          const Material& mat,
          const EffectiveFieldSum& heff,
          RelaxOptions opts,
          const MaterialField3D* matf,
          const SpinTorqueSum*   stt) {
    VectorField3D H(m.grid());

    const Real alpha = opts.alpha_relax;
    const Real dt    = opts.dt;

    for (int step = 0; step < opts.max_steps; ++step) {
        heff.compute(m, mat, H);

        // Check convergence: max |m × H|
        Real max_t = 0.0;
        for (Index i = 0; i < m.size(); ++i) {
            Vec3 t = m[i].cross(H[i]);
            Real t2 = t.dot(t);
            if (t2 > max_t) max_t = t2;
        }
        if (std::sqrt(max_t) < opts.threshold)
            return step;

        // STT contribution (if any) — add to effective H via dm approximation
        // Note: STT is added as a correction to the effective "force" field
        // For relax we still use pure damping torque direction; STT is omitted
        // as mumax3 Relax() does not apply STT.
        (void)stt;

        relax_step(m, H, alpha, dt, matf);
    }

    if (opts.throw_on_max_steps)
        throw std::runtime_error(
            "relax(): max_steps reached without convergence — "
            "try increasing max_steps or threshold");
    return opts.max_steps;
}

// ---------------------------------------------------------------------------
// minimize()  — steepest-descent with simple backtracking line search
// ---------------------------------------------------------------------------

int minimize(VectorField3D& m,
             const Material& mat,
             const EffectiveFieldSum& heff,
             MinimizeOptions opts,
             const MaterialField3D* matf) {
    VectorField3D H(m.grid());
    VectorField3D m_trial(m.grid());

    Real dt = opts.dt_init;
    const Real alpha = 1.0;   // pure damping direction

    for (int step = 0; step < opts.max_steps; ++step) {
        heff.compute(m, mat, H);

        // Check convergence
        Real max_t = 0.0;
        for (Index i = 0; i < m.size(); ++i) {
            Vec3 t = m[i].cross(H[i]);
            Real t2 = t.dot(t);
            if (t2 > max_t) max_t = t2;
        }
        if (std::sqrt(max_t) < opts.threshold)
            return step;

        const Real E0 = heff.total_energy(m, mat);

        // Trial step
        m_trial = m;
        relax_step(m_trial, H, alpha, dt, matf);
        Real E1 = heff.total_energy(m_trial, mat);

        if (E1 < E0) {
            // Accept: use result, grow step slightly
            m = m_trial;
            dt = std::min(opts.dt_max, dt * 1.2);
        } else {
            // Reject: shrink step, retry from same m
            dt *= 0.5;
            if (dt < opts.dt_min) {
                // At minimum step: accept anyway (near a saddle/flat region)
                m = m_trial;
                dt = opts.dt_init;
            }
        }
    }

    if (opts.throw_on_max_steps)
        throw std::runtime_error(
            "minimize(): max_steps reached without convergence");
    return opts.max_steps;
}

}  // namespace micromag
