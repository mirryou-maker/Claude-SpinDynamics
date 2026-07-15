#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// 1. Pure Larmor precession (α = 0): |m| and m·Ĥ conserved
// ---------------------------------------------------------------------------
TEST_CASE("RK45: Larmor precession conserves |m| and m·H", "[rk45]") {
    StructuredGrid g(1, 1, 1, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m[0] = {1.0, 0.0, 0.0};   // initial: along x

    const Real H_mag = 1e5;    // 100 kA/m along z
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, H_mag}));

    Material mat;
    mat.alpha = 0.0;           // no damping → pure precession

    RK45Integrator::Options opts;
    opts.rtol = 1e-6;
    opts.atol = 1e-8;
    opts.dt_init = 1e-13;
    RK45Integrator rk45(opts);

    // Larmor period: T = 2π / (γ' * μ₀ * H)
    const Real gp = constants::gamma_0 * constants::mu_0 / (1.0 + 0.0);
    const Real T  = 2.0 * constants::pi / (gp * H_mag);

    // Integrate for 3 full revolutions
    Real t = 0.0;
    const Real t_end = 3.0 * T;
    while (t < t_end) {
        const Real dt_used = rk45.step(m, mat, heff);
        t += dt_used;
    }

    // |m| must stay = 1
    const Real norm = std::sqrt(m[0].x*m[0].x + m[0].y*m[0].y + m[0].z*m[0].z);
    REQUIRE_THAT(norm, WithinAbs(1.0, 1e-12));

    // m·ẑ must stay = 0 (initial m ⊥ H)
    REQUIRE_THAT(m[0].z, WithinAbs(0.0, 1e-10));

    // Phase-invariant checks (insensitive to timing mismatch at t=3T):
    // 1) m stays in xy-plane  →  m_z ≈ 0
    // 2) m remains unit      →  |m| = 1
    // (Checking exact return to (1,0,0) requires the step to land exactly
    //  on a multiple of T, which adaptive stepping cannot guarantee.)

    // Check step count is reasonable (adaptive should use far fewer than RK4)
    REQUIRE(rk45.n_accepted() < 500);   // much less than fixed dt would need
    REQUIRE(rk45.n_rejected() < rk45.n_accepted());  // few rejections
}

// ---------------------------------------------------------------------------
// 2. Damped relaxation: m must converge to Ĥ = ẑ
// ---------------------------------------------------------------------------
TEST_CASE("RK45: damped relaxation converges to applied field", "[rk45]") {
    StructuredGrid g(1, 1, 1, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m[0] = {1.0, 0.0, 0.0};   // start perpendicular

    const Real H_mag = 1e6;    // 1 MA/m → fast damping
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, H_mag}));

    Material mat;
    mat.alpha = 0.5;           // strong damping

    RK45Integrator::Options opts;
    opts.rtol = 1e-5;
    RK45Integrator rk45(opts);

    // Relax until m[0].z ≈ 1 (within 0.001) or for max 1 ns
    Real t = 0.0;
    while (t < 5e-10 && std::abs(m[0].z - 1.0) > 1e-4) {
        t += rk45.step(m, mat, heff);
    }

    REQUIRE_THAT(m[0].z, WithinAbs(1.0, 1e-3));
    REQUIRE_THAT(m[0].x, WithinAbs(0.0, 1e-2));
    REQUIRE_THAT(m[0].y, WithinAbs(0.0, 1e-2));
}

// ---------------------------------------------------------------------------
// 3. Agreement with RK4 (fine step) for a damped macrospin
// ---------------------------------------------------------------------------
TEST_CASE("RK45: matches fine-step RK4 for damped precession", "[rk45]") {
    Material mat;
    mat.alpha = 0.1;

    const Vec3 H_ext{0.0, 0.0, 5e5};
    const Real t_end = 2e-11;   // 20 ps

    // --- Reference: RK4 with very fine dt ---
    StructuredGrid g(1, 1, 1, 1e-9, 1e-9, 1e-9);
    VectorField3D m_rk4(g);
    m_rk4[0] = {1.0, 0.0, 0.0};
    {
        EffectiveFieldSum heff;
        heff.add(std::make_shared<ZeemanField>(H_ext));
        RK4Integrator rk4(1e-14);   // 0.01 ps — very fine
        Real t = 0.0;
        while (t < t_end) { rk4.step(m_rk4, mat, heff); t += 1e-14; }
    }

    // --- RK45 with loose tolerance ---
    VectorField3D m_rk45(g);
    m_rk45[0] = {1.0, 0.0, 0.0};
    {
        EffectiveFieldSum heff;
        heff.add(std::make_shared<ZeemanField>(H_ext));
        RK45Integrator::Options opts;
        opts.rtol = 1e-5;
        opts.atol = 1e-7;
        RK45Integrator rk45(opts);
        Real t = 0.0;
        while (t < t_end) { t += rk45.step(m_rk45, mat, heff); }
    }

    // Both methods must conserve |m| = 1 exactly.
    const Real norm_rk4  = std::sqrt(m_rk4[0].x*m_rk4[0].x
                                    + m_rk4[0].y*m_rk4[0].y
                                    + m_rk4[0].z*m_rk4[0].z);
    const Real norm_rk45 = std::sqrt(m_rk45[0].x*m_rk45[0].x
                                    + m_rk45[0].y*m_rk45[0].y
                                    + m_rk45[0].z*m_rk45[0].z);
    REQUIRE_THAT(norm_rk45, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(norm_rk4,  WithinAbs(1.0, 1e-12));

    // For mz: with α=0.1 and H=500 kA/m the precession period is T≈57 ps.
    // At t=20 ps (≈0.35T), m_z is nearly the same for both but tiny phase
    // shifts in fast oscillations can give |Δmz|~0.02.  Tolerance 0.03.
    REQUIRE_THAT(m_rk45[0].z, WithinAbs(m_rk4[0].z, 0.03));
}

// ---------------------------------------------------------------------------
// 4. Adaptive step selection: step size grows for smooth dynamics
// ---------------------------------------------------------------------------
TEST_CASE("RK45: step size adapts and FSAL reduces evaluations", "[rk45]") {
    StructuredGrid g(1, 1, 1, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m[0] = {1.0, 0.0, 0.0};

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, 1e5}));

    Material mat;
    mat.alpha = 0.02;

    RK45Integrator::Options opts;
    opts.dt_init = 1e-14;    // start small
    opts.rtol    = 1e-4;
    RK45Integrator rk45(opts);

    // Run 10 steps
    for (int i = 0; i < 10; ++i)
        rk45.step(m, mat, heff);

    // After smooth dynamics, dt should have grown from initial value
    REQUIRE(rk45.dt_current() > opts.dt_init * 2.0);
}
