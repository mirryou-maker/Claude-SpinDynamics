#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/exchange.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Torque function unit tests
// ---------------------------------------------------------------------------

TEST_CASE("LLG torque: zero when m parallel H", "[llg]") {
    Vec3 t = llg_torque({0, 0, 1}, {0, 0, 1e5}, 0.02);
    REQUIRE_THAT(t.norm(), WithinAbs(0.0, 1.0));  // [1/s] units, small
}

TEST_CASE("LLG torque: precession direction m=+x H=+z", "[llg]") {
    // m × H = (0, -Hz, 0) → torque should have +y component
    Vec3 t = llg_torque({1, 0, 0}, {0, 0, 1e5}, 0.0);
    REQUIRE(t.y > 0);
    REQUIRE_THAT(t.x, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(t.z, WithinAbs(0.0, 1.0));
}

TEST_CASE("LLG torque: expected magnitude (no damping)", "[llg]") {
    // |dm/dt| = γ₀ μ₀ |H| for m ⊥ H, α = 0
    const Real H0 = 1e5;
    Vec3 t = llg_torque({1, 0, 0}, {0, 0, H0}, 0.0);
    Real expected = constants::gamma_0 * constants::mu_0 * H0;
    REQUIRE_THAT(t.norm(), WithinRel(expected, 1e-10));
}

// ---------------------------------------------------------------------------
// RK4 integrator tests
// ---------------------------------------------------------------------------

TEST_CASE("RK4: m stays fixed when aligned with H", "[integrator]") {
    StructuredGrid g(2, 2, 2, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({0, 0, 1});

    Material mat = Material::permalloy();
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 1e5}));

    RK4Integrator rk4(1e-13);
    for (int i = 0; i < 200; ++i)
        rk4.step(m, mat, heff);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(m[idx].x, WithinAbs(0.0, 1e-10));
        REQUIRE_THAT(m[idx].y, WithinAbs(0.0, 1e-10));
        REQUIRE_THAT(m[idx].z, WithinAbs(1.0, 1e-10));
    }
}

TEST_CASE("RK4: undamped precession quarter period", "[integrator]") {
    // m = (1,0,0), H = (0,0,H₀), α = 0
    // After T/4 = π/(2 γ₀ μ₀ H₀), m → (0,1,0)
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    Material mat = Material::permalloy();
    mat.alpha = 0.0;

    const Real H0    = 1e6;
    const Real omega = constants::gamma_0 * constants::mu_0 * H0;
    const Real dt    = constants::pi / (2.0 * omega) / 1000.0;  // T/4 / 1000

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, H0}));

    RK4Integrator rk4(dt);
    for (int i = 0; i < 1000; ++i)
        rk4.step(m, mat, heff);

    Vec3 m0 = m.at(0, 0, 0);
    REQUIRE_THAT(m0.x, WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(m0.y, WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(m0.z, WithinAbs(0.0, 1e-4));
}

TEST_CASE("RK4: undamped precession full period returns to start", "[integrator]") {
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    Material mat = Material::permalloy();
    mat.alpha = 0.0;

    const Real H0    = 1e6;
    const Real omega = constants::gamma_0 * constants::mu_0 * H0;
    const Real dt    = 2.0 * constants::pi / omega / 2000.0;  // T / 2000

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, H0}));

    RK4Integrator rk4(dt);
    for (int i = 0; i < 2000; ++i)
        rk4.step(m, mat, heff);

    Vec3 m0 = m.at(0, 0, 0);
    REQUIRE_THAT(m0.x, WithinAbs(1.0, 1e-3));
    REQUIRE_THAT(m0.y, WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(m0.z, WithinAbs(0.0, 1e-3));
}

TEST_CASE("RK4: energy decreases monotonically with damping", "[integrator]") {
    // m⊥H, α > 0: Zeeman energy must decrease each step
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    Material mat = Material::permalloy();
    mat.alpha = 0.5;

    auto zeeman = std::make_shared<ZeemanField>(Vec3{0, 0, 1e5});
    EffectiveFieldSum heff;
    heff.add(zeeman);

    RK4Integrator rk4(1e-12);
    Real E_prev = heff.total_energy(m, mat);

    for (int i = 0; i < 100; ++i) {
        rk4.step(m, mat, heff);
        Real E_now = heff.total_energy(m, mat);
        REQUIRE(E_now <= E_prev + 1e-50);
        E_prev = E_now;
    }
}

TEST_CASE("RK4: magnitude exactly 1 after every step", "[integrator]") {
    StructuredGrid g(4, 4, 2, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    m.set_vortex(4e-9, 4e-9, 4e-9);

    Material mat = Material::permalloy();
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 1e4}));
    heff.add(std::make_shared<ExchangeField>());

    RK4Integrator rk4(1e-13);
    for (int step = 0; step < 50; ++step) {
        rk4.step(m, mat, heff);
        for (Index idx = 0; idx < g.size(); ++idx)
            REQUIRE_THAT(m[idx].norm(), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("RK4: relaxation converges m toward H direction", "[integrator]") {
    // Start with m = +x, H = +z, high damping. After enough steps, m·z ≈ 1.
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    Material mat = Material::permalloy();
    mat.alpha = 0.9;

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 1e5}));

    RK4Integrator rk4(1e-12);
    for (int i = 0; i < 5000; ++i)
        rk4.step(m, mat, heff);

    REQUIRE(m.at(0, 0, 0).z > 0.99);
}
