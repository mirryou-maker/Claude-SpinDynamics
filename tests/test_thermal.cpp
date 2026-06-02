#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>
#include <numeric>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static Material py300() {
    Material m = Material::permalloy();
    return m;  // alpha=0.02, Ms=800kA/m
}

static StructuredGrid g1() {
    return StructuredGrid(1, 1, 1, 2.5e-9, 2.5e-9, 3.0e-9);
}

// ---------------------------------------------------------------------------
// T1-A: σ formula — units and magnitude
// ---------------------------------------------------------------------------
TEST_CASE("ThermalField: σ has correct units and magnitude", "[thermal]") {
    const auto mat  = py300();
    const auto grid = g1();
    const Real T    = 300.0;
    const Real dt   = 1e-13;  // 0.1 ps

    const Real sig = ThermalField::sigma(T, dt, mat, grid);

    // σ must be positive and finite
    REQUIRE(sig > 0.0);
    REQUIRE(std::isfinite(sig));

    // For Permalloy at 300 K, σ ~ O(100–10 000) A/m  (much less than Ms)
    REQUIRE(sig < mat.Ms);       // σ << Ms
    REQUIRE(sig > 1.0);          // σ > 1 A/m  (not zero)

    // Scaling checks: σ ∝ sqrt(T), σ ∝ 1/sqrt(dt), σ ∝ sqrt(α)
    const Real sig2T  = ThermalField::sigma(2*T, dt, mat, grid);
    const Real sig4dt = ThermalField::sigma(T, 4*dt, mat, grid);
    Material mat4a = mat; mat4a.alpha *= 4.0;
    const Real sig2a  = ThermalField::sigma(T, dt, mat4a, grid);

    REQUIRE_THAT(sig2T / sig, WithinRel(std::sqrt(2.0), 0.001));   // σ ∝ √T
    REQUIRE_THAT(sig4dt / sig, WithinRel(0.5, 0.001));             // σ ∝ 1/√dt
    REQUIRE_THAT(sig2a / sig, WithinRel(2.0, 0.001));              // σ ∝ √α
}

// ---------------------------------------------------------------------------
// T1-B: zero temperature → zero noise
// ---------------------------------------------------------------------------
TEST_CASE("ThermalField: T=0 gives zero noise", "[thermal]") {
    const auto mat  = py300();
    const auto grid = g1();

    ThermalField th(grid, 0.0, 1e-13);
    th.resample(mat);

    VectorField3D H(grid);
    H[0] = {0.0, 0.0, 0.0};
    th.accumulate(H, mat, H);

    REQUIRE_THAT(H[0].x, WithinAbs(0.0, 1e-30));
    REQUIRE_THAT(H[0].y, WithinAbs(0.0, 1e-30));
    REQUIRE_THAT(H[0].z, WithinAbs(0.0, 1e-30));
}

// ---------------------------------------------------------------------------
// T1-C: noise statistics — zero mean, correct variance
// ---------------------------------------------------------------------------
TEST_CASE("ThermalField: noise has zero mean and correct variance", "[thermal]") {
    // Use a larger 1D grid for statistical power
    StructuredGrid grid(500, 1, 1, 2.5e-9, 2.5e-9, 3.0e-9);
    const auto mat = py300();
    const Real T   = 300.0;
    const Real dt  = 1e-13;

    ThermalField th(grid, T, dt);
    const Real sig = ThermalField::sigma(T, dt, mat, grid);

    // Accumulate statistics over many resamples
    const int N_samples = 200;
    double sum_x = 0, sum_x2 = 0;
    for (int s = 0; s < N_samples; ++s) {
        th.resample(mat);
        VectorField3D H(grid);
        for (Index i = 0; i < grid.size(); ++i) H[i] = {0.0, 0.0, 0.0};
        th.accumulate(H, mat, H);
        for (Index i = 0; i < grid.size(); ++i) {
            sum_x  += H[i].x;
            sum_x2 += H[i].x * H[i].x;
        }
    }
    const double N_total = static_cast<double>(N_samples * grid.size());
    const double mean_x  = sum_x  / N_total;
    const double var_x   = sum_x2 / N_total;

    // Mean ≈ 0 within 3σ/√N
    const double mean_tol = 3.0 * sig / std::sqrt(N_total);
    REQUIRE(std::abs(mean_x) < mean_tol);

    // Variance ≈ σ²  within 10% (large sample → tight)
    REQUIRE_THAT(var_x, WithinRel(sig * sig, 0.10));
}

// ---------------------------------------------------------------------------
// T1-D: accumulate is idempotent between resamples (Stratonovich reuse)
// ---------------------------------------------------------------------------
TEST_CASE("ThermalField: same noise reused for predictor and corrector", "[thermal]") {
    const auto mat  = py300();
    const auto grid = g1();

    ThermalField th(grid, 300.0, 1e-13);
    th.resample(mat);

    // First call (predictor)
    VectorField3D H1(grid);
    H1[0] = {0.0, 0.0, 0.0};
    th.accumulate(H1, mat, H1);

    // Second call WITHOUT resample (corrector — same step)
    VectorField3D H2(grid);
    H2[0] = {0.0, 0.0, 0.0};
    th.accumulate(H2, mat, H2);

    REQUIRE_THAT(H2[0].x, WithinAbs(H1[0].x, 1e-30));
    REQUIRE_THAT(H2[0].y, WithinAbs(H1[0].y, 1e-30));
    REQUIRE_THAT(H2[0].z, WithinAbs(H1[0].z, 1e-30));
}

// ---------------------------------------------------------------------------
// T1-E: successive resamples give independent samples
// ---------------------------------------------------------------------------
TEST_CASE("ThermalField: consecutive resamples are independent", "[thermal]") {
    const auto mat  = py300();
    const auto grid = g1();

    ThermalField th(grid, 300.0, 1e-13);

    th.resample(mat);
    VectorField3D H1(grid); H1[0] = {0,0,0};
    th.accumulate(H1, mat, H1);

    th.resample(mat);    // new sample
    VectorField3D H2(grid); H2[0] = {0,0,0};
    th.accumulate(H2, mat, H2);

    // Samples should differ (probability of exact equality is zero)
    const bool same = (H2[0].x == H1[0].x &&
                       H2[0].y == H1[0].y &&
                       H2[0].z == H1[0].z);
    REQUIRE(!same);
}

// ---------------------------------------------------------------------------
// T1-F: reproducibility with same seed
// ---------------------------------------------------------------------------
TEST_CASE("ThermalField: same seed → same sequence", "[thermal]") {
    const auto mat  = py300();
    const auto grid = g1();

    ThermalField th_a(grid, 300.0, 1e-13, 99);
    ThermalField th_b(grid, 300.0, 1e-13, 99);

    th_a.resample(mat);
    th_b.resample(mat);

    VectorField3D Ha(grid), Hb(grid);
    Ha[0] = Hb[0] = {0,0,0};
    th_a.accumulate(Ha, mat, Ha);
    th_b.accumulate(Hb, mat, Hb);

    REQUIRE_THAT(Hb[0].x, WithinAbs(Ha[0].x, 1e-30));
    REQUIRE_THAT(Hb[0].y, WithinAbs(Ha[0].y, 1e-30));
    REQUIRE_THAT(Hb[0].z, WithinAbs(Ha[0].z, 1e-30));
}

// ===========================================================================
// T2: HeunIntegrator tests
// ===========================================================================

// ---------------------------------------------------------------------------
// T2-A: |m| = 1 preserved for all cells (with and without noise)
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegrator: |m|=1 preserved without thermal noise", "[thermal]") {
    StructuredGrid grid(4, 4, 2, 2.5e-9, 2.5e-9, 3.0e-9);
    VectorField3D m(grid);
    m.set_uniform({1.0, 0.0, 0.0});

    Material mat = py300();
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, 1e5}));

    HeunIntegrator heun(5e-14);
    for (int step = 0; step < 1000; ++step)
        heun.step(m, mat, heff);

    for (Index i = 0; i < m.size(); ++i) {
        const Real n = std::sqrt(m[i].x*m[i].x + m[i].y*m[i].y + m[i].z*m[i].z);
        REQUIRE_THAT(n, WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("HeunIntegrator: |m|=1 preserved with thermal noise", "[thermal]") {
    const auto grid = g1();
    VectorField3D m(grid);
    m.set_uniform({1.0, 0.0, 0.0});

    Material mat = py300();
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, 1e5}));

    ThermalField thermal(grid, 300.0, 5e-14);
    HeunIntegrator heun(5e-14);

    for (int step = 0; step < 1000; ++step)
        heun.step(m, mat, heff, &thermal);

    const Real n = std::sqrt(m[0].x*m[0].x + m[0].y*m[0].y + m[0].z*m[0].z);
    REQUIRE_THAT(n, WithinAbs(1.0, 1e-12));
}

// ---------------------------------------------------------------------------
// T2-B: T=0 (σ=0) is equivalent to deterministic Heun ODE
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegrator: T=0 gives same result as no thermal field", "[thermal]") {
    const auto grid = g1();
    Material mat = py300();
    mat.alpha = 0.5;   // strong damping → fast convergence, clean comparison

    Vec3 H0{0.0, 0.0, 5e5};
    const Real dt = 1e-13;
    const int N = 200;

    // Run without thermal
    VectorField3D m_no_th(grid);
    m_no_th.set_uniform({1.0, 0.0, 0.0});
    {
        EffectiveFieldSum heff;
        heff.add(std::make_shared<ZeemanField>(H0));
        HeunIntegrator heun(dt);
        for (int i = 0; i < N; ++i) heun.step(m_no_th, mat, heff);
    }

    // Run with T=0 thermal (σ=0 → no noise)
    VectorField3D m_T0(grid);
    m_T0.set_uniform({1.0, 0.0, 0.0});
    {
        EffectiveFieldSum heff;
        heff.add(std::make_shared<ZeemanField>(H0));
        ThermalField thermal(grid, 0.0, dt);
        HeunIntegrator heun(dt);
        for (int i = 0; i < N; ++i) heun.step(m_T0, mat, heff, &thermal);
    }

    REQUIRE_THAT(m_T0[0].x, WithinAbs(m_no_th[0].x, 1e-12));
    REQUIRE_THAT(m_T0[0].y, WithinAbs(m_no_th[0].y, 1e-12));
    REQUIRE_THAT(m_T0[0].z, WithinAbs(m_no_th[0].z, 1e-12));
}

// ---------------------------------------------------------------------------
// T2-C: T>0 gives different result from T=0 (noise has effect)
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegrator: T>0 trajectory differs from T=0", "[thermal]") {
    const auto grid = g1();
    Material mat = py300();
    const Real dt = 5e-14;
    const int N = 500;

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, 1e5}));

    VectorField3D m_cold(grid), m_hot(grid);
    m_cold.set_uniform({1.0, 0.0, 0.0});
    m_hot.set_uniform({1.0, 0.0, 0.0});

    {
        ThermalField th(grid, 0.0, dt);       // T=0
        HeunIntegrator heun(dt);
        for (int i = 0; i < N; ++i) heun.step(m_cold, mat, heff, &th);
    }
    {
        ThermalField th(grid, 5000.0, dt);    // very hot → large noise
        HeunIntegrator heun(dt);
        for (int i = 0; i < N; ++i) heun.step(m_hot, mat, heff, &th);
    }

    // Trajectories must diverge at high T
    const bool differs = (std::abs(m_hot[0].x - m_cold[0].x) > 1e-6 ||
                          std::abs(m_hot[0].y - m_cold[0].y) > 1e-6 ||
                          std::abs(m_hot[0].z - m_cold[0].z) > 1e-6);
    REQUIRE(differs);
}

// ---------------------------------------------------------------------------
// T2-D: Different seeds → different trajectories (RNG isolation)
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegrator: different seeds give different trajectories", "[thermal]") {
    const auto grid = g1();
    Material mat = py300();
    const Real dt = 5e-14;
    const int N = 100;

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, 1e5}));

    VectorField3D m_a(grid), m_b(grid);
    m_a.set_uniform({1.0, 0.0, 0.0});
    m_b.set_uniform({1.0, 0.0, 0.0});

    {
        ThermalField th(grid, 300.0, dt, 11);
        HeunIntegrator heun(dt);
        for (int i = 0; i < N; ++i) heun.step(m_a, mat, heff, &th);
    }
    {
        ThermalField th(grid, 300.0, dt, 99);  // different seed
        HeunIntegrator heun(dt);
        for (int i = 0; i < N; ++i) heun.step(m_b, mat, heff, &th);
    }

    const bool differs = (m_a[0].x != m_b[0].x ||
                          m_a[0].y != m_b[0].y ||
                          m_a[0].z != m_b[0].z);
    REQUIRE(differs);
}

// ---------------------------------------------------------------------------
// T2-E: Deterministic Heun converges correctly (m → Ĥ under damping)
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegrator: deterministic relaxation to applied field", "[thermal]") {
    const auto grid = g1();
    Material mat = py300();
    mat.alpha = 0.5;   // strong damping

    VectorField3D m(grid);
    m.set_uniform({1.0, 0.0, 0.0});   // start ⊥ field

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0.0, 0.0, 1e6}));

    const Real dt_h = 5e-14;
    HeunIntegrator heun(dt_h);
    for (int step = 0; step < 10000; ++step) {
        heun.step(m, mat, heff);
        if (std::abs(m[0].z - 1.0) < 1e-3) break;
    }

    // Must have relaxed to +z
    REQUIRE_THAT(m[0].z, WithinAbs(1.0, 1e-2));
}
