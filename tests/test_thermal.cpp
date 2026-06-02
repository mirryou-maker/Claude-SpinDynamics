#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>
#include <numeric>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/thermal_field.hpp"

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
