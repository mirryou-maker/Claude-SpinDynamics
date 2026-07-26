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
#include "micromag/anisotropy.hpp"
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

// ===========================================================================
// T3: Thermal equilibrium — Boltzmann distribution validation
// ===========================================================================

// ---------------------------------------------------------------------------
// T3-A: Energy equipartition (K=0, H_ext=0)
//
// Uniform distribution on sphere → <mx²> = <my²> = <mz²> = 1/3.
//
// Parameter choice: T=1e8 K (extreme), dt=100 ps, V=(5nm)³, α=0.5.
// At these values σ≈25 kA/m and each step rotates ~0.44 rad (≈25°),
// so the spin explores the full sphere in O(10) steps.
// 10×10×1 grid (100 independent spins) gives 1M samples in 1200 steps.
// ---------------------------------------------------------------------------
TEST_CASE("Thermal equilibrium: energy equipartition (K=0)", "[thermal]") {
    StructuredGrid grid(10, 10, 1, 5e-9, 5e-9, 5e-9);   // 100 cells

    Material mat   = Material::permalloy();
    mat.alpha      = 0.5;
    mat.K_uniaxial = 0.0;

    VectorField3D m(grid);
    m.set_uniform({1.0, 0.0, 0.0});

    // High T + large dt → large σ → rapid exploration of sphere
    const Real T  = 1e8;     // 100 MK — unphysical, but tests the distribution
    const Real dt = 1e-10;   // 100 ps

    EffectiveFieldSum heff;   // empty: only thermal noise
    ThermalField thermal(grid, T, dt);
    HeunIntegrator heun(dt);

    // Equilibration (200 steps ≈ many decorrelation times)
    for (int step = 0; step < 200; ++step)
        heun.step(m, mat, heff, &thermal);

    // Sampling: 1000 steps × 100 cells = 100,000 samples
    double sx2 = 0, sy2 = 0, sz2 = 0;
    const int N_sample = 1000;
    for (int step = 0; step < N_sample; ++step) {
        heun.step(m, mat, heff, &thermal);
        for (Index i = 0; i < m.size(); ++i) {
            sx2 += m[i].x * m[i].x;
            sy2 += m[i].y * m[i].y;
            sz2 += m[i].z * m[i].z;
        }
    }
    const double N   = N_sample * static_cast<double>(grid.size());
    const double mx2 = sx2 / N;
    const double my2 = sy2 / N;
    const double mz2 = sz2 / N;

    // Each component ≈ 1/3  (tolerance ±0.03 with 100 k samples)
    REQUIRE_THAT(mx2, WithinAbs(1.0/3.0, 0.03));
    REQUIRE_THAT(my2, WithinAbs(1.0/3.0, 0.03));
    REQUIRE_THAT(mz2, WithinAbs(1.0/3.0, 0.03));

    // Isotropy
    REQUIRE(std::abs(mx2 - my2) < 0.02);
    REQUIRE(std::abs(mx2 - mz2) < 0.02);

    // Sum must be 1 (|m|=1 by construction)
    REQUIRE_THAT(mx2 + my2 + mz2, WithinAbs(1.0, 1e-6));
}

// ---------------------------------------------------------------------------
// T3-A2: Field-coupled Langevin equilibrium — validates the ABSOLUTE sigma.
//
// A macrospin ensemble in a field H‖z relaxes to <m_z> = L(xi) = coth(xi) - 1/xi
// with xi = mu0 Ms V H / kB T. Unlike the field-FREE equipartition test above
// (which only checks isotropy and passes for ANY sigma scale), this pins sigma's
// absolute magnitude. It is the regression that guards the finite-T sigma fix
// (bare-mu0 -> mu0^2): with the old sigma this gave <mz>=1.0 (thermal ~1/mu0 too
// weak); the corrected sigma reproduces L(xi).
// ---------------------------------------------------------------------------
TEST_CASE("Thermal equilibrium: Langevin <mz> in a field (absolute sigma)",
          "[thermal]") {
    const double kB = 1.380649e-23, mu0 = 4e-7 * 3.14159265358979323846;
    const double Ms = 1e6, d = 2e-9, V = d*d*d, T = 300.0, xi = 3.0;
    const double Hz = xi * kB * T / (mu0 * Ms * V);
    const double L  = 1.0/std::tanh(xi) - 1.0/xi;      // ≈ 0.6716

    StructuredGrid grid(12, 12, 1, d, d, d);            // 144 independent spins
    Material mat; mat.Ms = Ms; mat.alpha = 0.5; mat.K_uniaxial = 0.0;
    VectorField3D m(grid); m.set_uniform({0, 0, 1});

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, static_cast<Real>(Hz)}));
    const Real dt = 1e-14;
    ThermalField thermal(grid, T, dt);
    HeunIntegrator heun(dt);

    for (int s = 0; s < 20000; ++s) heun.step(m, mat, heff, &thermal);  // equilibrate
    double acc = 0; int ns = 0;
    for (int s = 0; s < 25000; ++s) {
        heun.step(m, mat, heff, &thermal);
        if (s % 25 == 0) {
            double mz = 0; for (Index i = 0; i < m.size(); ++i) mz += m[i].z;
            acc += mz / m.size(); ++ns;
        }
    }
    const double mz_mean = acc / ns;
    INFO("<mz> = " << mz_mean << "  Langevin L(3) = " << L);
    REQUIRE_THAT(mz_mean, WithinAbs(L, 0.06));
}

// ---------------------------------------------------------------------------
// T3-B: Anisotropy drives deterministic relaxation to easy axis (T=0)
//
// Physical reason T=0 test is needed: at finite T, the thermal noise σ is
// always orders-of-magnitude smaller than H_ani for real materials (σ/H_ani
// ~ 10⁻⁴ for Permalloy at 300 K).  Therefore observing a Boltzmann bias in
// short unit tests is impractical.  Instead we verify the CORRECT DIRECTION
// of the anisotropy torque at T=0 — a necessary prerequisite for correct
// finite-T behaviour.
//
// K=1e4 J/m³, easy axis ẑ, start m along +x.
// τ_relax = 1/(α γ μ₀ H_ani) ≈ 570 ps = ~570 steps at dt=1 ps.
// After 3000 steps the spin must have reached |mz| > 0.9.
// ---------------------------------------------------------------------------
TEST_CASE("Anisotropy: T=0 relaxation toward easy axis", "[thermal]") {
    const StructuredGrid grid = g1();   // single spin

    Material mat   = Material::permalloy();
    mat.alpha      = 0.5;
    mat.K_uniaxial = 1e4;            // H_ani ≈ 20 kA/m
    mat.easy_axis  = {0.0, 0.0, 1.0};

    VectorField3D m(grid);
    // Start 10° from easy axis: H_ani ≠ 0 → clear anisotropy torque.
    // (mz=0 exactly gives zero torque — unstable equilibrium, no motion.)
    const Real tilt = 10.0 * constants::pi / 180.0;
    m.set_uniform({std::sin(tilt), 0.0, std::cos(tilt)});

    EffectiveFieldSum heff;
    heff.add(std::make_shared<UniaxialAnisotropyField>());

    HeunIntegrator heun(1e-12);

    // No thermal noise (T=0)
    for (int step = 0; step < 3000; ++step)
        heun.step(m, mat, heff);  // thermal=nullptr → T=0

    // Spin must have moved to ±z
    REQUIRE(std::abs(m[0].z) > 0.9);

    // |m| still 1
    const Real n = std::sqrt(m[0].x*m[0].x + m[0].y*m[0].y + m[0].z*m[0].z);
    REQUIRE_THAT(n, WithinAbs(1.0, 1e-12));
}

// ---------------------------------------------------------------------------
// T3-C: Anisotropy + noise — spin remains close to easy axis at low T
//
// With K=1e4 J/m³ and α=0.5:
//   τ_relax ≈ 570 ps  (time for anisotropy to pull spin back to ±z)
//   σ ≈ 4 kA/m at T=300K, dt=1ps  (thermal kick per step)
//   H_ani ≈ 20 kA/m  (restoring field near equator)
//
// Starting from ±z, the anisotropy holds the spin near the poles.
// After many steps: |<mz>| ≫ 0 even with thermal noise.
// This is NOT the equilibrium test (needs nanoseconds); it verifies that
// the Heun integrator combines anisotropy + noise with correct sign.
// ---------------------------------------------------------------------------
TEST_CASE("Anisotropy + noise: spin stays near easy axis at low T", "[thermal]") {
    const StructuredGrid grid = g1();

    Material mat   = Material::permalloy();
    mat.alpha      = 0.5;
    mat.K_uniaxial = 1e4;
    mat.easy_axis  = {0.0, 0.0, 1.0};

    VectorField3D m(grid);
    m.set_uniform({0.0, 0.0, 1.0});   // start at easy-axis pole

    // Confinement needs the barrier to beat kT: with the FDT-correct sigma,
    // K=1e4 J/m3 over (5nm)^3 gives Delta E = 1.25e-21 J, so T=5 K puts
    // Delta E / kT ~ 18 (strongly confined, <mz^2> ~ 0.88). (The old T=300 K
    // "passed" only because the pre-fix sigma was ~1/mu0 too weak; see the
    // finite-T sigma fix.)
    const Real T  = 5.0;
    const Real dt = 1e-12;

    EffectiveFieldSum heff;
    heff.add(std::make_shared<UniaxialAnisotropyField>());
    ThermalField thermal(grid, T, dt);
    HeunIntegrator heun(dt);

    double sz2 = 0;
    const int N = 5000;
    for (int step = 0; step < N; ++step) {
        heun.step(m, mat, heff, &thermal);
        sz2 += m[0].z * m[0].z;
    }
    const double mz2 = sz2 / N;

    INFO("<mz²> over " << N << " steps = " << mz2
         << "  (anisotropy keeps spin near ±z)");

    // Spin should remain predominantly along z due to anisotropy
    REQUIRE(mz2 > 0.85);

    // |m|=1 throughout
    const Real n = std::sqrt(m[0].x*m[0].x + m[0].y*m[0].y + m[0].z*m[0].z);
    REQUIRE_THAT(n, WithinAbs(1.0, 1e-12));
}

// ===========================================================================
// T4: Néel-Brown thermally activated switching
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: measure mean first-passage time from mz≈+1 to mz < -0.5.
// Returns mean steps over N_real realizations (timeout = max_steps).
// ---------------------------------------------------------------------------
static double mean_fpt(Real T_K, int N_real, int max_steps,
                        Real K_anis, Real dt_s) {
    const StructuredGrid grid(1, 1, 1, 5e-9, 5e-9, 5e-9);
    Material mat   = Material::permalloy();
    mat.alpha      = 0.5;
    mat.K_uniaxial = K_anis;
    mat.easy_axis  = {0.0, 0.0, 1.0};

    EffectiveFieldSum heff;
    heff.add(std::make_shared<UniaxialAnisotropyField>());

    long long total_steps = 0;
    int n_switched = 0;

    for (int r = 0; r < N_real; ++r) {
        VectorField3D m(grid);
        m.set_uniform({0.0, 0.0, 1.0});

        ThermalField thermal(grid, T_K, dt_s,
                              static_cast<unsigned>(r * 7919 + 1));
        HeunIntegrator heun(dt_s);

        // Equilibrate
        for (int s = 0; s < 100; ++s) heun.step(m, mat, heff, &thermal);

        // Wait for first crossing to -z hemisphere
        int steps = 0;
        while (m[0].z > -0.5 && steps < max_steps) {
            heun.step(m, mat, heff, &thermal);
            ++steps;
        }
        total_steps += steps;
        if (m[0].z <= -0.5) ++n_switched;
    }
    // Return mean steps (only for realizations that actually switched)
    return (n_switched > 0) ? static_cast<double>(total_steps) / n_switched
                            : static_cast<double>(max_steps);
}

// ---------------------------------------------------------------------------
// Note on Néel-Brown with physical Permalloy parameters
// -----------------------------------------------------
// For K=1e5 J/m³, V=(5nm)³, α=0.5, T=500K:
//   σ ≈ 432 A/m  but  H_ani ≈ 200 kA/m  →  σ/H_ani ≈ 0.002.
// The equilibration time ∝ (H_ani/σ)² ≈ 250 000 steps — impractical.
//
// Instead T4 tests the ATTEMPT FREQUENCY (K=0, free diffusion):
//   τ_attempt ∝ 1/σ² ∝ dt/T
// This is the τ₀ denominator of the Néel-Brown formula.
// At T_high = 10 × T_low:  τ_attempt(T_low)/τ_attempt(T_high) ≈ 10.
// Parameters: extreme T (1e7–1e8 K), dt=100 ps, K=0 so σ/H_ani is large.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// T4-A: Attempt frequency — lower T gives longer free-diffusion crossing time
//
// K=0 (no barrier): spin free-diffuses on the sphere driven only by H_th.
// Crossing time τ ∝ 1/D_θ ∝ 1/σ² ∝ dt/T.
// At T_high = 10 × T_low:  τ(T_low)/τ(T_high) ≈ 10.
// These parameters (T=1e7/1e8 K, dt=100ps) give σ≈8/25 kA/m — fast mixing.
// ---------------------------------------------------------------------------
TEST_CASE("Neel-Brown attempt freq: lower T gives longer crossing time", "[thermal]") {
    const Real K  = 0.0;       // no barrier — pure free diffusion
    const Real dt = 1e-12;     // 1 ps

    // With the FDT-correct sigma, PHYSICAL temperatures give measurable (non-
    // saturated) free-diffusion crossing times: tau ∝ 1/D_theta ∝ 1/sigma^2 ∝ 1/T.
    const double tau_low  = mean_fpt(100.0,  30, 20000, K, dt);   // T = 100 K
    const double tau_high = mean_fpt(1000.0, 30, 20000, K, dt);   // T = 1000 K

    INFO("tau(1e7K)=" << tau_low << "  tau(1e8K)=" << tau_high
          << "  ratio=" << tau_low/tau_high);

    // Lower T → slower diffusion → longer crossing time
    REQUIRE(tau_low > tau_high);
}

// ---------------------------------------------------------------------------
// T4-B: Scaling τ ∝ 1/T for free diffusion
//
// With K=0:  D_θ ∝ σ² ∝ T  →  τ_FPT ∝ 1/T
// At T_high = 10 × T_low:  ratio ≈ 10.
// Accept factor 2 tolerance (small N_r=30, Poisson statistics).
// ---------------------------------------------------------------------------
TEST_CASE("Neel-Brown: crossing time scales as 1/T for free diffusion", "[thermal]") {
    const Real K  = 0.0;
    const Real dt = 1e-12;
    const int  N_r = 50;

    const double tau_lo = mean_fpt(100.0,  N_r, 40000, K, dt);   // T = 100 K
    const double tau_hi = mean_fpt(1000.0, N_r, 40000, K, dt);   // T = 1000 K
    const double ratio  = tau_lo / tau_hi;

    INFO("tau(T=1e7K)=" << tau_lo << "  tau(T=1e8K)=" << tau_hi
          << "  ratio=" << ratio << "  expected≈10");

    // τ ∝ 1/T: ratio should be close to T_hi/T_lo = 10
    // Accept [3, 30] (factor-3 tolerance for N_r=50 realizations)
    REQUIRE(ratio > 3.0);
    REQUIRE(ratio < 30.0);
}
