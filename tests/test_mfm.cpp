#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>
#include <algorithm>

#include "micromag/mfm.hpp"
#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Test 1: Uniform mz → signal = 0  (k=0 mode zeroed)
// ---------------------------------------------------------------------------

TEST_CASE("MFMImage: uniform mz=1 gives zero signal (dipole)", "[mfm]") {
    StructuredGrid g(16, 16, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({0, 0, 1});

    MFMImage mfm(g, 50e-9, TipMode::Dipole);
    auto sig = mfm.compute(m, Material::permalloy());

    for (Real v : sig)
        REQUIRE_THAT(v, WithinAbs(0.0, 1e-3));
}

TEST_CASE("MFMImage: uniform mz=1 gives zero signal (monopole)", "[mfm]") {
    StructuredGrid g(16, 16, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({0, 0, 1});

    MFMImage mfm(g, 50e-9, TipMode::Monopole);
    auto sig = mfm.compute(m, Material::permalloy());

    for (Real v : sig)
        REQUIRE_THAT(v, WithinAbs(0.0, 1e-3));
}

TEST_CASE("MFMImage: uniform mz=0 gives zero signal", "[mfm]") {
    StructuredGrid g(16, 16, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});   // in-plane: mz = 0 everywhere

    MFMImage mfm(g, 50e-9);
    auto sig = mfm.compute(m, Material::permalloy());

    for (Real v : sig)
        REQUIRE_THAT(v, WithinAbs(0.0, 1e-30));
}

// ---------------------------------------------------------------------------
// Test 2: Domain wall — dipole peak at boundary, monopole sign change
//
// mz = +1 for ix < nx/2,  mz = -1 for ix >= nx/2
// Domain wall is at x = nx/2 * dx.
// ---------------------------------------------------------------------------

TEST_CASE("MFMImage: dipole signal peaks at domain wall", "[mfm]") {
    const Index nx = 32, ny = 8;
    StructuredGrid g(nx, ny, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);

    // Step profile in mz along x
    for (Index iz = 0; iz < 1; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix)
        m.at(ix, iy, iz) = {0, 0, (ix < nx/2) ? Real{1} : Real{-1}};

    MFMImage mfm(g, 10e-9, TipMode::Dipole);
    auto sig = mfm.compute(m, Material::permalloy());

    // Find the ix with maximum absolute signal (averaged over y)
    std::vector<Real> row_avg(nx, Real{0});
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix)
        row_avg[ix] += std::abs(sig[static_cast<std::size_t>(iy*nx + ix)]);
    for (Real& v : row_avg) v /= static_cast<Real>(ny);

    const auto peak_it = std::max_element(row_avg.begin(), row_avg.end());
    const Index peak_ix = static_cast<Index>(peak_it - row_avg.begin());

    // With periodic FFT there are two domain walls: at ix=nx/2 and at ix=0 (wrap).
    // Peak should be within 2 cells of either wall.
    const bool near_centre = std::abs(peak_ix - nx/2) <= 2;
    const bool near_edge   = (peak_ix <= 2 || peak_ix >= nx - 2);
    REQUIRE((near_centre || near_edge));

    // Signal away from the wall should be much smaller than at the wall
    const Real peak_val = *peak_it;
    REQUIRE(peak_val > 0.0);                        // non-trivial signal
    REQUIRE(row_avg[0] < 0.5 * peak_val);           // far from wall: small
}

TEST_CASE("MFMImage: monopole signal has opposite sign across domain wall", "[mfm]") {
    const Index nx = 32, ny = 4;
    StructuredGrid g(nx, ny, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);

    for (Index iz = 0; iz < 1; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix)
        m.at(ix, iy, iz) = {0, 0, (ix < nx/2) ? Real{1} : Real{-1}};

    MFMImage mfm(g, 10e-9, TipMode::Monopole);
    auto sig = mfm.compute(m, Material::permalloy());

    // Average signal in left quarter (should be positive) vs right quarter (negative)
    Real left_avg = Real{0}, right_avg = Real{0};
    for (Index iy = 0; iy < ny; ++iy) {
        for (Index ix = 2; ix < nx/4; ++ix)
            left_avg += sig[static_cast<std::size_t>(iy*nx + ix)];
        for (Index ix = 3*nx/4; ix < nx-2; ++ix)
            right_avg += sig[static_cast<std::size_t>(iy*nx + ix)];
    }

    // Opposite signs and both non-trivial
    REQUIRE(left_avg * right_avg < 0.0);
}

// ---------------------------------------------------------------------------
// Test 3: Larger lift → smoother (lower amplitude high-frequency content)
// ---------------------------------------------------------------------------

TEST_CASE("MFMImage: larger lift height suppresses high-frequency signal", "[mfm]") {
    const Index nx = 32, ny = 4;
    StructuredGrid g(nx, ny, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);

    // Checkerboard mz: alternating +1/-1 per cell → high spatial frequency
    for (Index iz = 0; iz < 1; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix)
        m.at(ix, iy, iz) = {0, 0, ((ix + iy) % 2 == 0) ? Real{1} : Real{-1}};

    MFMImage mfm_near(g, 5e-9,  TipMode::Dipole);
    MFMImage mfm_far (g, 50e-9, TipMode::Dipole);

    auto sig_near = mfm_near.compute(m, Material::permalloy());
    auto sig_far  = mfm_far .compute(m, Material::permalloy());

    // RMS of signal should be larger for near (high frequencies preserved)
    Real rms_near = Real{0}, rms_far = Real{0};
    for (Real v : sig_near) rms_near += v * v;
    for (Real v : sig_far)  rms_far  += v * v;

    REQUIRE(rms_near > rms_far * Real{10});   // near should be >> far
}

// ---------------------------------------------------------------------------
// Test 4: Monopole vs dipole produce different signals
// ---------------------------------------------------------------------------

TEST_CASE("MFMImage: monopole and dipole modes differ for non-uniform mz", "[mfm]") {
    const Index nx = 16, ny = 4;
    StructuredGrid g(nx, ny, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);

    for (Index iz = 0; iz < 1; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix)
        m.at(ix, iy, iz) = {0, 0, (ix < nx/2) ? Real{1} : Real{-1}};

    MFMImage mfm_mono(g, 10e-9, TipMode::Monopole);
    MFMImage mfm_dip (g, 10e-9, TipMode::Dipole);
    const Material mat = Material::permalloy();

    auto sig_mono = mfm_mono.compute(m, mat);
    auto sig_dip  = mfm_dip .compute(m, mat);

    // Signals should differ (max absolute difference > tolerance)
    Real max_diff = Real{0};
    for (std::size_t i = 0; i < sig_mono.size(); ++i)
        max_diff = std::max(max_diff, std::abs(sig_mono[i] - sig_dip[i]));

    REQUIRE(max_diff > Real{1e3});   // physically distinct signals
}

// ---------------------------------------------------------------------------
// Test 5: Signal scales linearly with Ms
// ---------------------------------------------------------------------------

TEST_CASE("MFMImage: signal scales linearly with Ms", "[mfm]") {
    const Index nx = 16, ny = 4;
    StructuredGrid g(nx, ny, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);

    for (Index iz = 0; iz < 1; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix)
        m.at(ix, iy, iz) = {0, 0, (ix < nx/2) ? Real{1} : Real{-1}};

    Material mat1 = Material::permalloy();
    Material mat2 = Material::permalloy();
    mat2.Ms *= 2.0;

    MFMImage mfm(g, 20e-9, TipMode::Dipole);
    auto sig1 = mfm.compute(m, mat1);
    auto sig2 = mfm.compute(m, mat2);

    // Each output value of sig2 should be exactly 2× sig1
    for (std::size_t i = 0; i < sig1.size(); ++i) {
        if (std::abs(sig1[i]) > 1e-6)   // skip near-zero cells
            REQUIRE_THAT(sig2[i] / sig1[i], WithinRel(2.0, 1e-6));
    }
}
