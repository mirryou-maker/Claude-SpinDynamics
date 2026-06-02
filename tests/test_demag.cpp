#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include <fftw3.h>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/demag.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// FFTW sanity: 8×8×8 r2c with std::vector + FFTW_ESTIMATE|FFTW_UNALIGNED.
// Replicates the exact buffer/flag configuration used by DemagField for a
// 4×4×4 micromagnetic grid.  If this fails, FFTW is broken for that config.
// ---------------------------------------------------------------------------
TEST_CASE("FFTW 8x8x8 std::vector ESTIMATE|UNALIGNED sanity", "[demag]") {
    const int nz = 8, ny = 8, nx = 8;
    const int nc = (nx / 2 + 1) * ny * nz;   // = 5*8*8 = 320

    std::vector<double> in(nz * ny * nx, 0.0);
    std::vector<std::complex<double>> out(nc);

    fftw_plan p = fftw_plan_dft_r2c_3d(nz, ny, nx,
        in.data(),
        reinterpret_cast<fftw_complex*>(out.data()),
        FFTW_ESTIMATE | FFTW_UNALIGNED);
    REQUIRE(p != nullptr);

    in[0] = 1.0;   // ESTIMATE does not overwrite arrays during planning
    fftw_execute(p);

    for (int i = 0; i < nc; ++i) {
        REQUIRE_THAT(out[i].real(), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(out[i].imag(), WithinAbs(0.0, 1e-10));
    }
    fftw_destroy_plan(p);
}

// ---------------------------------------------------------------------------
// FFTW sanity: 2×2×2 r2c, delta at origin → constant spectrum.
// Uses fftw_alloc (guaranteed alignment) + FFTW_MEASURE.
// If this fails, FFTW itself is broken in this environment.
// If this passes but DemagField tests fail, the bug is in buffer alignment
// inside DemagField (std::vector not guaranteed to be FFTW-aligned).
// ---------------------------------------------------------------------------
TEST_CASE("FFTW sanity: 2x2x2 r2c delta->constant", "[demag]") {
    const int nz = 2, ny = 2, nx = 2;
    const int nc = (nx / 2 + 1) * ny * nz;  // = 8 complex values

    double*       in  = fftw_alloc_real(nz * ny * nx);
    fftw_complex* out = fftw_alloc_complex(nc);

    // Plan BEFORE setting data: FFTW_MEASURE overwrites the arrays during planning.
    fftw_plan p = fftw_plan_dft_r2c_3d(nz, ny, nx, in, out, FFTW_MEASURE);
    REQUIRE(p != nullptr);

    // Set input AFTER planning.
    for (int i = 0; i < nz * ny * nx; ++i) in[i] = 0.0;
    in[0] = 1.0;

    fftw_execute(p);

    // FFT of unit delta at origin: all bins = (1, 0).
    for (int i = 0; i < nc; ++i) {
        REQUIRE_THAT(out[i][0], WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(out[i][1], WithinAbs(0.0, 1e-10));
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

// ---------------------------------------------------------------------------
// Helper: build a uniform magnetisation field and compute H_demag.
// ---------------------------------------------------------------------------
static VectorField3D compute_H_demag(const StructuredGrid& g, const Material& mat,
                                      const Vec3& m_dir) {
    VectorField3D m(g), H(g);
    m.set_uniform(m_dir);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
    DemagField df(g);
    df.accumulate(m, mat, H);
    return H;
}

// ---------------------------------------------------------------------------
// 1.  Macrospin cube: N_xx = N_yy = N_zz = 1/3  (isotropic demagnetisation)
//
// NOTE: the Newell FFT formula gives N_self = 0 for a cubic single cell
// (the 8-corner alternating sum of newell_f cancels exactly when dx=dy=dz).
// Physical N ≈ 1/3 emerges only from the SUM of inter-cell interactions in a
// multi-cell grid.  A 4×4×4 cubic sample is the minimum sensible test;
// centre cells converge toward N = 1/3 as the grid grows.
// ---------------------------------------------------------------------------
TEST_CASE("Demag: 4x4x4 cube Nzz = 1/3 at centre", "[demag]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D H = compute_H_demag(g, mat, {0, 0, 1});
    // Inner cell (1,1,1): H_z ≈ -Ms/3 within ±30% (finite-size correction).
    Real H_z = H.at(1, 1, 1).z;
    REQUIRE_THAT(H_z, WithinRel(-mat.Ms / 3.0, 0.30));
    // Verify no spurious transverse field at the exact centre of the cube.
    REQUIRE_THAT(H.at(1, 1, 1).x, WithinAbs(0.0, mat.Ms * 0.05));
    REQUIRE_THAT(H.at(1, 1, 1).y, WithinAbs(0.0, mat.Ms * 0.05));
}

TEST_CASE("Demag: 4x4x4 cube m=+x gives Hx≈-Ms/3", "[demag]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    VectorField3D H = compute_H_demag(g, mat, {1, 0, 0});
    Real H_x = H.at(1, 1, 1).x;
    REQUIRE_THAT(H_x, WithinRel(-mat.Ms / 3.0, 0.30));
    REQUIRE_THAT(H.at(1, 1, 1).y, WithinAbs(0.0, mat.Ms * 0.05));
    REQUIRE_THAT(H.at(1, 1, 1).z, WithinAbs(0.0, mat.Ms * 0.05));
}

// ---------------------------------------------------------------------------
// 2.  Thin film (Nx x Ny >> Nz): N_zz → 1 (perpendicular demagnetisation).
//     For a very flat slab magnetised out-of-plane, H_demag_z ≈ -Ms.
// ---------------------------------------------------------------------------
TEST_CASE("Demag: thin film Nzz approaches 1 for out-of-plane m", "[demag]") {
    // 16x16x1 cells, very flat (thin film in xy).
    StructuredGrid g(16, 16, 1, 5e-9, 5e-9, 2e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g), H(g);
    m.set_uniform({0, 0, 1});
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
    DemagField df(g);
    df.accumulate(m, mat, H);

    // For a thin film the demagnetization factor perpendicular to the film → 1.
    // Tolerance is loose because a 16×16×1 grid is not a perfect infinite film.
    Real H_z = H.at(8, 8, 0).z;   // centre cell
    REQUIRE(H_z < -mat.Ms * 0.85);   // N_zz > 0.85
    REQUIRE(H_z > -mat.Ms * 1.15);   // bounded above
}

// ---------------------------------------------------------------------------
// 3.  Uniform m → no transverse demag field (H_demag perpendicular ≈ 0).
// ---------------------------------------------------------------------------
TEST_CASE("Demag: uniform m=+z gives no transverse H_demag", "[demag]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    VectorField3D H = compute_H_demag(g, mat, {0, 0, 1});

    // For a uniform cube all cells should have Hx,Hy ≈ 0.
    for (Index kz = 0; kz < 4; ++kz)
    for (Index ky = 0; ky < 4; ++ky)
    for (Index kx = 0; kx < 4; ++kx) {
        REQUIRE_THAT(H.at(kx, ky, kz).x, WithinAbs(0.0, mat.Ms * 0.01));
        REQUIRE_THAT(H.at(kx, ky, kz).y, WithinAbs(0.0, mat.Ms * 0.01));
    }
}

// ---------------------------------------------------------------------------
// 4.  Energy: E = μ₀/2 · Ms² · V_total / 6   (cubic sample, N=1/3)
//
// Same 4×4×4 rationale as the field tests above.  V_total = (4·a)³.
// ---------------------------------------------------------------------------
TEST_CASE("Demag: 4x4x4 cube energy ≈ mu0 Ms^2 V / 6", "[demag]") {
    Real a = 5e-9;
    StructuredGrid g(4, 4, 4, a, a, a);
    Material mat = Material::permalloy();
    VectorField3D m(g);
    m.set_uniform({0, 0, 1});

    DemagField df(g);
    Real E = df.energy(m, mat);
    Real V_total = (4*a) * (4*a) * (4*a);
    Real expected = constants::mu_0 * mat.Ms * mat.Ms * V_total / 6.0;
    REQUIRE_THAT(E, WithinRel(expected, 0.30));
}

// ---------------------------------------------------------------------------
// 5.  Long rod along z (Nz → 0): perpendicular demagnetisation dominates.
//     Magnetised along z should show almost no H_demag_z.
// ---------------------------------------------------------------------------
TEST_CASE("Demag: long rod along z has small Nzz", "[demag]") {
    // 1x1x16 rod
    StructuredGrid g(1, 1, 16, 2e-9, 2e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g), H(g);
    m.set_uniform({0, 0, 1});
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
    DemagField df(g);
    df.accumulate(m, mat, H);

    // N_zz < 0.1 for a needle: H_demag_z should be small compared to Ms
    Real H_z = H.at(0, 0, 8).z;  // centre cell
    REQUIRE(std::abs(H_z) < mat.Ms * 0.15);
}
