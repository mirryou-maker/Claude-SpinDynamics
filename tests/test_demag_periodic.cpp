#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "micromag/demag_periodic.hpp"
#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
static Vec3 mean_field(const VectorField3D& H) {
    Vec3 s{0,0,0};
    for (Index i = 0; i < H.size(); ++i) { s.x+=H[i].x; s.y+=H[i].y; s.z+=H[i].z; }
    const double N = static_cast<double>(H.size());
    return {s.x/N, s.y/N, s.z/N};
}

// ---------------------------------------------------------------------------
// 1. Uniform magnetisation → H_demag = 0 everywhere (k=0 mode zeroed)
//    Physical interpretation: for periodic BC the uniform mode produces
//    no demagnetising field (toroidal / "ring" geometry analogy).
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldPeriodic: uniform m → zero demag field", "[demag_periodic]") {
    // 8×8×4 grid — not cubic so we can distinguish x/y/z in later tests
    const StructuredGrid grid(8, 8, 4, 5e-9, 5e-9, 5e-9);
    DemagFieldPeriodic demag(grid);
    const Material mat = Material::permalloy();

    // Uniform +x
    VectorField3D m(grid);  m.set_uniform({1,0,0});
    VectorField3D H(grid);  for (Index i=0;i<H.size();++i) H[i]={0,0,0};
    demag.accumulate(m, mat, H);

    for (Index i = 0; i < H.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, 1.0));   // 1 A/m tolerance on ~800 kA/m scale
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1.0));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1.0));
    }

    // Uniform +z
    m.set_uniform({0,0,1});
    for (Index i=0;i<H.size();++i) H[i]={0,0,0};
    demag.accumulate(m, mat, H);

    for (Index i = 0; i < H.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, 1.0));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1.0));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1.0));
    }
}

// ---------------------------------------------------------------------------
// 2. Uniform m → energy = 0
//    Follows directly from H = 0 for uniform m.
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldPeriodic: uniform m → zero energy", "[demag_periodic]") {
    const StructuredGrid grid(8, 8, 4, 5e-9, 5e-9, 5e-9);
    DemagFieldPeriodic demag(grid);
    const Material mat = Material::permalloy();

    VectorField3D m(grid);  m.set_uniform({1,0,0});
    REQUIRE_THAT(demag.energy(m, mat), WithinAbs(0.0, 1e-20));

    m.set_uniform({0,0,1});
    REQUIRE_THAT(demag.energy(m, mat), WithinAbs(0.0, 1e-20));
}

// ---------------------------------------------------------------------------
// 3. Non-uniform m → non-zero field (sanity that the FFT pipeline works)
//    Use alternating ±x magnetisation (checkerboard in x).
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldPeriodic: alternating mx → non-zero field", "[demag_periodic]") {
    const StructuredGrid grid(8, 4, 2, 5e-9, 5e-9, 5e-9);
    DemagFieldPeriodic demag(grid);
    const Material mat = Material::permalloy();

    // Alternating +x / -x along the x direction
    VectorField3D m(grid);
    for (Index iz=0; iz<grid.nz(); ++iz)
    for (Index iy=0; iy<grid.ny(); ++iy)
    for (Index ix=0; ix<grid.nx(); ++ix) {
        Index idx = ix + grid.nx()*(iy + grid.ny()*iz);
        m[idx] = (ix % 2 == 0) ? Vec3{1,0,0} : Vec3{-1,0,0};
    }

    VectorField3D H(grid);  for (Index i=0;i<H.size();++i) H[i]={0,0,0};
    demag.accumulate(m, mat, H);

    // RMS of Hx should be non-zero for this pattern
    double rms_Hx = 0.0;
    for (Index i=0; i<H.size(); ++i) rms_Hx += H[i].x*H[i].x;
    rms_Hx = std::sqrt(rms_Hx / static_cast<double>(H.size()));
    REQUIRE(rms_Hx > 1e3);  // must be at least 1 A/m
}

// ---------------------------------------------------------------------------
// 4. Non-uniform m → energy > 0
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldPeriodic: non-uniform m → positive energy", "[demag_periodic]") {
    const StructuredGrid grid(8, 4, 2, 5e-9, 5e-9, 5e-9);
    DemagFieldPeriodic demag(grid);
    const Material mat = Material::permalloy();

    VectorField3D m(grid);
    for (Index iz=0; iz<grid.nz(); ++iz)
    for (Index iy=0; iy<grid.ny(); ++iy)
    for (Index ix=0; ix<grid.nx(); ++ix) {
        Index idx = ix + grid.nx()*(iy + grid.ny()*iz);
        m[idx] = (ix % 2 == 0) ? Vec3{1,0,0} : Vec3{-1,0,0};
    }

    REQUIRE(demag.energy(m, mat) > 0.0);
}

// ---------------------------------------------------------------------------
// 5. Compare periodic vs open-BC demag for a large-cell thin configuration
//    where both should agree closely for the non-uniform part.
//    Specifically: for a single-period cosine wave in x, the periodic
//    demag field should be antisymmetric and zero-mean (H·x is odd in x).
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldPeriodic: cosine wave → antisymmetric Hx", "[demag_periodic]") {
    const int nx = 16;
    const StructuredGrid grid(nx, 4, 2, 5e-9, 5e-9, 5e-9);
    DemagFieldPeriodic demag(grid);
    const Material mat = Material::permalloy();

    // m = cos(2pi*ix/nx) * x̂  — one full period fits the cell
    VectorField3D m(grid);
    for (Index iz=0; iz<grid.nz(); ++iz)
    for (Index iy=0; iy<grid.ny(); ++iy)
    for (Index ix=0; ix<grid.nx(); ++ix) {
        Index idx = ix + grid.nx()*(iy + grid.ny()*iz);
        double val = std::cos(2.0*constants::pi*static_cast<double>(ix)/nx);
        m[idx] = Vec3{val, 0.0, 0.0};
    }

    VectorField3D H(grid);  for (Index i=0;i<H.size();++i) H[i]={0,0,0};
    demag.accumulate(m, mat, H);

    // Mean Hx must be (near) zero — cosine wave is zero-mean
    Vec3 avg = mean_field(H);
    REQUIRE_THAT(avg.x, WithinAbs(0.0, mat.Ms * 0.01));  // < 1% of Ms
    REQUIRE_THAT(avg.y, WithinAbs(0.0, mat.Ms * 0.01));
    REQUIRE_THAT(avg.z, WithinAbs(0.0, mat.Ms * 0.01));

    // H at ix=0 and ix=nx/2 should have opposite sign (antisymmetry of cosine demag)
    const Index i0    = 0 + grid.nx()*(0 + grid.ny()*0);
    const Index i_mid = nx/2 + grid.nx()*(0 + grid.ny()*0);
    REQUIRE(H[i0].x * H[i_mid].x < 0);   // opposite signs
}

// ---------------------------------------------------------------------------
// 6. Name and interface
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldPeriodic: name and IEffectiveField interface", "[demag_periodic]") {
    const StructuredGrid grid(4, 4, 2, 5e-9, 5e-9, 5e-9);
    DemagFieldPeriodic demag(grid);
    REQUIRE(std::string(demag.name()) == "DemagPeriodic");

    // Check it can be used via base pointer
    std::shared_ptr<IEffectiveField> ptr =
        std::make_shared<DemagFieldPeriodic>(grid);
    REQUIRE(std::string(ptr->name()) == "DemagPeriodic");
}
