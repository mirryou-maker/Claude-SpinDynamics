#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

TEST_CASE("VectorField3D set_uniform", "[field]") {
    StructuredGrid g(3, 3, 3, 1.0, 1.0, 1.0);
    VectorField3D f(g);
    f.set_uniform({1.0, 0.0, 0.0});

    REQUIRE_THAT(f.at(0, 0, 0).x, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(f.at(2, 2, 2).x, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(f.at(1, 1, 1).y, WithinAbs(0.0, 1e-12));
}

TEST_CASE("VectorField3D normalize", "[field]") {
    StructuredGrid g(2, 2, 2, 1.0, 1.0, 1.0);
    VectorField3D f(g);
    f.set_uniform({3.0, 4.0, 0.0});  // length 5
    f.normalize();

    Vec3 v = f.at(0, 0, 0);
    REQUIRE_THAT(v.norm(), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(v.x, WithinAbs(0.6, 1e-12));
    REQUIRE_THAT(v.y, WithinAbs(0.8, 1e-12));
}

TEST_CASE("VectorField3D vortex has unit length everywhere", "[field]") {
    StructuredGrid g(16, 16, 1, 1.0, 1.0, 1.0);
    VectorField3D f(g);
    f.set_vortex(8.0, 8.0, 4.0);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(f[idx].norm(), WithinAbs(1.0, 1e-6));
    }
}

// ---------------------------------------------------------------------------
// A1: ScalarField3D + component() + crop_into()
// ---------------------------------------------------------------------------

TEST_CASE("ScalarField3D set_uniform and access", "[field]") {
    StructuredGrid g(4, 3, 2, 1e-9, 1e-9, 1e-9);
    ScalarField3D f(g);
    f.set_uniform(3.14);
    REQUIRE(f.size() == g.size());
    REQUIRE_THAT(f[0],           WithinAbs(3.14, 1e-12));
    REQUIRE_THAT(f.at(3, 2, 1),  WithinAbs(3.14, 1e-12));
}

TEST_CASE("VectorField3D::component extracts correct values", "[field]") {
    StructuredGrid g(3, 2, 1, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({1.0, 2.0, 3.0});

    ScalarField3D sx = m.component(0);
    ScalarField3D sy = m.component(1);
    ScalarField3D sz = m.component(2);

    REQUIRE(sx.size() == g.size());
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(sx[i], WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(sy[i], WithinAbs(2.0, 1e-12));
        REQUIRE_THAT(sz[i], WithinAbs(3.0, 1e-12));
    }
}

TEST_CASE("VectorField3D::crop_into copies sub-region", "[field]") {
    // 6x4x2 source grid, fill with position-encoded values
    StructuredGrid src_g(6, 4, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D src(src_g);
    for (Index iz = 0; iz < 2; ++iz)
    for (Index iy = 0; iy < 4; ++iy)
    for (Index ix = 0; ix < 6; ++ix)
        src.at(ix, iy, iz) = {(double)ix, (double)iy, (double)iz};

    // Crop [1..3] x [0..2] x [0..1] = 3x3x2 = 18 cells
    StructuredGrid dst_g(3, 3, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D dst(dst_g);
    src.crop_into(dst, 1, 3, 0, 2, 0, 1);

    // dst[0,0,0] should equal src[1,0,0]
    REQUIRE_THAT(dst.at(0, 0, 0).x, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(dst.at(0, 0, 0).y, WithinAbs(0.0, 1e-12));
    // dst[2,2,1] should equal src[3,2,1]
    REQUIRE_THAT(dst.at(2, 2, 1).x, WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(dst.at(2, 2, 1).y, WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(dst.at(2, 2, 1).z, WithinAbs(1.0, 1e-12));
}
