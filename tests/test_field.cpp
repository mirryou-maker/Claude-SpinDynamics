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
