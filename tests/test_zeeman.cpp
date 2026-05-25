#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/zeeman.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Zeeman: H_eff equals H_ext at every cell", "[zeeman]") {
    StructuredGrid g(4, 5, 6, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({1, 0, 0});

    Vec3 H_ext{1e5, 2e5, -3e5};
    ZeemanField z(H_ext);
    Material mat = Material::permalloy();
    z.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].x, WithinAbs(H_ext.x, 1e-6));
        REQUIRE_THAT(H[idx].y, WithinAbs(H_ext.y, 1e-6));
        REQUIRE_THAT(H[idx].z, WithinAbs(H_ext.z, 1e-6));
    }
}

TEST_CASE("Zeeman: accumulate adds to existing H_out", "[zeeman]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({0, 0, 1});
    for (Index idx = 0; idx < g.size(); ++idx) H[idx] = {10, 20, 30};

    ZeemanField z({1, 2, 3});
    z.accumulate(m, Material::permalloy(), H);

    REQUIRE_THAT(H.at(0,0,0).x, WithinAbs(11.0, 1e-6));
    REQUIRE_THAT(H.at(0,0,0).y, WithinAbs(22.0, 1e-6));
    REQUIRE_THAT(H.at(0,0,0).z, WithinAbs(33.0, 1e-6));
}

TEST_CASE("Zeeman energy: aligned m gives -mu0 Ms |H| V", "[zeeman][energy]") {
    StructuredGrid g(4, 4, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    ZeemanField z({1e5, 0, 0});
    Material mat = Material::permalloy();
    Real E = z.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    Real expected = -constants::mu_0 * mat.Ms * 1e5 * V_total;
    REQUIRE_THAT(E, WithinRel(expected, 1e-12));
}

TEST_CASE("Zeeman energy: antiparallel m gives +mu0 Ms |H| V", "[zeeman][energy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({0, 0, -1});

    ZeemanField z({0, 0, 1e5});
    Material mat = Material::permalloy();
    Real E = z.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    Real expected = +constants::mu_0 * mat.Ms * 1e5 * V_total;
    REQUIRE_THAT(E, WithinRel(expected, 1e-12));
}

TEST_CASE("Zeeman energy: perpendicular m gives zero", "[zeeman][energy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    ZeemanField z({0, 0, 1e5});
    REQUIRE_THAT(z.energy(m, Material::permalloy()), WithinAbs(0.0, 1e-30));
}
