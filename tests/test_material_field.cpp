#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/material_field.hpp"
#include "micromag/grid.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// C1-1: MaterialField3D — per-cell material parameters
// ---------------------------------------------------------------------------

TEST_CASE("MaterialField3D: uniform construction matches Material", "[material]") {
    StructuredGrid g(4, 3, 2, 1e-9, 1e-9, 1e-9);
    Material mat = Material::cobalt();
    MaterialField3D matf(g, mat);

    REQUIRE(matf.size() == g.size());
    for (Index idx = 0; idx < matf.size(); ++idx) {
        REQUIRE(matf.Ms(idx)         == mat.Ms);
        REQUIRE(matf.A_exchange(idx) == mat.A_exchange);
        REQUIRE(matf.K_uniaxial(idx) == mat.K_uniaxial);
        REQUIRE(matf.alpha(idx)      == mat.alpha);
        REQUIRE(matf.easy_axis(idx).x == mat.easy_axis.x);
        REQUIRE(matf.easy_axis(idx).z == mat.easy_axis.z);
    }

    Material c5 = matf[5];
    REQUIRE(c5.Ms == mat.Ms);
    REQUIRE(c5.K_uniaxial == mat.K_uniaxial);
}

TEST_CASE("MaterialField3D: default construction uses default Material", "[material]") {
    StructuredGrid g(2, 2, 1, 1e-9, 1e-9, 1e-9);
    MaterialField3D matf(g);
    Material def{};

    REQUIRE(matf.Ms(0)         == def.Ms);
    REQUIRE(matf.A_exchange(0) == def.A_exchange);
    REQUIRE(matf.alpha(0)      == def.alpha);
}

TEST_CASE("MaterialField3D: set_uniform overwrites all cells", "[material]") {
    StructuredGrid g(3, 3, 1, 1e-9, 1e-9, 1e-9);
    MaterialField3D matf(g, Material::permalloy());

    matf.set_uniform(Material::iron());
    Material iron = Material::iron();
    for (Index idx = 0; idx < matf.size(); ++idx) {
        REQUIRE(matf.Ms(idx) == iron.Ms);
        REQUIRE(matf.K_uniaxial(idx) == iron.K_uniaxial);
    }
}

TEST_CASE("MaterialField3D: per-cell mutation via component fields", "[material]") {
    StructuredGrid g(4, 4, 1, 1e-9, 1e-9, 1e-9);
    MaterialField3D matf(g, Material::permalloy());

    // Vary K and easy_axis in the right half of the grid (Regions-style).
    const Real K_region = 5e5;
    const Vec3 axis_region{1, 0, 0};
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        if (ix >= g.nx() / 2) {
            Index idx = g.linear_index(ix, iy, 0);
            matf.K_field()[idx] = K_region;
            matf.easy_axis_field()[idx] = axis_region;
            matf.alpha_field()[idx] = 0.1;
        }
    }

    // Left half retains permalloy defaults.
    Material left = matf.at(0, 0, 0);
    REQUIRE_THAT(left.K_uniaxial, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(left.alpha, WithinAbs(0.02, 1e-12));

    // Right half sees the modified region values.
    Material right = matf.at(g.nx() - 1, 0, 0);
    REQUIRE_THAT(right.K_uniaxial, WithinAbs(K_region, 1e-12));
    REQUIRE_THAT(right.alpha, WithinAbs(0.1, 1e-12));
    REQUIRE_THAT(right.easy_axis.x, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(right.easy_axis.z, WithinAbs(0.0, 1e-12));

    // Ms untouched everywhere (still permalloy).
    REQUIRE_THAT(left.Ms, WithinAbs(right.Ms, 1e-6));
}

TEST_CASE("MaterialField3D: component fields share grid layout with VectorField3D", "[material]") {
    StructuredGrid g(5, 2, 2, 2e-9, 2e-9, 2e-9);
    MaterialField3D matf(g, Material::cobalt());

    REQUIRE(&matf.Ms_field().grid()        == &g);
    REQUIRE(&matf.A_field().grid()         == &g);
    REQUIRE(&matf.K_field().grid()         == &g);
    REQUIRE(&matf.alpha_field().grid()     == &g);
    REQUIRE(&matf.easy_axis_field().grid() == &g);
    REQUIRE(matf.Ms_field().size() == g.size());

    // Linear-index addressing matches grid().linear_index (x-fastest).
    Index idx = g.linear_index(2, 1, 1);
    matf.Ms_field()[idx] = 1.23e6;
    REQUIRE(matf.at(2, 1, 1).Ms == 1.23e6);
}
