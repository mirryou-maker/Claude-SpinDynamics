#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/anisotropy.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Anisotropy: K=0 gives zero H", "[anisotropy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({1, 0, 0});

    UniaxialAnisotropyField a;
    a.accumulate(m, Material::permalloy(), H);  // K=0

    for (Index idx = 0; idx < g.size(); ++idx)
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-30));
}

TEST_CASE("Anisotropy: m parallel easy-axis gives H = (2K/mu0 Ms) u", "[anisotropy]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({0, 0, 1});

    Material mat = Material::cobalt();  // easy_axis = (0,0,1)
    UniaxialAnisotropyField a;
    a.accumulate(m, mat, H);

    Real expected = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);
    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H[idx].z, WithinRel(expected, 1e-10));
        REQUIRE_THAT(H[idx].x, WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(H[idx].y, WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Anisotropy: m perpendicular easy-axis gives zero H", "[anisotropy]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({1, 0, 0});

    UniaxialAnisotropyField a;
    a.accumulate(m, Material::cobalt(), H);

    for (Index idx = 0; idx < g.size(); ++idx)
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Anisotropy energy: m parallel easy-axis gives -K V_total", "[anisotropy][energy]") {
    StructuredGrid g(4, 4, 4, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    m.set_uniform({0, 0, 1});

    Material mat = Material::cobalt();
    UniaxialAnisotropyField a;
    Real E = a.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    REQUIRE_THAT(E, WithinRel(-mat.K_uniaxial * V_total, 1e-12));
}

TEST_CASE("Anisotropy energy: m perpendicular easy-axis gives zero", "[anisotropy][energy]") {
    StructuredGrid g(3, 3, 3, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    REQUIRE_THAT(UniaxialAnisotropyField{}.energy(m, Material::cobalt()), WithinAbs(0.0, 1e-30));
}

TEST_CASE("Anisotropy energy: 45 deg gives -K V/2", "[anisotropy][energy]") {
    StructuredGrid g(2, 2, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    Real s = 1.0 / std::sqrt(2.0);
    m.set_uniform({s, 0, s});

    Material mat = Material::cobalt();
    UniaxialAnisotropyField a;
    Real E = a.energy(m, mat);

    Real V_total = static_cast<Real>(g.size()) * g.cell_volume();
    REQUIRE_THAT(E, WithinRel(-mat.K_uniaxial * 0.5 * V_total, 1e-10));
}
