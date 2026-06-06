#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/zeeman_spatial.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("ZeemanFieldSpatial: uniform H reproduces ZeemanField result", "[zeeman]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();
    const double Hx = 1e4;

    VectorField3D H_ext(g);
    H_ext.set_uniform({Hx, 0, 0});

    ZeemanFieldSpatial zs(H_ext);

    VectorField3D m(g); m.set_uniform({1, 0, 0});
    VectorField3D H_out(g);
    for (Index i = 0; i < H_out.size(); ++i) H_out[i] = {0,0,0};
    zs.accumulate(m, mat, H_out);

    for (Index i = 0; i < H_out.size(); ++i) {
        REQUIRE_THAT(H_out[i].x, WithinAbs(Hx, 1e-6));
        REQUIRE_THAT(H_out[i].y, WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("ZeemanFieldSpatial: spatially varying field", "[zeeman]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();

    VectorField3D H_ext(g);
    // Left half: H along +x, right half: H along -x
    for (Index iz=0; iz<1; ++iz)
    for (Index iy=0; iy<4; ++iy)
    for (Index ix=0; ix<4; ++ix)
        H_ext.at(ix, iy, iz) = {ix < 2 ? 1e4 : -1e4, 0, 0};

    ZeemanFieldSpatial zs(H_ext);
    VectorField3D m(g); m.set_uniform({1,0,0});
    VectorField3D H_out(g);
    for (Index i=0; i<H_out.size(); ++i) H_out[i]={0,0,0};
    zs.accumulate(m, mat, H_out);

    REQUIRE_THAT(H_out.at(0, 0, 0).x, WithinAbs( 1e4, 1e-6));
    REQUIRE_THAT(H_out.at(3, 0, 0).x, WithinAbs(-1e4, 1e-6));
}

TEST_CASE("ZeemanFieldSpatial: energy = -mu0*Ms*Sum(m.H)*dV", "[zeeman]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();
    const double Hx = 1e4;

    VectorField3D H_ext(g); H_ext.set_uniform({Hx, 0, 0});
    ZeemanFieldSpatial zs(H_ext);
    VectorField3D m(g); m.set_uniform({1, 0, 0});

    double E = zs.energy(m, mat);
    double E_expected = -constants::mu_0 * mat.Ms * Hx * g.size() * g.cell_volume();
    REQUIRE_THAT(E, WithinRel(E_expected, 0.001));
}

TEST_CASE("ZeemanFieldSpatial: set_field updates source at runtime", "[zeeman]") {
    StructuredGrid g(2, 2, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();

    VectorField3D H1(g); H1.set_uniform({1e4, 0, 0});
    VectorField3D H2(g); H2.set_uniform({0, 1e4, 0});

    ZeemanFieldSpatial zs(H1);
    VectorField3D m(g); m.set_uniform({1, 0, 0});
    VectorField3D H_out(g);
    for (Index i=0; i<H_out.size(); ++i) H_out[i]={0,0,0};
    zs.accumulate(m, mat, H_out);
    REQUIRE_THAT(H_out[0].x, WithinAbs(1e4, 1e-6));

    zs.set_field(H2);
    for (Index i=0; i<H_out.size(); ++i) H_out[i]={0,0,0};
    zs.accumulate(m, mat, H_out);
    REQUIRE_THAT(H_out[0].y, WithinAbs(1e4, 1e-6));
    REQUIRE_THAT(H_out[0].x, WithinAbs(0.0, 1e-6));
}
