// test_magnetoelastic.cpp — CPU MagnetoelasticField (B1/B2) physics tests
// Tag: [magnetoelastic]

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/magnetoelastic.hpp"
#include "micromag/material.hpp"
#include "micromag/types.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// H = -(2/μ₀Ms)[B1 mx exx + B2(my exy + mz exz), ...]

TEST_CASE("Magnetoelastic: zero strain → zero field", "[magnetoelastic]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    MagnetoelasticField me(-62.4e6, -27.1e6);   // Ni B1/B2, no strain set

    VectorField3D m(g); m.set_uniform({0.6, 0.0, 0.8}); m.normalize();
    VectorField3D H(g);
    for (Index i = 0; i < g.size(); ++i) H[i] = {0, 0, 0};
    me.accumulate(m, mat, H);

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("Magnetoelastic: uniaxial strain exx, m along x", "[magnetoelastic]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    const Real B1 = 6.96e6, B2 = -3.43e6;     // Fe
    const Real exx = 1e-3;
    MagnetoelasticField me(B1, B2);
    me.set_strain(exx);                        // only exx nonzero

    VectorField3D m(g); m.set_uniform({1.0, 0.0, 0.0});
    VectorField3D H(g);
    for (Index i = 0; i < g.size(); ++i) H[i] = {0, 0, 0};
    me.accumulate(m, mat, H);

    // H_x = -(2/μ₀Ms) B1 mx exx ; H_y = H_z = 0 (m has no y,z component)
    const Real expect_Hx = -2.0 / (constants::mu_0 * mat.Ms) * B1 * 1.0 * exx;
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinRel(expect_Hx, 1e-10));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Magnetoelastic: shear strain exy couples x/y", "[magnetoelastic]") {
    StructuredGrid g(3, 3, 1, 5e-9, 5e-9, 5e-9);
    Material mat = Material::cobalt();
    const Real B1 = -10.5e6, B2 = -8.0e6;
    const Real exy = 2e-3;
    MagnetoelasticField me(B1, B2);
    me.set_strain(0, 0, 0, exy, 0, 0);

    const Real inv = 1.0 / std::sqrt(2.0);
    VectorField3D m(g); m.set_uniform({inv, inv, 0.0});
    VectorField3D H(g);
    for (Index i = 0; i < g.size(); ++i) H[i] = {0, 0, 0};
    me.accumulate(m, mat, H);

    const Real pre = -2.0 / (constants::mu_0 * mat.Ms);
    // H_x = pre*B2*my*exy ; H_y = pre*B2*mx*exy ; H_z = 0
    const Real expect_Hx = pre * B2 * inv * exy;
    const Real expect_Hy = pre * B2 * inv * exy;
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinRel(expect_Hx, 1e-10));
        REQUIRE_THAT(H[i].y, WithinRel(expect_Hy, 1e-10));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Magnetoelastic: energy matches density integral", "[magnetoelastic]") {
    StructuredGrid g(4, 3, 2, 4e-9, 4e-9, 4e-9);
    Material mat = Material::permalloy();
    const Real B1 = 6.96e6, B2 = -3.43e6;
    MagnetoelasticField me(B1, B2);
    me.set_strain(1e-3, -5e-4, 2e-4, 3e-4, 1e-4, -2e-4);

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.3 + iy * 0.2 + iz * 0.1;
        m.at(ix, iy, iz) = {std::sin(phi), std::cos(phi) * 0.5, std::cos(phi)};
    }
    m.normalize();

    // Manual energy: e = B1(mx²exx+my²eyy+mz²ezz) + 2B2(mxmy exy+mymz eyz+mxmz exz)
    const Real exx = 1e-3, eyy = -5e-4, ezz = 2e-4, exy = 3e-4, exz = 1e-4, eyz = -2e-4;
    const Real dV = g.dx() * g.dy() * g.dz();
    Real E_manual = 0;
    for (Index i = 0; i < g.size(); ++i) {
        const Vec3& mi = m[i];
        E_manual += B1 * (mi.x*mi.x*exx + mi.y*mi.y*eyy + mi.z*mi.z*ezz)
                  + 2.0 * B2 * (mi.x*mi.y*exy + mi.y*mi.z*eyz + mi.x*mi.z*exz);
    }
    E_manual *= dV;

    REQUIRE_THAT(me.energy(m, mat), WithinRel(E_manual, 1e-10));
}
