// test_surface_anisotropy.cpp — CPU SurfaceAnisotropyField (interface PMA) tests
// Tag: [surface_anisotropy]

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/surface_anisotropy.hpp"
#include "micromag/types.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// H_s = (2 Ks / (μ₀ Ms t)) (m·n̂) n̂  on surface cells only (default: top+bottom layer)

TEST_CASE("SurfaceAnisotropy: only surface layers receive field", "[surface_anisotropy]") {
    StructuredGrid g(3, 3, 4, 5e-9, 5e-9, 2e-9);   // nz=4 → surface iz=0,3
    Material mat = Material::permalloy();
    const Real Ks = 1.2e-3;
    SurfaceAnisotropyField sa(Ks);                  // n_hat = z default

    VectorField3D m(g); m.set_uniform({0, 0, 1});
    VectorField3D H(g);
    for (Index i = 0; i < g.size(); ++i) H[i] = {0, 0, 0};
    sa.accumulate(m, mat, H);

    const Real t = g.dz();                            // cell thickness along z
    const Real prefac = 2.0 * Ks / (constants::mu_0 * mat.Ms * t);

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Vec3& Hi = H[g.linear_index(ix, iy, iz)];
        if (iz == 0 || iz == g.nz() - 1) {
            REQUIRE_THAT(Hi.z, WithinRel(prefac, 1e-10));
            REQUIRE_THAT(Hi.x, WithinAbs(0.0, 1e-6));
            REQUIRE_THAT(Hi.y, WithinAbs(0.0, 1e-6));
        } else {
            REQUIRE_THAT(Hi.z, WithinAbs(0.0, 1e-9));   // interior untouched
        }
    }
}

TEST_CASE("SurfaceAnisotropy: m ⊥ n_hat → zero field", "[surface_anisotropy]") {
    StructuredGrid g(3, 3, 4, 5e-9, 5e-9, 2e-9);
    Material mat = Material::permalloy();
    SurfaceAnisotropyField sa(1.2e-3);   // n_hat = z

    VectorField3D m(g); m.set_uniform({1, 0, 0});  // ⊥ z
    VectorField3D H(g);
    for (Index i = 0; i < g.size(); ++i) H[i] = {0, 0, 0};
    sa.accumulate(m, mat, H);

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("SurfaceAnisotropy: energy over surface cells", "[surface_anisotropy]") {
    StructuredGrid g(4, 3, 4, 5e-9, 5e-9, 2e-9);
    Material mat = Material::permalloy();
    const Real Ks = 9.5e-4;
    SurfaceAnisotropyField sa(Ks);

    VectorField3D m(g); m.set_uniform({0, 0, 1});  // along n̂

    // E = Σ_surface -Ks (m·n̂)² (dV/t); m·n̂=1, dV/t = dx*dy, surface cells = nx*ny*2
    const Real dxdy = g.dx() * g.dy();
    const Index n_surf = g.nx() * g.ny() * 2;
    const Real E_expect = -Ks * dxdy * static_cast<Real>(n_surf);

    REQUIRE_THAT(sa.energy(m, mat), WithinRel(E_expect, 1e-10));
}

TEST_CASE("SurfaceAnisotropy: n_hat=x selects x-faces", "[surface_anisotropy]") {
    StructuredGrid g(4, 3, 3, 2e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    const Real Ks = 1.0e-3;
    SurfaceAnisotropyField sa(Ks, {1, 0, 0});   // x-normal

    VectorField3D m(g); m.set_uniform({1, 0, 0});
    VectorField3D H(g);
    for (Index i = 0; i < g.size(); ++i) H[i] = {0, 0, 0};
    sa.accumulate(m, mat, H);

    const Real t = g.dx();
    const Real prefac = 2.0 * Ks / (constants::mu_0 * mat.Ms * t);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Real hx = H[g.linear_index(ix, iy, iz)].x;
        if (ix == 0 || ix == g.nx() - 1)
            REQUIRE_THAT(hx, WithinRel(prefac, 1e-10));
        else
            REQUIRE_THAT(hx, WithinAbs(0.0, 1e-9));
    }
}
