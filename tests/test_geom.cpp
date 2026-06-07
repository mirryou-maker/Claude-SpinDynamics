#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "micromag/geom_mask.hpp"
#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/exchange.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Index count_inside(const GeomMask& mask) {
    Index n = 0;
    for (Index i = 0; i < mask.size(); ++i)
        if (mask[i] >= Real{0.5}) ++n;
    return n;
}

static Index count_inside_layer(const GeomMask& mask, Index iz) {
    const auto& g = mask.grid();
    Index n = 0;
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix)
        if (mask(ix, iy, iz) >= Real{0.5}) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// B1-2: Primitive shape factory functions
// ---------------------------------------------------------------------------

TEST_CASE("GeomMask: ellipse — centre inside, corner outside", "[geom]") {
    // 20×20×1 grid, 5 nm cells → 100 nm × 100 nm box
    StructuredGrid g(20, 20, 1, 5e-9, 5e-9, 5e-9);
    auto mask = ellipse(g, 40e-9, 20e-9);  // a=40 nm, b=20 nm

    // Centre cell (ix=10, iy=10) is at centred coords (0,0) → inside
    REQUIRE(mask(10, 10, 0) == Real{1});
    // Corner (0,0) is at centred (-47.5 nm, -47.5 nm) → outside
    REQUIRE(mask(0, 0, 0) == Real{0});

    // Cell count ≈ π·a·b / (dx·dy) = π·8·4 ≈ 100.5 → expect 95–105
    const Index n = count_inside(mask);
    REQUIRE(n >= 95);
    REQUIRE(n <= 105);
}

TEST_CASE("GeomMask: circle — cell count ≈ π·r²/dx²", "[geom]") {
    StructuredGrid g(20, 20, 1, 5e-9, 5e-9, 5e-9);
    auto mask = circle(g, 30e-9);  // r=30 nm → 6 cells radius

    REQUIRE(mask(10, 10, 0) == Real{1});
    REQUIRE(mask(0,  0,  0) == Real{0});

    // π·(30/5)² = π·36 ≈ 113 cells
    const Index n = count_inside(mask);
    REQUIRE(n >= 108);
    REQUIRE(n <= 118);
}

TEST_CASE("GeomMask: rect — exact cell count", "[geom]") {
    StructuredGrid g(20, 20, 1, 5e-9, 5e-9, 5e-9);
    // lx=60 nm → 12 cells, ly=40 nm → 8 cells → exactly 96 cells
    auto mask = rect(g, 60e-9, 40e-9);

    REQUIRE(mask(10, 10, 0) == Real{1});
    REQUIRE(mask(0,  0,  0) == Real{0});
    REQUIRE(count_inside(mask) == 96);
}

TEST_CASE("GeomMask: cylinder — z-layer clipping", "[geom]") {
    // 20×20×4 grid, 5 nm cells → box height = 20 nm
    // box_cz = 10 nm; cylinder h=10 nm → hz=5 nm
    // iz=0: z_c = -7.5 nm → |z|>5 → outside
    // iz=1: z_c = -2.5 nm → inside
    // iz=2: z_c = +2.5 nm → inside
    // iz=3: z_c = +7.5 nm → outside
    StructuredGrid g(20, 20, 4, 5e-9, 5e-9, 5e-9);
    auto mask = cylinder(g, 30e-9, 10e-9);

    REQUIRE(mask(10, 10, 0) == Real{0});  // outside z
    REQUIRE(mask(10, 10, 1) == Real{1});  // inside
    REQUIRE(mask(10, 10, 2) == Real{1});  // inside
    REQUIRE(mask(10, 10, 3) == Real{0});  // outside z

    // Active layers iz=1,2 should each have circle-like cell counts
    const Index n1 = count_inside_layer(mask, 1);
    const Index n2 = count_inside_layer(mask, 2);
    REQUIRE(n1 == n2);
    REQUIRE(n1 >= 108);
    REQUIRE(n1 <= 118);

    // Inactive layers should be zero
    REQUIRE(count_inside_layer(mask, 0) == 0);
    REQUIRE(count_inside_layer(mask, 3) == 0);
}

// ---------------------------------------------------------------------------
// B1-1: Boolean combinators
// ---------------------------------------------------------------------------

TEST_CASE("GeomMask: union_ — max per cell", "[geom]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    GeomMask a(g), b(g);
    a.set_uniform(0.8);
    b.set_uniform(0.3);

    auto u = union_(a, b);
    for (Index i = 0; i < u.size(); ++i)
        REQUIRE_THAT(u[i], WithinAbs(0.8, 1e-12));
}

TEST_CASE("GeomMask: intersect_ — min per cell", "[geom]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    GeomMask a(g), b(g);
    a.set_uniform(0.8);
    b.set_uniform(0.3);

    auto res = intersect_(a, b);
    for (Index i = 0; i < res.size(); ++i)
        REQUIRE_THAT(res[i], WithinAbs(0.3, 1e-12));
}

TEST_CASE("GeomMask: sub_ — max(A-B,0) per cell", "[geom]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    GeomMask a(g), b(g);
    a.set_uniform(0.7);
    b.set_uniform(0.3);

    auto s = sub_(a, b);
    for (Index i = 0; i < s.size(); ++i)
        REQUIRE_THAT(s[i], WithinAbs(0.4, 1e-12));

    // sub_ floors at 0: B > A → 0
    b.set_uniform(1.0);
    auto s2 = sub_(a, b);
    for (Index i = 0; i < s2.size(); ++i)
        REQUIRE(s2[i] == Real{0});
}

TEST_CASE("GeomMask: boolean ops satisfy inclusion-exclusion", "[geom]") {
    StructuredGrid g(20, 20, 1, 5e-9, 5e-9, 5e-9);
    auto circ = circle(g, 30e-9);
    auto rec  = rect(g, 40e-9, 40e-9);

    auto u  = union_(circ, rec);
    auto in = intersect_(circ, rec);

    // Union ≥ each component
    REQUIRE(count_inside(u) >= count_inside(circ));
    REQUIRE(count_inside(u) >= count_inside(rec));

    // Intersection ≤ each component
    REQUIRE(count_inside(in) <= count_inside(circ));
    REQUIRE(count_inside(in) <= count_inside(rec));

    // Inclusion-exclusion: |A∪B| = |A| + |B| - |A∩B|
    REQUIRE(count_inside(u) ==
            count_inside(circ) + count_inside(rec) - count_inside(in));
}

// ---------------------------------------------------------------------------
// B1-3: Translate
// ---------------------------------------------------------------------------

TEST_CASE("GeomMask: translate — shifts by exact cell offset", "[geom]") {
    // 10×10×1, 5 nm cells; rect 20nm×20nm → 4×4=16 cells
    // centred at box centre: ix=3..6, iy=3..6
    StructuredGrid g(10, 10, 1, 5e-9, 5e-9, 5e-9);
    auto base = rect(g, 20e-9, 20e-9);

    // Shift +10 nm in x → +2 cells → ix=5..8
    auto shifted = translate(base, 10e-9, 0.0);
    REQUIRE(count_inside(shifted) == 16);
    REQUIRE(shifted(5, 3, 0) == Real{1});   // new position inside
    REQUIRE(shifted(3, 3, 0) == Real{0});   // old position now empty
}

TEST_CASE("GeomMask: translate — boundary clipping", "[geom]") {
    StructuredGrid g(10, 10, 1, 5e-9, 5e-9, 5e-9);
    auto base = rect(g, 20e-9, 20e-9);

    // Shift +40 nm → +8 cells → ix=11..14, all out of range → 0 cells
    auto shifted = translate(base, 40e-9, 0.0);
    REQUIRE(count_inside(shifted) == 0);
}

TEST_CASE("GeomMask: translate — negative shift stays in range", "[geom]") {
    StructuredGrid g(10, 10, 1, 5e-9, 5e-9, 5e-9);
    auto base = rect(g, 20e-9, 20e-9);

    // Shift -5 nm in y → -1 cell, still in range → still 16 cells
    auto shifted = translate(base, 0.0, -5e-9);
    REQUIRE(count_inside(shifted) == 16);
}

// ---------------------------------------------------------------------------
// B1-3: Rotate
// ---------------------------------------------------------------------------

TEST_CASE("GeomMask: rotate — identity (theta=0) is exact", "[geom]") {
    StructuredGrid g(20, 20, 1, 5e-9, 5e-9, 5e-9);
    auto base = ellipse(g, 40e-9, 20e-9);
    auto id   = rotate(base, 0.0);

    for (Index i = 0; i < base.size(); ++i)
        REQUIRE_THAT(id[i], WithinAbs(base[i], 1e-12));
}

TEST_CASE("GeomMask: rotate 90° swaps long and short axes", "[geom]") {
    // rect(40nm, 20nm): long axis along x (8 cols × 4 rows = 32 cells)
    StructuredGrid g(20, 20, 1, 5e-9, 5e-9, 5e-9);
    auto base  = rect(g, 40e-9, 20e-9);
    auto rot90 = rotate(base, constants::pi / 2.0);

    // Mass should be conserved within 5%
    Real mass_before = 0, mass_after = 0;
    for (Index i = 0; i < base.size(); ++i) {
        mass_before += base[i];
        mass_after  += rot90[i];
    }
    REQUIRE_THAT(mass_after, WithinRel(mass_before, 0.05));

    // After 90°, centre column should be longer than centre row
    const auto& gg = g;
    Index col_sum = 0, row_sum = 0;
    for (Index iy = 0; iy < gg.ny(); ++iy)
        if (rot90(10, iy, 0) >= Real{0.5}) ++col_sum;
    for (Index ix = 0; ix < gg.nx(); ++ix)
        if (rot90(ix, 10, 0) >= Real{0.5}) ++row_sum;
    REQUIRE(col_sum > row_sum);
}

// ---------------------------------------------------------------------------
// B1-4: apply_mask
// ---------------------------------------------------------------------------

TEST_CASE("GeomMask: apply_mask zeros m outside geometry", "[geom]") {
    StructuredGrid g(10, 10, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});

    auto mask = circle(g, 20e-9);
    m.apply_mask(mask);

    // Corner is outside → m = (0,0,0)
    REQUIRE_THAT(m.at(0, 0, 0).x, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(m.at(0, 0, 0).y, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(m.at(0, 0, 0).z, WithinAbs(0.0, 1e-15));

    // Centre is inside → m = (1,0,0)
    REQUIRE_THAT(m.at(5, 5, 0).x, WithinAbs(1.0, 1e-15));
}

TEST_CASE("GeomMask: apply_mask scales m by edge-smoothed value", "[geom]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    m.set_uniform({0, 1, 0});

    GeomMask mask(g);
    mask.set_uniform(0.5);
    m.apply_mask(mask);

    for (Index i = 0; i < m.size(); ++i)
        REQUIRE_THAT(m[i].y, WithinAbs(0.5, 1e-15));
}

// ---------------------------------------------------------------------------
// B1-5: ExchangeField mask-aware Neumann BC
// ---------------------------------------------------------------------------

TEST_CASE("GeomMask: exchange + mask — uniform m gives H=0 inside geometry", "[geom]") {
    StructuredGrid g(10, 10, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({1, 0, 0});

    auto mask = circle(g, 20e-9);
    m.apply_mask(mask);

    ExchangeField exch;
    exch.set_mask(&mask);
    exch.accumulate(m, Material::permalloy(), H);

    // Inside the circle, uniform m → H_exchange must be zero
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        if (mask(ix, iy, iz) < Real{0.5}) continue;
        REQUIRE_THAT(H.at(ix, iy, iz).norm(), WithinAbs(0.0, 1e-3));
    }
}

TEST_CASE("GeomMask: exchange + mask — Neumann BC is exact for step profile", "[geom]") {
    // 5×1×1 strip; step function m, mask active only on ix=1,2
    // At ix=1: left neighbour (ix=0) has mask=0 → Neumann ghost = self (m[1]=1)
    //          Laplacian = (m[2]-m[1]) / dx^2 + (ghost-m[1]) / dx^2
    //                    = (-1-1)/dx^2 + 0 = -2/dx^2
    StructuredGrid g(5, 1, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();
    const Real dx  = 5e-9;
    const Real pre = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);

    VectorField3D m(g), H(g);
    m.at(1, 0, 0) = { 1.0, 0, 0};
    m.at(2, 0, 0) = {-1.0, 0, 0};

    GeomMask mask(g);
    mask[1] = Real{1};
    mask[2] = Real{1};

    ExchangeField exch;
    exch.set_mask(&mask);
    exch.accumulate(m, mat, H);

    // Cells outside mask → H unchanged (stays zero)
    REQUIRE_THAT(H.at(0, 0, 0).x, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(H.at(3, 0, 0).x, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(H.at(4, 0, 0).x, WithinAbs(0.0, 1e-6));

    // ix=1: Neumann at left → expected -2/dx^2 * pre
    REQUIRE_THAT(H.at(1, 0, 0).x, WithinRel(pre * (-2.0 / (dx*dx)), 1e-6));

    // ix=2: Neumann at right → expected +2/dx^2 * pre
    REQUIRE_THAT(H.at(2, 0, 0).x, WithinRel(pre * ( 2.0 / (dx*dx)), 1e-6));
}

TEST_CASE("GeomMask: exchange + mask — clear_mask restores full behaviour", "[geom]") {
    StructuredGrid g(5, 1, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();
    VectorField3D m(g), H_masked(g), H_clear(g);
    m.set_uniform({1, 0, 0});

    GeomMask mask(g);
    mask.set_uniform(1.0);

    ExchangeField exch;
    exch.set_mask(&mask);
    exch.accumulate(m, mat, H_masked);

    exch.clear_mask();
    exch.accumulate(m, mat, H_clear);

    for (Index i = 0; i < g.size(); ++i)
        REQUIRE_THAT((H_masked[i] - H_clear[i]).norm(), WithinAbs(0.0, 1e-6));
}
