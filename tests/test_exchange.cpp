#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/exchange.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Exchange: uniform m gives zero H (Neumann)", "[exchange]") {
    StructuredGrid g(8, 8, 8, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({0, 0, 1});

    ExchangeField ex(BoundaryCondition::Neumann);
    ex.accumulate(m, Material::permalloy(), H);

    for (Index idx = 0; idx < g.size(); ++idx)
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-3));
}

TEST_CASE("Exchange: uniform m gives zero H (Periodic)", "[exchange]") {
    StructuredGrid g(8, 8, 8, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g), H(g);
    m.set_uniform({0.6, 0.8, 0});

    ExchangeField ex(BoundaryCondition::Periodic);
    ex.accumulate(m, Material::permalloy(), H);

    for (Index idx = 0; idx < g.size(); ++idx)
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-3));
}

TEST_CASE("Exchange energy: uniform m gives zero (both BC)", "[exchange][energy]") {
    StructuredGrid g(6, 6, 6, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g);
    m.set_uniform({1, 0, 0});
    Material mat = Material::permalloy();

    REQUIRE_THAT(ExchangeField(BoundaryCondition::Neumann).energy(m, mat),  WithinAbs(0.0, 1e-30));
    REQUIRE_THAT(ExchangeField(BoundaryCondition::Periodic).energy(m, mat), WithinAbs(0.0, 1e-30));
}

TEST_CASE("Exchange: A=0 gives zero H", "[exchange]") {
    StructuredGrid g(4, 4, 4, 1e-9, 1e-9, 1e-9);
    VectorField3D m(g), H(g);
    for (Index k = 0; k < g.nz(); ++k)
    for (Index j = 0; j < g.ny(); ++j)
    for (Index i = 0; i < g.nx(); ++i)
        m.at(i,j,k) = {static_cast<Real>(i), 0, 0};

    Material mat = Material::permalloy();
    mat.A_exchange = 0;
    ExchangeField ex;
    ex.accumulate(m, mat, H);

    for (Index idx = 0; idx < g.size(); ++idx)
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-30));
}

TEST_CASE("Exchange: 1D cosine matches analytical Laplacian (Periodic)",
          "[exchange][analytical]") {
    // m_x = cos(k x), ∇²m_x = -k² cos(k x)
    // → H_x = -(2A k² / μ₀ Ms) cos(k x)
    const Index N = 128;
    const Real  dx = 1e-10;
    const Real  L  = N * dx;
    const Real  kw = 2.0 * constants::pi / L;

    StructuredGrid g(N, 1, 1, dx, dx, dx);
    VectorField3D m(g), H(g);

    for (Index i = 0; i < N; ++i) {
        Real x = (static_cast<Real>(i) + 0.5) * dx;
        m.at(i, 0, 0) = {std::cos(kw * x), 0, 0};
    }

    Material mat = Material::permalloy();
    ExchangeField ex(BoundaryCondition::Periodic);
    ex.accumulate(m, mat, H);

    const Real amp = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms) * kw * kw;

    for (Index i = N/4; i < 3*N/4; i += 8) {
        Real x        = (static_cast<Real>(i) + 0.5) * dx;
        Real expected = -amp * std::cos(kw * x);
        REQUIRE_THAT(H.at(i,0,0).x, WithinRel(expected, 1e-2));
        REQUIRE_THAT(H.at(i,0,0).y, WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(H.at(i,0,0).z, WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("Exchange energy: 1D linear ramp has known value", "[exchange][energy]") {
    // m_x(i) = i * delta, gradient constant → E = A (delta/dx)² * (N-1) * V_cell
    const Index N     = 10;
    const Real  dx    = 1e-9;
    const Real  delta = 1e-3;

    StructuredGrid g(N, 1, 1, dx, dx, dx);
    VectorField3D m(g);
    for (Index i = 0; i < N; ++i)
        m.at(i, 0, 0) = {static_cast<Real>(i) * delta, 0, 0};

    Material mat = Material::permalloy();
    ExchangeField ex(BoundaryCondition::Neumann);
    Real E = ex.energy(m, mat);

    Real expected = mat.A_exchange * (N - 1) * (delta * delta) / (dx * dx) * g.cell_volume();
    REQUIRE_THAT(E, WithinRel(expected, 1e-12));
}
