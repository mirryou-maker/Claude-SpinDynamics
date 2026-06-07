#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>

#include "micromag/material_field.hpp"
#include "micromag/grid.hpp"
#include "micromag/exchange.hpp"
#include "micromag/demag.hpp"
#include "micromag/demag_periodic.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

namespace {
// Fill m with a smooth, non-uniform pattern (not normalised — fine for
// testing the raw H formula, which is linear in m).
void fill_smooth(VectorField3D& m) {
    const auto& g = m.grid();
    for (Index k = 0; k < g.nz(); ++k)
    for (Index j = 0; j < g.ny(); ++j)
    for (Index i = 0; i < g.nx(); ++i)
        m.at(i, j, k) = Vec3{0.1 * i, 0.1 * j, 1.0 - 0.05 * k};
}
}  // namespace

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

// ---------------------------------------------------------------------------
// C1-2: ExchangeField — per-cell A_exchange / Ms
// ---------------------------------------------------------------------------

TEST_CASE("ExchangeField: uniform MaterialField3D matches uniform Material", "[material][exchange]") {
    StructuredGrid g(6, 5, 4, 2e-9, 2e-9, 2e-9);
    Material mat = Material::cobalt();
    MaterialField3D matf(g, mat);

    VectorField3D m(g), H_uniform(g), H_perCell(g);
    fill_smooth(m);

    ExchangeField ex_uniform(BoundaryCondition::Neumann);
    ExchangeField ex_perCell(BoundaryCondition::Neumann);
    ex_perCell.set_material_field(&matf);

    ex_uniform.accumulate(m, mat, H_uniform);
    ex_perCell.accumulate(m, mat, H_perCell);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H_perCell[idx].x, WithinAbs(H_uniform[idx].x, 1e-6));
        REQUIRE_THAT(H_perCell[idx].y, WithinAbs(H_uniform[idx].y, 1e-6));
        REQUIRE_THAT(H_perCell[idx].z, WithinAbs(H_uniform[idx].z, 1e-6));
    }

    Real E_uniform = ex_uniform.energy(m, mat);
    Real E_perCell = ex_perCell.energy(m, mat);
    REQUIRE_THAT(E_perCell, WithinAbs(E_uniform, std::abs(E_uniform) * 1e-9 + 1e-30));
}

TEST_CASE("ExchangeField: region-boundary bond uses harmonic-mean stiffness", "[material][exchange]") {
    // 1D chain along x: cells {0,1} use material A1, cells {2,3} use A2.
    // ny=nz=1 + Neumann ⇒ y/z neighbours equal self ⇒ no y/z contribution.
    StructuredGrid g(4, 1, 1, 1e-9, 1e-9, 1e-9);
    const Real Ms = 8e5, A1 = 1.3e-11, A2 = 2.6e-11;

    Material base{};
    base.Ms = Ms;
    MaterialField3D matf(g, base);
    matf.A_field()[0] = A1;  matf.A_field()[1] = A1;
    matf.A_field()[2] = A2;  matf.A_field()[3] = A2;

    VectorField3D m(g), H(g);
    m.at(0,0,0) = {0.0, 0, 1.00};
    m.at(1,0,0) = {0.2, 0, 0.98};
    m.at(2,0,0) = {0.4, 0, 0.92};
    m.at(3,0,0) = {0.6, 0, 0.80};

    ExchangeField ex(BoundaryCondition::Neumann);
    ex.set_material_field(&matf);
    ex.accumulate(m, Material{}, H);

    const Real idx2 = 1.0 / (g.dx() * g.dx());
    const Real pre  = 2.0 / (constants::mu_0 * Ms);

    auto harmonic = [](Real a, Real b) { return 2.0 * a * b / (a + b); };

    // Cell 1: left bond within region A1 (harmonic(A1,A1)=A1),
    //         right bond crosses into region A2 (harmonic(A1,A2)).
    Vec3 expected_H1 =
        (m.at(0,0,0) - m.at(1,0,0)) * (harmonic(A1, A1) * idx2 * pre) +
        (m.at(2,0,0) - m.at(1,0,0)) * (harmonic(A1, A2) * idx2 * pre);

    REQUIRE_THAT(H.at(1,0,0).x, WithinAbs(expected_H1.x, 1e-6));
    REQUIRE_THAT(H.at(1,0,0).z, WithinAbs(expected_H1.z, 1e-6));

    // Cell 2: left bond crosses A2←A1 boundary, right bond within A2.
    Vec3 expected_H2 =
        (m.at(1,0,0) - m.at(2,0,0)) * (harmonic(A2, A1) * idx2 * pre) +
        (m.at(3,0,0) - m.at(2,0,0)) * (harmonic(A2, A2) * idx2 * pre);

    REQUIRE_THAT(H.at(2,0,0).x, WithinAbs(expected_H2.x, 1e-6));
    REQUIRE_THAT(H.at(2,0,0).z, WithinAbs(expected_H2.z, 1e-6));
}

TEST_CASE("ExchangeField: per-cell field still gives zero H for uniform m", "[material][exchange]") {
    StructuredGrid g(5, 5, 1, 2e-9, 2e-9, 2e-9);
    MaterialField3D matf(g, Material::permalloy());

    // Randomly-varying A and Ms — uniform m must still give zero exchange field
    // (no gradient ⇒ no torque, regardless of spatially-varying coefficients).
    for (Index idx = 0; idx < matf.size(); ++idx) {
        matf.A_field()[idx]  = 1e-11 * (1.0 + 0.3 * (idx % 5));
        matf.Ms_field()[idx] = 6e5   * (1.0 + 0.1 * (idx % 3));
    }

    VectorField3D m(g), H(g);
    m.set_uniform({0, 0.6, 0.8});

    ExchangeField ex(BoundaryCondition::Neumann);
    ex.set_material_field(&matf);
    ex.accumulate(m, Material{}, H);

    for (Index idx = 0; idx < g.size(); ++idx)
        REQUIRE_THAT(H[idx].norm(), WithinAbs(0.0, 1e-6));
}

// ---------------------------------------------------------------------------
// C1-3: DemagField / DemagFieldPeriodic — per-cell Ms
// ---------------------------------------------------------------------------

TEST_CASE("DemagField: uniform MaterialField3D matches uniform Material", "[material][demag]") {
    StructuredGrid g(8, 8, 4, 4e-9, 4e-9, 4e-9);
    Material mat = Material::permalloy();
    MaterialField3D matf(g, mat);

    VectorField3D m(g), H_uniform(g), H_perCell(g);
    fill_smooth(m);

    DemagField demag_uniform(g);
    DemagField demag_perCell(g);
    demag_perCell.set_material_field(&matf);

    demag_uniform.accumulate(m, mat, H_uniform);
    demag_perCell.accumulate(m, mat, H_perCell);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H_perCell[idx].x, WithinAbs(H_uniform[idx].x, 1e-6));
        REQUIRE_THAT(H_perCell[idx].y, WithinAbs(H_uniform[idx].y, 1e-6));
        REQUIRE_THAT(H_perCell[idx].z, WithinAbs(H_uniform[idx].z, 1e-6));
    }

    Real E_uniform = demag_uniform.energy(m, mat);
    Real E_perCell = demag_perCell.energy(m, mat);
    REQUIRE_THAT(E_perCell, WithinAbs(E_uniform, std::abs(E_uniform) * 1e-9 + 1e-30));
}

TEST_CASE("DemagField: per-cell Ms equals uniform Ms with pre-scaled m (M = Ms*m linearity)",
          "[material][demag]") {
    // The FFT pipeline only ever sees M_i = Ms_i * m_i. A per-cell field
    // Ms_i = Ms0*w_i acting on m must give exactly the same H and E as a
    // uniform Ms0 acting on a pre-scaled magnetisation m'_i = w_i * m_i —
    // a correctness check independent of the Newell-tensor numerics.
    StructuredGrid g(8, 6, 4, 3e-9, 3e-9, 3e-9);
    Material mat{};
    mat.Ms = 8e5;

    VectorField3D m(g), m_scaled(g);
    fill_smooth(m);

    MaterialField3D matf(g, mat);
    for (Index idx = 0; idx < g.size(); ++idx) {
        const Real w = 0.5 + 0.1 * static_cast<Real>(idx % 7);
        matf.Ms_field()[idx] = mat.Ms * w;
        m_scaled[idx] = m[idx] * w;
    }

    DemagField demag_perCell(g);
    demag_perCell.set_material_field(&matf);
    DemagField demag_scaled(g);

    VectorField3D H_perCell(g), H_scaled(g);
    demag_perCell.accumulate(m, mat, H_perCell);
    demag_scaled.accumulate(m_scaled, mat, H_scaled);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H_perCell[idx].x, WithinAbs(H_scaled[idx].x, 1e-6));
        REQUIRE_THAT(H_perCell[idx].y, WithinAbs(H_scaled[idx].y, 1e-6));
        REQUIRE_THAT(H_perCell[idx].z, WithinAbs(H_scaled[idx].z, 1e-6));
    }

    // E = -mu0/2 * Ms_i * m_i . H_i * dV
    //   per-cell: Ms_i = Ms0*w_i,  m_i,        H_i = H_perCell_i
    //   scaled:   Ms0,             m'_i = w_i*m_i,  H_i = H_scaled_i (== H_perCell_i)
    // Ms0*w_i*(m_i.H_i) == Ms0*(w_i*m_i).H_i  — identical sums ⇒ identical energy.
    Real E_perCell = demag_perCell.energy(m, mat);
    Real E_scaled  = demag_scaled.energy(m_scaled, mat);
    REQUIRE_THAT(E_perCell, WithinAbs(E_scaled, std::abs(E_scaled) * 1e-9 + 1e-30));
}

TEST_CASE("DemagFieldPeriodic: uniform MaterialField3D matches uniform Material",
          "[material][demag_periodic]") {
    StructuredGrid g(8, 8, 4, 4e-9, 4e-9, 4e-9);
    Material mat = Material::permalloy();
    MaterialField3D matf(g, mat);

    VectorField3D m(g), H_uniform(g), H_perCell(g);
    fill_smooth(m);

    DemagFieldPeriodic demag_uniform(g);
    DemagFieldPeriodic demag_perCell(g);
    demag_perCell.set_material_field(&matf);

    demag_uniform.accumulate(m, mat, H_uniform);
    demag_perCell.accumulate(m, mat, H_perCell);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H_perCell[idx].x, WithinAbs(H_uniform[idx].x, 1e-6));
        REQUIRE_THAT(H_perCell[idx].y, WithinAbs(H_uniform[idx].y, 1e-6));
        REQUIRE_THAT(H_perCell[idx].z, WithinAbs(H_uniform[idx].z, 1e-6));
    }

    Real E_uniform = demag_uniform.energy(m, mat);
    Real E_perCell = demag_perCell.energy(m, mat);
    REQUIRE_THAT(E_perCell, WithinAbs(E_uniform, std::abs(E_uniform) * 1e-9 + 1e-30));
}

TEST_CASE("DemagFieldPeriodic: per-cell Ms equals uniform Ms with pre-scaled m (M = Ms*m linearity)",
          "[material][demag_periodic]") {
    StructuredGrid g(8, 6, 4, 3e-9, 3e-9, 3e-9);
    Material mat{};
    mat.Ms = 8e5;

    VectorField3D m(g), m_scaled(g);
    fill_smooth(m);

    MaterialField3D matf(g, mat);
    for (Index idx = 0; idx < g.size(); ++idx) {
        const Real w = 0.5 + 0.1 * static_cast<Real>(idx % 7);
        matf.Ms_field()[idx] = mat.Ms * w;
        m_scaled[idx] = m[idx] * w;
    }

    DemagFieldPeriodic demag_perCell(g);
    demag_perCell.set_material_field(&matf);
    DemagFieldPeriodic demag_scaled(g);

    VectorField3D H_perCell(g), H_scaled(g);
    demag_perCell.accumulate(m, mat, H_perCell);
    demag_scaled.accumulate(m_scaled, mat, H_scaled);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H_perCell[idx].x, WithinAbs(H_scaled[idx].x, 1e-6));
        REQUIRE_THAT(H_perCell[idx].y, WithinAbs(H_scaled[idx].y, 1e-6));
        REQUIRE_THAT(H_perCell[idx].z, WithinAbs(H_scaled[idx].z, 1e-6));
    }

    Real E_perCell = demag_perCell.energy(m, mat);
    Real E_scaled  = demag_scaled.energy(m_scaled, mat);
    REQUIRE_THAT(E_perCell, WithinAbs(E_scaled, std::abs(E_scaled) * 1e-9 + 1e-30));
}

// ---------------------------------------------------------------------------
// C1-4: UniaxialAnisotropyField — per-cell K_uniaxial / easy_axis / Ms
// ---------------------------------------------------------------------------

TEST_CASE("UniaxialAnisotropyField: uniform MaterialField3D matches uniform Material",
          "[material][anisotropy]") {
    StructuredGrid g(5, 4, 3, 2e-9, 2e-9, 2e-9);
    Material mat = Material::cobalt();
    MaterialField3D matf(g, mat);

    VectorField3D m(g), H_uniform(g), H_perCell(g);
    fill_smooth(m);

    UniaxialAnisotropyField an_uniform, an_perCell;
    an_perCell.set_material_field(&matf);

    an_uniform.accumulate(m, mat, H_uniform);
    an_perCell.accumulate(m, mat, H_perCell);

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(H_perCell[idx].x, WithinAbs(H_uniform[idx].x, 1e-6));
        REQUIRE_THAT(H_perCell[idx].y, WithinAbs(H_uniform[idx].y, 1e-6));
        REQUIRE_THAT(H_perCell[idx].z, WithinAbs(H_uniform[idx].z, 1e-6));
    }

    Real E_uniform = an_uniform.energy(m, mat);
    Real E_perCell = an_perCell.energy(m, mat);
    REQUIRE_THAT(E_perCell, WithinAbs(E_uniform, std::abs(E_uniform) * 1e-9 + 1e-30));
}

TEST_CASE("UniaxialAnisotropyField: per-cell easy_axis gives region-dependent field & energy",
          "[material][anisotropy]") {
    // Left half: easy axis = x. Right half: easy axis = y. Same K, Ms everywhere
    // (mumax3 "Regions" style — e.g. randomly-oriented grains).
    StructuredGrid g(4, 2, 1, 2e-9, 2e-9, 2e-9);
    const Real Ms = 8e5, K = 4e5;

    Material base{};
    base.Ms = Ms;
    base.K_uniaxial = K;
    base.easy_axis = {1, 0, 0};

    MaterialField3D matf(g, base);
    const Vec3 axis_y{0, 1, 0};
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = g.nx() / 2; ix < g.nx(); ++ix)
        matf.easy_axis_field()[g.linear_index(ix, iy, 0)] = axis_y;

    VectorField3D m(g), H(g);
    fill_smooth(m);   // not normalised — H formula is linear in m, fine for this check

    UniaxialAnisotropyField an;
    an.set_material_field(&matf);
    an.accumulate(m, base, H);

    const Real prefactor = 2.0 * K / (constants::mu_0 * Ms);
    const Real dV = g.cell_volume();
    Real E_expected = 0;

    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Index idx = g.linear_index(ix, iy, 0);
        Vec3 u = (ix < g.nx() / 2) ? Vec3{1, 0, 0} : axis_y;
        Real c = m[idx].dot(u);
        Vec3 expected_H = u * (prefactor * c);

        REQUIRE_THAT(H[idx].x, WithinAbs(expected_H.x, 1e-6));
        REQUIRE_THAT(H[idx].y, WithinAbs(expected_H.y, 1e-6));
        REQUIRE_THAT(H[idx].z, WithinAbs(expected_H.z, 1e-6));

        E_expected += -K * c * c * dV;
    }

    Real E = an.energy(m, base);
    REQUIRE_THAT(E, WithinAbs(E_expected, std::abs(E_expected) * 1e-9 + 1e-30));
}

TEST_CASE("UniaxialAnisotropyField: per-cell K=0 silences anisotropy in that region only",
          "[material][anisotropy]") {
    StructuredGrid g(4, 2, 1, 2e-9, 2e-9, 2e-9);
    Material base = Material::cobalt();   // K=4.5e5, easy_axis = z
    MaterialField3D matf(g, base);

    // Zero out K in the left half (e.g. a non-magnetic / soft-region grain).
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx() / 2; ++ix)
        matf.K_field()[g.linear_index(ix, iy, 0)] = 0.0;

    VectorField3D m(g), H(g);
    m.set_uniform({0.6, 0.0, 0.8});   // non-zero component along cobalt's z easy-axis

    UniaxialAnisotropyField an;
    an.set_material_field(&matf);
    an.accumulate(m, base, H);

    for (Index iy = 0; iy < g.ny(); ++iy) {
        // Left half (K=0): no contribution.
        Index idx_left = g.linear_index(0, iy, 0);
        REQUIRE_THAT(H[idx_left].norm(), WithinAbs(0.0, 1e-12));

        // Right half (K=cobalt's): non-zero, along z.
        Index idx_right = g.linear_index(g.nx() - 1, iy, 0);
        REQUIRE(H[idx_right].norm() > 1e3);
        REQUIRE_THAT(H[idx_right].x, WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(H[idx_right].y, WithinAbs(0.0, 1e-12));
    }
}

// ---------------------------------------------------------------------------
// C1-5: Integrators — per-cell alpha damping
// ---------------------------------------------------------------------------

namespace {
// Common two-region alpha setup + uniform Zeeman field (no exchange/demag —
// cells evolve independently, so per-cell alpha is the only spatial coupling).
EffectiveFieldSum make_zeeman_only(Vec3 H_ext) {
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(H_ext));
    return heff;
}
}  // namespace

TEST_CASE("RK4Integrator: per-cell alpha matches independent uniform-alpha runs",
          "[material][integrator]") {
    // Without exchange/demag coupling, each cell relaxes independently under a
    // uniform Zeeman field — its trajectory depends only on its own alpha. A
    // two-region alpha field must therefore exactly reproduce, cell for cell,
    // two separate uniform-alpha runs (same fixed dt, no adaptive stepping).
    StructuredGrid g(4, 2, 1, 2e-9, 2e-9, 2e-9);
    const Real alpha_L = 0.01, alpha_R = 0.3;

    Material mat{};
    mat.Ms = 8e5;

    MaterialField3D matf(g, mat);
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix)
        matf.alpha_field()[g.linear_index(ix, iy, 0)] = (ix < g.nx() / 2) ? alpha_L : alpha_R;

    EffectiveFieldSum heff = make_zeeman_only({0, 0, 1e5});

    const Vec3 m0_raw{1.0, 0.0, 0.5};
    const Vec3 m0 = m0_raw / m0_raw.norm();

    VectorField3D m_perCell(g), m_L(g), m_R(g);
    m_perCell.set_uniform(m0);
    m_L.set_uniform(m0);
    m_R.set_uniform(m0);

    Material mat_L = mat; mat_L.alpha = alpha_L;
    Material mat_R = mat; mat_R.alpha = alpha_R;

    RK4Integrator rk_perCell(1e-13), rk_L(1e-13), rk_R(1e-13);
    rk_perCell.set_material_field(&matf);

    for (int i = 0; i < 50; ++i) {
        rk_perCell.step(m_perCell, mat,   heff);
        rk_L.step      (m_L,       mat_L, heff);
        rk_R.step      (m_R,       mat_R, heff);
    }

    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Index idx = g.linear_index(ix, iy, 0);
        const VectorField3D& ref = (ix < g.nx() / 2) ? m_L : m_R;
        REQUIRE_THAT(m_perCell[idx].x, WithinAbs(ref[idx].x, 1e-12));
        REQUIRE_THAT(m_perCell[idx].y, WithinAbs(ref[idx].y, 1e-12));
        REQUIRE_THAT(m_perCell[idx].z, WithinAbs(ref[idx].z, 1e-12));
    }

    // Sanity: stronger damping (alpha_R) relaxes closer to H=+z in the same time.
    REQUIRE(m_R[g.linear_index(g.nx() - 1, 0, 0)].z > m_L[g.linear_index(0, 0, 0)].z);
}

TEST_CASE("HeunIntegrator: per-cell alpha matches independent uniform-alpha runs (no thermal)",
          "[material][integrator]") {
    // Same independence argument as RK4: HeunIntegrator uses a fixed dt with
    // no adaptive stepping, so the per-cell run must exactly reproduce two
    // separate uniform-alpha deterministic-Heun runs (thermal = nullptr).
    StructuredGrid g(4, 2, 1, 2e-9, 2e-9, 2e-9);
    const Real alpha_L = 0.02, alpha_R = 0.4;

    Material mat{};
    mat.Ms = 8e5;

    MaterialField3D matf(g, mat);
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix)
        matf.alpha_field()[g.linear_index(ix, iy, 0)] = (ix < g.nx() / 2) ? alpha_L : alpha_R;

    EffectiveFieldSum heff = make_zeeman_only({0, 0, 1e5});

    const Vec3 m0_raw{1.0, 0.0, 0.5};
    const Vec3 m0 = m0_raw / m0_raw.norm();

    VectorField3D m_perCell(g), m_L(g), m_R(g);
    m_perCell.set_uniform(m0);
    m_L.set_uniform(m0);
    m_R.set_uniform(m0);

    Material mat_L = mat; mat_L.alpha = alpha_L;
    Material mat_R = mat; mat_R.alpha = alpha_R;

    HeunIntegrator heun_perCell(1e-13), heun_L(1e-13), heun_R(1e-13);
    heun_perCell.set_material_field(&matf);

    for (int i = 0; i < 50; ++i) {
        heun_perCell.step(m_perCell, mat,   heff);
        heun_L.step      (m_L,       mat_L, heff);
        heun_R.step      (m_R,       mat_R, heff);
    }

    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Index idx = g.linear_index(ix, iy, 0);
        const VectorField3D& ref = (ix < g.nx() / 2) ? m_L : m_R;
        REQUIRE_THAT(m_perCell[idx].x, WithinAbs(ref[idx].x, 1e-12));
        REQUIRE_THAT(m_perCell[idx].y, WithinAbs(ref[idx].y, 1e-12));
        REQUIRE_THAT(m_perCell[idx].z, WithinAbs(ref[idx].z, 1e-12));
    }
}

TEST_CASE("RK45Integrator: uniform MaterialField3D reproduces uniform Material exactly",
          "[material][integrator]") {
    // RK45's adaptive dt depends on the global error norm, so a per-cell
    // alpha field can only be compared bit-for-bit against the uniform path
    // when its values equal mat.alpha everywhere (identical physics ⇒
    // identical accept/reject sequence ⇒ identical trajectory).
    StructuredGrid g(4, 2, 1, 2e-9, 2e-9, 2e-9);
    Material mat = Material::permalloy();
    MaterialField3D matf(g, mat);

    EffectiveFieldSum heff = make_zeeman_only({0, 0, 1e5});

    const Vec3 m0_raw{1.0, 0.0, 0.5};
    const Vec3 m0 = m0_raw / m0_raw.norm();

    VectorField3D m_uniform(g), m_perCell(g);
    m_uniform.set_uniform(m0);
    m_perCell.set_uniform(m0);

    RK45Integrator rk_uniform, rk_perCell;
    rk_perCell.set_material_field(&matf);

    for (int i = 0; i < 30; ++i) {
        rk_uniform.step(m_uniform, mat, heff);
        rk_perCell.step(m_perCell, mat, heff);
    }

    for (Index idx = 0; idx < g.size(); ++idx) {
        REQUIRE_THAT(m_perCell[idx].x, WithinAbs(m_uniform[idx].x, 1e-12));
        REQUIRE_THAT(m_perCell[idx].y, WithinAbs(m_uniform[idx].y, 1e-12));
        REQUIRE_THAT(m_perCell[idx].z, WithinAbs(m_uniform[idx].z, 1e-12));
    }
}

TEST_CASE("RK45Integrator: per-cell alpha gives region-dependent relaxation rate",
          "[material][integrator]") {
    // Two regions with very different alpha relax toward H=+z from a common
    // tilted start; the higher-alpha region must end up closer to +z.
    StructuredGrid g(4, 2, 1, 2e-9, 2e-9, 2e-9);
    const Real alpha_L = 0.01, alpha_R = 0.5;

    Material mat{};
    mat.Ms = 8e5;

    MaterialField3D matf(g, mat);
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix)
        matf.alpha_field()[g.linear_index(ix, iy, 0)] = (ix < g.nx() / 2) ? alpha_L : alpha_R;

    EffectiveFieldSum heff = make_zeeman_only({0, 0, 1e5});

    const Vec3 m0_raw{1.0, 0.0, 0.3};
    const Vec3 m0 = m0_raw / m0_raw.norm();
    VectorField3D m(g);
    m.set_uniform(m0);

    RK45Integrator rk;
    rk.set_material_field(&matf);
    for (int i = 0; i < 60; ++i)
        rk.step(m, mat, heff);

    const Real mz_L = m[g.linear_index(0, 0, 0)].z;
    const Real mz_R = m[g.linear_index(g.nx() - 1, 0, 0)].z;
    REQUIRE(mz_R > mz_L);
}
