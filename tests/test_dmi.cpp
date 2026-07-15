#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/dmi.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================
// Helper: 3×3×1 grid with cell size 5 nm
// ============================================================
static StructuredGrid small_grid() {
    return StructuredGrid(3, 3, 1, 5e-9, 5e-9, 5e-9);
}

static Material py_mat() {
    Material m = Material::permalloy();
    m.K_uniaxial = 0.0;
    return m;
}

// ============================================================
// BulkDMIField tests
// ============================================================

TEST_CASE("BulkDMI: zero D gives zero field", "[dmi]") {
    auto g = small_grid();
    VectorField3D mv(g), H(g);
    mv.set_uniform({1, 0, 0});
    H.set_uniform({0, 0, 0});

    BulkDMIField dmi(0.0);
    dmi.accumulate(mv, py_mat(), H);

    for (Index i = 0; i < H.size(); ++i)
        REQUIRE(H[i].norm() < 1e-30);
}

TEST_CASE("BulkDMI: uniform m gives zero field", "[dmi]") {
    auto g = small_grid();
    VectorField3D mv(g), H(g);
    mv.set_uniform({1, 0, 0});
    H.set_uniform({0, 0, 0});

    BulkDMIField dmi(3e-3);
    dmi.accumulate(mv, py_mat(), H);

    // ∇×(const) = 0  →  H_DMI = 0
    for (Index i = 0; i < H.size(); ++i)
        REQUIRE(H[i].norm() < 1e-10);
}

TEST_CASE("BulkDMI: gradient in x produces Hz via curl", "[dmi]") {
    // Set up linear mz variation along x: mz(i) = 0.1*i*dx/L
    StructuredGrid g(5, 1, 1, 4e-9, 4e-9, 4e-9);
    VectorField3D mv(g), H(g);
    // mx=1 everywhere, mz varies linearly
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Real mz = static_cast<Real>(ix) * 0.1;
        Real mx = std::sqrt(std::max(0.0, 1.0 - mz*mz));
        mv[g.linear_index(ix, 0, 0)] = {mx, 0, mz};
    }

    H.set_uniform({0, 0, 0});
    Material mat = Material::permalloy();
    BulkDMIField dmi(3e-3);
    dmi.accumulate(mv, mat, H);

    // [∇×m]_y = ∂mx/∂z - ∂mz/∂x  →  for 1D in x and nz=1: -∂mz/∂x ≠ 0
    // Central interior cells (ix=1,2,3) should have H_y != 0
    REQUIRE(std::abs(H[g.linear_index(2, 0, 0)].y) > 1e5);
}

TEST_CASE("BulkDMI: energy is zero for uniform m", "[dmi]") {
    auto g = small_grid();
    VectorField3D mv(g);
    mv.set_uniform({0, 0, 1});

    BulkDMIField dmi(3e-3);
    Real E = dmi.energy(mv, py_mat());
    REQUIRE(std::abs(E) < 1e-30);
}

TEST_CASE("BulkDMI: name is correct", "[dmi]") {
    BulkDMIField dmi(3e-3);
    REQUIRE(std::string(dmi.name()) == "BulkDMI");
}

// ============================================================
// InterfacialDMIField tests
// ============================================================

TEST_CASE("InterfacialDMI: zero D gives zero field", "[dmi]") {
    auto g = small_grid();
    VectorField3D mv(g), H(g);
    mv.set_uniform({0, 0, 1});
    H.set_uniform({0, 0, 0});

    InterfacialDMIField dmi(0.0);
    dmi.accumulate(mv, py_mat(), H);

    for (Index i = 0; i < H.size(); ++i)
        REQUIRE(H[i].norm() < 1e-30);
}

TEST_CASE("InterfacialDMI: uniform mz gives zero Hz contribution", "[dmi]") {
    // Uniform m  →  ∂mz/∂x = ∂mz/∂y = 0, ∂mx/∂x = ∂my/∂y = 0  →  H = 0
    auto g = small_grid();
    VectorField3D mv(g), H(g);
    mv.set_uniform({0, 0, 1});
    H.set_uniform({0, 0, 0});

    InterfacialDMIField dmi(3e-3);
    dmi.accumulate(mv, py_mat(), H);

    for (Index i = 0; i < H.size(); ++i)
        REQUIRE(H[i].norm() < 1e-10);
}

TEST_CASE("InterfacialDMI: gradient of mz along x produces Hx", "[dmi]") {
    StructuredGrid g(5, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g), H(g);

    // mz varies linearly; normalise
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Real mz = static_cast<Real>(ix) * 0.1;
        Real mx = std::sqrt(std::max(0.0, 1.0 - mz*mz));
        mv[g.linear_index(ix, 0, 0)] = {mx, 0, mz};
    }

    H.set_uniform({0, 0, 0});
    InterfacialDMIField dmi(3e-3);
    dmi.accumulate(mv, Material::permalloy(), H);

    // Interior cells: Hx = (2D/mu0Ms) * ∂mz/∂x > 0 since dmz/dx > 0
    REQUIRE(H[g.linear_index(2, 0, 0)].x > 0.0);
}

TEST_CASE("InterfacialDMI: name is correct", "[dmi]") {
    InterfacialDMIField dmi(3e-3);
    REQUIRE(std::string(dmi.name()) == "InterfacialDMI");
}

// ============================================================
// ZhangLiSTT tests
// ============================================================

#include "micromag/spin_torque.hpp"

TEST_CASE("ZhangLiSTT: spin-drift velocity dimensional check", "[dmi]") {
    // u = P * mu_B * |J| / (e * Ms)
    // P=0.7, J=1e12 A/m², Ms=8e5 A/m
    // mu_B = 9.274e-24, e = 1.602e-19
    // u ~ 0.7 * 9.274e-24 * 1e12 / (1.602e-19 * 8e5) ~ 50 m/s (typical)
    Vec3 J{1e12, 0, 0};
    ZhangLiSTT zl(J, 0.7, 0.04);
    Real u = zl.u(8e5);
    REQUIRE(u > 1.0);    // > 1 m/s
    REQUIRE(u < 1e4);    // < 10 km/s
}

TEST_CASE("ZhangLiSTT: zero J gives zero torque", "[dmi]") {
    StructuredGrid g(3, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g), dm(g);
    mv.set_uniform({1, 0, 0});
    dm.set_uniform({0, 0, 0});

    ZhangLiSTT zl({0, 0, 0}, 0.7, 0.04);
    zl.accumulate(mv, Material::permalloy(), dm);

    for (Index i = 0; i < dm.size(); ++i)
        REQUIRE(dm[i].norm() < 1e-30);
}

TEST_CASE("ZhangLiSTT: uniform m gives zero torque", "[dmi]") {
    // (J·∇)m = 0 for uniform m  →  torque = 0
    StructuredGrid g(5, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g), dm(g);
    mv.set_uniform({0, 0, 1});
    dm.set_uniform({0, 0, 0});

    ZhangLiSTT zl({1e12, 0, 0}, 0.7, 0.04);
    zl.accumulate(mv, Material::permalloy(), dm);

    for (Index i = 0; i < dm.size(); ++i)
        REQUIRE(dm[i].norm() < 1e-10);
}

TEST_CASE("ZhangLiSTT: gradient produces non-zero torque", "[dmi]") {
    // Domain-wall-like profile: mz varies along x
    StructuredGrid g(7, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g), dm(g);
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Real mz = std::tanh(static_cast<Real>(ix - 3) * 0.5);
        Real mx = std::sqrt(std::max(0.0, 1.0 - mz*mz));
        mv[g.linear_index(ix, 0, 0)] = {mx, 0, mz};
    }
    dm.set_uniform({0, 0, 0});

    ZhangLiSTT zl({1e12, 0, 0}, 0.7, 0.04);
    zl.accumulate(mv, Material::permalloy(), dm);

    // At the wall centre (ix=3), gradient is maximal → torque > 0
    Real t_centre = dm[g.linear_index(3, 0, 0)].norm();
    REQUIRE(t_centre > 1e6);
}

// ============================================================
// Solver: relax() / max_torque() tests
// ============================================================

#include "micromag/solver.hpp"
#include "micromag/zeeman.hpp"

TEST_CASE("Solver: max_torque is zero for aligned m", "[solver]") {
    // m aligned with H_ext  →  m×H = 0  →  max_torque = 0
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    mv.set_uniform({0, 0, 1});

    Material mat = Material::permalloy();
    mat.K_uniaxial = 0.0;

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 1e6}));  // H along z

    Real t = max_torque(mv, mat, heff);
    REQUIRE(t < 1e-6);
}

TEST_CASE("Solver: relax converges single-domain macrospin", "[solver]") {
    // Single cell, easy axis along z, H along z
    // Start with m along x (not equilibrium)
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    mv.set_uniform({1, 0, 0});

    Material mat = Material::permalloy();
    mat.K_uniaxial = 1e4;
    mat.easy_axis  = {0, 0, 1};

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 8e4}));

    RelaxOptions opts;
    opts.threshold = 10.0;   // A/m
    opts.dt        = 5e-12;
    opts.max_steps = 20000;

    int n = relax(mv, mat, heff, opts);

    // After relax, m should be close to z-axis
    REQUIRE(n < opts.max_steps);
    REQUIRE(mv[0].z > 0.9);
    REQUIRE(max_torque(mv, mat, heff) < opts.threshold * 2);
}

TEST_CASE("Solver: minimize converges single-domain macrospin", "[solver]") {
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    mv.set_uniform({1, 0, 0});  // start off-axis

    Material mat = Material::permalloy();
    mat.K_uniaxial = 1e4;
    mat.easy_axis  = {0, 0, 1};

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 8e4}));

    MinimizeOptions opts;
    opts.threshold = 10.0;
    opts.max_steps = 10000;

    int n = minimize(mv, mat, heff, opts);

    REQUIRE(n < opts.max_steps);
    REQUIRE(mv[0].z > 0.9);
}

// ============================================================
// OVF I/O tests
// ============================================================

#include "micromag/ovf_io.hpp"
#include <filesystem>
#include <cstdlib>

TEST_CASE("OVF: save and load binary roundtrip", "[ovf]") {
    StructuredGrid g(4, 3, 2, 5e-9, 5e-9, 5e-9);
    VectorField3D m_orig(g);
    // Fill with recognisable values
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Index idx = g.linear_index(ix, iy, iz);
        m_orig[idx] = {static_cast<Real>(ix)*0.1,
                       static_cast<Real>(iy)*0.1,
                       static_cast<Real>(iz)*0.1};
        // Normalise (not strictly required for OVF but keep clean)
        Real n = m_orig[idx].norm();
        if (n > 0) m_orig[idx] = m_orig[idx] / n;
        else m_orig[idx] = {0, 0, 1};
    }

    const std::string fname = "test_ovf_binary.ovf";
    save_ovf(fname, m_orig, "test", OVFFormat::Binary8);

    VectorField3D m_loaded = load_ovf(fname);

    REQUIRE(m_loaded.grid().nx() == g.nx());
    REQUIRE(m_loaded.grid().ny() == g.ny());
    REQUIRE(m_loaded.grid().nz() == g.nz());
    REQUIRE(m_loaded.size() == m_orig.size());

    for (Index i = 0; i < m_orig.size(); ++i) {
        REQUIRE_THAT(m_loaded[i].x, WithinAbs(m_orig[i].x, 1e-12));
        REQUIRE_THAT(m_loaded[i].y, WithinAbs(m_orig[i].y, 1e-12));
        REQUIRE_THAT(m_loaded[i].z, WithinAbs(m_orig[i].z, 1e-12));
    }

    std::filesystem::remove(fname);
}

TEST_CASE("OVF: save and load text roundtrip", "[ovf]") {
    StructuredGrid g(3, 3, 1, 4e-9, 4e-9, 4e-9);
    VectorField3D m_orig(g);
    m_orig.set_uniform({1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)});

    const std::string fname = "test_ovf_text.ovf";
    save_ovf(fname, m_orig, "uniform_test", OVFFormat::Text);

    VectorField3D m_loaded = load_ovf(fname);

    REQUIRE(m_loaded.size() == m_orig.size());
    for (Index i = 0; i < m_orig.size(); ++i) {
        REQUIRE_THAT(m_loaded[i].x, WithinAbs(m_orig[i].x, 1e-12));
        REQUIRE_THAT(m_loaded[i].y, WithinAbs(m_orig[i].y, 1e-12));
        REQUIRE_THAT(m_loaded[i].z, WithinAbs(m_orig[i].z, 1e-12));
    }

    std::filesystem::remove(fname);
}
