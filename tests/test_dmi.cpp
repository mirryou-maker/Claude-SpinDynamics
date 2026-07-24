#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/dmi.hpp"
#include "micromag/material_field.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================
// Helper: 3×3×1 grid with cell size 5 nm
// ============================================================
static const StructuredGrid& gv_grid_alias(const VectorField3D& v) { return v.grid(); }

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
    // 3x3x3 so the centre cell is a true interior cell: with nz=1 every cell
    // sits on both z faces and the bulk z-surface BC field (correct physics,
    // mumax3-validated) is nonzero even at the in-plane centre.
    StructuredGrid g(3, 3, 3, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g), H(g);
    mv.set_uniform({1, 0, 0});
    H.set_uniform({0, 0, 0});

    BulkDMIField dmi(3e-3);
    dmi.accumulate(mv, py_mat(), H);

    // ∇×(const) = 0 in the interior, but with the free-boundary DMI
    // condition (default, mumax3-equivalent) the EDGE cells carry the
    // Rohart-Thiaville canting field -- a uniform state must NOT be a
    // spurious equilibrium. 3x3x1 grid: centre cell (1,1,0) is interior.
    const StructuredGrid& gr = mv.grid();
    REQUIRE(H[gr.linear_index(1, 1, 1)].norm() < 1e-10);   // interior: zero
    Real edge_norm = 0;
    for (Index i = 0; i < H.size(); ++i) edge_norm += H[i].norm();
    REQUIRE(edge_norm > 1e3);                              // edges: finite BC field

    // Legacy naive stencil (OpenBC): zero everywhere, as before the fix.
    VectorField3D H2(gv_grid_alias(mv));
    H2.set_uniform({0, 0, 0});
    BulkDMIField dmi_open(3e-3);
    dmi_open.set_open_bc(true);
    dmi_open.accumulate(mv, py_mat(), H2);
    for (Index i = 0; i < H2.size(); ++i)
        REQUIRE(H2[i].norm() < 1e-10);
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

    // Interior cell: zero. Edge cells: finite Rohart-Thiaville BC field
    // (uniform m_z film must cant at free edges; the pre-fix behaviour of
    // zero field everywhere was the spurious-equilibrium bug).
    const StructuredGrid& gr = mv.grid();
    REQUIRE(H[gr.linear_index(1, 1, 0)].norm() < 1e-10);
    Real edge_norm = 0;
    for (Index i = 0; i < H.size(); ++i) edge_norm += H[i].norm();
    REQUIRE(edge_norm > 1e3);

    // Closed form at an x-min edge cell (y-interior), uniform m = z_hat:
    //   H_x = +D/(mu0 Ms dx)   (exchange-ghost correction, s=-1, gamma_x=(-1,0,0))
    //   H_z = +D^2/(2 mu0 Ms A) (DMI gradient term, both x- and y- for corner;
    //         at a pure x-edge only the x contribution)
    {
        const Material mt = py_mat();
        const Real Dv = 3e-3;
        const Real Hx_ref = Dv / (constants::mu_0 * mt.Ms * gr.dx());
        const Vec3 He = H[gr.linear_index(0, 1, 0)];   // x-min edge, y-interior
        REQUIRE_THAT(He.x, Catch::Matchers::WithinRel(Hx_ref, 1e-9));
        const Real Hz_ref = Dv * Dv / (Real{2} * constants::mu_0 * mt.Ms * mt.A_exchange);
        REQUIRE_THAT(He.z, Catch::Matchers::WithinRel(Hz_ref, 1e-9));
    }

    // Legacy naive stencil (OpenBC): zero everywhere.
    VectorField3D H2(gv_grid_alias(mv));
    H2.set_uniform({0, 0, 0});
    InterfacialDMIField dmi_open(3e-3);
    dmi_open.set_open_bc(true);
    dmi_open.accumulate(mv, py_mat(), H2);
    for (Index i = 0; i < H2.size(); ++i)
        REQUIRE(H2[i].norm() < 1e-10);
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

TEST_CASE("ZhangLiSTT: thiaville_u scales u by 1/(1+xi^2)", "[dmi][spin_torque]") {
    // mumax3 compatibility switch: u_mumax3 = u_CS / (1 + xi^2).
    const Real xi = 0.5;
    ZhangLiSTT zl({1e12, 0, 0}, 0.7, xi);
    const Real u_cs = zl.u(8e5);
    zl.set_thiaville_u(true);
    const Real u_mx = zl.u(8e5);
    REQUIRE_THAT(u_mx * (1.0 + xi * xi), WithinRel(u_cs, 1e-12));
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

// ---------------------------------------------------------------------------
// CPU DMI set_material_field: per-cell Ms drives the 1/(mu0 Ms) prefactor
// (parity with the GPU DMI). Half the grid at 2x Ms -> H halved there.
// ---------------------------------------------------------------------------
TEST_CASE("InterfacialDMIField: set_material_field applies per-cell Ms", "[dmi]") {
    using namespace micromag;
    StructuredGrid g(8, 8, 1, 2e-9, 2e-9, 1e-9);
    Material mat; mat.Ms = 8e5; mat.A_exchange = 1.5e-11;
    const Real D = 2e-3;

    VectorField3D m(g);
    for (Index i = 0; i < g.size(); ++i) {
        const double t = 0.3 * static_cast<double>(i % 7);
        Vec3 v{std::sin(t), 0.2, std::cos(t)};
        m[i] = v / std::sqrt(v.dot(v));
    }

    InterfacialDMIField ref(D);
    VectorField3D H_ref(g); ref.accumulate(m, mat, H_ref);   // uniform Ms

    MaterialField3D matf(g, mat);
    for (Index j = 0; j < 8; ++j)
        for (Index i = 4; i < 8; ++i)
            matf.Ms_field()[i + 8 * j] = 2 * mat.Ms;
    InterfacialDMIField pc(D);
    pc.set_material_field(&matf);
    REQUIRE(pc.has_material_field());
    VectorField3D H_pc(g); pc.accumulate(m, mat, H_pc);

    bool any = false;
    for (Index j = 0; j < 8; ++j)
        for (Index i = 0; i < 8; ++i) {
            const Index idx = i + 8 * j;
            const double f = (i < 4) ? 1.0 : 0.5;   // H proportional to 1/Ms
            for (int c = 0; c < 3; ++c) {
                const double ref_c = (&H_ref[idx].x)[c];
                if (std::abs(ref_c) < 1.0) continue;
                any = true;
                REQUIRE_THAT((&H_pc[idx].x)[c], Catch::Matchers::WithinRel(f * ref_c, 1e-9));
            }
        }
    REQUIRE(any);
}

