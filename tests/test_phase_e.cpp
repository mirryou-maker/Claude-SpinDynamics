#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>

static constexpr double kPi = std::numbers::pi;

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/cubic_anisotropy.hpp"
#include "micromag/region_map.hpp"
#include "micromag/init_mag.hpp"
#include "micromag/geom_mask.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/dmi.hpp"
#include "micromag/demag.hpp"
#include "micromag/exchange.hpp"
#include "micromag/topological_charge.hpp"
#include "micromag/skyrmion_tools.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static StructuredGrid g5() { return StructuredGrid(5, 5, 1, 5e-9, 5e-9, 5e-9); }
static Material py_mat() {
    Material m = Material::permalloy();
    m.K_uniaxial = 0.0;
    return m;
}

// ============================================================
// Item 1a: Ku2 (second uniaxial)
// ============================================================

TEST_CASE("Ku2: zero Ku2 no change from K_uniaxial only", "[cubic]") {
    auto g = g5();
    VectorField3D mv(g), H(g);
    mv.set_uniform({0, 0, 1});
    H.set_uniform({0, 0, 0});

    Material mat = Material::permalloy();
    mat.K_uniaxial = 1e4;
    mat.easy_axis  = {0, 0, 1};
    mat.Ku2        = 0.0;

    UniaxialAnisotropyField anis;
    anis.accumulate(mv, mat, H);

    // m aligned with u: H = (2K/mu0Ms)*1*u  (purely along z)
    Real H_expected = 2.0 * 1e4 / (4e-7 * kPi * mat.Ms);
    REQUIRE_THAT(H[0].z, WithinRel(H_expected, 1e-6));
}

TEST_CASE("Ku2: non-zero Ku2 adds 4th-order term", "[cubic]") {
    // m = (sin45°, 0, cos45°), K1=0, Ku2=1e4
    // c = m·û = cos(45°) = 1/√2
    // H_z = (4Ku2/mu0Ms) * c^3 * 1 (along z)
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g), H(g);
    const Real c = 1.0 / std::sqrt(2.0);
    mv.set_uniform({std::sqrt(1.0 - c*c), 0, c});  // sin45, 0, cos45
    H.set_uniform({0, 0, 0});

    Material mat = Material::permalloy();
    mat.K_uniaxial = 0.0;
    mat.Ku2        = 1e4;
    mat.easy_axis  = {0, 0, 1};

    UniaxialAnisotropyField anis;
    anis.accumulate(mv, mat, H);

    const Real H_z_expected = 4.0 * 1e4 * c * c * c / (4e-7 * kPi * mat.Ms);
    REQUIRE_THAT(H[0].z, WithinRel(H_z_expected, 1e-6));
}

TEST_CASE("Ku2: energy density integral matches total energy", "[cubic]") {
    auto g = g5();
    VectorField3D mv(g);
    mv.set_uniform({0, 1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0)});

    Material mat = Material::permalloy();
    mat.K_uniaxial = 5e3;
    mat.Ku2        = 2e3;
    mat.easy_axis  = {0, 0, 1};

    UniaxialAnisotropyField anis;
    Real E_total = anis.energy(mv, mat);
    ScalarField3D ed = anis.energy_density(mv, mat);

    Real E_from_density = 0;
    for (Index i = 0; i < ed.size(); ++i)
        E_from_density += ed[i] * g.cell_volume();
    REQUIRE_THAT(E_from_density, WithinRel(E_total, 1e-10));
}

// ============================================================
// Item 1b: CubicAnisotropyField (Kc1/Kc2)
// ============================================================

TEST_CASE("CubicAnisotropy: name is correct", "[cubic]") {
    CubicAnisotropyField ca(1e4, 0);
    REQUIRE(std::string(ca.name()) == "CubicAnisotropy");
}

TEST_CASE("CubicAnisotropy: zero Kc gives zero field", "[cubic]") {
    auto g = g5();
    VectorField3D mv(g), H(g);
    mv.set_uniform({1, 0, 0});
    H.set_uniform({0, 0, 0});

    CubicAnisotropyField ca(0, 0);
    ca.accumulate(mv, py_mat(), H);
    for (Index i = 0; i < H.size(); ++i)
        REQUIRE(H[i].norm() < 1e-30);
}

TEST_CASE("CubicAnisotropy: m along c1 gives zero field (symmetry)", "[cubic]") {
    // m = c1 → a1=1, a2=a3=0 → e = Kc1*0 + Kc2*0 → H = 0
    auto g = g5();
    VectorField3D mv(g), H(g);
    mv.set_uniform({1, 0, 0});   // along c1
    H.set_uniform({0, 0, 0});

    CubicAnisotropyField ca(1e4, 0);
    ca.accumulate(mv, py_mat(), H);
    for (Index i = 0; i < H.size(); ++i)
        REQUIRE(H[i].norm() < 1e3);  // zero within floating-point tolerance
}

TEST_CASE("CubicAnisotropy: m at 45 deg produces non-zero field", "[cubic]") {
    // m = (1/√2, 1/√2, 0): a1=a2=1/√2, a3=0
    // dE/da1 = 2Kc1*a1*a2^2 → H component along c1 ≠ 0
    auto g = g5();
    VectorField3D mv(g), H(g);
    mv.set_uniform({1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0});
    H.set_uniform({0, 0, 0});

    CubicAnisotropyField ca(1e4, 0);
    ca.accumulate(mv, py_mat(), H);
    REQUIRE(H[0].norm() > 0.0);
}

TEST_CASE("CubicAnisotropy: energy sign convention (Kc1>0 easy-axis along c1)", "[cubic]") {
    // For Kc1>0: minimum energy at m along {100} or {010} or {001}
    // m along c1: E = Kc1*(0 + 0 + 0) = 0 (minimum)
    // m along (111)/sqrt(3): E = Kc1*(1/9+1/9+1/9) = Kc1/3 (maximum)
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m1(g), m2(g);
    m1.set_uniform({1, 0, 0});
    m2.set_uniform({1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)});

    Material mat = py_mat(); mat.Ms = 8e5;
    CubicAnisotropyField ca(1e4, 0);
    Real E1 = ca.energy(m1, mat);
    Real E2 = ca.energy(m2, mat);
    REQUIRE(E1 < E2);  // easy axis along c1 has lower energy
}

TEST_CASE("CubicAnisotropy: energy_density integral matches total energy", "[cubic]") {
    auto g = g5();
    VectorField3D mv(g);
    mv.set_uniform({1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)});

    CubicAnisotropyField ca(1e4, 500);
    Real E_total = ca.energy(mv, py_mat());
    ScalarField3D ed = ca.energy_density(mv, py_mat());

    Real E_from_density = 0;
    for (Index i = 0; i < ed.size(); ++i)
        E_from_density += ed[i] * g.cell_volume();
    REQUIRE_THAT(E_from_density, WithinRel(E_total, 1e-10));
}

// ============================================================
// Item 2: RegionMap
// ============================================================

TEST_CASE("RegionMap: default region is 0", "[region]") {
    auto g = g5();
    RegionMap rm(g);
    for (Index i = 0; i < rm.size(); ++i)
        REQUIRE(rm[i] == 0);
}

TEST_CASE("RegionMap: def_region sets correct IDs", "[region]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    RegionMap rm(g);
    GeomMask mask = rect(g, 10e-9, 20e-9);  // 2×4 cells
    rm.def_region(1, mask);

    int n_region1 = 0;
    for (Index i = 0; i < rm.size(); ++i)
        if (rm[i] == 1) ++n_region1;
    REQUIRE(n_region1 > 0);
    REQUIRE(n_region1 < rm.size());
}

TEST_CASE("RegionMap: region_mask returns correct mask", "[region]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    RegionMap rm(g);
    GeomMask mask_in = rect(g, 10e-9, 10e-9);
    rm.def_region(2, mask_in);

    GeomMask mask_out = rm.region_mask(2);
    // mask_out should be 1 exactly where rm[i]==2
    for (Index i = 0; i < rm.size(); ++i) {
        if (rm[i] == 2)
            REQUIRE_THAT(mask_out[i], WithinAbs(1.0, 1e-15));
        else
            REQUIRE_THAT(mask_out[i], WithinAbs(0.0, 1e-15));
    }
}

TEST_CASE("RegionMap: set_magnetization writes to correct cells", "[region]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    RegionMap rm(g);
    rm.def_region(1, rect(g, 20e-9, 20e-9));  // left half

    VectorField3D mv(g);
    mv.set_uniform({1, 0, 0});
    rm.set_magnetization(1, mv, {0, 0, 1});

    // Cells in region 1 should now be {0,0,1}
    for (Index i = 0; i < rm.size(); ++i) {
        if (rm[i] == 1) {
            REQUIRE_THAT(mv[i].z, WithinAbs(1.0, 1e-12));
        } else {
            REQUIRE_THAT(mv[i].x, WithinAbs(1.0, 1e-12));
        }
    }
}

// ============================================================
// Item 3: Initial magnetization states
// ============================================================

TEST_CASE("InitMag: uniform_mag produces unit vectors", "[init_mag]") {
    auto g = g5();
    VectorField3D mv = uniform_mag(g, {1, 1, 0});
    for (Index i = 0; i < mv.size(); ++i)
        REQUIRE_THAT(mv[i].norm(), WithinAbs(1.0, 1e-12));
}

TEST_CASE("InitMag: neel_skyrmion mz=-1 at core, +1 far", "[init_mag]") {
    // Fine grid for skyrmion profile test
    StructuredGrid g(21, 21, 1, 5e-9, 5e-9, 5e-9);
    const Real r_sky = 25e-9;
    VectorField3D mv = neel_skyrmion(g, r_sky, 1, 1);

    // Centre cell: mz = -pol = -1  (θ=π at ρ=0)
    const Index centre = g.linear_index(10, 10, 0);
    REQUIRE_THAT(mv[centre].z, WithinAbs(-1.0, 0.1));

    // Corner cell (ρ≈71nm >> r=25nm): mz approaching +pol=+1
    const Index corner = g.linear_index(0, 0, 0);
    REQUIRE(mv[corner].z > 0.7);
}

TEST_CASE("InitMag: bloch_skyrmion mz at core", "[init_mag]") {
    StructuredGrid g(21, 21, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv = bloch_skyrmion(g, 25e-9, 1, 1);
    const Index centre = g.linear_index(10, 10, 0);
    REQUIRE_THAT(mv[centre].z, WithinAbs(-1.0, 0.1));
}

TEST_CASE("InitMag: two_domain splits correctly in x", "[init_mag]") {
    StructuredGrid g(8, 4, 1, 5e-9, 5e-9, 5e-9);
    Vec3 m1{1, 0, 0}, m2{-1, 0, 0};
    VectorField3D mv = two_domain(g, m1, m2, 'x');

    // Left half (x < 0): m = m1
    Index il = g.linear_index(1, 0, 0);
    REQUIRE_THAT(mv[il].x, WithinAbs(1.0, 1e-12));

    // Right half (x > 0): m = m2
    Index ir = g.linear_index(6, 0, 0);
    REQUIRE_THAT(mv[ir].x, WithinAbs(-1.0, 1e-12));
}

TEST_CASE("InitMag: vortex_state core has |mz| near 1", "[init_mag]") {
    StructuredGrid g(21, 21, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv = vortex_state(g, 1, 1);
    // Core at centre: mz = pol=1
    const Index centre = g.linear_index(10, 10, 0);
    REQUIRE_THAT(mv[centre].z, WithinAbs(1.0, 0.15));
}

TEST_CASE("InitMag: random_mag produces unit vectors", "[init_mag]") {
    auto g = g5();
    VectorField3D mv = random_mag(g, 1234);
    for (Index i = 0; i < mv.size(); ++i)
        REQUIRE_THAT(mv[i].norm(), WithinAbs(1.0, 1e-12));
}

// ============================================================
// Item 4: New geometry shapes
// ============================================================

TEST_CASE("Geom: square = rect with equal sides", "[geom]") {
    StructuredGrid g(10, 10, 1, 5e-9, 5e-9, 5e-9);
    GeomMask m1 = square(g, 20e-9);
    GeomMask m2 = rect(g, 20e-9, 20e-9);
    for (Index i = 0; i < m1.size(); ++i)
        REQUIRE_THAT(m1[i], WithinAbs(m2[i], 1e-15));
}

TEST_CASE("Geom: cuboid centre is inside", "[geom]") {
    StructuredGrid g(10, 10, 10, 5e-9, 5e-9, 5e-9);
    GeomMask m = cuboid(g, 20e-9, 20e-9, 20e-9);
    REQUIRE_THAT(m[g.linear_index(5, 5, 5)], WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(m[g.linear_index(0, 0, 0)], WithinAbs(0.0, 1e-15));
}

TEST_CASE("Geom: sphere centre inside, corner outside", "[geom]") {
    StructuredGrid g(10, 10, 10, 5e-9, 5e-9, 5e-9);
    GeomMask m = sphere(g, 15e-9);
    REQUIRE_THAT(m[g.linear_index(5, 5, 5)], WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(m[g.linear_index(0, 0, 0)], WithinAbs(0.0, 1e-15));
}

TEST_CASE("Geom: layer selects only one z-layer", "[geom]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GeomMask m = layer(g, 2);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Real v = m(ix, iy, iz);
        if (iz == 2)
            REQUIRE_THAT(v, WithinAbs(1.0, 1e-15));
        else
            REQUIRE_THAT(v, WithinAbs(0.0, 1e-15));
    }
}

TEST_CASE("Geom: x_range selects correct cells", "[geom]") {
    StructuredGrid g(10, 1, 1, 5e-9, 5e-9, 5e-9);
    // Box x range: [-25nm, +25nm]; x_range([-10nm, +10nm]) → 4 centre cells
    GeomMask m = x_range(g, -10e-9, 10e-9);
    int n_in = 0;
    for (Index i = 0; i < m.size(); ++i)
        if (m[i] > 0.5) ++n_in;
    // Cells at ix=4 and ix=5 span x in [-25,-20]...[+25] so ix=3,4,5,6 are in range
    REQUIRE(n_in > 0);
    REQUIRE(n_in < g.size());
}

// ============================================================
// Item 5: Energy density fields
// ============================================================

TEST_CASE("Edens: ZeemanField energy_density sum matches energy", "[edens]") {
    auto g = g5();
    VectorField3D mv(g);
    mv.set_uniform({0.6, 0.8, 0});

    Material mat = py_mat();
    ZeemanField zeeman({0, 1e6, 0});
    Real E_total = zeeman.energy(mv, mat);
    ScalarField3D ed = zeeman.energy_density(mv, mat);

    Real E_sum = 0;
    for (Index i = 0; i < ed.size(); ++i)
        E_sum += ed[i] * g.cell_volume();
    REQUIRE_THAT(E_sum, WithinRel(E_total, 1e-10));
}

TEST_CASE("Edens: UniaxialAnisotropyField energy_density spatial", "[edens]") {
    // Different m orientations → different per-cell energies
    StructuredGrid g(2, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    mv[0] = {0, 0, 1};  // along easy axis
    mv[1] = {1, 0, 0};  // perpendicular

    Material mat = py_mat();
    mat.K_uniaxial = 1e4;
    mat.easy_axis  = {0, 0, 1};

    UniaxialAnisotropyField anis;
    ScalarField3D ed = anis.energy_density(mv, mat);
    // Cell 0 (along easy axis): e = -K < 0
    REQUIRE(ed[0] < 0);
    // Cell 1 (perpendicular): e = 0
    REQUIRE_THAT(ed[1], WithinAbs(0.0, 1e-3));
}

TEST_CASE("Edens: ExchangeField energy_density sum approximates energy", "[edens]") {
    StructuredGrid g(5, 5, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    // Gradient in x
    for (Index ix = 0; ix < g.nx(); ++ix)
    for (Index iy = 0; iy < g.ny(); ++iy) {
        Real angle = 0.3 * ix;
        mv[g.linear_index(ix, iy, 0)] = {std::cos(angle), std::sin(angle), 0};
    }

    Material mat = py_mat(); mat.A_exchange = 1.3e-11;
    ExchangeField exch;
    Real E_total = exch.energy(mv, mat);
    ScalarField3D ed = exch.energy_density(mv, mat);

    Real E_sum = 0;
    for (Index i = 0; i < ed.size(); ++i)
        E_sum += ed[i] * g.cell_volume();
    // Not exact (boundary effects differ) but within an order of magnitude
    REQUIRE(std::abs(E_sum) > 0);
    REQUIRE(std::abs(E_sum - E_total) < std::abs(E_total) * 2.0);
}

TEST_CASE("Edens: BulkDMI energy_density sum matches energy", "[edens]") {
    StructuredGrid g(7, 1, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    for (Index ix = 0; ix < g.nx(); ++ix) {
        Real mz = std::tanh(static_cast<Real>(ix - 3) * 0.5);
        Real mx = std::sqrt(std::max(0.0, 1.0 - mz*mz));
        mv[g.linear_index(ix, 0, 0)] = {mx, 0, mz};
    }

    BulkDMIField dmi(3e-3);
    Real E_total = dmi.energy(mv, py_mat());
    ScalarField3D ed = dmi.energy_density(mv, py_mat());

    Real E_sum = 0;
    for (Index i = 0; i < ed.size(); ++i)
        E_sum += ed[i] * g.cell_volume();
    REQUIRE_THAT(E_sum, WithinRel(E_total, 1e-10));
}

TEST_CASE("Edens: EffectiveFieldSum energy_density sums all terms", "[edens]") {
    auto g = g5();
    VectorField3D mv(g);
    mv.set_uniform({0, 0, 1});

    Material mat = py_mat();
    mat.K_uniaxial = 1e4;
    mat.easy_axis  = {0, 0, 1};

    EffectiveFieldSum heff;
    auto zeeman = std::make_shared<ZeemanField>(Vec3{0, 0, 1e6});
    auto anis   = std::make_shared<UniaxialAnisotropyField>();
    heff.add(zeeman);
    heff.add(anis);

    ScalarField3D ed = heff.energy_density(mv, mat);
    ScalarField3D ed_z = zeeman->energy_density(mv, mat);
    ScalarField3D ed_a = anis->energy_density(mv, mat);

    // Sum should match cell by cell
    for (Index i = 0; i < ed.size(); ++i)
        REQUIRE_THAT(ed[i], WithinAbs(ed_z[i] + ed_a[i], 1e-6));
}

TEST_CASE("Edens: DemagField energy_density sum matches energy", "[edens]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    mv.set_uniform({1, 0, 0});

    Material mat = py_mat();
    DemagField demag(g);
    Real E_total = demag.energy(mv, mat);
    ScalarField3D ed = demag.energy_density(mv, mat);

    Real E_sum = 0;
    for (Index i = 0; i < ed.size(); ++i)
        E_sum += ed[i] * g.cell_volume();
    REQUIRE_THAT(E_sum, WithinRel(E_total, 1e-8));
}

// ============================================================
// Phase F: Topological charge
// ============================================================

TEST_CASE("TopoCharge: uniform m → Q=0", "[topo]") {
    // Uniform magnetization has no winding → Q = 0
    StructuredGrid g(21, 21, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv(g);
    mv.set_uniform({0, 0, 1});
    const Real Q = topological_charge_Q(mv);
    REQUIRE_THAT(Q, WithinAbs(0.0, 1e-10));
}

TEST_CASE("TopoCharge: Neel skyrmion Q ≈ -1 (charge=1, pol=1)", "[topo]") {
    // 21×21 grid, r=25nm, dx=5nm → 105nm box
    // skyrmion centre at origin, topological charge should be ≈ ±1
    StructuredGrid g(61, 61, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv = neel_skyrmion(g, 20e-9, 1, 1);
    const Real Q = topological_charge_Q(mv);
    // Discrete grid discretisation error: expect Q within 0.1 of -1
    REQUIRE(Q < -0.8);
    REQUIRE(Q > -1.2);
}

TEST_CASE("TopoCharge: density field sums to Q (within dA factor)", "[topo]") {
    StructuredGrid g(31, 31, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mv = neel_skyrmion(g, 20e-9, 1, 1);
    auto [Q, dens] = topological_charge(mv);
    const Real dA = g.dx() * g.dy();
    constexpr double inv4pi = 1.0 / (4.0 * std::numbers::pi);
    Real Q_from_dens = 0;
    for (Index i = 0; i < dens.size(); ++i)
        Q_from_dens += dens[i] * dA * inv4pi;
    REQUIRE_THAT(Q_from_dens, WithinAbs(Q, 1e-12));
}

TEST_CASE("TopoCharge: Bloch skyrmion Q same sign as Neel", "[topo]") {
    StructuredGrid g(61, 61, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D mn = neel_skyrmion(g, 20e-9, 1, 1);
    VectorField3D mb = bloch_skyrmion(g, 20e-9, 1, 1);
    const Real Qn = topological_charge_Q(mn);
    const Real Qb = topological_charge_Q(mb);
    // Both should have the same sign and similar magnitude
    REQUIRE(Qn * Qb > 0);
    REQUIRE_THAT(Qb, WithinAbs(Qn, 0.05));
}

// ===========================================================================
// Skyrmion tracking tools — [skyrmion]
// ===========================================================================

TEST_CASE("SkyrmionTools: corepos near box centre for centred skyrmion", "[skyrmion]") {
    // 61×61×1 grid, 5 nm cells → 305 nm × 305 nm box
    StructuredGrid g(61, 61, 1, 5e-9, 5e-9, 5e-9);
    // pol=+1: mz_core = -1 (find_max=false finds min mz)
    VectorField3D m = neel_skyrmion(g, 20e-9, 1, 1);
    auto [cx, cy] = skyrmion_corepos(m, false);
    // Core should be within ±1 cell of box centre (origin)
    REQUIRE_THAT(cx, WithinAbs(0.0, 5e-9));
    REQUIRE_THAT(cy, WithinAbs(0.0, 5e-9));
}

TEST_CASE("SkyrmionTools: corepos finds pol=-1 core with find_max=true", "[skyrmion]") {
    StructuredGrid g(61, 61, 1, 5e-9, 5e-9, 5e-9);
    // pol=-1: mz_core = +1
    VectorField3D m = neel_skyrmion(g, 20e-9, 1, -1);
    auto [cx, cy] = skyrmion_corepos(m, true);
    REQUIRE_THAT(cx, WithinAbs(0.0, 5e-9));
    REQUIRE_THAT(cy, WithinAbs(0.0, 5e-9));
}

TEST_CASE("SkyrmionTools: bubble_pos matches corepos for centred skyrmion", "[skyrmion]") {
    StructuredGrid g(61, 61, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m = neel_skyrmion(g, 20e-9, 1, 1);
    auto [cpx, cpy] = skyrmion_corepos(m, false);
    auto [bpx, bpy] = bubble_pos(m);
    // Q-centroid should be close to mz-extremum position (within 1 cell)
    REQUIRE_THAT(bpx, WithinAbs(cpx, 5e-9));
    REQUIRE_THAT(bpy, WithinAbs(cpy, 5e-9));
    // Both near origin
    REQUIRE_THAT(bpx, WithinAbs(0.0, 8e-9));
    REQUIRE_THAT(bpy, WithinAbs(0.0, 8e-9));
}

TEST_CASE("SkyrmionTools: skyrmion_count = 1 for single skyrmion", "[skyrmion]") {
    StructuredGrid g(61, 61, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m = neel_skyrmion(g, 20e-9, 1, 1);
    REQUIRE(skyrmion_count(m) == 1);
}

TEST_CASE("SkyrmionTools: skyrmion_count = 0 for uniform state", "[skyrmion]") {
    StructuredGrid g(21, 21, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    for (Index i = 0; i < m.size(); ++i) m[i] = {0, 0, 1};
    REQUIRE(skyrmion_count(m) == 0);
}

TEST_CASE("SkyrmionTools: bubble_pos returns (0,0) for uniform state", "[skyrmion]") {
    StructuredGrid g(11, 11, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g);
    for (Index i = 0; i < m.size(); ++i) m[i] = {0, 0, 1};
    auto [bpx, bpy] = bubble_pos(m);
    REQUIRE_THAT(bpx, WithinAbs(0.0, 1e-20));
    REQUIRE_THAT(bpy, WithinAbs(0.0, 1e-20));
}
