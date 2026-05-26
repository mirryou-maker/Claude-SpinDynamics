#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/spin_torque.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================
// Helpers
// ============================================================

static StructuredGrid single_cell() {
    return StructuredGrid(1, 1, 1, 2e-9, 2e-9, 2e-9);
}

static Material cobalt_nm() {
    // Cobalt with 2 nm thickness (d used in STT/SOT formulas)
    return Material::cobalt();
}

// ============================================================
// a_J / a_SOT formula checks
// ============================================================

TEST_CASE("STT: a_J dimensional check", "[stt]") {
    // a_J = γ₀ ħ J P / (2 e Ms d)  should be ~10^8 – 10^10 [1/s]
    // for J = 1e12 A/m², P = 0.5, d = 2 nm, Ms = 1.4e6 A/m
    Real J = 1e12, P = 0.5, d = 2e-9;
    SlonczewskiSTT stt(J, P, d, {0, 0, 1});
    Real aJ = stt.a_J(1.4e6);
    REQUIRE(aJ > 1e7);
    REQUIRE(aJ < 1e12);
}

TEST_CASE("SOT: a_SOT dimensional check", "[sot]") {
    Real J_c = 1e12, theta_SH = 0.12, d = 2e-9;
    SpinOrbitTorque sot(J_c, theta_SH, d, {0, 1, 0});
    Real a = sot.a_SOT(1.4e6);
    REQUIRE(a > 1e7);
    REQUIRE(a < 1e12);
}

// ============================================================
// STT torque direction tests
// ============================================================

TEST_CASE("STT: zero torque when m parallel p", "[stt]") {
    // m ∥ p → m×p = 0 → both terms vanish
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm(g);
    m.set_uniform({0, 0, 1});

    SlonczewskiSTT stt(1e12, 0.5, 2e-9, {0, 0, 1});
    stt.accumulate(m, Material::cobalt(), dm);

    REQUIRE_THAT(dm.at(0,0,0).norm(), WithinAbs(0.0, 1.0));
}

TEST_CASE("STT: zero torque when m antiparallel p", "[stt]") {
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm(g);
    m.set_uniform({0, 0, -1});

    SlonczewskiSTT stt(1e12, 0.5, 2e-9, {0, 0, 1});
    stt.accumulate(m, Material::cobalt(), dm);

    REQUIRE_THAT(dm.at(0,0,0).norm(), WithinAbs(0.0, 1.0));
}

TEST_CASE("STT: damping-like torque direction (m=+x, p=+z)", "[stt]") {
    // m×(m×p): m=(1,0,0), p=(0,0,1)
    // m×p = (0,-1,0), m×(m×p) = (1,0,0)×(0,-1,0) = (0,0,-1)
    // With a_J > 0: dm/dt should have -z component
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm(g);
    m.set_uniform({1, 0, 0});

    SlonczewskiSTT stt(1e12, 0.5, 2e-9, {0, 0, 1}, /*beta=*/0.0);
    stt.accumulate(m, Material::cobalt(), dm);

    Vec3 t = dm.at(0, 0, 0);
    REQUIRE(t.z < 0);                        // pushed away from +z (antidamping)
    REQUIRE_THAT(t.x, WithinAbs(0.0, 1.0)); // no x/y component
    REQUIRE_THAT(t.y, WithinAbs(0.0, 1.0));
}

TEST_CASE("STT: field-like torque direction (m=+x, p=+z, beta=1)", "[stt]") {
    // b_J term: -β a_J (m×p), m×p = (0,-1,0) → −β a_J * (0,-1,0) = (0, β a_J, 0)
    // With beta=1 and a_J > 0: dm/dt should have +y component
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm(g);
    m.set_uniform({1, 0, 0});

    // beta = 1, pure field-like (set a_J direction to zero by using negative J
    // to cancel, then use only beta term) – simpler: just check sign with both terms
    SlonczewskiSTT stt(1e12, 0.5, 2e-9, {0, 0, 1}, /*beta=*/1.0);
    stt.accumulate(m, Material::cobalt(), dm);

    Vec3 t = dm.at(0, 0, 0);
    // DL: -z component; FL: +y component (since b_J = -β a_J and m×p = (0,-1,0))
    REQUIRE(t.y > 0);
}

TEST_CASE("STT: negative J reverses all torques", "[stt]") {
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm_pos(g), dm_neg(g);
    m.set_uniform({1, 0, 0});
    Material mat = Material::cobalt();

    SlonczewskiSTT stt_pos( 1e12, 0.5, 2e-9, {0, 0, 1}, 0.5);
    SlonczewskiSTT stt_neg(-1e12, 0.5, 2e-9, {0, 0, 1}, 0.5);
    stt_pos.accumulate(m, mat, dm_pos);
    stt_neg.accumulate(m, mat, dm_neg);

    Vec3 tp = dm_pos.at(0,0,0);
    Vec3 tn = dm_neg.at(0,0,0);
    REQUIRE_THAT(tp.x + tn.x, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(tp.y + tn.y, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(tp.z + tn.z, WithinAbs(0.0, 1.0));
}

// ============================================================
// SOT torque direction tests
// ============================================================

TEST_CASE("SOT: zero DL torque when m parallel sigma", "[sot]") {
    // m ∥ σ → m×σ = 0 → DL vanishes (FL also vanishes if η_FL=0)
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm(g);
    m.set_uniform({0, 1, 0});

    SpinOrbitTorque sot(1e12, 0.12, 2e-9, {0, 1, 0}, /*eta_DL=*/1.0, /*eta_FL=*/0.0);
    sot.accumulate(m, Material::cobalt(), dm);

    REQUIRE_THAT(dm.at(0,0,0).norm(), WithinAbs(0.0, 1.0));
}

TEST_CASE("SOT: DL torque direction (m=+z, sigma=+y)", "[sot]") {
    // m=(0,0,1), σ=(0,1,0)
    // m×σ = (0,0,1)×(0,1,0) = (-1,0,0)
    // m×(m×σ) = (0,0,1)×(-1,0,0) = (0,-1,0)
    // With a_SOT > 0, η_DL=1: dm/dt has -y component
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm(g);
    m.set_uniform({0, 0, 1});

    SpinOrbitTorque sot(1e12, 0.12, 2e-9, {0, 1, 0}, 1.0, 0.0);
    sot.accumulate(m, Material::cobalt(), dm);

    Vec3 t = dm.at(0, 0, 0);
    REQUIRE(t.y < 0);
    REQUIRE_THAT(t.x, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(t.z, WithinAbs(0.0, 1.0));
}

TEST_CASE("SOT: FL torque direction (m=+z, sigma=+y, eta_FL=1)", "[sot]") {
    // FL term: a_SOT η_FL (m×σ) = (0,0,1)×(0,1,0) = (-1,0,0)
    // dm/dt should have -x component
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm(g);
    m.set_uniform({0, 0, 1});

    SpinOrbitTorque sot(1e12, 0.12, 2e-9, {0, 1, 0}, /*eta_DL=*/0.0, /*eta_FL=*/1.0);
    sot.accumulate(m, Material::cobalt(), dm);

    Vec3 t = dm.at(0, 0, 0);
    REQUIRE(t.x < 0);
    REQUIRE_THAT(t.y, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(t.z, WithinAbs(0.0, 1.0));
}

TEST_CASE("SOT: negative theta_SH reverses torque", "[sot]") {
    StructuredGrid g = single_cell();
    VectorField3D m(g), dm_p(g), dm_n(g);
    m.set_uniform({0, 0, 1});
    Material mat = Material::cobalt();

    SpinOrbitTorque sot_p(1e12,  0.12, 2e-9, {0, 1, 0});
    SpinOrbitTorque sot_n(1e12, -0.12, 2e-9, {0, 1, 0});
    sot_p.accumulate(m, mat, dm_p);
    sot_n.accumulate(m, mat, dm_n);

    Vec3 tp = dm_p.at(0,0,0);
    Vec3 tn = dm_n.at(0,0,0);
    REQUIRE_THAT(tp.norm() - tn.norm(), WithinAbs(0.0, 1.0));
    REQUIRE_THAT(tp.x + tn.x, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(tp.y + tn.y, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(tp.z + tn.z, WithinAbs(0.0, 1.0));
}

// ============================================================
// Integration tests (macrospin dynamics)
// ============================================================

TEST_CASE("STT: RK4 backward integration recovers original state", "[stt][integrator]") {
    // Forward step with +J, then backward step with -J should approximately
    // return m to start (linear regime only, small step)
    StructuredGrid g = single_cell();
    VectorField3D m(g), m0(g);
    m.set_uniform({1, 0, 0});
    m0.set_uniform({1, 0, 0});

    Material mat = Material::cobalt();
    mat.alpha = 0.0;

    EffectiveFieldSum heff;  // no H_eff

    SpinTorqueSum stt_fwd, stt_bwd;
    stt_fwd.add(std::make_shared<SlonczewskiSTT>( 5e11, 0.5, 2e-9, Vec3{0,0,1}));
    stt_bwd.add(std::make_shared<SlonczewskiSTT>(-5e11, 0.5, 2e-9, Vec3{0,0,1}));

    const Real dt = 1e-14;  // very small step
    RK4Integrator rk4(dt);
    rk4.step(m, mat, heff, &stt_fwd);
    rk4.step(m, mat, heff, &stt_bwd);

    REQUIRE_THAT(m.at(0,0,0).x, WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(m.at(0,0,0).y, WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(m.at(0,0,0).z, WithinAbs(0.0, 1e-4));
}

TEST_CASE("STT: high current switches m from +z toward -z (macrospin)", "[stt][integrator]") {
    // m starts near +z, p = -z. Large negative a_J pushes m toward -z.
    // (a_J < 0 when J > 0 and p = -z? No: a_J sign = sign(J) only.
    //  For p = +z and m starting at +z, m×p = 0 → no torque.
    //  Use m slightly tilted and p = -z.
    StructuredGrid g = single_cell();
    VectorField3D m(g);
    // Start slightly off +z to break symmetry
    Real eps = 0.05;
    Vec3 m0 = {eps, 0, std::sqrt(1.0 - eps*eps)};
    m.set_uniform(m0);

    Material mat = Material::cobalt();
    mat.alpha = 0.05;
    mat.K_uniaxial = 0;  // no anisotropy for clean switching

    EffectiveFieldSum heff;
    // Large positive J, p = +z: a_J > 0 → antidamping w.r.t. +z
    // With α > 0 and large enough a_J, m should switch to -z
    SpinTorqueSum stt;
    stt.add(std::make_shared<SlonczewskiSTT>(5e12, 0.6, 2e-9, Vec3{0, 0, 1}));

    RK4Integrator rk4(5e-14);
    for (int i = 0; i < 3000; ++i)
        rk4.step(m, mat, heff, &stt);

    // m should have moved away from start (some dynamics happened)
    REQUIRE(m.at(0,0,0).z < m0.z);
}

TEST_CASE("SOT: current causes m to deviate from +z equilibrium", "[sot][integrator]") {
    // m = +z (OOP), σ = +y (x-current, z-normal interface, θ_SH > 0)
    // DL-SOT pushes m in -y direction
    StructuredGrid g = single_cell();
    VectorField3D m(g);
    m.set_uniform({0, 0, 1});

    Material mat = Material::cobalt();
    mat.alpha = 0.02;

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 1e5}));  // OOP field stabilises

    SpinTorqueSum sot_sum;
    sot_sum.add(std::make_shared<SpinOrbitTorque>(1e12, 0.30, 2e-9, Vec3{0,1,0},
                                                    /*eta_DL=*/1.0, /*eta_FL=*/0.0));

    RK4Integrator rk4(1e-13);
    for (int i = 0; i < 500; ++i)
        rk4.step(m, mat, heff, &sot_sum);

    // m_z should have decreased from 1 (SOT drives m away from +z)
    REQUIRE(m.at(0,0,0).z < 0.999);
    REQUIRE(m.at(0,0,0).norm() - 1.0 < 1e-10);
}

TEST_CASE("|m| = 1 preserved with both STT and SOT active", "[stt][sot][integrator]") {
    StructuredGrid g(2, 2, 1, 2e-9, 2e-9, 2e-9);
    VectorField3D m(g);
    m.set_uniform({0.6, 0.0, 0.8});

    Material mat = Material::cobalt();
    mat.alpha = 0.05;

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(Vec3{0, 0, 5e4}));

    SpinTorqueSum spin;
    spin.add(std::make_shared<SlonczewskiSTT>(1e12, 0.5, 2e-9, Vec3{0, 0, -1}));
    spin.add(std::make_shared<SpinOrbitTorque>(5e11, 0.12, 2e-9, Vec3{0, 1, 0}));

    RK4Integrator rk4(5e-14);
    for (int step = 0; step < 100; ++step) {
        rk4.step(m, mat, heff, &spin);
        for (Index idx = 0; idx < g.size(); ++idx)
            REQUIRE_THAT(m[idx].norm(), WithinAbs(1.0, 1e-11));
    }
}
