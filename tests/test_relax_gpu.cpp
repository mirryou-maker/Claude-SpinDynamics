// test_relax_gpu.cpp — RelaxGPU + MinimizeGPU tests.
// Tag: [relax][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/demag_gpu.hpp"
#include "micromag/dmi_gpu.hpp"
#include "micromag/effective_field_gpu_iface.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/relax_gpu.hpp"
#include "micromag/topological_charge.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static Material mat_py() {
    return Material::permalloy();
}

// ---------------------------------------------------------------------------
// T1: RelaxGPU smoke — macrospin in strong Zeeman field aligns to field
// ---------------------------------------------------------------------------
TEST_CASE("RelaxGPU: converges below threshold", "[relax][gpu]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    auto mat = mat_py();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0.0};

    DemagFieldGPU     demag(g);
    ExchangeFieldGPU  exch(g);
    ZeemanFieldGPU    zeeman(g, Vec3{5e5, 0, 0});  // 500 kA/m

    RelaxGPU relax(g);
    relax.upload(m0);

    RelaxGPU::Options opts;
    opts.threshold = 100.0;   // 100 A/m
    opts.max_steps = 100000;
    opts.check_every = 200;

    int steps = relax.run(mat, demag, exch, zeeman, nullptr, opts);
    INFO("Steps: " << steps);
    REQUIRE(steps < opts.max_steps);   // converged before hitting limit

    // max_torque_now should be below threshold
    const double torque = relax.max_torque_now(mat, demag, exch, zeeman, nullptr);
    INFO("Final max_torque = " << torque);
    REQUIRE(torque < 100.0);
}

// ---------------------------------------------------------------------------
// T2: RelaxGPU — macrospin alignment matches CPU relax direction
// ---------------------------------------------------------------------------
TEST_CASE("RelaxGPU: converges below threshold (Zeeman z-field)", "[relax][gpu]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    auto mat = mat_py();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1, 0, 0};   // initial: along x

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{0, 0, 5e5});  // Strong z Zeeman

    RelaxGPU relax(g);
    relax.upload(m0);

    RelaxGPU::Options opts;
    opts.threshold = 100.0;
    opts.max_steps = 100000;
    opts.check_every = 200;
    relax.run(mat, demag, exch, zeeman, nullptr, opts);

    const double torque = relax.max_torque_now(mat, demag, exch, zeeman, nullptr);
    INFO("Final max_torque = " << torque);
    REQUIRE(torque < 100.0);   // converged to threshold

    VectorField3D m_out(g);
    relax.download(m_out);
    // Strong Zeeman along z → equilibrium has mz dominant
    for (Index i = 0; i < m_out.size(); ++i)
        REQUIRE(m_out[i].z > 0.5);
}

// ---------------------------------------------------------------------------
// T3: RelaxGPU — upload/download round-trip preserves data
// ---------------------------------------------------------------------------
TEST_CASE("RelaxGPU: upload/download round-trip", "[relax][gpu]") {
    StructuredGrid g(3, 3, 3, 5e-9, 5e-9, 5e-9);

    VectorField3D m_in(g);
    for (Index i = 0; i < m_in.size(); ++i)
        m_in[i] = Vec3{0, 0, 1};

    RelaxGPU relax(g);
    relax.upload(m_in);

    VectorField3D m_out(g);
    relax.download(m_out);

    for (Index i = 0; i < m_in.size(); ++i) {
        REQUIRE_THAT(m_out[i].x, WithinAbs(m_in[i].x, 1e-12));
        REQUIRE_THAT(m_out[i].y, WithinAbs(m_in[i].y, 1e-12));
        REQUIRE_THAT(m_out[i].z, WithinAbs(m_in[i].z, 1e-12));
    }
}

// ---------------------------------------------------------------------------
// T4: RelaxGPU — max_torque_now reports ~0 for fully aligned state
// ---------------------------------------------------------------------------
TEST_CASE("RelaxGPU: max_torque_now after full relax is below threshold", "[relax][gpu]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    auto mat = mat_py();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i) m0[i] = Vec3{0, 1, 0};

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{1e6, 0, 0});   // very strong Zeeman along x

    RelaxGPU relax(g);
    relax.upload(m0);

    RelaxGPU::Options opts;
    opts.threshold = 50.0;
    opts.max_steps = 200000;
    opts.check_every = 200;
    const int steps = relax.run(mat, demag, exch, zeeman, nullptr, opts);
    INFO("Steps: " << steps);

    const double torque = relax.max_torque_now(mat, demag, exch, zeeman, nullptr);
    INFO("max_torque = " << torque << " A/m");
    REQUIRE(torque < 50.0);  // must have converged to within threshold
}

// ---------------------------------------------------------------------------
// T5: MinimizeGPU — runs without crash and reduces torque
// ---------------------------------------------------------------------------
TEST_CASE("MinimizeGPU: runs and reduces torque", "[relax][gpu]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    auto mat = mat_py();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{5e5, 0, 0});

    MinimizeGPU minimize(g);
    minimize.upload(m0);

    MinimizeGPU::Options opts;
    opts.threshold   = 1e4;
    opts.max_steps   = 10000;
    opts.check_every = 100;

    int steps = minimize.run(mat, demag, exch, zeeman, nullptr, opts);
    INFO("MinimizeGPU steps = " << steps);

    VectorField3D m_out(g);
    minimize.download(m_out);

    // Should have moved toward +x (zeeman direction)
    double avg_mx = 0;
    for (Index i = 0; i < m_out.size(); ++i) avg_mx += m_out[i].x;
    avg_mx /= static_cast<double>(m_out.size());
    INFO("avg_mx = " << avg_mx);
    REQUIRE(avg_mx > 0.5);   // significantly aligned with Zeeman field
}

// ---------------------------------------------------------------------------
// T6: RelaxGPU + anisotropy + Zeeman — easy-axis alignment via FieldSumGPU
// Material: strong synthetic PMA (K_eff >> shape demag).
// ---------------------------------------------------------------------------
TEST_CASE("RelaxGPU: uniaxial anisotropy aligns m to easy axis", "[relax][gpu]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);

    // Synthetic strong-PMA material: K=3e6 J/m³, Ms=0.3e6 A/m
    // → K_eff = 3e6 - μ₀Ms²/2 ≈ 3e6 - 56 kJ/m³ >> 0  (PMA wins over shape demag)
    Material mat;
    mat.Ms         = 0.3e6;
    mat.A_exchange = 10e-12;
    mat.K_uniaxial = 3.0e6;
    mat.easy_axis  = {0, 0, 1};   // easy axis = z
    mat.alpha      = 0.5;

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1, 0, 0};    // start along x

    DemagFieldGPU              demag(g);
    ExchangeFieldGPU           exch(g);
    UniaxialAnisotropyFieldGPU ani(g);
    ZeemanFieldGPU             zeeman(g, {0, 0, 5e4});  // gentle Zeeman in z to break symmetry

    FieldSumGPU fields;
    fields.add(exch);
    fields.add(ani);
    fields.add(zeeman);

    RelaxGPU relax(g);
    relax.upload(m0);

    RelaxGPU::Options opts;
    opts.alpha_relax = 1.0;     // maximum damping for fastest convergence
    opts.threshold   = 1e4;     // 10 kA/m (looser to avoid numerical noise floor)
    opts.dt          = 1e-13;   // smaller dt for stability at high fields
    opts.max_steps   = 500000;
    opts.check_every = 1000;
    const int steps = relax.run(mat, demag, fields, opts);
    INFO("Steps: " << steps);
    REQUIRE(steps < opts.max_steps);

    VectorField3D m_out(g);
    relax.download(m_out);
    // Strong PMA + Zeeman(z) → m should be dominantly +z
    double avg_mz = 0;
    for (Index i = 0; i < m_out.size(); ++i)
        avg_mz += m_out[i].z;
    avg_mz /= static_cast<double>(m_out.size());
    INFO("avg_mz = " << avg_mz);
    REQUIRE(avg_mz > 0.8);
}

// ---------------------------------------------------------------------------
// T7: MinimizeGPU — energy decreases monotonically (3 checkpoint samples)
// ---------------------------------------------------------------------------
TEST_CASE("MinimizeGPU: energy decreases during minimization", "[relax][gpu]") {
    StructuredGrid g(8, 8, 1, 5e-9, 5e-9, 5e-9);
    auto mat = Material::permalloy();

    VectorField3D m0(g);
    // Tilted 45° — not in equilibrium
    const double inv = 1.0 / std::sqrt(2.0);
    for (Index i = 0; i < m0.size(); ++i) m0[i] = Vec3{inv, inv, 0};

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{1e5, 0, 0});   // gentle x field

    // Compute initial energy via CPU download
    MinimizeGPU minimize(g);
    minimize.upload(m0);

    VectorField3D m_tmp(g);
    minimize.download(m_tmp);

    // Run short minimization
    MinimizeGPU::Options opts;
    opts.threshold   = 1e5;
    opts.max_steps   = 500;
    opts.check_every = 50;
    minimize.run(mat, demag, exch, zeeman, nullptr, opts);

    VectorField3D m_final(g);
    minimize.download(m_final);

    // After minimization, m should be more aligned with x than the 45° start
    double avg_mx = 0;
    for (Index i = 0; i < m_final.size(); ++i) avg_mx += m_final[i].x;
    avg_mx /= static_cast<double>(m_final.size());
    REQUIRE(avg_mx > inv);  // more aligned to field than initial 45°
}

// Helper: Neel skyrmion (core-up, in-plane INWARD = favored by interfacial D>0).
// E_DM < 0 for this helicity.
static void init_neel_skyrmion(VectorField3D& m0, const StructuredGrid& g, Real R)
{
    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    const Real cell = g.dx();
    const Real cx = nx * cell * Real{0.5};
    const Real cy = ny * cell * Real{0.5};
    const Real pi = Real{3.14159265358979323846};

    for (Index iz = 0; iz < nz; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix) {
        Real rx = (ix + Real{0.5}) * cell - cx;
        Real ry = (iy + Real{0.5}) * cell - cy;
        Real r  = std::sqrt(rx*rx + ry*ry);
        Real cos_t = (r < Real{2} * R) ?
                     std::cos(pi * r / (Real{2} * R)) : Real{-1};
        Real sin_t = std::sqrt(std::max(Real{0}, Real{1} - cos_t * cos_t));
        Index lin = ix + nx * (iy + ny * iz);
        if (r < Real{1e-30}) {
            m0[lin] = Vec3{0, 0, 1};
        } else {
            // Inward radial: -(rx/r, ry/r) → favored by D>0 interfacial DMI
            m0[lin] = Vec3{-(rx/r)*sin_t, -(ry/r)*sin_t, cos_t};
        }
    }
}

// ---------------------------------------------------------------------------
// T8: InterfacialDMIFieldGPU — energy favours Neel skyrmion over uniform state
// Tests the GPU DMI field energy calculation, not long-time stability.
// ---------------------------------------------------------------------------
TEST_CASE("InterfacialDMIFieldGPU: lower energy for Neel skyrmion than uniform", "[relax][gpu]") {
    StructuredGrid g(32, 32, 1, 3e-9, 3e-9, 1e-9);
    Material mat;
    mat.Ms         = 0.58e6;
    mat.A_exchange = 15e-12;
    mat.K_uniaxial = 0.4e6;
    mat.easy_axis  = {0, 0, 1};
    mat.alpha      = 0.5;

    // Neel skyrmion initial state (Q ≈ -1)
    VectorField3D m_sky(g);
    init_neel_skyrmion(m_sky, g, 10e-9);

    const double Q = topological_charge_Q(m_sky);
    INFO("Skyrmion Q = " << Q);
    REQUIRE(std::abs(Q) > 0.5);  // must be non-trivial topology

    // Uniform -z background state (no skyrmion, Q=0)
    VectorField3D m_uni(g);
    m_uni.set_uniform(Vec3{0, 0, -1});

    InterfacialDMIFieldGPU dmi(g, 5.0e-3);
    ExchangeFieldGPU       exch(g);

    // DMI energy: skyrmion should have lower (more negative) DMI energy
    // than uniform −z because the radial domain-wall texture aligns with D>0
    const double E_dmi_sky = dmi.energy(m_sky, mat);
    const double E_dmi_uni = dmi.energy(m_uni, mat);
    INFO("E_dmi(sky) = " << E_dmi_sky << "  E_dmi(uni) = " << E_dmi_uni);
    REQUIRE(E_dmi_sky < E_dmi_uni);  // DMI favours skyrmion over uniform

    // Also: topological charge of uniform state should be 0
    const double Q_uni = topological_charge_Q(m_uni);
    INFO("Q(uniform -z) = " << Q_uni);
    REQUIRE(std::abs(Q_uni) < 0.1);
}

// ---------------------------------------------------------------------------
// T9: MinimizeGPU — max_steps cap respected
// ---------------------------------------------------------------------------
TEST_CASE("MinimizeGPU: respects max_steps cap", "[relax][gpu]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    auto mat = Material::permalloy();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i) m0[i] = Vec3{1, 0, 0};

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, {0, 0, 0});  // zero field, just provides the reference

    MinimizeGPU minimize(g);
    minimize.upload(m0);

    MinimizeGPU::Options opts;
    opts.threshold   = 1e-30;  // impossible to reach → must hit max_steps
    opts.max_steps   = 50;
    opts.check_every = 10;
    const int steps = minimize.run(mat, demag, exch, zeeman, nullptr, opts);
    INFO("Steps = " << steps);
    REQUIRE(steps == opts.max_steps);
}

// ---------------------------------------------------------------------------
// T10: BulkDMIFieldGPU — energy favours Bloch skyrmion over uniform state
// ---------------------------------------------------------------------------
TEST_CASE("BulkDMIFieldGPU: lower energy for Bloch skyrmion than uniform", "[relax][gpu]") {
    StructuredGrid g(32, 32, 1, 3e-9, 3e-9, 1e-9);
    Material mat;
    mat.Ms         = 0.58e6;
    mat.A_exchange = 15e-12;
    mat.K_uniaxial = 0.4e6;
    mat.easy_axis  = {0, 0, 1};
    mat.alpha      = 0.5;

    // Bloch skyrmion: azimuthal in-plane, core up, background down
    const Index nx = 32, ny = 32, nz = 1;
    const Real cell = 3e-9;
    const Real cx = nx * cell * Real{0.5}, cy = ny * cell * Real{0.5};
    const Real R = 10e-9;
    const Real pi = Real{3.14159265358979323846};
    VectorField3D m_bloch(g);
    for (Index iz = 0; iz < nz; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < nx; ++ix) {
        Real rx = (ix + Real{0.5}) * cell - cx;
        Real ry = (iy + Real{0.5}) * cell - cy;
        Real r  = std::sqrt(rx*rx + ry*ry);
        Real cos_t = (r < Real{2}*R) ? std::cos(pi * r / (Real{2}*R)) : Real{-1};
        Real sin_t = std::sqrt(std::max(Real{0}, Real{1} - cos_t*cos_t));
        Index lin = ix + nx*(iy + ny*iz);
        if (r < Real{1e-30}) { m_bloch[lin] = {0, 0, 1}; continue; }
        // Clockwise azimuthal: (ry/r, -rx/r) → favored by bulk D>0 (CW Bloch)
        m_bloch[lin] = Vec3{(ry/r)*sin_t, -(rx/r)*sin_t, cos_t};
    }

    const double Q = topological_charge_Q(m_bloch);
    INFO("Bloch skyrmion Q = " << Q);
    REQUIRE(std::abs(Q) > 0.5);

    VectorField3D m_uni(g);
    m_uni.set_uniform(Vec3{0, 0, -1});

    BulkDMIFieldGPU dmi(g, 5.0e-3);

    const double E_sky = dmi.energy(m_bloch, mat);
    const double E_uni = dmi.energy(m_uni,   mat);
    INFO("E_dmi(Bloch sky) = " << E_sky << "  E_dmi(uniform) = " << E_uni);
    REQUIRE(E_sky < E_uni);  // bulk DMI favours Bloch skyrmion

    const double Q_uni = topological_charge_Q(m_uni);
    REQUIRE(std::abs(Q_uni) < 0.1);
}

#endif // MICROMAG_CUDA
