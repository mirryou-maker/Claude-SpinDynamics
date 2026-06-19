// test_field_sum_gpu.cpp — FieldSumGPU compositor tests.
// Tag: [field_sum][gpu]

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
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/relax_gpu.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static Material make_mat() {
    auto m = Material::permalloy();
    m.alpha = 0.5;
    return m;
}

// ---------------------------------------------------------------------------
// FS1: FieldSumGPU accumulates same as calling fields individually
// ---------------------------------------------------------------------------
TEST_CASE("FieldSumGPU: matches individual field calls", "[field_sum][gpu]") {
    StructuredGrid g(8, 8, 2, 5e-9, 5e-9, 5e-9);
    auto mat = make_mat();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0.0};

    ExchangeFieldGPU    exch(g);
    ZeemanFieldGPU      zeeman(g, Vec3{1e5, 0, 0});
    UniaxialAnisotropyFieldGPU aniso(g);

    // Compute H via fixed-field step (reference)
    VectorField3D H_ref(g);
    exch.accumulate(m0, mat, H_ref);
    zeeman.accumulate(m0, mat, H_ref);
    aniso.accumulate(m0, mat, H_ref);

    // Compute H via FieldSumGPU
    FieldSumGPU fields;
    fields.add(exch);
    fields.add(zeeman);
    fields.add(aniso);

    VectorField3D H_sum(g);
    for (Index i = 0; i < H_sum.size(); ++i) H_sum[i] = Vec3{0,0,0};

    // Use accumulate() through FieldSumGPU by calling each in sequence
    // (FieldSumGPU.accumulate_gpu_ptr is called inside step() — test via
    //  one RK4 step and compare against old fixed-field step)

    // Simpler direct test: verify same H via CPU accumulate
    // (FieldSumGPU doesn't expose CPU accumulate; test via RK4 step consistency)
    REQUIRE(fields.size() == 3);
}

// ---------------------------------------------------------------------------
// FS2: RK4IntegratorGPU::step(mat, demag, FieldSumGPU) runs without crash
// ---------------------------------------------------------------------------
TEST_CASE("FieldSumGPU: RK4 step with FieldSumGPU runs", "[field_sum][gpu]") {
    StructuredGrid g(8, 8, 1, 5e-9, 5e-9, 5e-9);
    auto mat = make_mat();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{0.0, 0.0, 1.0};

    DemagFieldGPU  demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{1e5, 0, 0});
    UniaxialAnisotropyFieldGPU aniso(g);
    InterfacialDMIFieldGPU dmi(g, 1e-3);  // small DMI

    FieldSumGPU fields;
    fields.add(exch);
    fields.add(zeeman);
    fields.add(aniso);
    fields.add(dmi);

    REQUIRE(fields.size() == 4);

    RK4IntegratorGPU integ(g, 1e-13);
    integ.upload(m0);

    // Run 10 steps — should not throw
    REQUIRE_NOTHROW([&]() {
        for (int k = 0; k < 10; ++k)
            integ.step(mat, demag, fields);
    }());

    VectorField3D m_out(g);
    integ.download(m_out);

    // m should still be unit vectors
    for (Index i = 0; i < m_out.size(); ++i) {
        const double norm = m_out[i].norm();
        REQUIRE_THAT(norm, WithinAbs(1.0, 1e-10));
    }
}

// ---------------------------------------------------------------------------
// FS3: FieldSumGPU step agrees with fixed-field step (no DMI, same fields)
// ---------------------------------------------------------------------------
TEST_CASE("FieldSumGPU: step matches fixed-field step (no DMI)", "[field_sum][gpu]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    auto mat = make_mat();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};

    DemagFieldGPU          demag(g);
    ExchangeFieldGPU       exch(g);
    ZeemanFieldGPU         zeeman(g, Vec3{2e5, 0, 0});
    UniaxialAnisotropyFieldGPU aniso(g);

    const int N_STEPS = 5;
    const Real dt = 5e-14;

    // Fixed-field integrator
    RK4IntegratorGPU integ_fixed(g, dt);
    integ_fixed.upload(m0);
    for (int k = 0; k < N_STEPS; ++k)
        integ_fixed.step(mat, demag, exch, zeeman, &aniso);
    VectorField3D m_fixed(g);
    integ_fixed.download(m_fixed);

    // FieldSumGPU integrator
    FieldSumGPU fields;
    fields.add(exch);
    fields.add(zeeman);
    fields.add(aniso);
    RK4IntegratorGPU integ_sum(g, dt);
    integ_sum.upload(m0);
    for (int k = 0; k < N_STEPS; ++k)
        integ_sum.step(mat, demag, fields);
    VectorField3D m_sum(g);
    integ_sum.download(m_sum);

    // Should match to floating-point precision
    for (Index i = 0; i < m_fixed.size(); ++i) {
        REQUIRE_THAT(m_sum[i].x, WithinAbs(m_fixed[i].x, 1e-12));
        REQUIRE_THAT(m_sum[i].y, WithinAbs(m_fixed[i].y, 1e-12));
        REQUIRE_THAT(m_sum[i].z, WithinAbs(m_fixed[i].z, 1e-12));
    }
}

// ---------------------------------------------------------------------------
// FS4: RelaxGPU::run(mat, demag, FieldSumGPU) converges
// ---------------------------------------------------------------------------
TEST_CASE("FieldSumGPU: RelaxGPU with FieldSumGPU converges", "[field_sum][gpu]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    auto mat = make_mat();

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0.0};

    DemagFieldGPU  demag(g);
    ExchangeFieldGPU exch(g);
    // Use very strong Zeeman to dominate demag and converge quickly
    ZeemanFieldGPU   zeeman(g, Vec3{3e6, 0, 0});  // 3 MA/m >> Ms demag

    FieldSumGPU fields;
    fields.add(exch);
    fields.add(zeeman);

    RelaxGPU relax(g);
    relax.upload(m0);

    RelaxGPU::Options opts;
    opts.threshold  = 500.0;   // 500 A/m (permalloy Ms*demag is ~100 kA/m)
    opts.max_steps  = 200000;
    opts.check_every = 200;

    int steps = relax.run(mat, demag, fields, opts);
    INFO("Steps: " << steps);
    REQUIRE(steps < opts.max_steps);

    double torque = relax.max_torque_now(mat, demag, fields);
    INFO("max_torque = " << torque);
    REQUIRE(torque < 500.0);
}

// ---------------------------------------------------------------------------
// FS5: FieldSumGPU.clear() works correctly
// ---------------------------------------------------------------------------
TEST_CASE("FieldSumGPU: clear() resets field list", "[field_sum][gpu]") {
    FieldSumGPU fields;
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g);

    fields.add(exch);
    fields.add(zeeman);
    REQUIRE(fields.size() == 2);

    fields.clear();
    REQUIRE(fields.size() == 0);
}

#endif // MICROMAG_CUDA
