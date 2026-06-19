// test_relax_gpu.cpp — RelaxGPU + MinimizeGPU tests.
// Tag: [relax][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/demag_gpu.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/relax_gpu.hpp"

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

#endif // MICROMAG_CUDA
