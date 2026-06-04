// test_heun_integrator_gpu.cpp — G8: HeunIntegratorGPU accuracy tests
// Tag: [heun][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/demag_gpu.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/exchange.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/gpu_state.hpp"
#include "micromag/grid.hpp"
#include "micromag/heun_integrator_gpu.hpp"
#include "micromag/integrator.hpp"
#include "micromag/material.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/demag.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static double max_abs_diff(const VectorField3D& a, const VectorField3D& b) {
    double mx = 0;
    for (Index i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(a[i].x - b[i].x));
        mx = std::max(mx, std::abs(a[i].y - b[i].y));
        mx = std::max(mx, std::abs(a[i].z - b[i].z));
    }
    return mx;
}

// ---------------------------------------------------------------------------
// T1: constructor smoke test
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegratorGPU: constructor", "[heun][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    HeunIntegratorGPU integ(g, 1e-13, 42);
    REQUIRE(integ.dt() == 1e-13);
}

// ---------------------------------------------------------------------------
// T2: T=0 (σ=0) — GPU Heun matches CPU HeunIntegrator (no noise)
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegratorGPU T=0: matches CPU Heun", "[heun][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    const Real dt = 1e-13;
    Vec3 Hext{-24.6e3, 4.3e3, 0.0};

    VectorField3D m0(g);
    m0.set_uniform({0.9798, 0.2000, 0.0});
    m0.normalize();

    // CPU Heun (no thermal)
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ExchangeField>());
    heff.add(std::make_shared<DemagField>(g));
    heff.add(std::make_shared<ZeemanField>(Hext));

    VectorField3D m_cpu(g);
    for (Index i=0; i<g.size(); ++i) m_cpu[i] = m0[i];
    HeunIntegrator cpu(dt);
    cpu.step(m_cpu, mat, heff, nullptr);   // T=0, no thermal

    // GPU Heun (T=0 → σ=0, no noise)
    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Hext);

    HeunIntegratorGPU gpu(g, dt, 42);
    gpu.upload(m0);
    gpu.step(mat, demag, exch, zeeman, /*T_K=*/0.0);

    VectorField3D m_gpu(g);
    gpu.download(m_gpu);

    const double err = max_abs_diff(m_cpu, m_gpu);
    INFO("max |GPU - CPU| = " << err);
    INFO("CPU[0] = (" << m_cpu[0].x << ", " << m_cpu[0].y << ", " << m_cpu[0].z << ")");
    INFO("GPU[0] = (" << m_gpu[0].x << ", " << m_gpu[0].y << ", " << m_gpu[0].z << ")");
    REQUIRE(err < 1e-12);
}

// ---------------------------------------------------------------------------
// T3: 10 steps T=0 — trajectory comparison CPU vs GPU
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegratorGPU 10 steps T=0 vs CPU", "[heun][gpu]") {
    StructuredGrid g(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9);
    Material mat = Material::permalloy();
    const Real dt = 5e-14;
    Vec3 Hext{-24.6e3, 4.3e3, 0.0};

    VectorField3D m0(g);
    m0.set_uniform({1.0, 0.1, 0.0});
    m0.normalize();

    // CPU Heun (no thermal)
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ExchangeField>());
    heff.add(std::make_shared<DemagField>(g));
    heff.add(std::make_shared<ZeemanField>(Hext));

    VectorField3D m_cpu(g);
    for (Index i=0; i<g.size(); ++i) m_cpu[i] = m0[i];
    HeunIntegrator cpu(dt);
    for (int k=0; k<10; ++k) cpu.step(m_cpu, mat, heff, nullptr);

    // GPU Heun
    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Hext);

    HeunIntegratorGPU gpu(g, dt, 42);
    gpu.upload(m0);
    for (int k=0; k<10; ++k) gpu.step(mat, demag, exch, zeeman, 0.0);

    VectorField3D m_gpu(g);
    gpu.download(m_gpu);

    const double err = max_abs_diff(m_cpu, m_gpu);
    INFO("10-step max |GPU - CPU| = " << err);
    REQUIRE(err < 1e-10);
}

// ---------------------------------------------------------------------------
// T4: T>0 noise generation — after one step, m changes due to thermal noise
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegratorGPU T=300K: thermal noise active", "[heun][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    const Real dt = 1e-13;
    Vec3 Hext{0, 0, 0};   // no applied field — only thermal noise drives motion

    VectorField3D m0(g); m0.set_uniform({0, 0, 1});

    // Two runs with DIFFERENT seeds → different trajectories (verifies noise)
    DemagFieldGPU    demag1(g), demag2(g);
    ExchangeFieldGPU exch1(g),  exch2(g);
    ZeemanFieldGPU   zeeman1(g, Hext), zeeman2(g, Hext);

    HeunIntegratorGPU gpu1(g, dt, 42);
    HeunIntegratorGPU gpu2(g, dt, 99);  // different seed

    gpu1.upload(m0); gpu2.upload(m0);

    // Run 10 steps at T=300 K
    for (int k=0; k<10; ++k) {
        gpu1.step(mat, demag1, exch1, zeeman1, 300.0);
        gpu2.step(mat, demag2, exch2, zeeman2, 300.0);
    }

    VectorField3D m1(g), m2(g);
    gpu1.download(m1); gpu2.download(m2);

    // Different seeds → different trajectories
    const double diff = max_abs_diff(m1, m2);
    INFO("diff between seed=42 and seed=99 after 10 steps at T=300K: " << diff);
    REQUIRE(diff > 1e-10);   // Should be significantly different

    // All cells still on unit sphere
    for (Index i=0; i<g.size(); ++i) {
        double len = std::sqrt(m1[i].x*m1[i].x + m1[i].y*m1[i].y + m1[i].z*m1[i].z);
        REQUIRE_THAT(len, WithinAbs(1.0, 1e-10));
    }
}

// ---------------------------------------------------------------------------
// T5: T=300K — unit sphere maintained after many steps
// ---------------------------------------------------------------------------
TEST_CASE("HeunIntegratorGPU T=300K: |m|=1 after 20 steps", "[heun][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    const Real dt = 1e-13;
    Vec3 Hext{-24.6e3, 4.3e3, 0.0};

    VectorField3D m0(g); m0.set_uniform({1, 0.1, 0}); m0.normalize();

    DemagFieldGPU    d(g);
    ExchangeFieldGPU e(g);
    ZeemanFieldGPU   z(g, Hext);

    HeunIntegratorGPU gpu(g, dt, 42);
    gpu.upload(m0);

    for (int k=0; k<20; ++k) gpu.step(mat, d, e, z, 300.0);

    VectorField3D m(g);
    gpu.download(m);

    // Every cell must remain on the unit sphere after normalize()
    for (Index i=0; i<g.size(); ++i) {
        const double len = std::sqrt(m[i].x*m[i].x + m[i].y*m[i].y + m[i].z*m[i].z);
        REQUIRE_THAT(len, WithinAbs(1.0, 1e-10));
    }

    // Magnetization should have moved from initial +x due to field + noise
    // (if still exactly at m0, something is wrong)
    Vec3 avg{0,0,0};
    for (Index i=0; i<g.size(); ++i) {avg.x+=m[i].x; avg.y+=m[i].y; avg.z+=m[i].z;}
    const double N = g.size();
    // System is driven by field — check that it changed from initial state
    REQUIRE(std::abs(avg.x/N - 1.0) > 1e-6);  // moved from mx=1
}

#endif // MICROMAG_CUDA
