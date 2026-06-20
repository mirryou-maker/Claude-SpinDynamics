// test_rk4_integrator_gpu.cpp — G6: RK4IntegratorGPU accuracy tests
// Runs a full LLG step (or N steps) with RK4IntegratorGPU and compares
// against the CPU RK4Integrator.
// Tag: [integ][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/demag.hpp"
#include "micromag/demag_gpu.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/exchange.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/integrator.hpp"
#include "micromag/material.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/zeeman.hpp"
#include "gpu_test_tol.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static double max_abs_diff(const VectorField3D& a, const VectorField3D& b) {
    double mx = 0.0;
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
TEST_CASE("RK4IntegratorGPU: constructor", "[integ][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    RK4IntegratorGPU integ(g, 1e-13);
    REQUIRE(integ.dt() == 1e-13);
}

// ---------------------------------------------------------------------------
// T2: one step (Zeeman only, macrospin) — GPU vs CPU
//
// Zeeman-only dynamics have an analytic solution (Larmor precession + damping).
// Comparing GPU and CPU verifies the G4+G5+G6 assembly.
// ---------------------------------------------------------------------------
TEST_CASE("RK4IntegratorGPU one step: Zeeman only vs CPU", "[integ][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    const Real dt = 1e-13;

    Vec3 Hext{-24.6e3, 4.3e3, 0.0};

    VectorField3D m0(g);
    m0.set_uniform({0.9798, 0.2000, 0.0});
    m0.normalize();

    // CPU: Exchange + Demag + Zeeman
    EffectiveFieldSum heff_all;
    heff_all.add(std::make_shared<ExchangeField>());
    heff_all.add(std::make_shared<DemagField>(g));  // expensive but correct
    heff_all.add(std::make_shared<ZeemanField>(Hext));

    VectorField3D m_cpu2(g);
    for (Index i=0; i<g.size(); ++i) m_cpu2[i] = m0[i];
    RK4Integrator cpu2(dt);
    cpu2.step(m_cpu2, mat, heff_all);

    // GPU: same fields
    DemagFieldGPU    gpu_demag(g);
    ExchangeFieldGPU exch_gpu(g);
    ZeemanFieldGPU   zeeman_gpu(g, Hext);

    RK4IntegratorGPU gpu_integ(g, dt);
    gpu_integ.upload(m0);
    gpu_integ.step(mat, gpu_demag, exch_gpu, zeeman_gpu);
    VectorField3D m_gpu(g);
    gpu_integ.download(m_gpu);

    const double err = max_abs_diff(m_cpu2, m_gpu);
    INFO("max |GPU - CPU| = " << err);
    INFO("CPU[0] = (" << m_cpu2[0].x << ", " << m_cpu2[0].y << ", " << m_cpu2[0].z << ")");
    INFO("GPU[0] = (" << m_gpu[0].x  << ", " << m_gpu[0].y  << ", " << m_gpu[0].z  << ")");
    REQUIRE(err < micromag::gtol(1e-12));
}

// ---------------------------------------------------------------------------
// T3: 10 steps (SP#4-like, Exchange + Demag + Zeeman) — trajectory comparison
// ---------------------------------------------------------------------------
TEST_CASE("RK4IntegratorGPU 10 steps vs CPU", "[integ][gpu]") {
    // SP#4 setup: 200×50×1, 2.5nm cells
    StructuredGrid g(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9);
    Material mat = Material::permalloy();
    const Real dt = 5e-14;   // 50 fs steps
    Vec3 Hext{-24.6e3, 4.3e3, 0.0};

    VectorField3D m0(g);
    m0.set_uniform({1.0, 0.1, 0.0});
    m0.normalize();

    // CPU reference
    EffectiveFieldSum heff_cpu;
    heff_cpu.add(std::make_shared<ExchangeField>());
    heff_cpu.add(std::make_shared<DemagField>(g));
    heff_cpu.add(std::make_shared<ZeemanField>(Hext));

    VectorField3D m_cpu(g);
    for (Index i=0; i<g.size(); ++i) m_cpu[i] = m0[i];
    RK4Integrator cpu(dt);
    for (int k=0; k<10; ++k) cpu.step(m_cpu, mat, heff_cpu);

    // GPU
    DemagFieldGPU    demag_gpu(g);
    ExchangeFieldGPU exch_gpu(g);
    ZeemanFieldGPU   zeeman_gpu(g, Hext);

    RK4IntegratorGPU gpu(g, dt);
    gpu.upload(m0);
    for (int k=0; k<10; ++k)
        gpu.step(mat, demag_gpu, exch_gpu, zeeman_gpu);

    VectorField3D m_gpu(g);
    gpu.download(m_gpu);

    const double err = max_abs_diff(m_cpu, m_gpu);
    INFO("10-step max |GPU - CPU| = " << err);
    INFO("CPU <mx> = " << [&](){ double s=0; for(Index i=0;i<g.size();++i) s+=m_cpu[i].x; return s/g.size(); }());
    INFO("GPU <mx> = " << [&](){ double s=0; for(Index i=0;i<g.size();++i) s+=m_gpu[i].x; return s/g.size(); }());
    REQUIRE(err < micromag::gtol(1e-10));
}

#endif // MICROMAG_CUDA
