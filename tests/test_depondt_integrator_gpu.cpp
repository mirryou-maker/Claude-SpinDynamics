// test_depondt_integrator_gpu.cpp — Task 1-A: DepondtMertensGPU
// Tag: [depondt][gpu]
//
// Covers the increment 1-A guarantees: exact |m|=1 conservation (roadmap
// invariant #1), correct T=0 damping toward the easy axis, the single-point
// thermal sigma scaling (1-D), and the loud-failure guard on the not-yet-wired
// finite-T path (1-B).

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/demag_gpu.hpp"
#include "micromag/depondt_integrator_gpu.hpp"
#include "micromag/effective_field_gpu_iface.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"

using namespace micromag;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

static double max_norm_dev(const VectorField3D& m) {
    double mx = 0;
    for (Index i = 0; i < m.size(); ++i) {
        const double n = std::sqrt(m[i].x*m[i].x + m[i].y*m[i].y + m[i].z*m[i].z);
        mx = std::max(mx, std::abs(n - 1.0));
    }
    return mx;
}

TEST_CASE("DepondtMertensGPU: constructor", "[depondt][gpu]") {
    StructuredGrid g(4, 4, 1, 2e-9, 2e-9, 2e-9);
    DepondtMertensGPU integ(g, 1e-13);
    REQUIRE(integ.dt() == 1e-13);
    REQUIRE_THROWS_AS(DepondtMertensGPU(g, -1e-13), std::invalid_argument);
}

// Invariant #1: rotation conserves |m| to machine precision, and the T=0
// dynamics damp a tilted macrospin toward the easy axis (+z).
TEST_CASE("DepondtMertensGPU T=0: |m| exact + damps to easy axis",
          "[depondt][gpu]") {
    StructuredGrid g(4, 4, 1, 2e-9, 2e-9, 2e-9);
    Material mat; mat.Ms = 580e3; mat.A_exchange = 13e-12;
    mat.K_uniaxial = 5e5; mat.easy_axis = Vec3{0, 0, 1}; mat.alpha = 0.1;

    VectorField3D m0(g);
    m0.set_uniform({std::sin(0.6), 0.0, std::cos(0.6)});   // tilted 0.6 rad from +z
    m0.normalize();

    ZeroDemagGPU     demag;   // isolate the integrator from demag shape physics
    ExchangeFieldGPU exch(g);
    UniaxialAnisotropyFieldGPU aniso(g);
    FieldSumGPU fields; fields.add(exch); fields.add(aniso);

    DepondtMertensGPU integ(g, 1e-13);
    integ.upload(m0);
    for (int k = 0; k < 20000; ++k) integ.step(mat, demag, fields, /*T_K=*/0.0);

    VectorField3D m_out(g);
    integ.download(m_out);

    const double dev = max_norm_dev(m_out);
    INFO("max ||m|-1| = " << dev);
    REQUIRE(dev < 1e-10);                 // invariant #1 (no renormalize)

    double mz = 0; for (Index i = 0; i < m_out.size(); ++i) mz += m_out[i].z;
    mz /= m_out.size();
    INFO("<mz> after damping = " << mz);
    REQUIRE(mz > 0.99);                    // relaxed to easy axis
}

// Roadmap 1-D: sigma is the single 1/sqrt(dt) point. Doubling dt scales it by
// 1/sqrt(2); this is the anchor of the "no silent rescale" regression.
TEST_CASE("DepondtMertensGPU: therm_sigma ~ 1/sqrt(dt)", "[depondt][gpu]") {
    Material mat; mat.Ms = 580e3; mat.alpha = 0.1;
    const double s1 = DepondtMertensGPU::therm_sigma(mat, 1e-13, 2e-9,2e-9,2e-9, 300.0);
    const double s2 = DepondtMertensGPU::therm_sigma(mat, 2e-13, 2e-9,2e-9,2e-9, 300.0);
    REQUIRE(s1 > 0.0);
    REQUIRE_THAT(s1 / s2, WithinRel(std::sqrt(2.0), 1e-9));
    REQUIRE(DepondtMertensGPU::therm_sigma(mat, 1e-13, 2e-9,2e-9,2e-9, 0.0) == 0.0);
}

// 1-B: finite-T thermal path (device Philox), free-spin infrastructure check —
// |m| stays exact under the thermal field and the noise is active. The ABSOLUTE
// field-coupled scale is checked separately by the Langevin test below.
TEST_CASE("DepondtMertensGPU T>0: |m| exact + noise active (free spin)",
          "[depondt][gpu]") {
    StructuredGrid g(4, 4, 1, 2e-9, 2e-9, 2e-9);
    Material mat; mat.Ms = 580e3; mat.A_exchange = 0.0; mat.alpha = 0.5;
    VectorField3D m0(g); m0.set_uniform({0, 0, 1});

    ZeroDemagGPU     demag;
    ZeemanFieldGPU   zeeman(g, Vec3{0, 0, 0});   // no field — pure diffusion
    FieldSumGPU fields; fields.add(zeeman);

    DepondtMertensGPU integ(g, 5e-13, /*seed=*/7);
    integ.upload(m0);
    for (int k = 0; k < 4000; ++k) integ.step(mat, demag, fields, /*T_K=*/1e7);

    VectorField3D m_out(g);
    integ.download(m_out);
    REQUIRE(max_norm_dev(m_out) < 1e-10);          // |m| exact under noise

    double moved = 0;                               // spin left +z (noise active)
    for (Index i = 0; i < m_out.size(); ++i)
        moved += m_out[i].x*m_out[i].x + m_out[i].y*m_out[i].y;
    moved /= m_out.size();
    INFO("<mx^2+my^2> = " << moved);
    REQUIRE(moved > 1e-6);
}

// 1-C: adaptive step control keeps |m| exact and actually changes dt.
TEST_CASE("DepondtMertensGPU: adaptive keeps |m| exact and adapts dt",
          "[depondt][gpu]") {
    StructuredGrid g(4, 4, 1, 2e-9, 2e-9, 2e-9);
    Material mat; mat.Ms = 580e3; mat.A_exchange = 13e-12;
    mat.K_uniaxial = 5e5; mat.easy_axis = Vec3{0, 0, 1}; mat.alpha = 0.1;

    VectorField3D m0(g);
    m0.set_uniform({std::sin(0.6), 0.0, std::cos(0.6)}); m0.normalize();

    ZeroDemagGPU demag; ExchangeFieldGPU exch(g);
    UniaxialAnisotropyFieldGPU aniso(g);
    FieldSumGPU fields; fields.add(exch); fields.add(aniso);

    DepondtMertensGPU integ(g, 1e-14);
    integ.options().adaptive = true;
    integ.options().rtol = 1e-3; integ.options().atol = 1e-4;
    integ.options().dt_max = 1e-12; integ.options().dt_min = 1e-16;
    const Real dt0 = integ.dt();
    integ.upload(m0);
    for (int k = 0; k < 3000; ++k) integ.step(mat, demag, fields, /*T_K=*/0.0);

    VectorField3D m_out(g); integ.download(m_out);
    REQUIRE(max_norm_dev(m_out) < 1e-10);          // |m| exact under adaptive
    INFO("dt0 = " << dt0 << "  dt_final = " << integ.dt());
    REQUIRE(integ.dt() != dt0);                     // controller moved dt
    double mz = 0; for (Index i=0;i<m_out.size();++i) mz += m_out[i].z;
    REQUIRE(mz / m_out.size() > 0.99);              // still relaxes to easy axis
}

// 1-B FDT: field-coupled Langevin equilibrium validates the ABSOLUTE thermal
// σ (not just its dt-scaling). A macrospin ensemble in a field H‖ẑ must relax to
// ⟨m_z⟩ = L(ξ) = coth ξ − 1/ξ with ξ = μ₀ Ms V H/k_B T. This is the test the
// existing (field-free) equipartition test could not do; it pins σ's absolute
// scale for DepondtMertensGPU.
TEST_CASE("DepondtMertensGPU FDT: Langevin <mz> in a field", "[depondt][gpu]") {
    const double kB = 1.380649e-23, mu0 = 4e-7 * 3.14159265358979323846;
    const double Ms = 1e6, dx = 2e-9, V = dx*dx*dx, T = 300.0;
    const double xi = 3.0;
    const double Hz = xi * kB * T / (mu0 * Ms * V);
    const double L  = 1.0/std::tanh(xi) - 1.0/xi;   // ≈ 0.6716

    StructuredGrid g(16, 16, 1, dx, dx, dx);         // 256 independent spins
    Material mat; mat.Ms = Ms; mat.A_exchange = 0.0; mat.alpha = 0.5;
    VectorField3D m0(g); m0.set_uniform({0, 0, 1});

    ZeroDemagGPU   demag;
    ZeemanFieldGPU zeeman(g, Vec3{0, 0, static_cast<Real>(Hz)});
    FieldSumGPU fields; fields.add(zeeman);

    DepondtMertensGPU integ(g, 1e-14, /*seed=*/5);
    integ.upload(m0);
    for (int k = 0; k < 25000; ++k) integ.step(mat, demag, fields, T);  // equilibrate

    double acc = 0; int nsamp = 0;
    VectorField3D m(g);
    for (int k = 0; k < 30000; ++k) {
        integ.step(mat, demag, fields, T);
        if (k % 50 == 0) {
            integ.download(m);
            double mz = 0; for (Index i = 0; i < m.size(); ++i) mz += m[i].z;
            acc += mz / m.size(); ++nsamp;
        }
    }
    const double mz_mean = acc / nsamp;
    INFO("<mz> = " << mz_mean << "  Langevin L(3) = " << L);
    REQUIRE(max_norm_dev(m) < 1e-10);
    REQUIRE_THAT(mz_mean, WithinAbs(L, 0.06));
}

#endif // MICROMAG_CUDA
