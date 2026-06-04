// test_rk4_gpu.cpp — G4 (LLG torque) + G5 (RK4 stage kernels) tests
// Tags: [llg][gpu], [rk4][gpu]
//
// Integration test: one full RK4 step (macrospin, Zeeman only) vs CPU integrator.

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/effective_field.hpp"
#include "micromag/field.hpp"
#include "micromag/gpu_state.hpp"
#include "micromag/grid.hpp"
#include "micromag/integrator.hpp"    // llg_torque, RK4Integrator
#include "micromag/material.hpp"
#include "micromag/rk4_gpu.hpp"
#include "micromag/types.hpp"
#include "micromag/zeeman.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================================
// G4 — LLG torque kernel tests
// ============================================================================

// ---------------------------------------------------------------------------
// T1: m ∥ H → dm/dt = 0  (no torque in equilibrium)
// ---------------------------------------------------------------------------
TEST_CASE("LLG torque GPU: m ∥ H → zero torque", "[llg][gpu]") {
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    GPUMagState state(g);
    VectorField3D m(g); m.set_uniform({0, 0, 1});   // m = +z
    state.upload(m);

    VectorField3D H(g); H[0] = {0, 0, 1e5};         // H = +z
    state.upload_H(H);

    launch_llg_torque(state.d_ki(), state.d_m(), state.d_H(),
                       mat.alpha, (int)state.N(), state.stream());
    state.sync();

    VectorField3D ki(g);
    state.download_ki(ki);

    REQUIRE_THAT(ki[0].x, WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(ki[0].y, WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(ki[0].z, WithinAbs(0.0, 1e-3));
}

// ---------------------------------------------------------------------------
// T2: single cell — GPU torque matches CPU llg_torque() exactly
// ---------------------------------------------------------------------------
TEST_CASE("LLG torque GPU matches CPU: single cell", "[llg][gpu]") {
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    // Arbitrary non-equilibrium state
    const Vec3 m_vec{0.6, 0.8, 0.0};
    const Vec3 H_vec{-24.6e3, 4.3e3, 0.0};

    // CPU reference
    const Vec3 ki_cpu = llg_torque(m_vec, H_vec, mat.alpha);

    // GPU
    GPUMagState state(g);
    VectorField3D m(g); m[0] = m_vec;
    state.upload(m);
    VectorField3D H(g); H[0] = H_vec;
    state.upload_H(H);

    launch_llg_torque(state.d_ki(), state.d_m(), state.d_H(),
                       mat.alpha, (int)state.N(), state.stream());
    state.sync();

    VectorField3D ki(g);
    state.download_ki(ki);

    INFO("CPU ki = (" << ki_cpu.x << ", " << ki_cpu.y << ", " << ki_cpu.z << ")");
    INFO("GPU ki = (" << ki[0].x  << ", " << ki[0].y  << ", " << ki[0].z  << ")");
    REQUIRE_THAT(ki[0].x, WithinRel(ki_cpu.x, 1e-8));
    REQUIRE_THAT(ki[0].y, WithinRel(ki_cpu.y, 1e-8));
    REQUIRE_THAT(ki[0].z, WithinAbs(ki_cpu.z, std::abs(ki_cpu.y)*1e-8 + 1.0));
}

// ---------------------------------------------------------------------------
// T3: multi-cell — GPU torque field matches CPU for all cells
// ---------------------------------------------------------------------------
TEST_CASE("LLG torque GPU matches CPU: 8×6×4 grid", "[llg][gpu]") {
    StructuredGrid g(8, 6, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g), H(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix*0.3 + iy*0.2;
        m.at(ix,iy,iz) = {std::cos(phi), std::sin(phi), 0.0};
        // Spatially varying H (simulates non-uniform demag/exchange)
        H.at(ix,iy,iz) = {-24.6e3 + ix*100.0, 4.3e3 - iy*50.0, iz*200.0};
    }

    // CPU reference: compute ki[i] = llg_torque(m[i], H[i], alpha) for all i
    VectorField3D ki_cpu(g);
    for (Index i=0; i<g.size(); ++i)
        ki_cpu[i] = llg_torque(m[i], H[i], mat.alpha);

    // GPU
    GPUMagState state(g);
    state.upload(m);
    state.upload_H(H);

    launch_llg_torque(state.d_ki(), state.d_m(), state.d_H(),
                       mat.alpha, (int)state.N(), state.stream());
    state.sync();

    VectorField3D ki_gpu(g);
    state.download_ki(ki_gpu);

    // Relative error: |GPU - CPU| / max(|CPU|, 1.0 s⁻¹)
    // Floor of 1.0 prevents division by zero for near-zero components.
    double max_rel = 0.0;
    for (Index i=0; i<g.size(); ++i)
    for (int c=0; c<3; ++c) {
        const double ref  = (&ki_cpu[i].x)[c];
        const double got  = (&ki_gpu[i].x)[c];
        max_rel = std::max(max_rel, std::abs(ref-got) / std::max(std::abs(ref), 1.0));
    }
    INFO("max relative err = " << max_rel);
    REQUIRE(max_rel < 1e-8);
}

// ============================================================================
// G5 — RK4 stage kernel tests
// ============================================================================

// ---------------------------------------------------------------------------
// T4: rk4_stage — m_out = m0 + scale * ki (arithmetic check)
// ---------------------------------------------------------------------------
TEST_CASE("RK4 stage kernel: m_out = m0 + scale*ki", "[rk4][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);
    const double dt = 1e-13;

    // m0 = +z, ki = (1, 0, -1) at every cell
    VectorField3D m0(g), ki_v(g);
    m0.set_uniform({0, 0, 1});
    for (Index i=0; i<g.size(); ++i) ki_v[i] = {1e11, 0, -1e11};

    state.upload(m0);    // d_m = m0
    state.save_m0();     // d_m0 = m0
    state.sync();

    state.upload_H(ki_v); // misuse d_H_ as ki placeholder for this test

    // m_stage = m0 + dt/2 * ki  →  use d_m as output, d_m0 as m0, d_H as ki
    launch_rk4_stage(state.d_m(), state.d_m0(), state.d_H(),
                      dt * 0.5, (int)state.N(), state.stream());
    state.sync();

    VectorField3D m_stage(g);
    state.download(m_stage);

    // Expected: m_stage[i] = (0 + dt/2*1e11, 0, 1 + dt/2*(-1e11))
    //                      = (0 + 5e-3,       0, 1 - 5e-3)
    const double expected_x = 0.0 + (dt*0.5) * 1e11;
    const double expected_z = 1.0 + (dt*0.5) * (-1e11);
    for (Index i=0; i<g.size(); ++i) {
        REQUIRE_THAT(m_stage[i].x, WithinRel(expected_x, 1e-10));
        REQUIRE_THAT(m_stage[i].y, WithinAbs(0.0, 1e-20));
        REQUIRE_THAT(m_stage[i].z, WithinRel(expected_z, 1e-10));
    }
}

// ---------------------------------------------------------------------------
// T5: rk4_accumulate — k_acc += weight * ki (arithmetic check)
// ---------------------------------------------------------------------------
TEST_CASE("RK4 accumulate kernel: k_acc += weight*ki", "[rk4][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    // k_acc starts at zero (fresh state, zero_k_acc)
    state.zero_k_acc();
    state.sync();

    // ki = (2, -1, 3) at every cell
    VectorField3D ki_v(g);
    for (Index i=0; i<g.size(); ++i) ki_v[i] = {2e11, -1e11, 3e11};
    state.upload_H(ki_v);  // misuse d_H_ as ki

    // accumulate with weight = 1/6
    const double w = 1.0/6.0;
    launch_rk4_accumulate(state.d_k_acc(), state.d_H(),
                           w, (int)state.N(), state.stream());
    state.sync();

    VectorField3D k_acc(g);
    state.download_k_acc(k_acc);

    // Expected: k_acc[i] = 1/6 * (2, -1, 3) * 1e11
    for (Index i=0; i<g.size(); ++i) {
        REQUIRE_THAT(k_acc[i].x, WithinRel(w * 2e11, 1e-12));
        REQUIRE_THAT(k_acc[i].y, WithinRel(w * (-1e11), 1e-12));
        REQUIRE_THAT(k_acc[i].z, WithinRel(w * 3e11, 1e-12));
    }

    // Accumulate again with weight = 2/6 → total = (1/6 + 2/6)*ki
    const double w2 = 2.0/6.0;
    launch_rk4_accumulate(state.d_k_acc(), state.d_H(),
                           w2, (int)state.N(), state.stream());
    state.sync();
    state.download_k_acc(k_acc);

    REQUIRE_THAT(k_acc[0].x, WithinRel((w+w2) * 2e11, 1e-12));
}

// ---------------------------------------------------------------------------
// T6: rk4_finalize — m_new = m0 + dt * k_acc (arithmetic check)
// ---------------------------------------------------------------------------
TEST_CASE("RK4 finalize kernel: m_new = m0 + dt*k_acc", "[rk4][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);
    const double dt = 1e-13;

    // m0 = (0.6, 0.8, 0), k_acc = (1, -1, 0) × 1e11 at every cell
    VectorField3D m0(g), ka(g);
    m0.set_uniform({0.6, 0.8, 0.0});
    for (Index i=0; i<g.size(); ++i) ka[i] = {1e11, -1e11, 0.0};

    state.upload(m0);   state.save_m0();   state.sync();   // d_m0 = m0
    state.zero_k_acc(); state.sync();

    // Put k_acc into d_k_acc by accumulate with weight=1
    state.upload_H(ka);
    launch_rk4_accumulate(state.d_k_acc(), state.d_H(),
                           1.0, (int)state.N(), state.stream());
    state.sync();

    launch_rk4_finalize(state.d_m(), state.d_m0(), state.d_k_acc(),
                         dt, (int)state.N(), state.stream());
    state.sync();

    VectorField3D m_new(g);
    state.download(m_new);

    const double ex = 0.6 + dt * 1e11;
    const double ey = 0.8 + dt * (-1e11);
    for (Index i=0; i<g.size(); ++i) {
        REQUIRE_THAT(m_new[i].x, WithinRel(ex, 1e-12));
        REQUIRE_THAT(m_new[i].y, WithinRel(ey, 1e-12));
        REQUIRE_THAT(m_new[i].z, WithinAbs(0.0, 1e-25));
    }
}

// ---------------------------------------------------------------------------
// T7: normalize — |m| = 1 after normalize, direction preserved
// ---------------------------------------------------------------------------
TEST_CASE("Normalize kernel: unit sphere clamp", "[rk4][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    // Non-normalised m (length ≠ 1)
    VectorField3D m_in(g);
    for (Index i=0; i<g.size(); ++i) m_in[i] = {3.0, 4.0, 0.0};  // |m| = 5
    state.upload(m_in);

    launch_normalize(state.d_m(), (int)state.N(), state.stream());
    state.sync();

    VectorField3D m_out(g);
    state.download(m_out);

    for (Index i=0; i<g.size(); ++i) {
        const double len = std::sqrt(m_out[i].x*m_out[i].x +
                                     m_out[i].y*m_out[i].y +
                                     m_out[i].z*m_out[i].z);
        REQUIRE_THAT(len, WithinAbs(1.0, 1e-14));
        // Direction preserved: (3,4,0)/5 = (0.6, 0.8, 0)
        REQUIRE_THAT(m_out[i].x, WithinAbs(0.6, 1e-14));
        REQUIRE_THAT(m_out[i].y, WithinAbs(0.8, 1e-14));
    }
}

// ============================================================================
// Integration test: one full RK4 step (Zeeman only) — GPU vs CPU
// ============================================================================
// Runs a 4×4×4 macrospin under H_ext = SP#4 Field A conditions.
// GPU manually executes the 4 RK4 stages using G4+G5 kernels.
// CPU uses RK4Integrator.  Results must agree within double precision.
// ============================================================================

TEST_CASE("G4+G5 integration: one RK4 step vs CPU (Zeeman only)", "[rk4][gpu][llg]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    const double dt   = 1e-13;
    const Vec3   Hext{-24.6e3, 4.3e3, 0.0};

    // Initial state: uniform m ≈ (0.98, 0.20, 0)
    VectorField3D m0(g);
    m0.set_uniform({0.9798, 0.2000, 0.0});
    m0.normalize();

    // ------------------------------------------------------------------
    // CPU reference: one RK4 step with ZeemanField only
    // ------------------------------------------------------------------
    EffectiveFieldSum heff_cpu;
    heff_cpu.add(std::make_shared<ZeemanField>(Hext));

    VectorField3D m_cpu(g);
    for (Index i=0; i<g.size(); ++i) m_cpu[i] = m0[i];

    RK4Integrator cpu_integ(dt);
    cpu_integ.step(m_cpu, mat, heff_cpu);

    // ------------------------------------------------------------------
    // GPU: manual 4-stage RK4 using G4+G5 kernels
    //
    // Stage weights (Butcher tableau):
    //   k1: eval at m0,            accumulate 1/6,  stage m0 + dt/2*k1
    //   k2: eval at m0+dt/2*k1,    accumulate 2/6,  stage m0 + dt/2*k2
    //   k3: eval at m0+dt/2*k2,    accumulate 2/6,  stage m0 + dt*k3
    //   k4: eval at m0+dt*k3,      accumulate 1/6,  no stage update
    // finalize: m_new = m0 + dt * k_acc
    // normalize: |m| = 1
    // ------------------------------------------------------------------
    GPUMagState state(g);
    state.upload(m0);
    state.save_m0();
    state.zero_k_acc();
    state.sync();

    void* s = state.stream();

    // Helper: compute H_ext (uniform) into d_H — same as ZeemanField for all cells
    auto set_H = [&]() {
        // Fill d_H with H_ext (uniform Zeeman: H[i] = Hext for all i)
        VectorField3D H(g); for (Index i=0; i<g.size(); ++i) H[i] = Hext;
        state.upload_H(H);
    };

    const double alpha = mat.alpha;
    const int N = (int)state.N();

    // Stage 1
    set_H();
    launch_llg_torque(state.d_ki(), state.d_m(), state.d_H(), alpha, N, s);
    launch_rk4_accumulate(state.d_k_acc(), state.d_ki(), 1.0/6.0, N, s);
    launch_rk4_stage(state.d_m(), state.d_m0(), state.d_ki(), dt*0.5, N, s);
    state.sync();

    // Stage 2
    set_H();
    launch_llg_torque(state.d_ki(), state.d_m(), state.d_H(), alpha, N, s);
    launch_rk4_accumulate(state.d_k_acc(), state.d_ki(), 2.0/6.0, N, s);
    launch_rk4_stage(state.d_m(), state.d_m0(), state.d_ki(), dt*0.5, N, s);
    state.sync();

    // Stage 3
    set_H();
    launch_llg_torque(state.d_ki(), state.d_m(), state.d_H(), alpha, N, s);
    launch_rk4_accumulate(state.d_k_acc(), state.d_ki(), 2.0/6.0, N, s);
    launch_rk4_stage(state.d_m(), state.d_m0(), state.d_ki(), dt, N, s);
    state.sync();

    // Stage 4
    set_H();
    launch_llg_torque(state.d_ki(), state.d_m(), state.d_H(), alpha, N, s);
    launch_rk4_accumulate(state.d_k_acc(), state.d_ki(), 1.0/6.0, N, s);

    // Finalize + normalize
    launch_rk4_finalize(state.d_m(), state.d_m0(), state.d_k_acc(), dt, N, s);
    launch_normalize(state.d_m(), N, s);
    state.sync();

    VectorField3D m_gpu(g);
    state.download(m_gpu);

    // Compare GPU and CPU results
    double max_err = 0.0;
    for (Index i=0; i<g.size(); ++i) {
        max_err = std::max(max_err, std::abs(m_gpu[i].x - m_cpu[i].x));
        max_err = std::max(max_err, std::abs(m_gpu[i].y - m_cpu[i].y));
        max_err = std::max(max_err, std::abs(m_gpu[i].z - m_cpu[i].z));
    }
    INFO("max |GPU - CPU| = " << max_err);
    INFO("GPU[0] = (" << m_gpu[0].x << ", " << m_gpu[0].y << ", " << m_gpu[0].z << ")");
    INFO("CPU[0] = (" << m_cpu[0].x << ", " << m_cpu[0].y << ", " << m_cpu[0].z << ")");
    REQUIRE(max_err < 1e-12);
}

#endif // MICROMAG_CUDA
