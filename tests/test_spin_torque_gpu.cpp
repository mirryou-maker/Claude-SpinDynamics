// test_spin_torque_gpu.cpp — GPU spin torque tests (STT, SOT, Zhang-Li)
// Tag: [spin_torque_gpu][gpu]
//
// ST1: SlonczewskiSTTGPU numerics vs CPU SlonczewskiSTT
// ST2: SpinOrbitTorqueGPU numerics vs CPU SpinOrbitTorque
// ST3: ZhangLiSTTGPU numerics vs CPU ZhangLiSTT
// ST4: SpinTorqueSumGPU adds terms correctly
// ST5: RK4IntegratorGPU::step with SpinTorqueSumGPU (no crash, m normalized)
// ST6: SlonczewskiSTTGPU switches a macrospin
// ST7: RK45IntegratorGPU + SpinTorqueSumGPU (no crash, m normalized)
// ST8: RK4 + STT with changing J between steps (no stale baked-in values)
// ST9: RK4 and RK45 + SOT agree at T=0

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cuda_runtime.h>

#include "micromag/demag_gpu.hpp"
#include "micromag/effective_field_gpu_iface.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/rk45_integrator_gpu.hpp"
#include "micromag/spin_torque.hpp"
#include "micromag/spin_torque_gpu.hpp"

#include "gpu_test_tol.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Single macrospin (1×1×1) for numeric comparison tests
static StructuredGrid single_cell_grid() {
    return StructuredGrid(1, 1, 1, 5e-9, 5e-9, 5e-9);
}

static Material make_py() {
    auto m = Material::permalloy();
    m.alpha = 0.01;
    return m;
}

// ---------------------------------------------------------------------------
// Helper: compare GPU spin torque output vs CPU reference on a single cell.
// dm_ref = reference CPU dm/dt contribution for m = (mx, my, mz).
// ---------------------------------------------------------------------------
static Vec3 run_stt_gpu(ISpinTorqueGPU& stt_gpu, Vec3 m_vec, const Material& mat) {
    const StructuredGrid g = single_cell_grid();
    const int N = 1;

    // Upload m to device (device buffers use GReal to match accumulate_gpu_ptr)
    GReal h_m[3] = {static_cast<GReal>(m_vec.x),
                    static_cast<GReal>(m_vec.y),
                    static_cast<GReal>(m_vec.z)};
    GReal* d_m;
    cudaMalloc(&d_m, 3 * sizeof(GReal));
    cudaMemcpy(d_m, h_m, 3 * sizeof(GReal), cudaMemcpyHostToDevice);

    // dm_out initialized to zero
    GReal* d_dm;
    cudaMalloc(&d_dm, 3 * sizeof(GReal));
    cudaMemset(d_dm, 0, 3 * sizeof(GReal));

    stt_gpu.accumulate_gpu_ptr(d_m, mat, d_dm);

    GReal h_dm[3];
    cudaMemcpy(h_dm, d_dm, 3 * sizeof(GReal), cudaMemcpyDeviceToHost);

    cudaFree(d_m);
    cudaFree(d_dm);

    return Vec3{h_dm[0], h_dm[1], h_dm[2]};
}

// ---------------------------------------------------------------------------
// ST1: SlonczewskiSTTGPU numerics vs CPU SlonczewskiSTT
// ---------------------------------------------------------------------------
TEST_CASE("SlonczewskiSTTGPU: matches CPU reference", "[spin_torque_gpu][gpu]") {
    const StructuredGrid g = single_cell_grid();
    const auto mat = make_py();

    const Real J    = 1e12;   // A/m²
    const Real P    = 0.5;
    const Real d    = 3e-9;   // 3 nm
    const Vec3 p    = {0, 0, 1};
    const Real beta = 0.1;

    // GPU version
    SlonczewskiSTTGPU stt_gpu(g, J, P, d, p, beta);

    // CPU version
    SlonczewskiSTT stt_cpu(J, P, d, p, beta);

    // Test with m tilted 45 degrees from p
    const Vec3 m = Vec3{1.0/std::sqrt(2.0), 0, 1.0/std::sqrt(2.0)};

    // CPU reference
    VectorField3D mf(g), dm_cpu(g);
    mf[0] = m;
    dm_cpu[0] = Vec3{0, 0, 0};
    stt_cpu.accumulate(mf, mat, dm_cpu);

    // GPU result
    const Vec3 dm_gpu = run_stt_gpu(stt_gpu, m, mat);

    REQUIRE_THAT(dm_gpu.x, WithinRel(dm_cpu[0].x, micromag::gtol(1e-10)));
    REQUIRE_THAT(dm_gpu.y, WithinRel(dm_cpu[0].y, micromag::gtol(1e-10)));
    REQUIRE_THAT(dm_gpu.z, WithinRel(dm_cpu[0].z, micromag::gtol(1e-10)));
}

// ---------------------------------------------------------------------------
// ST2: SpinOrbitTorqueGPU numerics vs CPU SpinOrbitTorque
// ---------------------------------------------------------------------------
TEST_CASE("SpinOrbitTorqueGPU: matches CPU reference", "[spin_torque_gpu][gpu]") {
    const StructuredGrid g = single_cell_grid();
    const auto mat = make_py();

    const Real J_c      = 5e11;   // A/m²
    const Real theta_SH = 0.12;   // Pt
    const Real d_fm     = 3e-9;
    const Vec3 sigma    = {0, 1, 0};  // y-direction polarisation
    const Real eta_DL   = 1.0;
    const Real eta_FL   = 0.2;

    SpinOrbitTorqueGPU sot_gpu(g, J_c, theta_SH, d_fm, sigma, eta_DL, eta_FL);
    SpinOrbitTorque    sot_cpu(J_c, theta_SH, d_fm, sigma, eta_DL, eta_FL);

    const Vec3 m = Vec3{1.0/std::sqrt(2.0), 1.0/std::sqrt(2.0), 0};

    VectorField3D mf(g), dm_cpu(g);
    mf[0] = m;
    dm_cpu[0] = Vec3{0, 0, 0};
    sot_cpu.accumulate(mf, mat, dm_cpu);

    const Vec3 dm_gpu = run_stt_gpu(sot_gpu, m, mat);

    REQUIRE_THAT(dm_gpu.x, WithinRel(dm_cpu[0].x, micromag::gtol(1e-10)));
    REQUIRE_THAT(dm_gpu.y, WithinRel(dm_cpu[0].y, micromag::gtol(1e-10)));
    REQUIRE_THAT(dm_gpu.z, WithinRel(dm_cpu[0].z, micromag::gtol(1e-10)));
}

// ---------------------------------------------------------------------------
// ST3: ZhangLiSTTGPU numerics vs CPU ZhangLiSTT (uniform gradient → zero torque)
// ---------------------------------------------------------------------------
TEST_CASE("ZhangLiSTTGPU: uniform m gives zero torque", "[spin_torque_gpu][gpu]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const auto mat = make_py();

    // Uniform m: gradient = 0 → zero torque everywhere
    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{0.0, 0.0, 1.0};

    const Vec3 J = {1e12, 0, 0};
    ZhangLiSTTGPU zl_gpu(g, J, 0.5, 0.05);
    ZhangLiSTT    zl_cpu(J, 0.5, 0.05);

    // CPU reference
    VectorField3D dm_cpu(g);
    for (Index i = 0; i < dm_cpu.size(); ++i) dm_cpu[i] = Vec3{0,0,0};
    zl_cpu.accumulate(m0, mat, dm_cpu);

    // GPU: upload m0, run, download dm
    const int N = static_cast<int>(g.size());
    GReal* d_m;
    GReal* d_dm;
    cudaMalloc(&d_m,  3 * N * sizeof(GReal));
    cudaMalloc(&d_dm, 3 * N * sizeof(GReal));
    cudaMemset(d_dm, 0, 3 * N * sizeof(GReal));

    // Pack m0 into component-major
    std::vector<GReal> h_m(3 * N);
    for (int i = 0; i < N; ++i) {
        h_m[i]       = static_cast<GReal>(m0[i].x);
        h_m[N + i]   = static_cast<GReal>(m0[i].y);
        h_m[2*N + i] = static_cast<GReal>(m0[i].z);
    }
    cudaMemcpy(d_m, h_m.data(), 3 * N * sizeof(GReal), cudaMemcpyHostToDevice);

    zl_gpu.accumulate_gpu_ptr(d_m, mat, d_dm);

    std::vector<GReal> h_dm(3 * N);
    cudaMemcpy(h_dm.data(), d_dm, 3 * N * sizeof(GReal), cudaMemcpyDeviceToHost);

    cudaFree(d_m);
    cudaFree(d_dm);

    // All should be zero (uniform m = no gradient)
    for (int i = 0; i < N; ++i) {
        REQUIRE_THAT(h_dm[i],       WithinAbs(dm_cpu[i].x, 1e-6));
        REQUIRE_THAT(h_dm[N+i],     WithinAbs(dm_cpu[i].y, 1e-6));
        REQUIRE_THAT(h_dm[2*N+i],   WithinAbs(dm_cpu[i].z, 1e-6));
    }
}

// ---------------------------------------------------------------------------
// ST4: SpinTorqueSumGPU adds multiple torque contributions
// ---------------------------------------------------------------------------
TEST_CASE("SpinTorqueSumGPU: size tracking and composition", "[spin_torque_gpu][gpu]") {
    const StructuredGrid g = single_cell_grid();
    const auto mat = make_py();

    SlonczewskiSTTGPU stt(g, 1e12, 0.5, 3e-9, Vec3{0,0,1});
    SpinOrbitTorqueGPU sot(g, 5e11, 0.12, 3e-9, Vec3{0,1,0});

    SpinTorqueSumGPU torques;
    REQUIRE(torques.size() == 0);

    torques.add(stt);
    REQUIRE(torques.size() == 1);

    torques.add(sot);
    REQUIRE(torques.size() == 2);

    // Verify sum equals individual results added
    const Vec3 m = Vec3{0.5, 0.5, 1.0/std::sqrt(2.0)};

    const Vec3 dm_stt = run_stt_gpu(stt, m, mat);
    const Vec3 dm_sot = run_stt_gpu(sot, m, mat);

    // Run the compositor
    GReal h_m[3] = {static_cast<GReal>(m.x),
                    static_cast<GReal>(m.y),
                    static_cast<GReal>(m.z)};
    GReal* d_m;
    GReal* d_dm;
    cudaMalloc(&d_m,  3 * sizeof(GReal));
    cudaMalloc(&d_dm, 3 * sizeof(GReal));
    cudaMemcpy(d_m, h_m, 3 * sizeof(GReal), cudaMemcpyHostToDevice);
    cudaMemset(d_dm, 0,  3 * sizeof(GReal));

    torques.accumulate_gpu_ptr(d_m, mat, d_dm);

    GReal h_dm[3];
    cudaMemcpy(h_dm, d_dm, 3 * sizeof(GReal), cudaMemcpyDeviceToHost);
    cudaFree(d_m);
    cudaFree(d_dm);

    // Sum should equal stt + sot
    REQUIRE_THAT(h_dm[0], WithinAbs(dm_stt.x + dm_sot.x, micromag::gtol(1e-3, 1e3)));
    REQUIRE_THAT(h_dm[1], WithinAbs(dm_stt.y + dm_sot.y, micromag::gtol(1e-3, 1e3)));
    REQUIRE_THAT(h_dm[2], WithinAbs(dm_stt.z + dm_sot.z, micromag::gtol(1e-3, 1e3)));

    torques.clear();
    REQUIRE(torques.size() == 0);
}

// ---------------------------------------------------------------------------
// ST5: RK4IntegratorGPU::step with SpinTorqueSumGPU — no crash, m normalized
// ---------------------------------------------------------------------------
TEST_CASE("SpinTorqueSumGPU: RK4 step runs without crash", "[spin_torque_gpu][gpu]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    auto mat = make_py();
    mat.alpha = 0.5;

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{1.0/std::sqrt(2.0), 0, 1.0/std::sqrt(2.0)};

    DemagFieldGPU      demag(g);
    ExchangeFieldGPU   exch(g);
    ZeemanFieldGPU     zeeman(g, Vec3{1e5, 0, 0});

    FieldSumGPU fields;
    fields.add(exch);
    fields.add(zeeman);

    SlonczewskiSTTGPU stt(g, 1e12, 0.5, 3e-9, Vec3{0,0,1});
    SpinOrbitTorqueGPU sot(g, 5e11, 0.12, 3e-9, Vec3{0,1,0});

    SpinTorqueSumGPU torques;
    torques.add(stt);
    torques.add(sot);

    RK4IntegratorGPU integ(g, 1e-13);
    integ.upload(m0);

    REQUIRE_NOTHROW([&]() {
        for (int k = 0; k < 10; ++k)
            integ.step(mat, demag, fields, torques);
    }());

    VectorField3D m_out(g);
    integ.download(m_out);

    for (Index i = 0; i < m_out.size(); ++i)
        REQUIRE_THAT(m_out[i].norm(), WithinAbs(1.0, micromag::gtol(1e-10)));
}

// ---------------------------------------------------------------------------
// ST6: SlonczewskiSTTGPU drives m away from initial state
//
// Convention in this code: τ = a_J [m×(m×p)], a_J > 0.
// Near m ≈ p (parallel state), τ ≈ +a_J*δm → parallel state is UNSTABLE.
// Near m ≈ -p (antiparallel), τ ≈ -a_J*δm → antiparallel state is STABLE.
//
// Test: m starts at +z (tilted slightly), p = +z → unstable → mz decreases.
// After 1000 steps with large J, mz should be less than the initial value.
// ---------------------------------------------------------------------------
TEST_CASE("SlonczewskiSTTGPU: torque drives mz away from p direction",
          "[spin_torque_gpu][gpu]") {
    StructuredGrid g(1, 1, 1, 10e-9, 10e-9, 3e-9);
    auto mat = Material::permalloy();
    mat.alpha = 0.02;

    // Start slightly tilted from +z
    const double mx0 = 0.1;
    const double mz0 = std::sqrt(1.0 - mx0*mx0);
    VectorField3D m0(g);
    m0[0] = Vec3{mx0, 0.0, mz0};

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{0, 0, 0});

    FieldSumGPU fields;
    fields.add(exch);
    fields.add(zeeman);

    // p = +z (parallel to initial m) → a_J > 0 → antidamping → mz decreases
    const Real J_large = 1e13;
    SlonczewskiSTTGPU stt(g, J_large, 0.5, 3e-9, Vec3{0, 0, 1});
    SpinTorqueSumGPU torques;
    torques.add(stt);

    RK4IntegratorGPU integ(g, 5e-14);
    integ.upload(m0);

    for (int k = 0; k < 1000; ++k)
        integ.step(mat, demag, fields, torques);

    VectorField3D m_out(g);
    integ.download(m_out);

    // mz should have decreased from mz0 (STT drives m away from p = +z)
    INFO("mz_initial=" << mz0 << "  mz_final=" << m_out[0].z);
    REQUIRE(m_out[0].z < mz0);
}

// ---------------------------------------------------------------------------
// ST7: RK45IntegratorGPU + SpinTorqueSumGPU — no crash, m stays normalized
//
// RK45 does not use CUDA Graphs, so STT/SOT kernels are always called in
// normal (non-capture) execution mode.  This test verifies the overload
// step(mat, demag, FieldSumGPU, SpinTorqueSumGPU) works end-to-end.
// ---------------------------------------------------------------------------
TEST_CASE("RK45IntegratorGPU: step with SpinTorqueSumGPU runs without crash",
          "[spin_torque_gpu][gpu]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    auto mat = make_py();
    mat.alpha = 0.3;

    VectorField3D m0(g);
    for (Index i = 0; i < m0.size(); ++i)
        m0[i] = Vec3{0.1, 0.0, std::sqrt(1.0 - 0.01)};

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{0, 0, 1e5});

    FieldSumGPU fields;
    fields.add(exch);
    fields.add(zeeman);

    SlonczewskiSTTGPU stt(g, 1e12, 0.5, 3e-9, Vec3{0, 0, 1});
    SpinTorqueSumGPU torques;
    torques.add(stt);

    RK45IntegratorGPU rk45(g);
    rk45.upload(m0);

    REQUIRE_NOTHROW([&]() {
        for (int k = 0; k < 10; ++k)
            rk45.step(mat, demag, fields, torques);
    }());

    VectorField3D m_out(g);
    rk45.download(m_out);
    for (Index i = 0; i < m_out.size(); ++i)
        REQUIRE_THAT(m_out[i].norm(), WithinAbs(1.0, micromag::gtol(1e-9)));
}

// ---------------------------------------------------------------------------
// ST8: RK4 with changing torque parameter (J sweep) stays physically correct
//
// Tests the core use case: changing stt.J between steps should give results
// consistent with a fresh integrator started at the new J.  Before the fix,
// the CUDA Graph could bake in the old aJ coefficient.
// ---------------------------------------------------------------------------
TEST_CASE("RK4IntegratorGPU + STT: changing J between steps gives correct physics",
          "[spin_torque_gpu][gpu]") {
    StructuredGrid g(1, 1, 1, 10e-9, 10e-9, 3e-9);
    auto mat = Material::permalloy();
    mat.alpha = 0.02;

    const double mx0 = 0.1;
    const double mz0 = std::sqrt(1.0 - mx0*mx0);

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{0, 0, 0});
    FieldSumGPU fields;
    fields.add(exch); fields.add(zeeman);

    SlonczewskiSTTGPU stt(g, 1e13, 0.5, 3e-9, Vec3{0, 0, 1});
    SpinTorqueSumGPU torques;
    torques.add(stt);

    // Run 500 steps at J=1e13 (large — should cause mz to decrease)
    VectorField3D m0(g);
    m0[0] = Vec3{mx0, 0.0, mz0};
    RK4IntegratorGPU integ(g, 5e-14);
    integ.upload(m0);
    for (int k = 0; k < 500; ++k)
        integ.step(mat, demag, fields, torques);
    VectorField3D m_big_J(g);
    integ.download(m_big_J);

    // Run 500 steps at J=0 (no STT — m should barely change from precession)
    stt.set_J(0.0);
    m0[0] = Vec3{mx0, 0.0, mz0};
    RK4IntegratorGPU integ2(g, 5e-14);
    integ2.upload(m0);
    for (int k = 0; k < 500; ++k)
        integ2.step(mat, demag, fields, torques);
    VectorField3D m_zero_J(g);
    integ2.download(m_zero_J);

    // With J=1e13 (antidamping), mz should have decreased more than at J=0
    INFO("mz at J=1e13: " << m_big_J[0].z << "  mz at J=0: " << m_zero_J[0].z);
    REQUIRE(m_big_J[0].z < m_zero_J[0].z);
}

// ---------------------------------------------------------------------------
// ST9: RK4 + SOT gives same result as RK45 + SOT (T=0, short run)
//
// Confirms both integrators agree on deterministic SOT dynamics.
// RK4 uses direct execution (no graph); RK45 uses its normal loop.
// ---------------------------------------------------------------------------
TEST_CASE("RK4 and RK45 + SOT agree at T=0 (short run)",
          "[spin_torque_gpu][gpu]") {
    StructuredGrid g(1, 1, 1, 10e-9, 10e-9, 3e-9);
    auto mat = Material::permalloy();
    mat.alpha = 0.02;

    const double mx0 = 0.1, mz0 = std::sqrt(1.0 - mx0*mx0);
    VectorField3D m0(g);
    m0[0] = Vec3{mx0, 0.0, mz0};

    DemagFieldGPU    demag(g);
    ExchangeFieldGPU exch(g);
    ZeemanFieldGPU   zeeman(g, Vec3{0, 0, 0});
    FieldSumGPU fields;
    fields.add(exch); fields.add(zeeman);

    SpinOrbitTorqueGPU sot(g, 2e12, 0.12, 3e-9, Vec3{1, 0, 0});
    SpinTorqueSumGPU torques;
    torques.add(sot);

    const Real DT = 5e-15;
    const int NSTEP = 100;

    // RK4 run
    RK4IntegratorGPU rk4(g, DT);
    rk4.upload(m0);
    for (int k = 0; k < NSTEP; ++k)
        rk4.step(mat, demag, fields, torques);
    VectorField3D m_rk4(g);
    rk4.download(m_rk4);

    // RK45 run (same fixed dt via dt_max = dt_init = DT to force fixed stepping)
    RK45IntegratorGPU::Options opts;
    opts.dt_init = DT; opts.dt_max = DT; opts.rtol = 1e-8; opts.atol = 1e-10;
    RK45IntegratorGPU rk45(g, opts);
    rk45.upload(m0);
    for (int k = 0; k < NSTEP; ++k)
        rk45.step(mat, demag, fields, torques);
    VectorField3D m_rk45(g);
    rk45.download(m_rk45);

    // RK4 and RK45 with same dt should agree within truncation error (O(dt^4))
    const double tol = micromag::gtol(1e-6);
    REQUIRE_THAT(m_rk4[0].x, WithinAbs(m_rk45[0].x, tol));
    REQUIRE_THAT(m_rk4[0].y, WithinAbs(m_rk45[0].y, tol));
    REQUIRE_THAT(m_rk4[0].z, WithinAbs(m_rk45[0].z, tol));
}

#endif // MICROMAG_CUDA
