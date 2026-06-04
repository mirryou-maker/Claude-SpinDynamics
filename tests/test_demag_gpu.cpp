// test_demag_gpu.cpp — GPU demag accuracy tests
// Verifies DemagFieldGPU gives results matching CPU DemagField
// within double-precision floating-point tolerance (~1e-5 relative).
//
// Tag: [demag][gpu]
// Build: cmake --preset windows-msvc-cuda

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/demag.hpp"
#include "micromag/demag_gpu.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Compute H_demag from CPU and GPU in one call, return both
static std::pair<VectorField3D, VectorField3D>
compute_both(const StructuredGrid& g, const Material& mat, const Vec3& m_dir)
{
    VectorField3D m(g); m.set_uniform(m_dir);
    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0, 0, 0};

    DemagField    cpu(g); cpu.accumulate(m, mat, H_cpu);
    DemagFieldGPU gpu(g); gpu.accumulate(m, mat, H_gpu);
    return {H_cpu, H_gpu};
}

// Per-cell max absolute and relative errors
static double max_abs_diff(const VectorField3D& a, const VectorField3D& b)
{
    double mx = 0;
    for (Index i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(a[i].x - b[i].x));
        mx = std::max(mx, std::abs(a[i].y - b[i].y));
        mx = std::max(mx, std::abs(a[i].z - b[i].z));
    }
    return mx;
}
static double max_rel_diff(const VectorField3D& a, const VectorField3D& b, double tol_abs)
{
    double mx = 0;
    for (Index i = 0; i < a.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            const double av = (&a[i].x)[c], bv = (&b[i].x)[c];
            const double ref = std::max(std::abs(av), tol_abs);
            mx = std::max(mx, std::abs(av - bv) / ref);
        }
    }
    return mx;
}

// ---------------------------------------------------------------------------
// T1: DemagFieldGPU constructor — basic smoke test
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU: constructor + name", "[demag][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    DemagFieldGPU gpu(g);
    REQUIRE(std::string(gpu.name()) == "DemagFieldGPU");
}

// ---------------------------------------------------------------------------
// T2: 4×4×4 cube, m = +z  — diagonal H_z component
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU matches CPU: 4x4x4 cube m=+z", "[demag][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    auto [H_cpu, H_gpu] = compute_both(g, mat, {0, 0, 1});

    // Max relative error across all cells/components (tol_abs = 1 A/m)
    const double rel_err = max_rel_diff(H_cpu, H_gpu, 1.0);
    INFO("max relative error = " << rel_err);
    REQUIRE(rel_err < 1e-5);

    // Spot-check physical result at centre (verifies GPU didn't crash)
    REQUIRE_THAT(H_gpu.at(1,1,1).z, WithinRel(-mat.Ms/3.0, 0.30));
}

// ---------------------------------------------------------------------------
// T3: 4×4×4 cube, m = +x  — off-axis component
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU matches CPU: 4x4x4 cube m=+x", "[demag][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    auto [H_cpu, H_gpu] = compute_both(g, mat, {1, 0, 0});

    const double rel_err = max_rel_diff(H_cpu, H_gpu, 1.0);
    INFO("max relative error = " << rel_err);
    REQUIRE(rel_err < 1e-5);

    // Diagonal component at centre
    REQUIRE_THAT(H_gpu.at(1,1,1).x, WithinRel(-mat.Ms/3.0, 0.30));
}

// ---------------------------------------------------------------------------
// T4: 16×16×1 thin film  — N_zz → 1
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU matches CPU: 16x16x1 thin film", "[demag][gpu]") {
    StructuredGrid g(16, 16, 1, 5e-9, 5e-9, 2e-9);
    Material mat = Material::permalloy();

    auto [H_cpu, H_gpu] = compute_both(g, mat, {0, 0, 1});

    const double rel_err = max_rel_diff(H_cpu, H_gpu, 1.0);
    INFO("max relative error = " << rel_err);
    REQUIRE(rel_err < 1e-5);

    // Physical check: thin film should be near -Ms
    REQUIRE(H_gpu.at(8,8,0).z < -mat.Ms * 0.85);
}

// ---------------------------------------------------------------------------
// T5: 1×1×16 long rod  — N_zz → 0
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU matches CPU: 1x1x16 long rod", "[demag][gpu]") {
    StructuredGrid g(1, 1, 16, 2e-9, 2e-9, 5e-9);
    Material mat = Material::permalloy();

    auto [H_cpu, H_gpu] = compute_both(g, mat, {0, 0, 1});

    const double rel_err = max_rel_diff(H_cpu, H_gpu, 1.0);
    INFO("max relative error = " << rel_err);
    REQUIRE(rel_err < 1e-5);
}

// ---------------------------------------------------------------------------
// T6: 5×5×5 cube, m = +z  — zero transverse at symmetric centre
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU matches CPU: 5x5x5 no transverse H", "[demag][gpu]") {
    StructuredGrid g(5, 5, 5, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    auto [H_cpu, H_gpu] = compute_both(g, mat, {0, 0, 1});

    const double rel_err = max_rel_diff(H_cpu, H_gpu, 1.0);
    INFO("max relative error = " << rel_err);
    REQUIRE(rel_err < 1e-5);

    // Symmetric centre: H_x and H_y must vanish (same check as CPU test)
    REQUIRE_THAT(H_gpu.at(2,2,2).x, WithinAbs(0.0, mat.Ms * 0.001));
    REQUIRE_THAT(H_gpu.at(2,2,2).y, WithinAbs(0.0, mat.Ms * 0.001));
}

// ---------------------------------------------------------------------------
// T7: Energy matches CPU
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU energy matches CPU", "[demag][gpu]") {
    const Real a = 5e-9;
    StructuredGrid g(4, 4, 4, a, a, a);
    Material mat = Material::permalloy();

    VectorField3D m(g); m.set_uniform({0, 0, 1});

    DemagField    cpu(g);
    DemagFieldGPU gpu(g);

    const Real E_cpu = cpu.energy(m, mat);
    const Real E_gpu = gpu.energy(m, mat);

    INFO("E_cpu=" << E_cpu << "  E_gpu=" << E_gpu);
    // Both non-zero, agree within 0.01%
    REQUIRE(E_cpu != 0.0);
    REQUIRE_THAT(E_gpu, WithinRel(E_cpu, 1e-4));
}

// ---------------------------------------------------------------------------
// T8: Accumulate is additive — calling twice gives 2× single result
// ---------------------------------------------------------------------------
TEST_CASE("DemagFieldGPU: accumulate is additive", "[demag][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    VectorField3D m(g); m.set_uniform({0, 0, 1});

    DemagFieldGPU gpu(g);

    VectorField3D H1(g), H2(g);
    for (Index i = 0; i < g.size(); ++i) H1[i] = H2[i] = {0, 0, 0};

    gpu.accumulate(m, mat, H1);           // one call
    gpu.accumulate(m, mat, H2);           // same call — adds to H2
    gpu.accumulate(m, mat, H2);           // second call — H2 = 2× H1

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H2[i].z, WithinRel(2.0 * H1[i].z, 1e-10));
    }
}

#endif // MICROMAG_CUDA
