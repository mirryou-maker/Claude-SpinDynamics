// test_demag_periodic_gpu.cpp — DemagFieldPeriodicGPU accuracy tests
// Verifies GPU periodic demag matches CPU DemagFieldPeriodic within
// floating-point tolerance and validates k=0 zeroing (uniform m -> H=0).
//
// Tag: [demag_periodic][gpu]
// Build: cmake --preset windows-msvc-cuda

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdlib>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/demag_periodic.hpp"
#include "micromag/demag_periodic_gpu.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Helper: compute H_demag for both CPU and GPU, return pair
// ---------------------------------------------------------------------------
static std::pair<VectorField3D, VectorField3D>
demag_per_both(const StructuredGrid& g, const Material& mat, const Vec3& m_dir,
               int n_rep = 2)
{
    VectorField3D m(g); m.set_uniform(m_dir);
    VectorField3D Hc(g), Hg(g);
    for (Index i = 0; i < g.size(); ++i) Hc[i] = Hg[i] = {0, 0, 0};

    DemagFieldPeriodic    cpu(g, n_rep); cpu.accumulate(m, mat, Hc);
    DemagFieldPeriodicGPU gpu(g, n_rep); gpu.accumulate(m, mat, Hg);
    return {Hc, Hg};
}

static double max_abs(const VectorField3D& a, const VectorField3D& b) {
    double mx = 0;
    for (Index i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(a[i].x - b[i].x));
        mx = std::max(mx, std::abs(a[i].y - b[i].y));
        mx = std::max(mx, std::abs(a[i].z - b[i].z));
    }
    return mx;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("DemagFieldPeriodicGPU: uniform m gives H=0 (k=0 zeroed)", "[demag_periodic][gpu]") {
    // Periodic BC: uniform magnetisation has no demag field (shape-independent)
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    DemagFieldPeriodicGPU demag(g, 2);
    VectorField3D m(g); m.set_uniform(Vec3{1, 0, 0});
    VectorField3D H(g);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
    demag.accumulate(m, mat, H);

    // All H components should be < 1 A/m (numerical noise level)
    double Hmax = 0;
    for (Index i = 0; i < H.size(); ++i)
        Hmax = std::max({Hmax, std::abs(H[i].x), std::abs(H[i].y), std::abs(H[i].z)});
    REQUIRE_THAT(Hmax, WithinAbs(0.0, 1.0));  // < 1 A/m noise
}

TEST_CASE("DemagFieldPeriodicGPU: matches CPU for Mx excitation (8x8x4)", "[demag_periodic][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    auto [Hc, Hg] = demag_per_both(g, mat, Vec3{1, 0, 0});
    REQUIRE_THAT(max_abs(Hc, Hg), WithinAbs(0.0, 1e4));   // <10kA/m ~ 1e-5 relative
}

TEST_CASE("DemagFieldPeriodicGPU: matches CPU for Mz excitation (8x8x4)", "[demag_periodic][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    auto [Hc, Hg] = demag_per_both(g, mat, Vec3{0, 0, 1});
    REQUIRE_THAT(max_abs(Hc, Hg), WithinAbs(0.0, 1e4));
}

TEST_CASE("DemagFieldPeriodicGPU: matches CPU for random m (16x12x2)", "[demag_periodic][gpu]") {
    StructuredGrid g(16, 12, 2, 4e-9, 4e-9, 4e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);
    // Use a deterministic non-uniform m (sine-wave pattern)
    const Index N = g.nx(), ny = g.ny(), nz = g.nz();
    for (Index iz = 0; iz < nz; ++iz)
    for (Index iy = 0; iy < ny; ++iy)
    for (Index ix = 0; ix < N;  ++ix) {
        const double theta = 2*3.14159*ix/N;
        m[g.linear_index(ix,iy,iz)] = Vec3{std::cos(theta), std::sin(theta), 0};
    }

    VectorField3D Hc(g), Hg(g);
    for (Index i = 0; i < g.size(); ++i) Hc[i] = Hg[i] = {0, 0, 0};
    DemagFieldPeriodic    cpu(g); cpu.accumulate(m, mat, Hc);
    DemagFieldPeriodicGPU gpu(g); gpu.accumulate(m, mat, Hg);

    REQUIRE_THAT(max_abs(Hc, Hg), WithinAbs(0.0, 5e4));  // <50kA/m ~1e-4 rel
}

TEST_CASE("DemagFieldPeriodicGPU: energy matches CPU", "[demag_periodic][gpu]") {
    StructuredGrid g(8, 8, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);
    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    for (Index iz=0; iz<nz; ++iz)
    for (Index iy=0; iy<ny; ++iy)
    for (Index ix=0; ix<nx; ++ix) {
        double phi = 2*3.14159*ix/nx + 3.14159*iy/ny;
        m[g.linear_index(ix,iy,iz)] = Vec3{std::cos(phi), std::sin(phi), 0};
    }

    DemagFieldPeriodic    cpu(g);
    DemagFieldPeriodicGPU gpu(g);
    const Real E_cpu = cpu.energy(m, mat);
    const Real E_gpu = gpu.energy(m, mat);

    // Use absolute tolerance: energy magnitudes can be very small
    const Real E_ref = std::max(std::abs(E_cpu), std::abs(E_gpu));
    REQUIRE_THAT(std::abs(E_gpu - E_cpu), WithinAbs(0.0, 1e-4 * E_ref + 1e-30));
}

TEST_CASE("DemagFieldPeriodicGPU: accumulate is additive (H_out += H_demag)", "[demag_periodic][gpu]") {
    // accumulate() adds to existing H_out, not overwrite
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    VectorField3D m(g); m.set_uniform(Vec3{0, 1, 0});

    DemagFieldPeriodicGPU demag(g);

    VectorField3D H1(g), H2(g);
    for (Index i=0; i<g.size(); ++i) H1[i] = H2[i] = {1e5, 0, 0};  // pre-filled

    demag.accumulate(m, mat, H1);  // should ADD to pre-filled

    // Compare: H1 should equal H2 + H_demag
    VectorField3D Hd(g);
    for (Index i=0; i<g.size(); ++i) Hd[i] = {0, 0, 0};
    demag.accumulate(m, mat, Hd);

    for (Index i=0; i<g.size(); ++i) {
        REQUIRE_THAT(H1[i].x, WithinAbs(H2[i].x + Hd[i].x, 1.0));
        REQUIRE_THAT(H1[i].y, WithinAbs(H2[i].y + Hd[i].y, 1.0));
        REQUIRE_THAT(H1[i].z, WithinAbs(H2[i].z + Hd[i].z, 1.0));
    }
}

#endif // MICROMAG_CUDA
