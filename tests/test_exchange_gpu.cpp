// test_exchange_gpu.cpp — ExchangeFieldGPU accuracy tests
// Compares GPU 6-point Laplacian against CPU ExchangeField.
// Tag: [exchange][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/exchange.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"

#include "gpu_test_tol.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Max component-wise relative error, |ref| floored at tol_abs
static double max_rel_diff(const VectorField3D& ref, const VectorField3D& got,
                             double tol_abs = 1.0) {
    double mx = 0.0;
    for (Index i = 0; i < ref.size(); ++i)
    for (int c = 0; c < 3; ++c) {
        const double rv = (&ref[i].x)[c];
        const double gv = (&got[i].x)[c];
        mx = std::max(mx, std::abs(rv - gv) / std::max(std::abs(rv), tol_abs));
    }
    return mx;
}

// ---------------------------------------------------------------------------
// T1: constructor smoke test
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU: constructor + name", "[exchange][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    ExchangeFieldGPU exch(g);
    REQUIRE(std::string(exch.name()) == "ExchangeFieldGPU");
}

// ---------------------------------------------------------------------------
// T2: single cell → zero exchange (no neighbours)
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU: single cell → zero", "[exchange][gpu]") {
    StructuredGrid g(1, 1, 1, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);   m.set_uniform({0, 0, 1});
    VectorField3D H(g);   H[0] = {0, 0, 0};

    ExchangeFieldGPU exch(g);
    exch.accumulate(m, mat, H);

    REQUIRE_THAT(H[0].x, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(H[0].y, WithinAbs(0.0, 1.0));
    REQUIRE_THAT(H[0].z, WithinAbs(0.0, 1.0));
}

// ---------------------------------------------------------------------------
// T3: uniform m → zero exchange everywhere (Laplacian of constant = 0)
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU: uniform m → zero field", "[exchange][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);   m.set_uniform({0, 0, 1});
    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    ExchangeField    cpu;
    ExchangeFieldGPU gpu(g);
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    // Both should be exactly zero; check CPU==GPU for symmetry
    const double err = max_rel_diff(H_cpu, H_gpu, 1.0);
    REQUIRE(err < 1e-10);
    // Also check magnitude directly
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H_gpu[i].z, WithinAbs(0.0, 1.0));
    }
}

// ---------------------------------------------------------------------------
// T4: non-uniform m (spin-wave along x) — main accuracy test
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU matches CPU: spin-wave along x", "[exchange][gpu]") {
    StructuredGrid g(12, 6, 4, 4e-9, 4e-9, 4e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.3;   // 0.3 rad/cell spiral along x
        m.at(ix, iy, iz) = {std::sin(phi), 0.0, std::cos(phi)};
    }

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    ExchangeField    cpu;
    ExchangeFieldGPU gpu(g);
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    const double err = max_rel_diff(H_cpu, H_gpu, mat.Ms * 1e-4);
    INFO("max_rel_err = " << err);
    REQUIRE(err < micromag::gtol(1e-8, 5e-2));
}

// ---------------------------------------------------------------------------
// T5: non-uniform along y and z (checks fy, fz terms separately)
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU matches CPU: spiral along z", "[exchange][gpu]") {
    StructuredGrid g(4, 4, 10, 5e-9, 5e-9, 3e-9);  // dz ≠ dx to test fz
    Material mat = Material::permalloy();

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = iz * 0.25;
        m.at(ix, iy, iz) = {std::cos(phi), std::sin(phi), 0.0};
    }

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    ExchangeField    cpu;
    ExchangeFieldGPU gpu(g);
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    const double err = max_rel_diff(H_cpu, H_gpu, mat.Ms * 1e-4);
    INFO("max_rel_err = " << err);
    REQUIRE(err < micromag::gtol(1e-8, 5e-2));
}

// ---------------------------------------------------------------------------
// T6: 16×16×1 thin film — 2D exchange (only ±x, ±y active)
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU matches CPU: 16×16×1 thin film", "[exchange][gpu]") {
    StructuredGrid g(16, 16, 1, 5e-9, 5e-9, 3e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.2 + iy * 0.15;
        m.at(ix, iy, 0) = {std::cos(phi), std::sin(phi), 0.0};
    }

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    ExchangeField    cpu;
    ExchangeFieldGPU gpu(g);
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    const double err = max_rel_diff(H_cpu, H_gpu, mat.Ms * 1e-4);
    INFO("max_rel_err = " << err);
    REQUIRE(err < micromag::gtol(1e-8, 5e-2));
}

// ---------------------------------------------------------------------------
// T7: 1×1×8 rod — only ±z exchange active; checks Neumann at rod ends
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU matches CPU: 1×1×8 rod along z", "[exchange][gpu]") {
    StructuredGrid g(1, 1, 8, 3e-9, 3e-9, 3e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz) {
        double phi = iz * 0.4;
        m.at(0, 0, iz) = {std::sin(phi), 0.0, std::cos(phi)};
    }

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    ExchangeField    cpu;
    ExchangeFieldGPU gpu(g);
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    const double err = max_rel_diff(H_cpu, H_gpu, mat.Ms * 1e-4);
    INFO("max_rel_err = " << err);
    REQUIRE(err < micromag::gtol(1e-8, 5e-2));

    // Neumann BC: ends (iz=0 and iz=7) should have only one-sided exchange
    // Verify by checking CPU == GPU (not by hardcoding the value)
}

// ---------------------------------------------------------------------------
// T8: additivity — calling accumulate twice gives 2× single result
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU: accumulate is additive", "[exchange][gpu]") {
    StructuredGrid g(6, 6, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.2 + iy * 0.1;
        m.at(ix, iy, iz) = {std::cos(phi), std::sin(phi), 0.0};
    }

    VectorField3D H1(g), H2(g);
    for (Index i = 0; i < g.size(); ++i) H1[i] = H2[i] = {0,0,0};

    ExchangeFieldGPU gpu(g);
    gpu.accumulate(m, mat, H1);           // one call
    gpu.accumulate(m, mat, H2);           // first call
    gpu.accumulate(m, mat, H2);           // second call → H2 = 2 × H1

    for (Index i = 0; i < g.size(); ++i) {
        if (std::abs(H1[i].x) > 1.0)
            REQUIRE_THAT(H2[i].x, WithinRel(2.0 * H1[i].x, 1e-10));
        if (std::abs(H1[i].y) > 1.0)
            REQUIRE_THAT(H2[i].y, WithinRel(2.0 * H1[i].y, 1e-10));
    }
}

// ---------------------------------------------------------------------------
// T9: energy matches CPU
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU energy matches CPU", "[exchange][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.3;
        m.at(ix, iy, iz) = {std::sin(phi), 0.0, std::cos(phi)};
    }

    ExchangeField    cpu;
    ExchangeFieldGPU gpu(g);

    const Real E_cpu = cpu.energy(m, mat);
    const Real E_gpu = gpu.energy(m, mat);

    INFO("E_cpu=" << E_cpu << "  E_gpu=" << E_gpu);
    REQUIRE(E_cpu != 0.0);
    REQUIRE_THAT(E_gpu, WithinRel(E_cpu, 1e-10));
}

// ---------------------------------------------------------------------------
// T8: Periodic BC — GPU matches CPU ExchangeField (BoundaryCondition::Periodic)
// ---------------------------------------------------------------------------
TEST_CASE("ExchangeFieldGPU: periodic BC matches CPU", "[exchange][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();

    // Sine-wave m pattern: non-trivial gradient at all cells including boundary
    VectorField3D m(g);
    const Index nx=g.nx(), ny=g.ny(), nz=g.nz();
    for (Index iz=0; iz<nz; ++iz)
    for (Index iy=0; iy<ny; ++iy)
    for (Index ix=0; ix<nx; ++ix) {
        double theta = 2*3.14159*ix/nx;
        m[g.linear_index(ix,iy,iz)] = Vec3{std::cos(theta), std::sin(theta), 0};
    }

    VectorField3D Hc(g), Hg(g);
    for (Index i=0; i<g.size(); ++i) Hc[i] = Hg[i] = {0,0,0};

    ExchangeField cpu(BoundaryCondition::Periodic);
    cpu.accumulate(m, mat, Hc);

    ExchangeFieldGPU gpu(g, BoundaryCondition::Periodic);
    gpu.accumulate(m, mat, Hg);

    REQUIRE_THAT(max_rel_diff(Hc, Hg, 1.0), WithinAbs(0.0, micromag::gtol(1e-6, 5e-2)));
}

TEST_CASE("ExchangeFieldGPU: uniform m → zero exchange (periodic BC)", "[exchange][gpu]") {
    StructuredGrid g(6, 6, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    VectorField3D m(g); m.set_uniform({0, 0, 1});
    VectorField3D H(g);
    for (Index i=0; i<H.size(); ++i) H[i] = {0,0,0};

    ExchangeFieldGPU gpu(g, BoundaryCondition::Periodic);
    gpu.accumulate(m, mat, H);

    double hmax = 0;
    for (Index i=0; i<H.size(); ++i)
        hmax = std::max({hmax, std::abs(H[i].x), std::abs(H[i].y), std::abs(H[i].z)});
    REQUIRE_THAT(hmax, WithinAbs(0.0, 1.0));
}

#endif // MICROMAG_CUDA
