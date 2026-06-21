// test_field_kernels_gpu.cpp — G2: GPU Zeeman + Anisotropy accuracy tests
// Compares GPU implementations against CPU counterparts.
// Tags: [zeeman][gpu], [anisotropy][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/anisotropy.hpp"
#include "micromag/field.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/zeeman.hpp"
#include "gpu_test_tol.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

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

// ============================================================================
// ZeemanFieldGPU tests
// ============================================================================

TEST_CASE("ZeemanFieldGPU: constructor + name", "[zeeman][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    ZeemanFieldGPU zeeman(g, {1e3, 0, 0});
    REQUIRE(std::string(zeeman.name()) == "ZeemanFieldGPU");
}

TEST_CASE("ZeemanFieldGPU: H_ext=0 → zero contribution", "[zeeman][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    VectorField3D m(g);   m.set_uniform({0, 0, 1});
    VectorField3D H(g);   for (Index i=0; i<g.size(); ++i) H[i]={0,0,0};

    ZeemanFieldGPU zeeman(g, {0, 0, 0});
    zeeman.accumulate(m, mat, H);

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, 1.0));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1.0));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1.0));
    }
}

TEST_CASE("ZeemanFieldGPU matches CPU: uniform H", "[zeeman][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    Vec3 H_ext{-24.6e3, 4.3e3, 0.0};

    VectorField3D m(g);   m.set_uniform({1, 0, 0});
    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i=0; i<g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    ZeemanField    cpu(H_ext);
    ZeemanFieldGPU gpu(g, H_ext);
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    // H should be exactly H_ext at every cell
    const double err = max_rel_diff(H_cpu, H_gpu, std::abs(H_ext.x));
    INFO("max_rel_err = " << err);
    REQUIRE(err < 1e-14);   // exact double equality expected (pure CPU path)
}

TEST_CASE("ZeemanFieldGPU: H_ext independent of m", "[zeeman][gpu]") {
    // Two different m states → same H_out increment
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    Vec3 H_ext{0, 0, 1e4};

    VectorField3D m1(g); m1.set_uniform({1, 0, 0});
    VectorField3D m2(g); m2.set_uniform({0, 1, 0});
    VectorField3D H1(g), H2(g);
    for (Index i=0; i<g.size(); ++i) H1[i] = H2[i] = {0,0,0};

    ZeemanFieldGPU gpu(g, H_ext);
    gpu.accumulate(m1, mat, H1);
    gpu.accumulate(m2, mat, H2);

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H1[i].z, WithinRel(H2[i].z, 1e-14));
    }
}

TEST_CASE("ZeemanFieldGPU: additive", "[zeeman][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    Vec3 H_ext{1e4, 2e3, -5e3};

    VectorField3D m(g); m.set_uniform({0, 0, 1});
    VectorField3D H1(g), H2(g);
    for (Index i=0; i<g.size(); ++i) H1[i] = H2[i] = {0,0,0};

    ZeemanFieldGPU gpu(g, H_ext);
    gpu.accumulate(m, mat, H1);
    gpu.accumulate(m, mat, H2);
    gpu.accumulate(m, mat, H2);   // H2 = 2 × H1

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H2[i].x, WithinRel(2.0 * H1[i].x, 1e-14));
        REQUIRE_THAT(H2[i].z, WithinRel(2.0 * H1[i].z, 1e-14));
    }
}

TEST_CASE("ZeemanFieldGPU energy matches CPU", "[zeeman][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    Vec3 H_ext{-24.6e3, 4.3e3, 0};

    VectorField3D m(g); m.set_uniform({0, 0, 1});
    ZeemanField    cpu(H_ext);
    ZeemanFieldGPU gpu(g, H_ext);

    REQUIRE_THAT(gpu.energy(m, mat), WithinRel(cpu.energy(m, mat), 1e-12));
}

// ============================================================================
// UniaxialAnisotropyFieldGPU tests
// ============================================================================

TEST_CASE("UniaxialAnisotropyFieldGPU: constructor + name", "[anisotropy][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    UniaxialAnisotropyFieldGPU aniso(g);
    REQUIRE(std::string(aniso.name()) == "UniaxialAnisotropyFieldGPU");
}

TEST_CASE("UniaxialAnisotropyFieldGPU: K=0 → zero", "[anisotropy][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    mat.K_uniaxial = 0.0;

    VectorField3D m(g); m.set_uniform({0.5, 0.5, 0.707107}); m.normalize();
    VectorField3D H(g); for (Index i=0; i<g.size(); ++i) H[i]={0,0,0};

    UniaxialAnisotropyFieldGPU aniso(g);
    aniso.accumulate(m, mat, H);

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, 1.0));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1.0));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1.0));
    }
}

TEST_CASE("UniaxialAnisotropyFieldGPU: m along easy axis → max H", "[anisotropy][gpu]") {
    // m ∥ û → H = 2K/(μ₀Ms) × û
    StructuredGrid g(2, 2, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    mat.K_uniaxial  = 1e5;
    mat.easy_axis   = {0, 0, 1};

    VectorField3D m(g); m.set_uniform({0, 0, 1});
    VectorField3D H(g); for (Index i=0; i<g.size(); ++i) H[i]={0,0,0};

    UniaxialAnisotropyFieldGPU aniso(g);
    aniso.accumulate(m, mat, H);

    const double expected_Hz = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0,      expected_Hz * 1e-10));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0,      expected_Hz * 1e-10));
        REQUIRE_THAT(H[i].z, WithinRel(expected_Hz, micromag::gtol(1e-10)));
    }
}

TEST_CASE("UniaxialAnisotropyFieldGPU: m ⊥ easy axis → zero H", "[anisotropy][gpu]") {
    // m ⊥ û → m·û = 0 → H = 0
    StructuredGrid g(2, 2, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    mat.K_uniaxial = 1e5;
    mat.easy_axis  = {0, 0, 1};   // easy axis = z

    VectorField3D m(g); m.set_uniform({1, 0, 0});   // m = x ⊥ z
    VectorField3D H(g); for (Index i=0; i<g.size(); ++i) H[i]={0,0,0};

    UniaxialAnisotropyFieldGPU aniso(g);
    aniso.accumulate(m, mat, H);

    const double scale = 2.0 * mat.K_uniaxial / (constants::mu_0 * mat.Ms);
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, scale * 1e-10));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, scale * 1e-10));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, scale * 1e-10));
    }
}

TEST_CASE("UniaxialAnisotropyFieldGPU matches CPU: cobalt", "[anisotropy][gpu]") {
    // Cobalt has large K_uniaxial (7e5 J/m³) — non-trivial anisotropy
    StructuredGrid g(8, 6, 4, 3e-9, 3e-9, 3e-9);
    Material mat = Material::cobalt();

    // Arbitrary non-uniform m
    VectorField3D m(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix*0.3 + iy*0.2;
        m.at(ix,iy,iz) = {std::sin(phi)*std::cos(iz*0.1),
                           std::sin(phi)*std::sin(iz*0.1),
                           std::cos(phi)};
    }

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i=0; i<g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    UniaxialAnisotropyField    cpu;
    UniaxialAnisotropyFieldGPU gpu(g);
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    const double scale = 2.0*mat.K_uniaxial/(constants::mu_0*mat.Ms);
    const double err = max_rel_diff(H_cpu, H_gpu, scale * 1e-4);
    INFO("max_rel_err = " << err);
    REQUIRE(err < micromag::gtol(1e-8));
}

TEST_CASE("UniaxialAnisotropyFieldGPU: additive", "[anisotropy][gpu]") {
    StructuredGrid g(6, 6, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::cobalt();

    VectorField3D m(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix*0.25 + iy*0.1;
        m.at(ix,iy,iz) = {std::cos(phi), std::sin(phi), 0.0};
    }

    VectorField3D H1(g), H2(g);
    for (Index i=0; i<g.size(); ++i) H1[i] = H2[i] = {0,0,0};

    UniaxialAnisotropyFieldGPU gpu(g);
    gpu.accumulate(m, mat, H1);
    gpu.accumulate(m, mat, H2);
    gpu.accumulate(m, mat, H2);   // H2 = 2 × H1

    const double scale = 2.0*mat.K_uniaxial/(constants::mu_0*mat.Ms);
    for (Index i=0; i<g.size(); ++i) {
        if (std::abs(H1[i].x) > scale*1e-6)
            REQUIRE_THAT(H2[i].x, WithinRel(2.0*H1[i].x, 1e-10));
        if (std::abs(H1[i].z) > scale*1e-6)
            REQUIRE_THAT(H2[i].z, WithinRel(2.0*H1[i].z, 1e-10));
    }
}

TEST_CASE("UniaxialAnisotropyFieldGPU: 2nd-order Ku2 matches CPU", "[anisotropy][gpu]") {
    // Validate the GPU 2nd-order uniaxial term H += (4Ku2/μ₀Ms)(m·û)³ û
    // against the CPU reference. Ku2 was previously silently dropped on GPU.
    StructuredGrid g(8, 6, 4, 3e-9, 3e-9, 3e-9);
    Material mat = Material::cobalt();
    mat.K_uniaxial = 5.2e5;
    mat.Ku2        = 1.5e5;            // strong 2nd-order term
    mat.easy_axis  = {0.0, 0.0, 1.0};

    VectorField3D m(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix*0.3 + iy*0.2;
        m.at(ix,iy,iz) = {std::sin(phi)*std::cos(iz*0.1),
                           std::sin(phi)*std::sin(iz*0.1),
                           std::cos(phi)};
    }

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i=0; i<g.size(); ++i) H_cpu[i] = H_gpu[i] = {0,0,0};

    UniaxialAnisotropyField    cpu;
    UniaxialAnisotropyFieldGPU gpu(g);
    cpu.accumulate(m, mat, H_cpu);   // CPU includes Ku2
    gpu.accumulate(m, mat, H_gpu);   // GPU host path (now includes Ku2)

    const double scale = 2.0*mat.K_uniaxial/(constants::mu_0*mat.Ms);
    const double err = max_rel_diff(H_cpu, H_gpu, scale * 1e-4);
    INFO("max_rel_err (host path) = " << err);
    REQUIRE(err < micromag::gtol(1e-7));

    // Sanity: with Ku2=0 the GPU result must differ (proves Ku2 contributes)
    Material mat_k1 = mat; mat_k1.Ku2 = 0.0;
    VectorField3D H_k1(g);
    for (Index i=0; i<g.size(); ++i) H_k1[i] = {0,0,0};
    gpu.accumulate(m, mat_k1, H_k1);
    double maxdiff = 0.0;
    for (Index i=0; i<g.size(); ++i)
        maxdiff = std::max(maxdiff, std::abs(H_gpu[i].z - H_k1[i].z));
    INFO("Ku2 contribution magnitude = " << maxdiff);
    REQUIRE(maxdiff > scale * 1e-3);
}

TEST_CASE("UniaxialAnisotropyFieldGPU energy matches CPU", "[anisotropy][gpu]") {
    StructuredGrid g(6, 6, 4, 5e-9, 5e-9, 5e-9);
    Material mat = Material::cobalt();

    VectorField3D m(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix*0.3;
        m.at(ix,iy,iz) = {std::sin(phi), 0, std::cos(phi)};
    }

    UniaxialAnisotropyField    cpu;
    UniaxialAnisotropyFieldGPU gpu(g);

    const Real E_cpu = cpu.energy(m, mat);
    const Real E_gpu = gpu.energy(m, mat);
    INFO("E_cpu=" << E_cpu << "  E_gpu=" << E_gpu);
    REQUIRE(E_cpu != 0.0);
    REQUIRE_THAT(E_gpu, WithinRel(E_cpu, 1e-10));
}

#endif // MICROMAG_CUDA
