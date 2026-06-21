// test_magnetoelastic_gpu.cpp — MagnetoelasticFieldGPU vs CPU (uniform strain)
// Tag: [magnetoelastic][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/magnetoelastic.hpp"
#include "micromag/magnetoelastic_gpu.hpp"
#include "micromag/material.hpp"
#include "gpu_test_tol.hpp"

using namespace micromag;
using Catch::Matchers::WithinRel;

static double max_abs_diff_me(const VectorField3D& a, const VectorField3D& b) {
    double mx = 0.0;
    for (Index i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(a[i].x - b[i].x));
        mx = std::max(mx, std::abs(a[i].y - b[i].y));
        mx = std::max(mx, std::abs(a[i].z - b[i].z));
    }
    return mx;
}

TEST_CASE("MagnetoelasticFieldGPU: name + zero strain", "[magnetoelastic][gpu]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    Material mat = Material::permalloy();
    MagnetoelasticFieldGPU me(-62.4e6, -27.1e6, g);
    REQUIRE(std::string(me.name()) == "MagnetoelasticFieldGPU");

    VectorField3D m(g); m.set_uniform({0.6, 0, 0.8}); m.normalize();
    VectorField3D H(g);
    for (Index i = 0; i < g.size(); ++i) H[i] = {0, 0, 0};
    me.accumulate(m, mat, H);   // no strain set → zero
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE(std::abs(H[i].x) < 1e-6);
        REQUIRE(std::abs(H[i].y) < 1e-6);
        REQUIRE(std::abs(H[i].z) < 1e-6);
    }
}

TEST_CASE("MagnetoelasticFieldGPU matches CPU: full strain tensor", "[magnetoelastic][gpu]") {
    StructuredGrid g(8, 6, 4, 3e-9, 3e-9, 3e-9);
    Material mat = Material::cobalt();
    const Real B1 = -10.5e6, B2 = -8.0e6;

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.3 + iy * 0.2;
        m.at(ix, iy, iz) = {std::sin(phi) * std::cos(iz * 0.1),
                            std::sin(phi) * std::sin(iz * 0.1),
                            std::cos(phi)};
    }
    m.normalize();

    MagnetoelasticField    cpu(B1, B2);
    MagnetoelasticFieldGPU gpu(B1, B2, g);
    cpu.set_strain(1e-3, -5e-4, 2e-4, 3e-4, 1e-4, -2e-4);
    gpu.set_strain(1e-3, -5e-4, 2e-4, 3e-4, 1e-4, -2e-4);

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0, 0, 0};
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    // Field magnitude scale for relative tolerance
    const double scale = 2.0 * std::abs(B1) / (constants::mu_0 * mat.Ms) * 1e-3;
    const double err = max_abs_diff_me(H_cpu, H_gpu);
    INFO("max |GPU - CPU| = " << err << "  scale = " << scale);
    REQUIRE(err < scale * micromag::gtol(1e-4));
}

TEST_CASE("MagnetoelasticFieldGPU energy matches CPU", "[magnetoelastic][gpu]") {
    StructuredGrid g(6, 6, 4, 4e-9, 4e-9, 4e-9);
    Material mat = Material::permalloy();
    const Real B1 = 6.96e6, B2 = -3.43e6;

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.25;
        m.at(ix, iy, iz) = {std::sin(phi), 0, std::cos(phi)};
    }

    MagnetoelasticField    cpu(B1, B2);
    MagnetoelasticFieldGPU gpu(B1, B2, g);
    cpu.set_strain(1e-3, 5e-4, -2e-4, 1e-4, 0, 0);
    gpu.set_strain(1e-3, 5e-4, -2e-4, 1e-4, 0, 0);

    const Real E_cpu = cpu.energy(m, mat);
    const Real E_gpu = gpu.energy(m, mat);
    INFO("E_cpu=" << E_cpu << "  E_gpu=" << E_gpu);
    REQUIRE(E_cpu != 0.0);
    REQUIRE_THAT(E_gpu, WithinRel(E_cpu, 1e-6));
}

#endif // MICROMAG_CUDA
