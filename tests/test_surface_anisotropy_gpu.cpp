// test_surface_anisotropy_gpu.cpp — SurfaceAnisotropyFieldGPU vs CPU
// Tag: [surface_anisotropy][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/surface_anisotropy.hpp"
#include "micromag/surface_anisotropy_gpu.hpp"
#include "gpu_test_tol.hpp"

using namespace micromag;
using Catch::Matchers::WithinRel;

static double max_abs_diff_sa(const VectorField3D& a, const VectorField3D& b) {
    double mx = 0.0;
    for (Index i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(a[i].x - b[i].x));
        mx = std::max(mx, std::abs(a[i].y - b[i].y));
        mx = std::max(mx, std::abs(a[i].z - b[i].z));
    }
    return mx;
}

TEST_CASE("SurfaceAnisotropyFieldGPU: name", "[surface_anisotropy][gpu]") {
    StructuredGrid g(3, 3, 4, 5e-9, 5e-9, 2e-9);
    SurfaceAnisotropyFieldGPU sa(1.2e-3, g);
    REQUIRE(std::string(sa.name()) == "SurfaceAnisotropyFieldGPU");
}

TEST_CASE("SurfaceAnisotropyFieldGPU matches CPU: z-normal", "[surface_anisotropy][gpu]") {
    StructuredGrid g(8, 6, 4, 4e-9, 4e-9, 2e-9);
    Material mat = Material::permalloy();
    const Real Ks = 1.2e-3;

    VectorField3D m(g);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        double phi = ix * 0.3 + iy * 0.2;
        m.at(ix, iy, iz) = {std::sin(phi) * 0.5, std::cos(phi) * 0.3, std::cos(phi)};
    }
    m.normalize();

    SurfaceAnisotropyField    cpu(Ks);
    SurfaceAnisotropyFieldGPU gpu(Ks, g);

    VectorField3D H_cpu(g), H_gpu(g);
    for (Index i = 0; i < g.size(); ++i) H_cpu[i] = H_gpu[i] = {0, 0, 0};
    cpu.accumulate(m, mat, H_cpu);
    gpu.accumulate(m, mat, H_gpu);

    const double scale = 2.0 * Ks / (constants::mu_0 * mat.Ms * g.dz());
    const double err = max_abs_diff_sa(H_cpu, H_gpu);
    INFO("max |GPU - CPU| = " << err << "  scale = " << scale);
    REQUIRE(err < scale * micromag::gtol(1e-4));
}

TEST_CASE("SurfaceAnisotropyFieldGPU energy matches CPU", "[surface_anisotropy][gpu]") {
    StructuredGrid g(6, 6, 4, 4e-9, 4e-9, 2e-9);
    Material mat = Material::permalloy();
    const Real Ks = 9.5e-4;

    VectorField3D m(g); m.set_uniform({0.2, 0.0, 0.98}); m.normalize();

    SurfaceAnisotropyField    cpu(Ks);
    SurfaceAnisotropyFieldGPU gpu(Ks, g);

    const Real E_cpu = cpu.energy(m, mat);
    const Real E_gpu = gpu.energy(m, mat);
    INFO("E_cpu=" << E_cpu << "  E_gpu=" << E_gpu);
    REQUIRE(E_cpu != 0.0);
    REQUIRE_THAT(E_gpu, WithinRel(E_cpu, 1e-6));
}

#endif // MICROMAG_CUDA
