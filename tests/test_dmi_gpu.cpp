// test_dmi_gpu.cpp — BulkDMIFieldGPU + InterfacialDMIFieldGPU accuracy tests.
// Compares GPU output against CPU BulkDMIField / InterfacialDMIField.
// Tag: [dmi][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/dmi.hpp"
#include "micromag/dmi_gpu.hpp"
#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static Material mat_py() {
    Material m = Material::permalloy();
    m.K_uniaxial = 0.0;
    return m;
}

// Max component-wise relative error, floor at tol_abs
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
// Bulk DMI
// ---------------------------------------------------------------------------

TEST_CASE("BulkDMIFieldGPU: uniform m -> zero field", "[dmi][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 5e-9);
    auto mat = mat_py();
    const Real D = 1e-3;
    VectorField3D m(g); m.set_uniform({0, 0, 1});
    VectorField3D H(g); for (Index i=0; i<H.size(); ++i) H[i]={0,0,0};

    BulkDMIFieldGPU gpu(g, D);
    gpu.accumulate(m, mat, H);

    for (Index i = 0; i < H.size(); ++i)
        REQUIRE(H[i].norm() < 1.0);   // A/m tolerance (should be ~0)
}

TEST_CASE("BulkDMIFieldGPU: matches CPU (spiral along x)", "[dmi][gpu]") {
    StructuredGrid g(12, 6, 4, 5e-9, 5e-9, 5e-9);
    auto mat = mat_py();
    const Real D = 2e-3;

    VectorField3D m(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix * 0.25;
        m.at(ix,iy,iz) = {std::cos(phi), std::sin(phi), 0};
    }

    VectorField3D Hc(g), Hg(g);
    for (Index i=0; i<g.size(); ++i) Hc[i]=Hg[i]={0,0,0};

    BulkDMIField    cpu(D);
    BulkDMIFieldGPU gpu(g, D);
    cpu.accumulate(m, mat, Hc);
    gpu.accumulate(m, mat, Hg);

    const double tol = mat.Ms * 1e-4;
    REQUIRE_THAT(max_rel_diff(Hc, Hg, tol), WithinAbs(0.0, 1e-6));
}

TEST_CASE("BulkDMIFieldGPU: matches CPU (3-D vortex)", "[dmi][gpu]") {
    StructuredGrid g(8, 8, 6, 5e-9, 5e-9, 4e-9);
    auto mat = mat_py();
    const Real D = 1.5e-3;

    VectorField3D m(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi_x = ix * 0.2, phi_z = iz * 0.15;
        m.at(ix,iy,iz) = {std::cos(phi_x), std::sin(phi_z), std::sin(phi_x)*std::cos(phi_z)};
        m.at(ix,iy,iz) = m.at(ix,iy,iz) / m.at(ix,iy,iz).norm();
    }

    VectorField3D Hc(g), Hg(g);
    for (Index i=0; i<g.size(); ++i) Hc[i]=Hg[i]={0,0,0};

    BulkDMIField    cpu(D);
    BulkDMIFieldGPU gpu(g, D);
    cpu.accumulate(m, mat, Hc);
    gpu.accumulate(m, mat, Hg);

    REQUIRE_THAT(max_rel_diff(Hc, Hg, mat.Ms*1e-4), WithinAbs(0.0, 1e-6));
}

TEST_CASE("BulkDMIFieldGPU: D=0 -> no accumulation", "[dmi][gpu]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    auto mat = mat_py();
    VectorField3D m(g); m.set_uniform({1,0,0});
    VectorField3D H(g); for (Index i=0; i<H.size(); ++i) H[i]={1e6,2e6,3e6};

    VectorField3D H_ref(g);
    for (Index i=0; i<H_ref.size(); ++i) H_ref[i]={1e6,2e6,3e6};

    BulkDMIFieldGPU gpu(g, 0.0);
    gpu.accumulate(m, mat, H);

    // H should be unchanged
    REQUIRE_THAT(max_rel_diff(H_ref, H, 1e3), WithinAbs(0.0, 1e-10));
}

// ---------------------------------------------------------------------------
// Interfacial DMI
// ---------------------------------------------------------------------------

TEST_CASE("InterfacialDMIFieldGPU: uniform m -> zero field", "[dmi][gpu]") {
    StructuredGrid g(8, 8, 1, 5e-9, 5e-9, 1e-9);
    auto mat = mat_py();
    VectorField3D m(g); m.set_uniform({0, 0, 1});
    VectorField3D H(g); for (Index i=0; i<H.size(); ++i) H[i]={0,0,0};

    InterfacialDMIFieldGPU gpu(g, 2e-3);
    gpu.accumulate(m, mat, H);

    double hmax = 0;
    for (Index i=0; i<H.size(); ++i)
        hmax = std::max({hmax, std::abs(H[i].x), std::abs(H[i].y), std::abs(H[i].z)});
    REQUIRE_THAT(hmax, WithinAbs(0.0, 1.0));
}

TEST_CASE("InterfacialDMIFieldGPU: matches CPU (thin film Neel spiral)", "[dmi][gpu]") {
    StructuredGrid g(12, 10, 1, 5e-9, 5e-9, 1e-9);
    auto mat = mat_py();
    const Real D = 2.5e-3;

    VectorField3D m(g);
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix * 0.3 + iy * 0.2;
        m.at(ix,iy,0) = {std::cos(phi)*0.5, std::sin(phi)*0.5,
                         std::cos(0.5*phi)};
        m.at(ix,iy,0) = m.at(ix,iy,0) / m.at(ix,iy,0).norm();
    }

    VectorField3D Hc(g), Hg(g);
    for (Index i=0; i<g.size(); ++i) Hc[i]=Hg[i]={0,0,0};

    InterfacialDMIField    cpu(D);
    InterfacialDMIFieldGPU gpu(g, D);
    cpu.accumulate(m, mat, Hc);
    gpu.accumulate(m, mat, Hg);

    REQUIRE_THAT(max_rel_diff(Hc, Hg, mat.Ms*1e-4), WithinAbs(0.0, 1e-6));
}

TEST_CASE("InterfacialDMIFieldGPU: matches CPU (3D body, z-gradient ignored)", "[dmi][gpu]") {
    StructuredGrid g(8, 8, 4, 5e-9, 5e-9, 3e-9);
    auto mat = mat_py();
    const Real D = 1.8e-3;

    VectorField3D m(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix*0.25 + iy*0.15;
        m.at(ix,iy,iz) = {std::sin(phi), 0, std::cos(phi)};
    }

    VectorField3D Hc(g), Hg(g);
    for (Index i=0; i<g.size(); ++i) Hc[i]=Hg[i]={0,0,0};

    InterfacialDMIField    cpu(D);
    InterfacialDMIFieldGPU gpu(g, D);
    cpu.accumulate(m, mat, Hc);
    gpu.accumulate(m, mat, Hg);

    REQUIRE_THAT(max_rel_diff(Hc, Hg, mat.Ms*1e-4), WithinAbs(0.0, 1e-6));
}

TEST_CASE("InterfacialDMIFieldGPU: additivity", "[dmi][gpu]") {
    StructuredGrid g(6, 6, 1, 5e-9, 5e-9, 1e-9);
    auto mat = mat_py();
    const Real D = 2e-3;

    VectorField3D m(g);
    for (Index ix=0; ix<g.nx(); ++ix)
    for (Index iy=0; iy<g.ny(); ++iy) {
        double phi = ix*0.3;
        m.at(ix,iy,0) = {std::cos(phi), std::sin(phi), 0};
    }

    VectorField3D H1(g), H2(g);
    for (Index i=0; i<g.size(); ++i) H1[i]=H2[i]={0,0,0};

    InterfacialDMIFieldGPU gpu(g, D);
    gpu.accumulate(m, mat, H1);
    gpu.accumulate(m, mat, H2);
    gpu.accumulate(m, mat, H2);  // H2 = 2*H1

    for (Index i=0; i<g.size(); ++i) {
        if (std::abs(H1[i].x) > 1e3)
            REQUIRE_THAT(H2[i].x, WithinRel(2.0*H1[i].x, 1e-9));
        if (std::abs(H1[i].z) > 1e3)
            REQUIRE_THAT(H2[i].z, WithinRel(2.0*H1[i].z, 1e-9));
    }
}

#endif // MICROMAG_CUDA
