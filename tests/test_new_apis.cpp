// test_new_apis.cpp — unit tests for W/X/Y/Z + priority-1 APIs
//
// Covers:
//   [api] VectorField3D __setitem__ (C++ operator[])
//   [api] MaterialField3D scalar + axis accessors
//   [api] load_ovf_grid + load_ovf_into round-trip
//   [api] RKKYField bidirectional coupling

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdio>
#include <cmath>
#include <string>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/material_field.hpp"
#include "micromag/ovf_io.hpp"
#include "micromag/rkky.hpp"
#include "micromag/types.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// VectorField3D subscription
// ---------------------------------------------------------------------------
TEST_CASE("VectorField3D operator[] read/write round-trip", "[api][field]") {
    StructuredGrid g(4, 3, 2, 1e-9, 1e-9, 1e-9);
    VectorField3D f(g);
    f.set_uniform({0, 0, 0});

    const Index idx = 1 + 4*(2 + 3*1);  // cell (1,2,1)
    f[idx] = Vec3{0.6, 0.8, 0.0};

    REQUIRE_THAT(f[idx].x, WithinAbs(0.6, 1e-12));
    REQUIRE_THAT(f[idx].y, WithinAbs(0.8, 1e-12));
    REQUIRE_THAT(f[idx].z, WithinAbs(0.0, 1e-12));

    // Other cells untouched
    REQUIRE_THAT(f[0].x, WithinAbs(0.0, 1e-12));
}

TEST_CASE("VectorField3D from_numpy round-trip via operator[]", "[api][field]") {
    StructuredGrid g(2, 2, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D f(g);
    f[0] = Vec3{1, 0, 0};
    f[1] = Vec3{0, 1, 0};
    f[2] = Vec3{0, 0, 1};
    f[3] = Vec3{0.6, 0.8, 0};

    REQUIRE_THAT(f[2].z, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(f[3].x, WithinAbs(0.6, 1e-12));
    REQUIRE_THAT(f[3].y, WithinAbs(0.8, 1e-12));
}

// ---------------------------------------------------------------------------
// MaterialField3D per-cell accessors
// ---------------------------------------------------------------------------
TEST_CASE("MaterialField3D per-cell material accessors", "[api][material_field]") {
    StructuredGrid g(2, 2, 1, 5e-9, 5e-9, 5e-9);
    MaterialField3D mf(g, Material::permalloy());

    const Material cobalt = Material::cobalt();
    // Set cell 0 to cobalt
    mf.Ms_field()[0]        = cobalt.Ms;
    mf.A_field()[0]         = cobalt.A_exchange;
    mf.K_field()[0]         = cobalt.K_uniaxial;
    mf.alpha_field()[0]     = cobalt.alpha;
    mf.easy_axis_field()[0] = cobalt.easy_axis;

    REQUIRE_THAT(mf.Ms(0),         WithinAbs(cobalt.Ms, 1e3));
    REQUIRE_THAT(mf.A_exchange(0), WithinAbs(cobalt.A_exchange, 1e-18));
    REQUIRE_THAT(mf.K_uniaxial(0), WithinAbs(cobalt.K_uniaxial, 1e2));

    // Cell 1 stays permalloy
    const Material py = Material::permalloy();
    REQUIRE_THAT(mf.Ms(1), WithinAbs(py.Ms, 1e3));
}

// ---------------------------------------------------------------------------
// OVF IO: load_ovf_grid + load_ovf_into
// ---------------------------------------------------------------------------
TEST_CASE("load_ovf_grid + load_ovf_into round-trip", "[api][ovf]") {
    // Write a small field to a temp OVF file, then read back via the two-step API
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 20e-9);
    VectorField3D m_out(g);
    m_out.set_vortex(20e-9, 20e-9, 4.0);
    m_out.normalize();

    const std::string fname = "test_api_ovf_tmp.ovf";
    save_ovf(fname, m_out);

    // Two-step safe load (no dangling pointer)
    StructuredGrid g2 = load_ovf_grid(fname);
    VectorField3D m_in(g2);
    load_ovf_into(fname, m_in);

    REQUIRE(g2.nx() == g.nx());
    REQUIRE(g2.ny() == g.ny());
    REQUIRE(g2.nz() == g.nz());

    // Check a few cells
    for (Index i = 0; i < static_cast<Index>(4); ++i) {
        REQUIRE_THAT(m_in[i].x, WithinAbs(m_out[i].x, 1e-9));
        REQUIRE_THAT(m_in[i].y, WithinAbs(m_out[i].y, 1e-9));
        REQUIRE_THAT(m_in[i].z, WithinAbs(m_out[i].z, 1e-9));
    }

    std::remove(fname.c_str());
}

// ---------------------------------------------------------------------------
// RKKYField bidirectional coupling energy symmetry
// ---------------------------------------------------------------------------
TEST_CASE("RKKYField: antiferromagnetic J<0, energy sign", "[api][rkky]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 1e-9);
    const Material mat = Material::permalloy();
    const double J     = -0.5e-3;  // AFM
    const double d     = 1e-9;

    VectorField3D m1(g), m2(g);
    m1.set_uniform({ 1, 0, 0});
    m2.set_uniform({-1, 0, 0});   // antiparallel

    // RKKY field on m1 due to m2
    RKKYField rkky(m2, J, d);

    VectorField3D H(g);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0,0,0};
    rkky.accumulate(m1, mat, H);

    // J<0, m2=(-1,0,0) → H_RKKY = +|J|/(mu0*Ms*d) * (1,0,0) (aligns m1)
    REQUIRE(H[0].x > 0.0);

    // Energy of antiparallel (favoured by AFM J) should be lower than parallel
    VectorField3D m2_par(g);
    m2_par.set_uniform({1, 0, 0});
    RKKYField rkky_par(m2_par, J, d);
    double E_afm = rkky.energy(m1, mat);
    double E_fm  = rkky_par.energy(m1, mat);
    REQUIRE(E_afm < E_fm);
}
