#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <memory>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/material.hpp"
#include "micromag/rkky.hpp"

using namespace micromag;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

static Vec3 mean_H(const VectorField3D& H) {
    Vec3 s{0,0,0};
    for (Index i = 0; i < H.size(); ++i) { s.x+=H[i].x; s.y+=H[i].y; s.z+=H[i].z; }
    const double N = static_cast<double>(H.size());
    return {s.x/N, s.y/N, s.z/N};
}

TEST_CASE("RKKYField: antiferromagnetic J<0 → field opposes m_ref", "[rkky]") {
    // m_ref = +x, J < 0 (antiferromagnetic) → H should point in -x
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();

    VectorField3D m1(g), m2(g);
    m1.set_uniform({1, 0, 0});
    m2.set_uniform({1, 0, 0});

    const Real J = -1e-3;   // antiferromagnetic [J/m²]
    const Real d = 1e-9;
    RKKYField rkky(m2, J, d);   // field on m1 due to m2

    VectorField3D H(g); for (Index i=0;i<H.size();++i) H[i]={0,0,0};
    rkky.accumulate(m1, mat, H);

    // H = J/(mu_0*Ms*d) * m_ref; J<0 → coeff < 0 → H in -x
    Vec3 avg = mean_H(H);
    const double mu_0 = 4 * 3.14159265358979 * 1e-7;
    const double coeff = J / (mu_0 * mat.Ms * d);
    REQUIRE_THAT(avg.x, WithinRel(coeff, 0.001));   // coeff < 0 → H in -x ✓
    REQUIRE_THAT(avg.y, WithinAbs(0.0, std::abs(coeff)*1e-10));
}

TEST_CASE("RKKYField: ferromagnetic J>0 → field aligns with m_ref", "[rkky]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();

    VectorField3D m1(g), m2(g);
    m1.set_uniform({0, 1, 0});   // m2 along +y
    m2.set_uniform({0, 1, 0});

    const Real J = +1e-3;   // ferromagnetic
    const Real d = 1e-9;
    RKKYField rkky(m2, J, d);

    VectorField3D H(g); for (Index i=0;i<H.size();++i) H[i]={0,0,0};
    rkky.accumulate(m1, mat, H);

    Vec3 avg = mean_H(H);
    // J>0 → coeff > 0 → H in +y (aligns m1 with m2) ✓
    REQUIRE(avg.y > 0.0);
    REQUIRE_THAT(avg.x, WithinAbs(0.0, avg.y * 1e-10));
}

TEST_CASE("RKKYField: energy antiparallel < energy parallel (AFM)", "[rkky]") {
    // For antiferromagnetic coupling (J<0), antiparallel has lower energy.
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();

    VectorField3D m_ref(g); m_ref.set_uniform({1, 0, 0});

    const Real J = -1e-3;
    const Real d = 1e-9;
    RKKYField rkky(m_ref, J, d);

    VectorField3D m_par(g);  m_par.set_uniform({ 1, 0, 0});  // parallel
    VectorField3D m_anti(g); m_anti.set_uniform({-1, 0, 0}); // antiparallel

    const double E_par  = rkky.energy(m_par,  mat);
    const double E_anti = rkky.energy(m_anti, mat);
    REQUIRE(E_anti < E_par);   // antiparallel is lower energy ✓
}

TEST_CASE("RKKYField: energy formula E = -J/(2d) * m.m_ref * V", "[rkky]") {
    StructuredGrid g(4, 4, 1, 5e-9, 5e-9, 5e-9);
    const Material mat = Material::permalloy();

    VectorField3D m_ref(g); m_ref.set_uniform({1, 0, 0});
    VectorField3D m(g);     m.set_uniform({1, 0, 0});   // parallel: m.m_ref = 1

    const Real J = -1e-3;
    const Real d = 1e-9;
    RKKYField rkky(m_ref, J, d);

    const double E = rkky.energy(m, mat);
    const double dV = g.cell_volume();
    // E = -J/(2d) * N * dV   (m.m_ref = 1 for all cells)
    const double E_expected = -double(J) / (2.0 * d) * g.size() * dV;
    REQUIRE_THAT(E, WithinRel(E_expected, 0.001));
}

TEST_CASE("RKKYField: J property update", "[rkky]") {
    StructuredGrid g(2, 2, 1, 5e-9, 5e-9, 5e-9);
    VectorField3D m(g); m.set_uniform({1,0,0});
    RKKYField rkky(m, -1e-3, 1e-9);
    REQUIRE_THAT(rkky.J(), WithinAbs(-1e-3, 1e-15));
    rkky.set_J(-2e-3);
    REQUIRE_THAT(rkky.J(), WithinAbs(-2e-3, 1e-15));
}
