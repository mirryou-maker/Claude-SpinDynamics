// Diagnostic Step 1: Verify Newell kernel values WITHOUT FFT.
// Duplicates newell_f/nxx locally so we can call them without modifying DemagField.
// Goal: Is the NEWELL FORMULA itself correct?

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/demag.hpp"
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"

using namespace micromag;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// --- Standalone Newell formulas (copied from demag.cpp for independent verification) ---

static double local_newell_f(double x, double y, double z) {
    double x2=x*x, y2=y*y, z2=z*z;
    double r = std::sqrt(x2+y2+z2);
    if (r == 0.0) return 0.0;
    double val = 0.0;
    double d_xz = std::sqrt(x2+z2);
    if (d_xz > 0.0) val += y*(z2-x2)*0.5*std::asinh(y/d_xz);
    double d_xy = std::sqrt(x2+y2);
    if (d_xy > 0.0) val += z*(y2-x2)*0.5*std::asinh(z/d_xy);
    if (std::abs(x) > 0.0) val -= x*y*z*std::atan(y*z/(x*r));
    val += (2.0*x2-y2-z2)*r/6.0;
    return val;
}

static double local_nxx(double x, double y, double z, double dx, double dy, double dz) {
    double sum = 0.0;
    for (int sx : {-1,1}) for (int sy : {-1,1}) for (int sz : {-1,1}) {
        double cx = x + sx*dx*0.5;
        double cy = y + sy*dy*0.5;
        double cz = z + sz*dz*0.5;
        sum += sx*sy*sz * local_newell_f(cx, cy, cz);
    }
    return -sum / (4.0 * 3.14159265358979323846 * dx * dy * dz);
}

// N_zz(x,y,z) = nxx(z, y, x, dz, dy, dx)
static double local_Nzz(double x, double y, double z, double dx, double dy, double dz) {
    return local_nxx(z, y, x, dz, dy, dx);
}

// --- Tests ---

TEST_CASE("DIAG1-A: nxx(0,0,0) self-term = 0 for cubic cell", "[diag]") {
    double a = 5e-9;
    double val = local_nxx(0, 0, 0, a, a, a);
    INFO("nxx(0,0,0,a,a,a) = " << val << "  (expected exactly 0 by 8-corner cancellation)");
    REQUIRE_THAT(val, WithinAbs(0.0, 1e-30));
}

TEST_CASE("DIAG1-B: nxx adjacent cubic cell along x is negative", "[diag]") {
    double a = 5e-9;
    double val = local_nxx(a, 0, 0, a, a, a);
    INFO("nxx(a,0,0,a,a,a) = " << val << "  (should be negative ~-0.0505)");
    // Known result: for two adjacent unit cubes, Nxx(1,0,0) < 0
    REQUIRE(val < 0.0);
    REQUIRE(val > -0.2);  // sanity bound
}

TEST_CASE("DIAG1-C: Direct sum N_zz over 4x4x4 cube, center cell (1,1,1)", "[diag]") {
    // If Newell formula is correct, this sum should be ~1/3 for a cubic sample.
    double a = 5e-9;
    double Nzz_sum = 0.0;
    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    for (int k = 0; k < 4; k++) {
        Nzz_sum += local_Nzz((1-i)*a, (1-j)*a, (1-k)*a, a, a, a);
    }
    INFO("Direct sum N_zz at (1,1,1) = " << Nzz_sum << "  (expected ~1/3 = " << 1.0/3.0 << ")");
    REQUIRE_THAT(Nzz_sum, WithinRel(1.0/3.0, 0.30));
}

TEST_CASE("DIAG1-D: Direct sum N_zz matches FFT DemagField result", "[diag]") {
    // KEY TEST: compare direct sum (ground truth) vs FFT result.
    // If they differ  -> bug is in the FFT pipeline.
    // If they agree (both wrong) -> bug is in Newell kernel.
    double a = 5e-9;
    StructuredGrid g(4, 4, 4, a, a, a);
    Material mat = Material::permalloy();

    double Nzz_direct = 0.0;
    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    for (int k = 0; k < 4; k++) {
        Nzz_direct += local_Nzz((1-i)*a, (1-j)*a, (1-k)*a, a, a, a);
    }
    double H_z_direct = -Nzz_direct * mat.Ms;

    VectorField3D m(g), H(g);
    m.set_uniform({0, 0, 1});
    for (Index idx = 0; idx < H.size(); ++idx) H[idx] = {0, 0, 0};
    DemagField df(g);
    df.accumulate(m, mat, H);
    double H_z_fft = H.at(1, 1, 1).z;

    INFO("Nzz_direct = " << Nzz_direct
         << "  H_z_direct = " << H_z_direct
         << "  H_z_fft = " << H_z_fft);

    REQUIRE_THAT(H_z_fft, WithinRel(H_z_direct, 0.01));
}
