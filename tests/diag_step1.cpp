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
    x = std::abs(x); y = std::abs(y); z = std::abs(z);   // 6D formula requires abs
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

// 6D double-cell integral: corners at (n+ia-id)*dx, sign=(-1)^(ia+ib+ic+id+ie+ig)
static double local_nxx(double x, double y, double z, double dx, double dy, double dz) {
    const int nx = static_cast<int>(std::round(x / dx));
    const int ny = static_cast<int>(std::round(y / dy));
    const int nz = static_cast<int>(std::round(z / dz));
    double sum = 0.0;
    for (int ia : {0,1}) for (int ib : {0,1}) for (int ic : {0,1})
    for (int id : {0,1}) for (int ie : {0,1}) for (int ig : {0,1}) {
        const int sign = ((ia+ib+ic+id+ie+ig) % 2 == 0) ? 1 : -1;
        sum += sign * local_newell_f(
            (nx+ia-id)*dx, (ny+ib-ie)*dy, (nz+ic-ig)*dz);
    }
    return +sum / (4.0 * 3.14159265358979323846 * dx * dy * dz);
}

// N_zz(x,y,z) = nxx(z, y, x, dz, dy, dx)
static double local_Nzz(double x, double y, double z, double dx, double dy, double dz) {
    return local_nxx(z, y, x, dz, dy, dx);
}

// --- Tests ---

TEST_CASE("DIAG1-A: nxx(0,0,0) self-term = 1/3 for cubic cell", "[diag]") {
    // With the 6D double-cell integral the self-demagnetisation factor of
    // a cubic cell is exactly 1/3 (isotropic).
    double a = 5e-9;
    double val = local_nxx(0, 0, 0, a, a, a);
    INFO("nxx(0,0,0,a,a,a) = " << val << "  (expected 1/3 = " << 1.0/3.0 << ")");
    REQUIRE_THAT(val, WithinRel(1.0/3.0, 0.01));
}

TEST_CASE("DIAG1-B: nxx adjacent cubic cell along x is negative", "[diag]") {
    // nxx(a,0,0): contribution of a z-axis-aligned source cell at (a,0,0)
    // to K_zz at origin.  The z-component of a z-dipole field along its
    // own axis is positive (H_z > 0), which reduces the demagnetising effect,
    // so K_zz < 0 for this neighbour.  Analytic 6D value ≈ -0.135.
    double a = 5e-9;
    double val = local_nxx(a, 0, 0, a, a, a);
    INFO("nxx(a,0,0,a,a,a) = " << val << "  (expected negative ~-0.135)");
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
