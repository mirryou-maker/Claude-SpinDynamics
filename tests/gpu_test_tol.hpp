#pragma once
// gpu_test_tol.hpp — precision-aware tolerance helper for GPU tests.
//
// The GPU test suite asserts agreement with double-precision CPU references.
// When the library is built with MICROMAG_FLOAT32=ON (P11), GPU kernels run in
// single precision (~1e-7 relative epsilon), so tolerances tuned for double
// (1e-10 … 1e-15) can no longer be met. gtol() keeps the original strict
// tolerance for the default double build and substitutes a relaxed one only in
// the float32 build, so each build stays as strict as its precision allows.
//
//   REQUIRE_THAT(norm, WithinAbs(1.0, gtol(1e-10)));          // f32 → 1e-5
//   REQUIRE_THAT(max_rel_diff(...), WithinAbs(0.0, gtol(1e-6, 5e-2)));

#include "micromag/gpu_real.hpp"

namespace micromag {

// f64_tol: tolerance used in the default (double) build.
// f32_tol: tolerance used when MICROMAG_FLOAT32=ON (defaults to 1e-5, the
//          typical single-precision relative floor for these kernels).
constexpr double gtol(double f64_tol, double f32_tol = 1e-5) {
#ifdef MICROMAG_FLOAT32
    (void)f64_tol;
    return f32_tol;
#else
    (void)f32_tol;
    return f64_tol;
#endif
}

}  // namespace micromag
