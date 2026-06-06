#pragma once

// demag_periodic.hpp — DemagFieldPeriodic: FFT demag with fully-periodic BC
//
// Implements the magnetostatic demag field for a simulation cell with
// periodic boundary conditions in all three dimensions.  The cell is
// treated as one tile in an infinite 3-D lattice.
//
// Algorithm
// ---------
// 1. Kernel precomputation (once, in constructor):
//      N_αβ^periodic(r) = Σ_{n} N_αβ^Newell(r + n·L)
//    where n = (n₁,n₂,n₃) ranges over ±n_rep image cells per side and
//    L = (nx·dx, ny·dy, nz·dz) is the period.
//    The FFT of this periodic kernel is stored in K_αβ (complex, half-range).
//    k = 0 mode is set to zero (standard periodic-BC convention: the uniform
//    magnetisation component produces no demag field).
//
// 2. accumulate() at runtime:
//      FFT(Mx,My,Mz) → pointwise Ĥ_α = -Σ_β K̂_αβ M̂_β → IFFT → H_out
//    No zero-padding; FFT size equals grid size → 8× smaller FFT than OBC.
//
// Comparison with DemagField (open BC)
// -------------------------------------
// DemagField  : padded to 2N (linear convolution via circular)
// DemagFieldPeriodic: no padding (genuinely circular convolution)
//
// Physical meaning
// -----------------
// Uniform magnetisation → H_demag = 0  (k=0 zeroed)
// Non-uniform modes    → identical physics to open BC for those wave vectors
//
// Parameters
// ----------
// n_rep  : image cells per side (each dimension).
//          n_rep=2 → 5³=125 images, sufficient for most grids.
//          Increase for small cells (nx<8) if accuracy matters.

#include <complex>
#include <vector>

#include "effective_field.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "types.hpp"

// Forward-declare FFTW plan type without including fftw3.h in the header.
struct fftw_plan_s;

namespace micromag {

class DemagFieldPeriodic : public IEffectiveField {
public:
    explicit DemagFieldPeriodic(const StructuredGrid& grid, int n_rep = 2);
    ~DemagFieldPeriodic();

    DemagFieldPeriodic(const DemagFieldPeriodic&)            = delete;
    DemagFieldPeriodic& operator=(const DemagFieldPeriodic&) = delete;

    void       accumulate(const VectorField3D& m, const Material& mat,
                           VectorField3D& H_out) const override;
    Real       energy    (const VectorField3D& m, const Material& mat) const override;
    const char* name     () const override { return "DemagPeriodic"; }

private:
    Index  nx_, ny_, nz_;
    double dx_, dy_, dz_;
    int    n_rep_;
    Index  fft_nx_;   // = nx/2 + 1  (r2c half-range)

    // Precomputed kernel in frequency space [fft_nx × ny × nz]
    std::vector<std::complex<double>> K_xx_, K_yy_, K_zz_;
    std::vector<std::complex<double>> K_xy_, K_xz_, K_yz_;

    // Scratch buffers (shared between accumulate calls via mutable)
    mutable std::vector<double>               r_buf_;
    mutable std::vector<std::complex<double>> c_buf_;

    fftw_plan_s* plan_fwd_ = nullptr;
    fftw_plan_s* plan_inv_ = nullptr;

    // Newell (1993) helpers
    static double newell_f(double x, double y, double z);
    static double newell_g(double x, double y, double z);
    double nxx(double x, double y, double z,
               double dx, double dy, double dz) const;
    double nxy(double x, double y, double z,
               double dx, double dy, double dz) const;

    void precompute_kernel();
};

}  // namespace micromag
