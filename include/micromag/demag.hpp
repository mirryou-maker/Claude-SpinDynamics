#pragma once

#include <complex>
#include <vector>

#include "effective_field.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

// Forward-declare FFTW plan handle to avoid including fftw3.h in headers.
struct fftw_plan_s;

namespace micromag {

// FFT-based demagnetization field using the Newell (1993) analytical tensor.
//
// H_demag = -N * M  (SI, H in A/m)
// where N is the 3x3 symmetric demagnetization tensor computed via zero-padded
// FFT convolution.  Kernel is precomputed once at construction; runtime cost
// is 6 forward FFTs + 6 pointwise products + 3 inverse FFTs per step.
class DemagField : public IEffectiveField {
public:
    explicit DemagField(const StructuredGrid& grid);
    ~DemagField();

    // Non-copyable (owns FFTW plans).
    DemagField(const DemagField&)            = delete;
    DemagField& operator=(const DemagField&) = delete;

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "DemagField"; }

private:
    // Grid geometry (stored for convenience).
    Index nx_, ny_, nz_;   // real-space dimensions
    Real  dx_, dy_, dz_;
    Index pad_nx_, pad_ny_, pad_nz_;   // zero-padded FFT dimensions
    Index fft_nx_;                      // pad_nx_/2+1 (r2c output)

    // Precomputed demag kernel in frequency space.
    // 6 independent components: xx, yy, zz, xy, xz, yz.
    // Size: pad_nz_ * pad_ny_ * fft_nx_
    std::vector<std::complex<double>> K_xx_, K_yy_, K_zz_;
    std::vector<std::complex<double>> K_xy_, K_xz_, K_yz_;

    // Scratch arrays for forward/inverse transforms (mutable for const accumulate).
    mutable std::vector<double>              r_buf_;    // real input  (padded)
    mutable std::vector<std::complex<double>> c_buf_;   // complex FFT (padded r2c)

    // FFTW plans (forward r2c and inverse c2r, reused each call).
    fftw_plan_s* plan_fwd_ = nullptr;
    fftw_plan_s* plan_inv_ = nullptr;

    // Helpers -----------------------------------------------------------------
    void precompute_kernel();

    // Newell (1993) analytical formulas for N_xx, N_xy components of a cuboid.
    static double newell_f(double x, double y, double z);
    static double newell_g(double x, double y, double z);
    static double nxx(double x, double y, double z, double dx, double dy, double dz);
    static double nxy(double x, double y, double z, double dx, double dy, double dz);
};

}  // namespace micromag
