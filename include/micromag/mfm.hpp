#pragma once

// mfm.hpp — MFM (Magnetic Force Microscopy) image simulation
//
// Computes the MFM signal at a given lift height h above the sample surface
// using Fourier-space propagation of the magnetic stray field.
//
// Algorithm (2D R2C FFT, no zero-padding — same conventions as DemagFieldPeriodic):
//   1. Sum mz over all z layers: σ(x,y) = Ms * dz * sum_z(m.z)  [surface charge]
//   2. Forward 2D R2C FFT of σ → σ_hat(kx,ky)
//   3. Multiply by tip kernel K(k, h):
//        Monopole: K = exp(-|k|·h)          [∝ Hz at height h]
//        Dipole:   K = |k|·exp(-|k|·h)      [∝ ∂Hz/∂z at height h]
//      k=0 mode → 0 (uniform magnetisation has no stray field)
//   4. Inverse FFT + normalise → real-space MFM signal (nx×ny)
//
// Usage:
//   MFMImage mfm(grid, 50e-9);              // dipole tip, 50 nm lift
//   auto signal = mfm.compute(m, mat);      // returns vector<Real> size nx*ny
//   // signal[iy * nx + ix] is the MFM signal at cell (ix, iy)

#include <complex>
#include <vector>

#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

struct fftw_plan_s;

namespace micromag {

enum class TipMode { Monopole, Dipole };

class MFMImage {
public:
    // lift_m : tip lift height above the top sample surface [m]
    // tip    : Monopole ∝ Hz(h),  Dipole ∝ ∂Hz/∂z(h)  (default: Dipole)
    MFMImage(const StructuredGrid& grid,
             Real lift_m,
             TipMode tip = TipMode::Dipole);
    ~MFMImage();

    // Non-copyable (owns FFTW plans).
    MFMImage(const MFMImage&)            = delete;
    MFMImage& operator=(const MFMImage&) = delete;

    // Compute MFM signal map.
    // Returns a flat vector of size nx*ny in row-major order: index = iy*nx + ix.
    // Units: proportional to Hz [A/m] (monopole) or ∂Hz/∂z [A/m²] (dipole).
    std::vector<Real> compute(const VectorField3D& m,
                              const Material& mat) const;

    Real    lift() const { return lift_m_; }
    TipMode tip()  const { return tip_; }

private:
    Index  nx_, ny_, nz_;
    Real   dx_, dy_, dz_;
    Real   lift_m_;
    TipMode tip_;
    Index  fft_nx_;   // = nx/2 + 1  (R2C half-range in x)

    // Precomputed real-valued kernel K(kx, ky) stored at R2C output positions.
    // Size: ny * fft_nx
    std::vector<Real> kernel_;

    // Scratch buffers reused each compute() call.
    mutable std::vector<double>               r_buf_;  // ny * nx  real
    mutable std::vector<std::complex<double>> c_buf_;  // ny * fft_nx  complex

    fftw_plan_s* plan_fwd_ = nullptr;
    fftw_plan_s* plan_inv_ = nullptr;

    void precompute_kernel();
};

}  // namespace micromag
