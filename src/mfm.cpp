#include "micromag/mfm.hpp"

#include <cmath>
#include <stdexcept>
#include <thread>

#include <fftw3.h>

#include "micromag/types.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// Constructor — allocate buffers, create FFTW plans, precompute kernel.
// ---------------------------------------------------------------------------

MFMImage::MFMImage(const StructuredGrid& grid, Real lift_m, TipMode tip)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      lift_m_(lift_m), tip_(tip),
      fft_nx_(nx_ / 2 + 1)
{
    const std::size_t real_size    = static_cast<std::size_t>(nx_ * ny_);
    const std::size_t complex_size = static_cast<std::size_t>(fft_nx_ * ny_);

    kernel_.resize(complex_size, Real{0});
    r_buf_.resize(real_size, 0.0);
    c_buf_.resize(complex_size);

    { static bool s_inited = false;
      if (!s_inited) { fftw_init_threads(); s_inited = true; }
      fftw_plan_with_nthreads(static_cast<int>(std::thread::hardware_concurrency())); }

    fftw_import_wisdom_from_filename("fftw_wisdom.dat");

    // 2D R2C and C2R plans on (ny, nx) — x is fastest (last argument in FFTW).
    const unsigned flags = FFTW_MEASURE;
    plan_fwd_ = reinterpret_cast<fftw_plan_s*>(
        fftw_plan_dft_r2c_2d(
            static_cast<int>(ny_), static_cast<int>(nx_),
            r_buf_.data(),
            reinterpret_cast<fftw_complex*>(c_buf_.data()),
            flags));
    plan_inv_ = reinterpret_cast<fftw_plan_s*>(
        fftw_plan_dft_c2r_2d(
            static_cast<int>(ny_), static_cast<int>(nx_),
            reinterpret_cast<fftw_complex*>(c_buf_.data()),
            r_buf_.data(),
            flags));

    if (!plan_fwd_ || !plan_inv_)
        throw std::runtime_error("MFMImage: FFTW plan creation failed");

    fftw_export_wisdom_to_filename("fftw_wisdom.dat");

    precompute_kernel();
}

MFMImage::~MFMImage() {
    if (plan_fwd_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_fwd_));
    if (plan_inv_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_inv_));
}

// ---------------------------------------------------------------------------
// precompute_kernel
//
// For each k-point (kx, ky) in the R2C output:
//   |k| = sqrt(kx^2 + ky^2)
//
//   Monopole: K = exp(-|k| * h)
//   Dipole:   K = |k| * exp(-|k| * h)
//   k = 0:    K = 0  (uniform mz has no stray field above the surface)
//
// k-vector conventions (FFTW R2C on (ny, nx), x-fastest):
//   kx[ix] = 2π * ix / (nx * dx)           for ix = 0 .. nx/2
//   ky[iy] = 2π * iy / (ny * dy)           for iy = 0 .. ny/2
//          = 2π * (iy - ny) / (ny * dy)    for iy > ny/2
// ---------------------------------------------------------------------------

void MFMImage::precompute_kernel() {
    const Real two_pi_over_Lx =
        Real{2} * constants::pi / (static_cast<Real>(nx_) * dx_);
    const Real two_pi_over_Ly =
        Real{2} * constants::pi / (static_cast<Real>(ny_) * dy_);

    for (Index iy = 0; iy < ny_; ++iy) {
        const Real ky = (iy <= ny_ / 2)
            ? static_cast<Real>(iy)        * two_pi_over_Ly
            : static_cast<Real>(iy - ny_)  * two_pi_over_Ly;

        for (Index ix = 0; ix < fft_nx_; ++ix) {
            const Real kx = static_cast<Real>(ix) * two_pi_over_Lx;
            const Real k  = std::sqrt(kx * kx + ky * ky);

            Real val = Real{0};
            if (k > Real{0}) {
                const Real prop = std::exp(-k * lift_m_);
                val = (tip_ == TipMode::Monopole) ? prop : k * prop;
            }
            kernel_[static_cast<std::size_t>(iy * fft_nx_ + ix)] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// compute — returns flat vector [iy * nx + ix] of MFM signal values.
// ---------------------------------------------------------------------------

std::vector<Real> MFMImage::compute(const VectorField3D& m,
                                     const Material& mat) const {
    const std::size_t N2  = static_cast<std::size_t>(nx_ * ny_);
    const std::size_t Nc  = static_cast<std::size_t>(fft_nx_ * ny_);

    // Step 1: build effective surface charge σ(x,y) = Ms * dz * Σ_z mz(x,y,z).
    // This is the integrated z-magnetization weighted by the cell height.
    std::fill(r_buf_.begin(), r_buf_.end(), 0.0);
    for (Index iz = 0; iz < nz_; ++iz)
    for (Index iy = 0; iy < ny_; ++iy)
    for (Index ix = 0; ix < nx_; ++ix)
        r_buf_[static_cast<std::size_t>(iy * nx_ + ix)] += m.at(ix, iy, iz).z;

    // Prefactor: Ms * dz / 2 (from the Green's function for a magnetic surface charge).
    const double pre = mat.Ms * static_cast<double>(dz_) * 0.5;
    for (std::size_t i = 0; i < N2; ++i)
        r_buf_[i] *= pre;

    // Step 2: forward 2D R2C FFT.
    fftw_execute(reinterpret_cast<fftw_plan>(plan_fwd_));

    // Step 3: multiply by precomputed real kernel in k-space.
    for (std::size_t i = 0; i < Nc; ++i)
        c_buf_[i] *= static_cast<double>(kernel_[i]);

    // Step 4: inverse 2D C2R FFT and normalise.
    fftw_execute(reinterpret_cast<fftw_plan>(plan_inv_));
    const double norm = 1.0 / static_cast<double>(nx_ * ny_);
    for (std::size_t i = 0; i < N2; ++i)
        r_buf_[i] *= norm;

    return std::vector<Real>(r_buf_.begin(), r_buf_.end());
}

}  // namespace micromag
