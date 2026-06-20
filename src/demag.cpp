#include "micromag/demag.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <thread>

#include <fftw3.h>

#include "micromag/types.hpp"
#include "micromag/material_field.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// Newell (1993) helper functions for the analytical demag tensor.
// Reference: Newell, Williams & Dunlop, JGR 1993, doi:10.1029/93JB00694
//
// These compute the antiderivatives used in the closed-form integrals.
// ---------------------------------------------------------------------------

// f(x,y,z) = antiderivative used for diagonal components N_xx.
double DemagField::newell_f(double x, double y, double z) {
    x = std::abs(x); y = std::abs(y); z = std::abs(z);
    double x2 = x * x, y2 = y * y, z2 = z * z;
    double r  = std::sqrt(x2 + y2 + z2);
    if (r == 0.0) return 0.0;

    double val = 0.0;
    // Term: y*(z²-x²)/2 * asinh(y/sqrt(x²+z²))
    double d_xz = std::sqrt(x2 + z2);
    if (d_xz > 0.0) val += y * (z2 - x2) * 0.5 * std::asinh(y / d_xz);
    // Term: z*(y²-x²)/2 * asinh(z/sqrt(x²+y²))
    double d_xy = std::sqrt(x2 + y2);
    if (d_xy > 0.0) val += z * (y2 - x2) * 0.5 * std::asinh(z / d_xy);
    // Term: -xyz * atan(y*z/(x*r))
    // Use std::atan (range -π/2..π/2), NOT atan2.  Newell (1993) eq. (B2)
    // uses the standard arctangent; atan2 gives wrong values for x < 0,
    // shifting the result by ±π and corrupting the 8-corner alternating sum.
    if (std::abs(x) > 0.0) val -= x * y * z * std::atan(y * z / (x * r));
    // Term: (2x²-y²-z²)*r/6
    val += (2.0 * x2 - y2 - z2) * r / 6.0;
    return val;
}

// g(x,y,z) = antiderivative used for off-diagonal component N_xy.
double DemagField::newell_g(double x, double y, double z) {
    z = std::abs(z);
    double x2 = x * x, y2 = y * y, z2 = z * z;
    double r  = std::sqrt(x2 + y2 + z2);
    if (r == 0.0) return 0.0;

    double val = 0.0;
    // Term: x*y*z * asinh(z/sqrt(x²+y²))
    double d_xy = std::sqrt(x2 + y2);
    if (d_xy > 0.0) val += x * y * z * std::asinh(z / d_xy);
    // Term: y/6*(3z²-y²) * asinh(x/sqrt(y²+z²))
    double d_yz = std::sqrt(y2 + z2);
    if (d_yz > 0.0) val += y * (3.0 * z2 - y2) / 6.0 * std::asinh(x / d_yz);
    // Term: x/6*(3z²-x²) * asinh(y/sqrt(x²+z²))
    double d_xz = std::sqrt(x2 + z2);
    if (d_xz > 0.0) val += x * (3.0 * z2 - x2) / 6.0 * std::asinh(y / d_xz);
    // Term: -z³/6 * atan(x*y/(z*r))
    if (std::abs(z) > 0.0) val -= z * z2 / 6.0 * std::atan(x * y / (z * r));
    // Term: -z*y²/2 * atan(x*z/(y*r))
    if (std::abs(y) > 0.0) val -= z * y2 * 0.5 * std::atan(x * z / (y * r));
    // Term: -z*x²/2 * atan(y*z/(x*r))
    if (std::abs(x) > 0.0) val -= z * x2 * 0.5 * std::atan(y * z / (x * r));
    // Term: -x*y*r/3
    val -= x * y * r / 3.0;
    return val;
}

// Closed-form integral for diagonal tensor component N_xx(x,y,z,dx,dy,dz).
// Uses the exact Newell (1993) 6D double-cell integral (64-term alternating sum).
double DemagField::nxx(double x, double y, double z,
                        double dx, double dy, double dz) {
    const int nx = static_cast<int>(std::round(x / dx));
    const int ny = static_cast<int>(std::round(y / dy));
    const int nz = static_cast<int>(std::round(z / dz));
    double sum = 0.0;
    for (int ia : {0, 1}) for (int ib : {0, 1}) for (int ic : {0, 1})
    for (int id : {0, 1}) for (int ie : {0, 1}) for (int ig : {0, 1}) {
        const int sign = ((ia + ib + ic + id + ie + ig) % 2 == 0) ? 1 : -1;
        sum += sign * newell_f(
            (nx + ia - id) * dx,
            (ny + ib - ie) * dy,
            (nz + ic - ig) * dz);
    }
    return +sum / (4.0 * constants::pi * dx * dy * dz);
}

// Closed-form integral for off-diagonal component N_xy(x,y,z,dx,dy,dz).
// Uses the exact Newell (1993) 6D double-cell integral (64-term alternating sum).
double DemagField::nxy(double x, double y, double z,
                        double dx, double dy, double dz) {
    const int nx = static_cast<int>(std::round(x / dx));
    const int ny = static_cast<int>(std::round(y / dy));
    const int nz = static_cast<int>(std::round(z / dz));
    double sum = 0.0;
    for (int ia : {0, 1}) for (int ib : {0, 1}) for (int ic : {0, 1})
    for (int id : {0, 1}) for (int ie : {0, 1}) for (int ig : {0, 1}) {
        const int sign = ((ia + ib + ic + id + ie + ig) % 2 == 0) ? 1 : -1;
        sum += sign * newell_g(
            (nx + ia - id) * dx,
            (ny + ib - ie) * dy,
            (nz + ic - ig) * dz);
    }
    return +sum / (4.0 * constants::pi * dx * dy * dz);
}

// ---------------------------------------------------------------------------
// Constructor — allocate FFTW plans and precompute kernel.
// ---------------------------------------------------------------------------

DemagField::DemagField(const StructuredGrid& grid)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz())
{
    // Zero-pad to 2N in each dimension (linear convolution via circular).
    pad_nx_ = 2 * nx_;
    pad_ny_ = 2 * ny_;
    pad_nz_ = 2 * nz_;
    fft_nx_ = pad_nx_ / 2 + 1;

    std::size_t real_size    = static_cast<std::size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    std::size_t complex_size = static_cast<std::size_t>(fft_nx_ * pad_ny_ * pad_nz_);

    // Allocate kernel arrays.
    K_xx_.resize(complex_size);  K_yy_.resize(complex_size);  K_zz_.resize(complex_size);
    K_xy_.resize(complex_size);  K_xz_.resize(complex_size);  K_yz_.resize(complex_size);

    // Scratch buffers for runtime use.
    r_buf_.resize(real_size, 0.0);
    c_buf_.resize(complex_size);

    // Batch buffers for 3-component accumulate (all 3 M/H components at once).
    r_buf_3_.resize(3 * real_size, 0.0);
    c_buf_3_.resize(3 * complex_size);

    // Multi-threaded FFTW: init once; set thread count before every plan call.
    // Thread count matches logical CPUs — fftw_execute() uses a thread pool.
    { static bool s_inited = false;
      if (!s_inited) { fftw_init_threads(); s_inited = true; }
      fftw_plan_with_nthreads(static_cast<int>(std::thread::hardware_concurrency())); }

    // Import FFTW wisdom from a previous run (eliminates re-measurement overhead
    // on repeat runs; FFTW silently ignores invalid/missing files).
    fftw_import_wisdom_from_filename("fftw_wisdom.dat");

    // FFTW_MEASURE probes multiple algorithms at plan creation and picks the
    // fastest. std::vector<double> on MSVC x64 is ≥16-byte aligned so SIMD
    // paths are safe. Plan overhead: ~O(0.1s) per grid; paid once per run.
    const unsigned fftw_flags = FFTW_MEASURE;

    // Single-component plans — used only by precompute_kernel().
    plan_fwd_ = fftw_plan_dft_r2c_3d(
        static_cast<int>(pad_nz_),
        static_cast<int>(pad_ny_),
        static_cast<int>(pad_nx_),
        r_buf_.data(),
        reinterpret_cast<fftw_complex*>(c_buf_.data()),
        fftw_flags);

    plan_inv_ = fftw_plan_dft_c2r_3d(
        static_cast<int>(pad_nz_),
        static_cast<int>(pad_ny_),
        static_cast<int>(pad_nx_),
        reinterpret_cast<fftw_complex*>(c_buf_.data()),
        r_buf_.data(),
        fftw_flags);

    if (!plan_fwd_ || !plan_inv_)
        throw std::runtime_error("DemagField: FFTW plan creation failed");

    // 3-component batch plans — used by accumulate() to process Mx/My/Mz
    // in two calls instead of six (3 forward + 3 inverse → 1+1).
    const int n[3] = { static_cast<int>(pad_nz_),
                       static_cast<int>(pad_ny_),
                       static_cast<int>(pad_nx_) };

    plan_fwd_3_ = reinterpret_cast<fftw_plan_s*>(
        fftw_plan_many_dft_r2c(
            3, n, 3,
            r_buf_3_.data(),              nullptr, 1, static_cast<int>(real_size),
            reinterpret_cast<fftw_complex*>(c_buf_3_.data()),
                                          nullptr, 1, static_cast<int>(complex_size),
            fftw_flags));

    plan_inv_3_ = reinterpret_cast<fftw_plan_s*>(
        fftw_plan_many_dft_c2r(
            3, n, 3,
            reinterpret_cast<fftw_complex*>(c_buf_3_.data()),
                                          nullptr, 1, static_cast<int>(complex_size),
            r_buf_3_.data(),              nullptr, 1, static_cast<int>(real_size),
            fftw_flags));

    if (!plan_fwd_3_ || !plan_inv_3_)
        throw std::runtime_error("DemagField: FFTW batch plan creation failed");

    // Export updated wisdom so subsequent runs skip the measurement phase.
    fftw_export_wisdom_to_filename("fftw_wisdom.dat");

    precompute_kernel();
}

DemagField::~DemagField() {
    if (plan_fwd_)   fftw_destroy_plan(plan_fwd_);
    if (plan_inv_)   fftw_destroy_plan(plan_inv_);
    if (plan_fwd_3_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_fwd_3_));
    if (plan_inv_3_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_inv_3_));
}

// ---------------------------------------------------------------------------
// precompute_kernel — fill K_xx … K_yz in frequency space.
// ---------------------------------------------------------------------------

void DemagField::precompute_kernel() {
    // Fill r_buf_ with kernel values and FFT into c_buf_ using the plan's own
    // buffers.  Using fftw_execute (not the new-array form) avoids the alignment
    // requirement that fftw_execute_dft_r2c imposes on caller-supplied arrays;
    // FFTW compiled with AVX needs 32-byte alignment which std::vector does not
    // guarantee, causing silent corruption when new-array execute is used.

    auto put = [&](Index px, Index py, Index pz, double v) {
        if (px < 0) px += pad_nx_;
        if (py < 0) py += pad_ny_;
        if (pz < 0) pz += pad_nz_;
        r_buf_[static_cast<std::size_t>(px + pad_nx_ * (py + pad_ny_ * pz))] = v;
    };

    // Lambda: fill r_buf_ with one kernel component, FFT → c_buf_, store in K_dest.
    auto fill_and_fft = [&](std::vector<std::complex<double>>& K_dest,
                             auto kernel_fn) {
        std::fill(r_buf_.begin(), r_buf_.end(), 0.0);

        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            double val = kernel_fn(static_cast<double>(kx) * dx_,
                                   static_cast<double>(ky) * dy_,
                                   static_cast<double>(kz) * dz_);
            put( kx,  ky,  kz,  val);
            if (kx > 0) put(-kx,  ky,  kz, val);
            if (ky > 0) put( kx, -ky,  kz, val);
            if (kz > 0) put( kx,  ky, -kz, val);
            if (kx > 0 && ky > 0) put(-kx, -ky,  kz, val);
            if (kx > 0 && kz > 0) put(-kx,  ky, -kz, val);
            if (ky > 0 && kz > 0) put( kx, -ky, -kz, val);
            if (kx > 0 && ky > 0 && kz > 0) put(-kx, -ky, -kz, val);
        }

        fftw_execute(plan_fwd_);
        K_dest = c_buf_;
    };

    // Diagonal components (even symmetry in all directions).
    fill_and_fft(K_xx_, [&](double x, double y, double z) {
        return nxx(x, y, z, dx_, dy_, dz_);
    });
    fill_and_fft(K_yy_, [&](double x, double y, double z) {
        return nxx(y, x, z, dy_, dx_, dz_);   // N_yy by symmetry
    });
    fill_and_fft(K_zz_, [&](double x, double y, double z) {
        return nxx(z, y, x, dz_, dy_, dx_);   // N_zz by symmetry
    });

    // Off-diagonal components (mixed parity): sx/sy/sz = sign flip when that
    // index negates.  N_xy is odd in x and y, even in z → sx=-1, sy=-1, sz=+1.
    auto fill_and_fft_offdiag = [&](std::vector<std::complex<double>>& K_dest,
                                     int sx, int sy, int sz,
                                     auto kernel_fn) {
        std::fill(r_buf_.begin(), r_buf_.end(), 0.0);
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            double val = kernel_fn(static_cast<double>(kx) * dx_,
                                   static_cast<double>(ky) * dy_,
                                   static_cast<double>(kz) * dz_);
            for (int ix : {0, 1})
            for (int iy : {0, 1})
            for (int iz : {0, 1}) {
                if (ix && kx == 0) continue;
                if (iy && ky == 0) continue;
                if (iz && kz == 0) continue;
                double sign = (ix ? (double)sx : 1.0)
                            * (iy ? (double)sy : 1.0)
                            * (iz ? (double)sz : 1.0);
                put(ix ? -kx : kx, iy ? -ky : ky, iz ? -kz : kz, sign * val);
            }
        }
        fftw_execute(plan_fwd_);
        K_dest = c_buf_;
    };

    // N_xy: odd in x and y, even in z  → sx=-1, sy=-1, sz=+1
    fill_and_fft_offdiag(K_xy_, -1, -1, +1,
        [&](double x, double y, double z) { return nxy(x, y, z, dx_, dy_, dz_); });
    // N_xz: odd in x and z, even in y  → sx=-1, sy=+1, sz=-1
    fill_and_fft_offdiag(K_xz_, -1, +1, -1,
        [&](double x, double y, double z) { return nxy(x, z, y, dx_, dz_, dy_); });
    // N_yz: odd in y and z, even in x  → sx=+1, sy=-1, sz=-1
    fill_and_fft_offdiag(K_yz_, +1, -1, -1,
        [&](double x, double y, double z) { return nxy(y, z, x, dy_, dz_, dx_); });
}

// ---------------------------------------------------------------------------
// accumulate — compute H_demag and add to H_out.
// ---------------------------------------------------------------------------

void DemagField::accumulate(const VectorField3D& m,
                             const Material& mat,
                             VectorField3D& H_out) const {
    const double Ms = mat.Ms;
    const std::size_t real_size  = static_cast<std::size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    const std::size_t c_total    = static_cast<std::size_t>(fft_nx_ * pad_ny_ * pad_nz_);
    const double norm_factor     = 1.0 / static_cast<double>(real_size);

    // Step 1: pack Mx/My/Mz into the 3-component real buffer (zero-padded).
    // r_buf_3_ layout: [Mx_padded | My_padded | Mz_padded]
    std::fill(r_buf_3_.begin(), r_buf_3_.end(), 0.0);
    // Flattened over the unpadded grid (src == linear index) so OpenMP scales
    // on thin-z grids; each iteration writes its own padded dst → race-free.
    const Index Ncells = nx_ * ny_ * nz_;
    #pragma omp parallel for schedule(static) if(Ncells > 4096)
    for (Index src = 0; src < Ncells; ++src) {
        const Index kx   = src % nx_;
        const Index trow = src / nx_;
        const Index ky   = trow % ny_;
        const Index kz   = trow / ny_;
        const std::size_t dst = static_cast<std::size_t>(kx + pad_nx_ * (ky + pad_ny_ * kz));
        const Vec3&  v       = m[src];
        const double Ms_cell = matf_ ? matf_->Ms(src) : Ms;
        r_buf_3_[dst]                 = Ms_cell * v.x;
        r_buf_3_[real_size  + dst]    = Ms_cell * v.y;
        r_buf_3_[2*real_size + dst]   = Ms_cell * v.z;
    }

    // Step 2: batch forward FFT — 3 transforms in one call.
    // c_buf_3_ layout after: [Mx_f | My_f | Mz_f]
    fftw_execute(reinterpret_cast<fftw_plan>(plan_fwd_3_));

    // Step 3: pointwise tensor multiply  H_f = -N * M_f  (all 3 rows at once).
    // Read Mx/My/Mz before overwriting (in-place within c_buf_3_).
    #pragma omp parallel for schedule(static) if(c_total > 4096)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(c_total); ++i) {
        const std::complex<double> Mx = c_buf_3_[i];
        const std::complex<double> My = c_buf_3_[c_total     + i];
        const std::complex<double> Mz = c_buf_3_[2*c_total   + i];
        c_buf_3_[i]               = -(K_xx_[i]*Mx + K_xy_[i]*My + K_xz_[i]*Mz);
        c_buf_3_[c_total   + i]   = -(K_xy_[i]*Mx + K_yy_[i]*My + K_yz_[i]*Mz);
        c_buf_3_[2*c_total + i]   = -(K_xz_[i]*Mx + K_yz_[i]*My + K_zz_[i]*Mz);
    }

    // Step 4: batch inverse FFT — 3 transforms in one call.
    // r_buf_3_ layout after: [Hx_padded | Hy_padded | Hz_padded]
    fftw_execute(reinterpret_cast<fftw_plan>(plan_inv_3_));

    // Step 5: unpack and accumulate into H_out (all 3 components, one loop).
    // Flattened over the unpadded grid (dst == linear index); each iteration
    // writes its own H_out[dst] → race-free under OpenMP.
    const Index Ncells2 = nx_ * ny_ * nz_;
    #pragma omp parallel for schedule(static) if(Ncells2 > 4096)
    for (Index dst = 0; dst < Ncells2; ++dst) {
        const Index kx   = dst % nx_;
        const Index trow = dst / nx_;
        const Index ky   = trow % ny_;
        const Index kz   = trow / ny_;
        const std::size_t src = static_cast<std::size_t>(kx + pad_nx_ * (ky + pad_ny_ * kz));
        H_out[dst].x += r_buf_3_[src]                  * norm_factor;
        H_out[dst].y += r_buf_3_[real_size   + src]    * norm_factor;
        H_out[dst].z += r_buf_3_[2*real_size + src]    * norm_factor;
    }
}

// ---------------------------------------------------------------------------
// energy — magnetostatic energy  E = -μ₀/2 * Ms * Σ m·H_demag * dV
// ---------------------------------------------------------------------------

Real DemagField::energy(const VectorField3D& m, const Material& mat) const {
    const StructuredGrid& g = m.grid();
    // Compute H_demag into a temporary field.
    VectorField3D H(g);
    // Zero H first (accumulate adds to existing).
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
    accumulate(m, mat, H);

    Real E = 0.0;
    const Real dV = g.cell_volume();
    for (Index i = 0; i < m.size(); ++i) {
        const Real Ms_cell = matf_ ? matf_->Ms(i) : mat.Ms;
        E -= constants::mu_0 * Ms_cell * m[i].dot(H[i]) * dV;
    }
    return 0.5 * E;   // factor 1/2 avoids double-counting
}

ScalarField3D DemagField::energy_density(const VectorField3D& m,
                                          const Material& mat) const {
    const StructuredGrid& g = m.grid();
    VectorField3D H(g);
    for (Index i = 0; i < H.size(); ++i) H[i] = {0, 0, 0};
    accumulate(m, mat, H);

    ScalarField3D edens(g);
    for (Index i = 0; i < m.size(); ++i) {
        const Real Ms_cell = matf_ ? matf_->Ms(i) : mat.Ms;
        // e_i = -μ₀/2 * Ms_i * (m_i · H_demag_i)  [J/m³]
        edens[i] = -0.5 * constants::mu_0 * Ms_cell * m[i].dot(H[i]);
    }
    return edens;
}

}  // namespace micromag
