#include "micromag/demag.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include <fftw3.h>

#include "micromag/types.hpp"

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

    // FFTW_UNALIGNED prevents FFTW from selecting SIMD (AVX/SSE) algorithms
    // that require stricter alignment than std::vector guarantees.
    // We use fftw_execute (not new-array execute) so this flag is compatible.
    const unsigned fftw_flags = FFTW_ESTIMATE | FFTW_UNALIGNED;

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

    precompute_kernel();
}

DemagField::~DemagField() {
    if (plan_fwd_) fftw_destroy_plan(plan_fwd_);
    if (plan_inv_) fftw_destroy_plan(plan_inv_);
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
    const std::size_t pad_total = static_cast<std::size_t>(pad_nx_ * pad_ny_ * pad_nz_);
    const std::size_t c_total   = static_cast<std::size_t>(fft_nx_ * pad_ny_ * pad_nz_);
    const double norm_factor    = 1.0 / static_cast<double>(pad_total);

    // We need per-component FFTs of Mx, My, Mz.
    // Store them in temporary arrays.
    std::vector<std::complex<double>> Mx_f(c_total), My_f(c_total), Mz_f(c_total);

    // Helper: forward FFT of one magnetisation component.
    auto fft_component = [&](int comp, std::vector<std::complex<double>>& out) {
        // Zero-fill real buffer, copy component into unpadded region.
        std::fill(r_buf_.begin(), r_buf_.end(), 0.0);
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            std::size_t src = static_cast<std::size_t>(kx + nx_ * (ky + ny_ * kz));
            std::size_t dst = static_cast<std::size_t>(kx + pad_nx_ * (ky + pad_ny_ * kz));
            const Vec3& v = m[static_cast<Index>(src)];
            r_buf_[dst] = Ms * (comp == 0 ? v.x : comp == 1 ? v.y : v.z);
        }
        fftw_execute(plan_fwd_);
        out = c_buf_;
    };

    fft_component(0, Mx_f);
    fft_component(1, My_f);
    fft_component(2, Mz_f);

    // For each output component Hx,Hy,Hz: sum over tensor row.
    // H_x = -(N_xx*Mx + N_xy*My + N_xz*Mz)   (H_demag = -N*M)
    auto ifft_and_add = [&](const std::vector<std::complex<double>>& Ka,
                             const std::vector<std::complex<double>>& Kb,
                             const std::vector<std::complex<double>>& Kc,
                             const std::vector<std::complex<double>>& Ma,
                             const std::vector<std::complex<double>>& Mb,
                             const std::vector<std::complex<double>>& Mc,
                             int out_comp) {
        // Pointwise multiply and sum.
        for (std::size_t i = 0; i < c_total; ++i)
            c_buf_[i] = Ka[i] * Ma[i] + Kb[i] * Mb[i] + Kc[i] * Mc[i];

        // Inverse FFT.
        fftw_execute(plan_inv_);

        // Copy unpadded region to H_out, applying sign (-N*M) and normalisation.
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            std::size_t src = static_cast<std::size_t>(kx + pad_nx_ * (ky + pad_ny_ * kz));
            Index       dst = kx + nx_ * (ky + ny_ * kz);
            double h_val = -r_buf_[src] * norm_factor;
            Vec3& hv = H_out[dst];
            if (out_comp == 0) hv.x += h_val;
            else if (out_comp == 1) hv.y += h_val;
            else                   hv.z += h_val;
        }
    };

    ifft_and_add(K_xx_, K_xy_, K_xz_, Mx_f, My_f, Mz_f, 0);  // Hx
    ifft_and_add(K_xy_, K_yy_, K_yz_, Mx_f, My_f, Mz_f, 1);  // Hy
    ifft_and_add(K_xz_, K_yz_, K_zz_, Mx_f, My_f, Mz_f, 2);  // Hz
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
    for (Index i = 0; i < m.size(); ++i)
        E -= constants::mu_0 * mat.Ms * m[i].dot(H[i]) * dV;
    return 0.5 * E;   // factor 1/2 avoids double-counting
}

}  // namespace micromag
