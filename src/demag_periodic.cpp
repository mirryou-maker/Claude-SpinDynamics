#include "micromag/demag_periodic.hpp"

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
// Newell (1993) helpers — identical to those in demag.cpp but scoped here.
// ---------------------------------------------------------------------------

double DemagFieldPeriodic::newell_f(double x, double y, double z) {
    x = std::abs(x); y = std::abs(y); z = std::abs(z);
    const double x2 = x*x, y2 = y*y, z2 = z*z;
    const double r  = std::sqrt(x2 + y2 + z2);
    if (r == 0.0) return 0.0;

    double val = 0.0;
    const double d_xz = std::sqrt(x2 + z2);
    if (d_xz > 0.0) val += y*(z2 - x2)*0.5*std::asinh(y/d_xz);
    const double d_xy = std::sqrt(x2 + y2);
    if (d_xy > 0.0) val += z*(y2 - x2)*0.5*std::asinh(z/d_xy);
    if (std::abs(x) > 0.0) val -= x*y*z*std::atan(y*z/(x*r));
    val += (2.0*x2 - y2 - z2)*r/6.0;
    return val;
}

double DemagFieldPeriodic::newell_g(double x, double y, double z) {
    z = std::abs(z);
    const double x2 = x*x, y2 = y*y, z2 = z*z;
    const double r  = std::sqrt(x2 + y2 + z2);
    if (r == 0.0) return 0.0;

    double val = 0.0;
    const double d_xy = std::sqrt(x2 + y2);
    if (d_xy > 0.0) val += x*y*z*std::asinh(z/d_xy);
    const double d_yz = std::sqrt(y2 + z2);
    if (d_yz > 0.0) val += y*(3.0*z2 - y2)/6.0*std::asinh(x/d_yz);
    const double d_xz = std::sqrt(x2 + z2);
    if (d_xz > 0.0) val += x*(3.0*z2 - x2)/6.0*std::asinh(y/d_xz);
    if (std::abs(z) > 0.0) val -= z*z2/6.0*std::atan(x*y/(z*r));
    if (std::abs(y) > 0.0) val -= z*y2*0.5*std::atan(x*z/(y*r));
    if (std::abs(x) > 0.0) val -= z*x2*0.5*std::atan(y*z/(x*r));
    val -= x*y*r/3.0;
    return val;
}

double DemagFieldPeriodic::nxx(double x, double y, double z,
                                 double dx, double dy, double dz) const {
    const int nx = static_cast<int>(std::round(x/dx));
    const int ny = static_cast<int>(std::round(y/dy));
    const int nz = static_cast<int>(std::round(z/dz));
    double sum = 0.0;
    for (int ia:{0,1}) for (int ib:{0,1}) for (int ic:{0,1})
    for (int id:{0,1}) for (int ie:{0,1}) for (int ig:{0,1}) {
        const int sign = ((ia+ib+ic+id+ie+ig)%2 == 0) ? 1 : -1;
        sum += sign*newell_f((nx+ia-id)*dx, (ny+ib-ie)*dy, (nz+ic-ig)*dz);
    }
    return sum/(4.0*constants::pi*dx*dy*dz);
}

double DemagFieldPeriodic::nxy(double x, double y, double z,
                                 double dx, double dy, double dz) const {
    const int nx = static_cast<int>(std::round(x/dx));
    const int ny = static_cast<int>(std::round(y/dy));
    const int nz = static_cast<int>(std::round(z/dz));
    double sum = 0.0;
    for (int ia:{0,1}) for (int ib:{0,1}) for (int ic:{0,1})
    for (int id:{0,1}) for (int ie:{0,1}) for (int ig:{0,1}) {
        const int sign = ((ia+ib+ic+id+ie+ig)%2 == 0) ? 1 : -1;
        sum += sign*newell_g((nx+ia-id)*dx, (ny+ib-ie)*dy, (nz+ic-ig)*dz);
    }
    return sum/(4.0*constants::pi*dx*dy*dz);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DemagFieldPeriodic::DemagFieldPeriodic(const StructuredGrid& grid, int n_rep)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      n_rep_(n_rep),
      fft_nx_(nx_/2 + 1)
{
    const std::size_t real_size    = static_cast<std::size_t>(nx_*ny_*nz_);
    const std::size_t complex_size = static_cast<std::size_t>(fft_nx_*ny_*nz_);

    K_xx_.resize(complex_size); K_yy_.resize(complex_size); K_zz_.resize(complex_size);
    K_xy_.resize(complex_size); K_xz_.resize(complex_size); K_yz_.resize(complex_size);

    r_buf_.resize(real_size, 0.0);
    c_buf_.resize(complex_size);

    { static bool s_inited = false;
      if (!s_inited) { fftw_init_threads(); s_inited = true; }
      fftw_plan_with_nthreads(static_cast<int>(std::thread::hardware_concurrency())); }

    fftw_import_wisdom_from_filename("fftw_wisdom.dat");

    const unsigned flags = FFTW_MEASURE;
    plan_fwd_ = reinterpret_cast<fftw_plan_s*>(
        fftw_plan_dft_r2c_3d(
            static_cast<int>(nz_), static_cast<int>(ny_), static_cast<int>(nx_),
            r_buf_.data(),
            reinterpret_cast<fftw_complex*>(c_buf_.data()),
            flags));
    plan_inv_ = reinterpret_cast<fftw_plan_s*>(
        fftw_plan_dft_c2r_3d(
            static_cast<int>(nz_), static_cast<int>(ny_), static_cast<int>(nx_),
            reinterpret_cast<fftw_complex*>(c_buf_.data()),
            r_buf_.data(),
            flags));

    if (!plan_fwd_ || !plan_inv_)
        throw std::runtime_error("DemagFieldPeriodic: FFTW plan creation failed");

    fftw_export_wisdom_to_filename("fftw_wisdom.dat");

    precompute_kernel();
}

DemagFieldPeriodic::~DemagFieldPeriodic() {
    if (plan_fwd_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_fwd_));
    if (plan_inv_) fftw_destroy_plan(reinterpret_cast<fftw_plan>(plan_inv_));
}

// ---------------------------------------------------------------------------
// precompute_kernel
//
// For each kernel component K_αβ:
//   1. Fill r_buf_ with N_αβ^periodic(r) = Σ_n N_αβ^Newell(r + n·L)
//      using minimum-image convention for r and summing ±n_rep images.
//   2. Forward FFT → c_buf_ → store in K_αβ.
//   3. Zero the k=0 mode (K_αβ[0] = 0): uniform magnetisation has no demag
//      field in the periodic-BC convention.
// ---------------------------------------------------------------------------

void DemagFieldPeriodic::precompute_kernel() {
    const double Lx = nx_*dx_, Ly = ny_*dy_, Lz = nz_*dz_;

    // fill_k: compute image-sum kernel component into r_buf_, FFT, zero k=0.
    auto fill_k = [&](std::vector<std::complex<double>>& K_dest,
                       auto kernel_fn) {
        for (Index kz = 0; kz < nz_; ++kz)
        for (Index ky = 0; ky < ny_; ++ky)
        for (Index kx = 0; kx < nx_; ++kx) {
            // Minimum-image displacement (wraps at half-period)
            const double x0 = (kx <= nx_/2) ? kx*dx_ : (kx-nx_)*dx_;
            const double y0 = (ky <= ny_/2) ? ky*dy_ : (ky-ny_)*dy_;
            const double z0 = (kz <= nz_/2) ? kz*dz_ : (kz-nz_)*dz_;

            double val = 0.0;
            for (int n1 = -n_rep_; n1 <= n_rep_; ++n1)
            for (int n2 = -n_rep_; n2 <= n_rep_; ++n2)
            for (int n3 = -n_rep_; n3 <= n_rep_; ++n3)
                val += kernel_fn(x0 + n1*Lx, y0 + n2*Ly, z0 + n3*Lz);

            r_buf_[static_cast<std::size_t>(kx + nx_*(ky + ny_*kz))] = val;
        }

        fftw_execute(reinterpret_cast<fftw_plan>(plan_fwd_));
        K_dest = c_buf_;
        K_dest[0] = 0.0;   // k=0: no demag field for uniform m (periodic convention)
    };

    // Diagonal (even symmetry — image sum handles it naturally)
    fill_k(K_xx_, [&](double x, double y, double z) {
        return nxx(x, y, z, dx_, dy_, dz_);
    });
    fill_k(K_yy_, [&](double x, double y, double z) {
        return nxx(y, x, z, dy_, dx_, dz_);
    });
    fill_k(K_zz_, [&](double x, double y, double z) {
        return nxx(z, y, x, dz_, dy_, dx_);
    });

    // Off-diagonal (nxy handles signed arguments correctly)
    fill_k(K_xy_, [&](double x, double y, double z) {
        return nxy(x, y, z, dx_, dy_, dz_);
    });
    fill_k(K_xz_, [&](double x, double y, double z) {
        return nxy(x, z, y, dx_, dz_, dy_);
    });
    fill_k(K_yz_, [&](double x, double y, double z) {
        return nxy(y, z, x, dy_, dz_, dx_);
    });
}

// ---------------------------------------------------------------------------
// accumulate — H_demag = -N·M, add to H_out
// ---------------------------------------------------------------------------

void DemagFieldPeriodic::accumulate(const VectorField3D& m,
                                     const Material& mat,
                                     VectorField3D& H_out) const {
    const double Ms      = mat.Ms;
    const std::size_t N  = static_cast<std::size_t>(nx_*ny_*nz_);
    const std::size_t Nc = static_cast<std::size_t>(fft_nx_*ny_*nz_);
    const double norm    = 1.0 / static_cast<double>(N);

    // Forward FFT of one magnetisation component
    std::vector<std::complex<double>> Mx_f(Nc), My_f(Nc), Mz_f(Nc);

    auto fft_comp = [&](int c, std::vector<std::complex<double>>& out) {
        #pragma omp parallel for schedule(static) if(N > 4096)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
            const Vec3& v = m[static_cast<Index>(i)];
            const double Ms_cell = matf_ ? matf_->Ms(static_cast<Index>(i)) : Ms;
            r_buf_[static_cast<std::size_t>(i)] = Ms_cell * (c == 0 ? v.x : c == 1 ? v.y : v.z);
        }
        fftw_execute(reinterpret_cast<fftw_plan>(plan_fwd_));
        out = c_buf_;
    };

    fft_comp(0, Mx_f);
    fft_comp(1, My_f);
    fft_comp(2, Mz_f);

    // Inverse FFT of one output component and add to H_out
    auto ifft_add = [&](const std::vector<std::complex<double>>& Ka,
                         const std::vector<std::complex<double>>& Kb,
                         const std::vector<std::complex<double>>& Kc,
                         const std::vector<std::complex<double>>& Ma,
                         const std::vector<std::complex<double>>& Mb,
                         const std::vector<std::complex<double>>& Mc,
                         int out_comp) {
        #pragma omp parallel for schedule(static) if(Nc > 4096)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(Nc); ++i)
            c_buf_[static_cast<std::size_t>(i)] = Ka[i]*Ma[i] + Kb[i]*Mb[i] + Kc[i]*Mc[i];
        fftw_execute(reinterpret_cast<fftw_plan>(plan_inv_));
        #pragma omp parallel for schedule(static) if(N > 4096)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
            const double h = -r_buf_[static_cast<std::size_t>(i)] * norm;
            Vec3& hv = H_out[static_cast<Index>(i)];
            if      (out_comp == 0) hv.x += h;
            else if (out_comp == 1) hv.y += h;
            else                    hv.z += h;
        }
    };

    ifft_add(K_xx_, K_xy_, K_xz_, Mx_f, My_f, Mz_f, 0);  // Hx
    ifft_add(K_xy_, K_yy_, K_yz_, Mx_f, My_f, Mz_f, 1);  // Hy
    ifft_add(K_xz_, K_yz_, K_zz_, Mx_f, My_f, Mz_f, 2);  // Hz
}

// ---------------------------------------------------------------------------
// energy  E = -μ₀/2 · Ms · Σ m·H_demag · dV
// ---------------------------------------------------------------------------

Real DemagFieldPeriodic::energy(const VectorField3D& m,
                                 const Material& mat) const {
    VectorField3D H(m.grid());
    for (Index i = 0; i < H.size(); ++i) H[i] = {0,0,0};
    accumulate(m, mat, H);

    Real E = 0.0;
    const Real dV = m.grid().cell_volume();
    for (Index i = 0; i < m.size(); ++i) {
        const Real Ms_cell = matf_ ? matf_->Ms(i) : mat.Ms;
        E -= constants::mu_0 * Ms_cell * m[i].dot(H[i]) * dV;
    }
    return 0.5*E;
}

}  // namespace micromag
