// spin_torque_gpu.cu ??GPU spin torque CUDA kernels.
//
// All buffers: component-major [3횞N] on device.
// Kernel conventions: idx ??[0, N), x-comp at [idx], y at [N+idx], z at [2N+idx].
//
// Each kernel adds to d_dm_out (uses +=) so it can be called after LLG torque.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/gpu_real.hpp"
#include "micromag/spin_torque_gpu.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA(stt): ") + cudaGetErrorString(_e)); \
} while(0)

namespace micromag {

// ---------------------------------------------------------------------------
// SlonczewskiSTT kernel
//
// ? = a_J [m횞(m횞p?)] + b_J [m횞p?]
// ---------------------------------------------------------------------------
// base = γ₀ħJ / (2·e·Ms·d)   [1/s]; the polarisation enters through the
// mumax3 angular efficiency ε(m·p) = P·Λ²/((Λ²+1)+(Λ²−1)(m·p)). Λ→∞ recovers
// the constant-ε (=P) form; Λ=1 gives ε=P/2 (mumax3 default). Field-like term
// b_J = −β·a_J (a_J is now angle-dependent).
__global__ static void kernel_slonczewski(
    GReal* dm_out, const GReal* m, int N,
    double base, double Pval, double lam2, double beta,
    double px, double py, double pz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double mx = m[i], my = m[N+i], mz = m[2*N+i];

    const double mdotp = mx*px + my*py + mz*pz;
    const double eps   = Pval * lam2 / ((lam2 + 1.0) + (lam2 - 1.0) * mdotp);
    const double aJ = base * eps;
    const double bJ = -beta * aJ;

    const double mxpx = my*pz - mz*py;
    const double mxpy = mz*px - mx*pz;
    const double mxpz = mx*py - my*px;

    const double mxmxpx = my*mxpz - mz*mxpy;
    const double mxmxpy = mz*mxpx - mx*mxpz;
    const double mxmxpz = mx*mxpy - my*mxpx;

    dm_out[i]     += aJ*mxmxpx + bJ*mxpx;
    dm_out[N+i]   += aJ*mxmxpy + bJ*mxpy;
    dm_out[2*N+i] += aJ*mxmxpz + bJ*mxpz;
}

// ---------------------------------------------------------------------------
// SOT kernel
//
// ? = a_SOT [管_DL m횞(m횞??) + 管_FL (m횞??)]
// ---------------------------------------------------------------------------
__global__ static void kernel_sot(
    GReal* dm_out, const GReal* m, int N,
    double a_etaDL, double a_etaFL,
    double sx, double sy, double sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double mx = m[i], my = m[N+i], mz = m[2*N+i];

    const double mxsx = my*sz - mz*sy;
    const double mxsy = mz*sx - mx*sz;
    const double mxsz = mx*sy - my*sx;

    const double mxmxsx = my*mxsz - mz*mxsy;
    const double mxmxsy = mz*mxsx - mx*mxsz;
    const double mxmxsz = mx*mxsy - my*mxsx;

    dm_out[i]     += a_etaDL*mxmxsx + a_etaFL*mxsx;
    dm_out[N+i]   += a_etaDL*mxmxsy + a_etaFL*mxsy;
    dm_out[2*N+i] += a_etaDL*mxmxsz + a_etaFL*mxsz;
}

// ---------------------------------------------------------------------------
// Zhang-Li STT kernel
//
// ? = u [(캔쨌??m ??刮 m횞(캔쨌??m]
// Finite differences: one-sided at boundaries (same as CPU).
// ---------------------------------------------------------------------------
__global__ static void kernel_zhangli(
    GReal* dm_out, const GReal* m,
    int nx, int ny, int nz, int N,
    double dx, double dy, double dz,
    double u_val, double xi,
    double jhat_x, double jhat_y, double jhat_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int ix = idx % nx;
    const int iy = (idx / nx) % ny;
    const int iz = idx / (nx * ny);

    double gx = 0.0, gy = 0.0, gz = 0.0;

    // x-direction
    if (jhat_x != 0.0) {
        int xm, xp;
        double dxeff;
        if (ix == 0)         { xm = idx;   xp = idx+1; dxeff = dx; }
        else if (ix == nx-1) { xm = idx-1; xp = idx;   dxeff = dx; }
        else                 { xm = idx-1; xp = idx+1; dxeff = 2.0*dx; }
        const double c = jhat_x / dxeff;
        gx += c * (m[xp]     - m[xm]);
        gy += c * (m[N+xp]   - m[N+xm]);
        gz += c * (m[2*N+xp] - m[2*N+xm]);
    }

    // y-direction
    if (jhat_y != 0.0) {
        int ym, yp;
        double dyeff;
        if (iy == 0)         { ym = idx;    yp = idx+nx; dyeff = dy; }
        else if (iy == ny-1) { ym = idx-nx; yp = idx;    dyeff = dy; }
        else                 { ym = idx-nx; yp = idx+nx; dyeff = 2.0*dy; }
        const double c = jhat_y / dyeff;
        gx += c * (m[yp]     - m[ym]);
        gy += c * (m[N+yp]   - m[N+ym]);
        gz += c * (m[2*N+yp] - m[2*N+ym]);
    }

    // z-direction
    if (jhat_z != 0.0) {
        const int stride = nx * ny;
        int zm, zp;
        double dzeff;
        if (iz == 0)         { zm = idx;        zp = idx+stride; dzeff = dz; }
        else if (iz == nz-1) { zm = idx-stride; zp = idx;        dzeff = dz; }
        else                 { zm = idx-stride; zp = idx+stride; dzeff = 2.0*dz; }
        const double c = jhat_z / dzeff;
        gx += c * (m[zp]     - m[zm]);
        gy += c * (m[N+zp]   - m[N+zm]);
        gz += c * (m[2*N+zp] - m[2*N+zm]);
    }

    // ? = u [(캔쨌??m ??刮 m횞(캔쨌??m]
    const double mx = m[idx], my_v = m[N+idx], mz_v = m[2*N+idx];

    const double cross_x = my_v*gz - mz_v*gy;
    const double cross_y = mz_v*gx - mx*gz;
    const double cross_z = mx*gy   - my_v*gx;

    dm_out[idx]     += u_val * (gx - xi * cross_x);
    dm_out[N+idx]   += u_val * (gy - xi * cross_y);
    dm_out[2*N+idx] += u_val * (gz - xi * cross_z);
}

// ---------------------------------------------------------------------------
// SlonczewskiSTTGPU
// ---------------------------------------------------------------------------

SlonczewskiSTTGPU::SlonczewskiSTTGPU(
    const StructuredGrid& grid, Real J, Real P, Real d, Vec3 p, Real beta,
    Real Lambda)
    : N_(grid.size()), J_(J), P_(P), d_(d), beta_(beta), lambda_(Lambda)
{
    const Real n = p.norm();
    p_ = (n > 1e-30) ? p / n : Vec3{0, 0, 1};
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

SlonczewskiSTTGPU::~SlonczewskiSTTGPU() {
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

double SlonczewskiSTTGPU::a_J(double Ms, double mdotp) const {
    const double base = constants::gamma_0 * constants::hbar * J_
                        / (2.0 * constants::e_charge * Ms * d_);
    const double lam2 = double(lambda_) * double(lambda_);
    const double eps  = double(P_) * lam2 / ((lam2 + 1.0) + (lam2 - 1.0) * mdotp);
    return base * eps;
}

void SlonczewskiSTTGPU::accumulate_gpu_ptr(
    const GReal* d_m, const Material& mat, GReal* d_dm_out) const
{
    const int N = static_cast<int>(N_);
    const double base = constants::gamma_0 * constants::hbar * double(J_)
                        / (2.0 * constants::e_charge * mat.Ms * double(d_));
    const double lam2 = double(lambda_) * double(lambda_);
    auto s = static_cast<cudaStream_t>(stream_);
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    kernel_slonczewski<<<blocks, threads, 0, s>>>(
        d_dm_out,
        d_m,
        N, base, double(P_), lam2, double(beta_), p_.x, p_.y, p_.z);
    // Note: no cudaGetLastError() here — that call is forbidden during CUDA Graph
    // capture and would cause the capture to fail.  Kernel launch errors are
    // caught by the integrator's cudaStreamSynchronize() at the end of step().
}

// ---------------------------------------------------------------------------
// SpinOrbitTorqueGPU
// ---------------------------------------------------------------------------

SpinOrbitTorqueGPU::SpinOrbitTorqueGPU(
    const StructuredGrid& grid,
    Real J_c, Real theta_SH, Real d_fm, Vec3 sigma,
    Real eta_DL, Real eta_FL)
    : N_(grid.size()), J_c_(J_c), theta_SH_(theta_SH), d_fm_(d_fm),
      eta_DL_(eta_DL), eta_FL_(eta_FL)
{
    const Real n = sigma.norm();
    sigma_ = (n > 1e-30) ? sigma / n : Vec3{0, 1, 0};
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

SpinOrbitTorqueGPU::~SpinOrbitTorqueGPU() {
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

double SpinOrbitTorqueGPU::a_SOT(double Ms) const {
    return constants::gamma_0 * constants::hbar * J_c_ * theta_SH_
           / (2.0 * constants::e_charge * Ms * d_fm_);
}

void SpinOrbitTorqueGPU::accumulate_gpu_ptr(
    const GReal* d_m, const Material& mat, GReal* d_dm_out) const
{
    const int N = static_cast<int>(N_);
    const double a = a_SOT(mat.Ms);
    auto s = static_cast<cudaStream_t>(stream_);
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    kernel_sot<<<blocks, threads, 0, s>>>(
        d_dm_out,
        d_m,
        N, a * eta_DL_, a * eta_FL_,
        sigma_.x, sigma_.y, sigma_.z);
    // No cudaGetLastError() — see SlonczewskiSTTGPU::accumulate_gpu_ptr.
}

// ---------------------------------------------------------------------------
// ZhangLiSTTGPU
// ---------------------------------------------------------------------------

ZhangLiSTTGPU::ZhangLiSTTGPU(
    const StructuredGrid& grid, Vec3 J, Real P, Real xi)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      J_(J), P_(P), xi_(xi)
{
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = static_cast<void*>(s);
}

ZhangLiSTTGPU::~ZhangLiSTTGPU() {
    if (stream_ && stream_owned_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

double ZhangLiSTTGPU::u(double Ms) const {
    const double Jmag = J_.norm();
    double u_val = P_ * constants::mu_B * Jmag / (constants::e_charge * Ms);
    if (thiaville_u_) u_val /= (1.0 + xi_ * xi_);   // mumax3 convention
    return u_val;
}

void ZhangLiSTTGPU::accumulate_gpu_ptr(
    const GReal* d_m, const Material& mat, GReal* d_dm_out) const
{
    const double u_val = u(mat.Ms);
    if (u_val == 0.0 || J_.norm() < 1e-30) return;

    const double Jmag  = J_.norm();
    const double jhatx = J_.x / Jmag;
    const double jhaty = J_.y / Jmag;
    const double jhatz = J_.z / Jmag;

    const int N = static_cast<int>(nx_ * ny_ * nz_);
    auto s = static_cast<cudaStream_t>(stream_);
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    kernel_zhangli<<<blocks, threads, 0, s>>>(
        d_dm_out,
        d_m,
        static_cast<int>(nx_), static_cast<int>(ny_), static_cast<int>(nz_), N,
        dx_, dy_, dz_, u_val, xi_,
        jhatx, jhaty, jhatz);
    // No cudaGetLastError() — see SlonczewskiSTTGPU::accumulate_gpu_ptr.
}

}  // namespace micromag

#endif // MICROMAG_CUDA


