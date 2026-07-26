// batched_llg_gpu.cu — Task 2 Phase 2.1: multi-cell replica-batched LLG.
// Batched Depondt–Mertens step over R replicas × N cells with local fields
// (exchange Neumann + uniaxial + Zeeman) + Slonczewski STT + FDT thermal.
// See the header for layout. Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/batched_llg_gpu.hpp"
#include "micromag/gpu_real.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA error: ") +      \
                                     cudaGetErrorString(_e));           \
    } while (0)

namespace micromag {
namespace {

constexpr int TPB = 256;
inline int nblocks(long n) { return int((n + TPB - 1) / TPB); }

// H_eff = H_exchange(Neumann) + H_uniaxial + H_Zeeman  for every (replica,cell).
// One thread per (r,cell). Reads neighbours within the SAME replica.
__global__ void field_kernel(GReal* __restrict__ H, const GReal* __restrict__ m,
                             int R, int nx, int ny, int nz,
                             double pre_x, double pre_y, double pre_z,   // 2A/(μ0Ms d²)
                             double two_K1_over_mu0Ms, double ex, double ey, double ez,
                             double Hx, double Hy, double Hz)
{
    const long N = long(nx) * ny * nz;
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= R * N) return;
    const int r = int(tid / N);
    const long idx = tid - long(r) * N;
    const long base = long(r) * 3 * N;

    const int iz = int(idx / (long(nx) * ny));
    const int iy = int((idx / nx) % ny);
    const int ix = int(idx % nx);

    const double mx = m[base + 0*N + idx];
    const double my = m[base + 1*N + idx];
    const double mz = m[base + 2*N + idx];

    // exchange Laplacian (Neumann: out-of-range bond uses centre → contributes 0)
    double lx = 0, ly = 0, lz = 0;
    #define ADDB(cond, nidx, pre) if (cond) { long ni = (nidx); \
        lx += (double(m[base+0*N+ni]) - mx) * (pre); \
        ly += (double(m[base+1*N+ni]) - my) * (pre); \
        lz += (double(m[base+2*N+ni]) - mz) * (pre); }
    ADDB(ix+1 <  nx, idx + 1,              pre_x)
    ADDB(ix-1 >= 0,  idx - 1,              pre_x)
    ADDB(iy+1 <  ny, idx + nx,             pre_y)
    ADDB(iy-1 >= 0,  idx - nx,             pre_y)
    ADDB(iz+1 <  nz, idx + long(nx)*ny,    pre_z)
    ADDB(iz-1 >= 0,  idx - long(nx)*ny,    pre_z)
    #undef ADDB

    // uniaxial: (2K1/μ0Ms)(m·û)û
    const double mdotu = mx*ex + my*ey + mz*ez;
    const double ax = two_K1_over_mu0Ms * mdotu;

    H[base + 0*N + idx] = static_cast<GReal>(lx + ax*ex + Hx);
    H[base + 1*N + idx] = static_cast<GReal>(ly + ax*ey + Hy);
    H[base + 2*N + idx] = static_cast<GReal>(lz + ax*ez + Hz);
}

// H += σ_r · η,  σ_r per replica (per-cell T = replica T). Philox keyed by
// (seed, subsequence = r*N+cell, offset). Same offset in predictor & corrector.
__global__ void thermal_kernel(GReal* __restrict__ H, const double* __restrict__ Tarr,
                               int R, long N, double Ms, double alpha,
                               double V, double dt, unsigned seed,
                               unsigned long long offset)
{
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= R * N) return;
    const int r = int(tid / N);
    const long idx = tid - long(r) * N;
    const double T = Tarr[r];
    if (!(T > 0.0) || !(Ms > 0.0)) return;
    const double mu0 = 4.0e-7 * 3.14159265358979323846;
    const double gamma0 = 1.760859630e11;
    const double sigma = sqrt(2.0 * alpha * 1.380649e-23 * T /
                              (mu0*mu0 * Ms * gamma0 * V * dt));
    curandStatePhilox4_32_10_t st;
    curand_init((unsigned long long)seed, (unsigned long long)tid, offset, &st);
    const long base = long(r) * 3 * N;
    H[base + 0*N + idx] = static_cast<GReal>(double(H[base+0*N+idx]) + sigma*curand_normal_double(&st));
    H[base + 1*N + idx] = static_cast<GReal>(double(H[base+1*N+idx]) + sigma*curand_normal_double(&st));
    H[base + 2*N + idx] = static_cast<GReal>(double(H[base+2*N+idx]) + sigma*curand_normal_double(&st));
}

// ω = gp(H + α m×H) + ω_STT,  ω_STT = −a_J(m×p̂) − b_J p̂,
// a_J = base_r·ε(m·p̂), base_r = γ₀ħ J[r]/(2 e Ms d), b_J = −β a_J.
__global__ void omega_kernel(GReal* __restrict__ w, const GReal* __restrict__ m,
                             const GReal* __restrict__ H, const double* __restrict__ Jarr,
                             int R, long N, double gp, double alpha,
                             double g0hbar_over_2eMsd, double Pval, double lam2,
                             double beta, double px, double py, double pz)
{
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= R * N) return;
    const int r = int(tid / N);
    const long idx = tid - long(r) * N;
    const long base = long(r) * 3 * N;
    const double mx = m[base+0*N+idx], my = m[base+1*N+idx], mz = m[base+2*N+idx];
    const double hx = H[base+0*N+idx], hy = H[base+1*N+idx], hz = H[base+2*N+idx];
    // m×H
    const double cx = my*hz - mz*hy, cy = mz*hx - mx*hz, cz = mx*hy - my*hx;
    double wx = gp*(hx + alpha*cx), wy = gp*(hy + alpha*cy), wz = gp*(hz + alpha*cz);
    // STT
    const double mdotp = mx*px + my*py + mz*pz;
    const double eps = Pval * lam2 / ((lam2 + 1.0) + (lam2 - 1.0)*mdotp);
    const double aj = g0hbar_over_2eMsd * Jarr[r] * eps;
    const double bj = -beta * aj;
    const double mpx = my*pz - mz*py, mpy = mz*px - mx*pz, mpz = mx*py - my*px;
    wx += -aj*mpx - bj*px; wy += -aj*mpy - bj*py; wz += -aj*mpz - bj*pz;
    w[base+0*N+idx] = static_cast<GReal>(wx);
    w[base+1*N+idx] = static_cast<GReal>(wy);
    w[base+2*N+idx] = static_cast<GReal>(wz);
}

// Rodrigues rotation of m_in about ω̂ by θ=|ω|dt → m_out (component-major, R·N).
__global__ void rotate_kernel(GReal* __restrict__ m_out, const GReal* __restrict__ m_in,
                              const GReal* __restrict__ w, int R, long N, double dt)
{
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= R * N) return;
    const int r = int(tid / N);
    const long idx = tid - long(r) * N;
    const long base = long(r) * 3 * N;
    const double wx = w[base+0*N+idx], wy = w[base+1*N+idx], wz = w[base+2*N+idx];
    const double mx = m_in[base+0*N+idx], my = m_in[base+1*N+idx], mz = m_in[base+2*N+idx];
    const double wn = sqrt(wx*wx + wy*wy + wz*wz);
    if (wn < 1e-30) {
        m_out[base+0*N+idx] = static_cast<GReal>(mx);
        m_out[base+1*N+idx] = static_cast<GReal>(my);
        m_out[base+2*N+idx] = static_cast<GReal>(mz);
        return;
    }
    const double th = wn*dt, s = sin(th), c = cos(th);
    const double ex = wx/wn, ey = wy/wn, ez = wz/wn;
    const double kx = ey*mz - ez*my, ky = ez*mx - ex*mz, kz = ex*my - ey*mx;
    const double edm = ex*mx + ey*my + ez*mz;
    m_out[base+0*N+idx] = static_cast<GReal>(mx*c + kx*s + ex*edm*(1.0-c));
    m_out[base+1*N+idx] = static_cast<GReal>(my*c + ky*s + ey*edm*(1.0-c));
    m_out[base+2*N+idx] = static_cast<GReal>(mz*c + kz*s + ez*edm*(1.0-c));
}

__global__ void avg_kernel(GReal* __restrict__ out, const GReal* __restrict__ a,
                           const GReal* __restrict__ b, long n) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<GReal>(0.5*(double(a[i]) + double(b[i])));
}

}  // namespace

// ---------------------------------------------------------------------------
BatchedLLGGPU::BatchedLLGGPU(int R, const StructuredGrid& grid,
                             const BatchedLLGConfig& cfg, Real dt, unsigned seed)
    : R_(R), nx_(int(grid.nx())), ny_(int(grid.ny())), nz_(int(grid.nz())),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()), dt_(dt), seed_(seed), cfg_(cfg)
{
    if (R <= 0)        throw std::invalid_argument("BatchedLLGGPU: R must be > 0");
    if (dt <= Real{0}) throw std::invalid_argument("BatchedLLGGPU: dt must be > 0");
    if (cfg.Ms <= Real{0}) throw std::invalid_argument("BatchedLLGGPU: Ms must be > 0");
    N_ = nx_ * ny_ * nz_;

    const Real en = std::sqrt(cfg_.easy.dot(cfg_.easy));
    if (en > Real{1e-30}) cfg_.easy = cfg_.easy / en;
    const Real pn = std::sqrt(cfg_.p.dot(cfg_.p));
    if (pn > Real{1e-30}) cfg_.p = cfg_.p / pn;

    const size_t n3 = size_t(R_) * 3 * N_;
    CUDA_CHECK(cudaMalloc(&d_m_,  n3 * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_m0_, n3 * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H_,  n3 * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_w1_, n3 * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_w2_, n3 * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_J_,  size_t(R_) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_T_,  size_t(R_) * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_J_, 0, size_t(R_) * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_T_, 0, size_t(R_) * sizeof(double)));
    set_uniform(cfg_.easy.x, cfg_.easy.y, cfg_.easy.z);
}

BatchedLLGGPU::~BatchedLLGGPU() {
    if (d_m_) cudaFree(d_m_);  if (d_m0_) cudaFree(d_m0_); if (d_H_) cudaFree(d_H_);
    if (d_w1_) cudaFree(d_w1_); if (d_w2_) cudaFree(d_w2_);
    if (d_J_) cudaFree(d_J_);  if (d_T_) cudaFree(d_T_);
}

void BatchedLLGGPU::set_J(const std::vector<double>& J) {
    if (int(J.size()) != R_) throw std::invalid_argument("set_J: size != R");
    CUDA_CHECK(cudaMemcpy(d_J_, J.data(), J.size()*sizeof(double), cudaMemcpyHostToDevice));
}
void BatchedLLGGPU::set_T(const std::vector<double>& T) {
    if (int(T.size()) != R_) throw std::invalid_argument("set_T: size != R");
    CUDA_CHECK(cudaMemcpy(d_T_, T.data(), T.size()*sizeof(double), cudaMemcpyHostToDevice));
}

void BatchedLLGGPU::set_uniform(double mx, double my, double mz) {
    double n = std::sqrt(mx*mx + my*my + mz*mz);
    if (n > 1e-30) { mx/=n; my/=n; mz/=n; }
    std::vector<GReal> buf(size_t(R_) * 3 * N_);
    for (int r = 0; r < R_; ++r) {
        GReal* b = buf.data() + size_t(r)*3*N_;
        for (int i = 0; i < N_; ++i) { b[0*N_+i]=GReal(mx); b[1*N_+i]=GReal(my); b[2*N_+i]=GReal(mz); }
    }
    CUDA_CHECK(cudaMemcpy(d_m_, buf.data(), buf.size()*sizeof(GReal), cudaMemcpyHostToDevice));
}

void BatchedLLGGPU::set_state(const std::vector<double>& m) {
    if (int(m.size()) != R_ * 3 * N_) throw std::invalid_argument("set_state: size != R*3N");
    std::vector<GReal> buf(m.size());
    for (int r = 0; r < R_; ++r) {
        const double* s = m.data() + size_t(r)*3*N_;
        GReal* b = buf.data() + size_t(r)*3*N_;
        for (int i = 0; i < N_; ++i) {
            double x=s[0*N_+i], y=s[1*N_+i], z=s[2*N_+i];
            double n = std::sqrt(x*x+y*y+z*z); if (n>1e-30){x/=n;y/=n;z/=n;}
            b[0*N_+i]=GReal(x); b[1*N_+i]=GReal(y); b[2*N_+i]=GReal(z);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_m_, buf.data(), buf.size()*sizeof(GReal), cudaMemcpyHostToDevice));
}

void BatchedLLGGPU::substep_(unsigned long long noise_offset) {
    const long N = N_;
    const long RN = long(R_) * N;
    const long RN3 = long(R_) * 3 * N;
    const double alpha = double(cfg_.alpha);
    const double gp = constants::gamma_0 * constants::mu_0 / (1.0 + alpha*alpha);
    const double preM = 2.0 * double(cfg_.A) / (constants::mu_0 * double(cfg_.Ms));
    const double pre_x = preM / (dx_*dx_), pre_y = preM / (dy_*dy_), pre_z = preM / (dz_*dz_);
    const double twoK = 2.0 * double(cfg_.K1) / (constants::mu_0 * double(cfg_.Ms));
    const double g0hbar = constants::gamma_0 * constants::hbar /
                          (2.0 * constants::e_charge * double(cfg_.Ms) * double(cfg_.d_free));
    const double lam2 = double(cfg_.Lambda) * double(cfg_.Lambda);
    const double V = dx_*dy_*dz_;
    const double dt = double(dt_);

    GReal* m  = static_cast<GReal*>(d_m_);
    GReal* m0 = static_cast<GReal*>(d_m0_);
    GReal* H  = static_cast<GReal*>(d_H_);
    GReal* w1 = static_cast<GReal*>(d_w1_);
    GReal* w2 = static_cast<GReal*>(d_w2_);
    const double* J = static_cast<const double*>(d_J_);
    const double* T = static_cast<const double*>(d_T_);

    CUDA_CHECK(cudaMemcpy(m0, m, RN3*sizeof(GReal), cudaMemcpyDeviceToDevice));
    // predictor
    field_kernel<<<nblocks(RN), TPB>>>(H, m0, R_, nx_, ny_, nz_, pre_x, pre_y, pre_z,
        twoK, cfg_.easy.x, cfg_.easy.y, cfg_.easy.z, cfg_.H_ext.x, cfg_.H_ext.y, cfg_.H_ext.z);
    thermal_kernel<<<nblocks(RN), TPB>>>(H, T, R_, N, double(cfg_.Ms), alpha, V, dt, seed_, noise_offset);
    omega_kernel<<<nblocks(RN), TPB>>>(w1, m0, H, J, R_, N, gp, alpha, g0hbar,
        double(cfg_.P), lam2, double(cfg_.beta), cfg_.p.x, cfg_.p.y, cfg_.p.z);
    rotate_kernel<<<nblocks(RN), TPB>>>(m, m0, w1, R_, N, dt);
    // corrector (same noise realisation)
    field_kernel<<<nblocks(RN), TPB>>>(H, m, R_, nx_, ny_, nz_, pre_x, pre_y, pre_z,
        twoK, cfg_.easy.x, cfg_.easy.y, cfg_.easy.z, cfg_.H_ext.x, cfg_.H_ext.y, cfg_.H_ext.z);
    thermal_kernel<<<nblocks(RN), TPB>>>(H, T, R_, N, double(cfg_.Ms), alpha, V, dt, seed_, noise_offset);
    omega_kernel<<<nblocks(RN), TPB>>>(w2, m, H, J, R_, N, gp, alpha, g0hbar,
        double(cfg_.P), lam2, double(cfg_.beta), cfg_.p.x, cfg_.p.y, cfg_.p.z);
    avg_kernel<<<nblocks(RN3), TPB>>>(w1, w1, w2, RN3);
    rotate_kernel<<<nblocks(RN), TPB>>>(m, m0, w1, R_, N, dt);
    CUDA_CHECK(cudaGetLastError());
}

void BatchedLLGGPU::run(int n_steps) {
    for (int s = 0; s < n_steps; ++s) {
        substep_((unsigned long long)step_index_);
        ++step_index_;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<double> BatchedLLGGPU::get_state() const {
    std::vector<GReal> buf(size_t(R_) * 3 * N_);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_m_, buf.size()*sizeof(GReal), cudaMemcpyDeviceToHost));
    return std::vector<double>(buf.begin(), buf.end());
}

std::vector<double> BatchedLLGGPU::get_avg_m() const {
    std::vector<double> s = get_state();
    std::vector<double> avg(size_t(R_) * 3, 0.0);
    for (int r = 0; r < R_; ++r) {
        const double* b = s.data() + size_t(r)*3*N_;
        double sx=0, sy=0, sz=0;
        for (int i = 0; i < N_; ++i) { sx+=b[0*N_+i]; sy+=b[1*N_+i]; sz+=b[2*N_+i]; }
        avg[r*3+0]=sx/N_; avg[r*3+1]=sy/N_; avg[r*3+2]=sz/N_;
    }
    return avg;
}

}  // namespace micromag

#endif // MICROMAG_CUDA
