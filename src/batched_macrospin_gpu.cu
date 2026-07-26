// batched_macrospin_gpu.cu — Task 2 Phase 2.0: replica-batched macrospin.
// One thread per replica advances a full Depondt–Mertens step (predictor +
// corrector) for one single-cell macrospin. See the header for the physics.
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <string>

#include "micromag/batched_macrospin_gpu.hpp"
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
inline int nblocks(int n) { return (n + TPB - 1) / TPB; }

struct D3 { double x, y, z; };
__device__ inline D3 cross(const D3& a, const D3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__device__ inline double dot(const D3& a, const D3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// ω(m) = ω_LLG + ω_STT.
//   ω_LLG = gp·(H + α m×H),        H = H_ani(m) + H_ext (+ σ·η if with_noise)
//   ω_STT = −a_J(m×p̂) − b_J p̂     (so ω_STT×m = a_J[m×(m×p̂)] + b_J[m×p̂])
// gp = γ₀μ₀/(1+α²).  a_J = base·ε(m·p̂), base = γ₀ħJ/(2 e Ms d), b_J = −β a_J.
__device__ inline D3 omega_of(const D3& m, double gp, double alpha,
                              double two_K1_over_mu0Ms, const D3& easy,
                              const D3& Hext, const D3& eta, double sigma,
                              double aj_base, double P, double lam2, double beta,
                              const D3& p)
{
    // H_eff (A/m)
    const double mdotu = dot(m, easy);
    D3 H = {two_K1_over_mu0Ms * mdotu * easy.x + Hext.x + sigma * eta.x,
            two_K1_over_mu0Ms * mdotu * easy.y + Hext.y + sigma * eta.y,
            two_K1_over_mu0Ms * mdotu * easy.z + Hext.z + sigma * eta.z};
    const D3 mxH = cross(m, H);
    D3 w = {gp * (H.x + alpha * mxH.x),
            gp * (H.y + alpha * mxH.y),
            gp * (H.z + alpha * mxH.z)};
    // STT
    const double mdotp = dot(m, p);
    const double eps = P * lam2 / ((lam2 + 1.0) + (lam2 - 1.0) * mdotp);
    const double aj = aj_base * eps;
    const double bj = -beta * aj;
    const D3 mxp = cross(m, p);
    w.x += -aj * mxp.x - bj * p.x;
    w.y += -aj * mxp.y - bj * p.y;
    w.z += -aj * mxp.z - bj * p.z;
    return w;
}

// Rodrigues rotation of m about ω̂ by θ = |ω|·dt (norm-exact; identity at |ω|→0).
__device__ inline D3 rotate(const D3& m, const D3& w, double dt) {
    const double wn = sqrt(w.x*w.x + w.y*w.y + w.z*w.z);
    if (wn < 1e-30) return m;
    const double th = wn * dt, s = sin(th), c = cos(th);
    const D3 e = {w.x/wn, w.y/wn, w.z/wn};
    const D3 k = cross(e, m);
    const double edm = dot(e, m);
    return {m.x*c + k.x*s + e.x*edm*(1.0 - c),
            m.y*c + k.y*s + e.y*edm*(1.0 - c),
            m.z*c + k.z*s + e.z*edm*(1.0 - c)};
}

__global__ void macrospin_step_kernel(
    GReal* __restrict__ m, const double* __restrict__ Jarr,
    const double* __restrict__ Tarr, int R,
    double Ms, double alpha, double gp, double V,
    double two_K1_over_mu0Ms, double ex, double ey, double ez,
    double Hx, double Hy, double Hz,
    double gamma0_hbar_over_2eMsd,   // base without J: γ₀ħ/(2 e Ms d)
    double P, double lam2, double beta, double px, double py, double pz,
    double dt, unsigned seed,
    // retire/refill state (Phase 2.4): per-replica RNG stream + counters.
    int* __restrict__ active, int* __restrict__ reason,
    long long* __restrict__ scount, const unsigned* __restrict__ rngid,
    int retire_on, double mz_switch, double mz_reset)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    if (active[r] == 0) return;   // retired: frozen, no update
    // per-replica RNG stream: subsequence folds the stream id (bumped on refill),
    // offset is this replica's OWN active-step counter (continues after refill).
    const unsigned long long sub = (unsigned long long)r
                                 + (unsigned long long)rngid[r] * (unsigned long long)R;
    const unsigned long long off = (unsigned long long)scount[r];

    D3 mv = {double(m[r*3+0]), double(m[r*3+1]), double(m[r*3+2])};
    const D3 easy = {ex, ey, ez};
    const D3 Hext = {Hx, Hy, Hz};
    const D3 p    = {px, py, pz};
    const double aj_base = gamma0_hbar_over_2eMsd * Jarr[r];

    // thermal σ = √(2α k_B T/(μ₀² Ms γ₀ V dt))  (per-replica T; mirrors
    // DepondtMertensGPU::therm_sigma — the μ₀² is the A/m field-unit correction).
    const double T = Tarr[r];
    double sigma = 0.0;
    if (T > 0.0 && Ms > 0.0) {
        const double mu0 = 4.0e-7 * 3.14159265358979323846;
        const double gamma0 = 1.760859630e11;
        const double num = 2.0 * alpha * 1.380649e-23 * T;
        const double den = mu0 * mu0 * Ms * gamma0 * V * dt;
        sigma = sqrt(num / den);
    }

    // one Philox draw of 3 normals per replica per step (reused predictor+corrector)
    D3 eta = {0, 0, 0};
    if (sigma > 0.0) {
        curandStatePhilox4_32_10_t st;
        curand_init((unsigned long long)seed, sub, off, &st);
        eta.x = curand_normal_double(&st);
        eta.y = curand_normal_double(&st);
        eta.z = curand_normal_double(&st);
    }

    // predictor
    D3 w1 = omega_of(mv, gp, alpha, two_K1_over_mu0Ms, easy, Hext, eta, sigma,
                     aj_base, P, lam2, beta, p);
    D3 mstar = rotate(mv, w1, dt);
    // corrector (same noise realisation)
    D3 w2 = omega_of(mstar, gp, alpha, two_K1_over_mu0Ms, easy, Hext, eta, sigma,
                     aj_base, P, lam2, beta, p);
    D3 wbar = {0.5*(w1.x+w2.x), 0.5*(w1.y+w2.y), 0.5*(w1.z+w2.z)};
    D3 mnew = rotate(mv, wbar, dt);

    m[r*3+0] = static_cast<GReal>(mnew.x);
    m[r*3+1] = static_cast<GReal>(mnew.y);
    m[r*3+2] = static_cast<GReal>(mnew.z);

    // this replica took one active step
    scount[r] += 1;
    // stop check on the new state (variable-length trial retirement)
    if (retire_on) {
        if (mnew.z < mz_switch)      { active[r] = 0; reason[r] = 1; }  // SWITCHED
        else if (mnew.z > mz_reset)  { active[r] = 0; reason[r] = 2; }  // RESET
    }
}

}  // namespace

// ---------------------------------------------------------------------------
BatchedMacrospinGPU::BatchedMacrospinGPU(int R, const BatchedMacrospinConfig& cfg,
                                         Real dt, unsigned seed)
    : R_(R), dt_(dt), seed_(seed), cfg_(cfg)
{
    if (R <= 0)        throw std::invalid_argument("BatchedMacrospinGPU: R must be > 0");
    if (dt <= Real{0}) throw std::invalid_argument("BatchedMacrospinGPU: dt must be > 0");
    if (cfg.Ms <= Real{0}) throw std::invalid_argument("BatchedMacrospinGPU: Ms must be > 0");
    if (cfg.d_free <= Real{0}) throw std::invalid_argument("BatchedMacrospinGPU: d_free must be > 0");

    // normalise easy axis & polarisation
    const Real en = std::sqrt(cfg_.easy.dot(cfg_.easy));
    if (en > Real{1e-30}) cfg_.easy = cfg_.easy / en;
    const Real pn = std::sqrt(cfg_.p.dot(cfg_.p));
    if (pn > Real{1e-30}) cfg_.p = cfg_.p / pn;

    CUDA_CHECK(cudaMalloc(&d_m_, size_t(R_) * 3 * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_J_, size_t(R_) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_T_, size_t(R_) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_active_, size_t(R_) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reason_, size_t(R_) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scount_, size_t(R_) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_rngid_,  size_t(R_) * sizeof(unsigned)));

    // initial state m = +easy, J = 0, T = 0
    std::vector<GReal> m0(size_t(R_) * 3);
    for (int r = 0; r < R_; ++r) {
        m0[r*3+0] = static_cast<GReal>(cfg_.easy.x);
        m0[r*3+1] = static_cast<GReal>(cfg_.easy.y);
        m0[r*3+2] = static_cast<GReal>(cfg_.easy.z);
    }
    CUDA_CHECK(cudaMemcpy(d_m_, m0.data(), m0.size() * sizeof(GReal),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_J_, 0, size_t(R_) * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_T_, 0, size_t(R_) * sizeof(double)));
    // all replicas active, reason 0, step counter 0, RNG stream id 0
    std::vector<int> ones(R_, 1);
    CUDA_CHECK(cudaMemcpy(d_active_, ones.data(), size_t(R_)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_reason_, 0, size_t(R_) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_scount_, 0, size_t(R_) * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_rngid_,  0, size_t(R_) * sizeof(unsigned)));
}

BatchedMacrospinGPU::~BatchedMacrospinGPU() {
    if (d_m_) cudaFree(d_m_);
    if (d_J_) cudaFree(d_J_);
    if (d_T_) cudaFree(d_T_);
    if (d_active_) cudaFree(d_active_);
    if (d_reason_) cudaFree(d_reason_);
    if (d_scount_) cudaFree(d_scount_);
    if (d_rngid_)  cudaFree(d_rngid_);
}

void BatchedMacrospinGPU::set_J(const std::vector<double>& J) {
    if (int(J.size()) != R_)
        throw std::invalid_argument("BatchedMacrospinGPU::set_J: size != R");
    CUDA_CHECK(cudaMemcpy(d_J_, J.data(), J.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void BatchedMacrospinGPU::set_T(const std::vector<double>& T) {
    if (int(T.size()) != R_)
        throw std::invalid_argument("BatchedMacrospinGPU::set_T: size != R");
    CUDA_CHECK(cudaMemcpy(d_T_, T.data(), T.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void BatchedMacrospinGPU::set_state(const std::vector<double>& m) {
    if (int(m.size()) != R_ * 3)
        throw std::invalid_argument("BatchedMacrospinGPU::set_state: size != R*3");
    std::vector<GReal> buf(size_t(R_) * 3);
    for (int r = 0; r < R_; ++r) {
        double x = m[r*3+0], y = m[r*3+1], z = m[r*3+2];
        double n = std::sqrt(x*x + y*y + z*z);
        if (n > 1e-30) { x /= n; y /= n; z /= n; }
        buf[r*3+0] = static_cast<GReal>(x);
        buf[r*3+1] = static_cast<GReal>(y);
        buf[r*3+2] = static_cast<GReal>(z);
    }
    CUDA_CHECK(cudaMemcpy(d_m_, buf.data(), buf.size() * sizeof(GReal),
                          cudaMemcpyHostToDevice));
}

void BatchedMacrospinGPU::run(int n_steps) {
    const double alpha = double(cfg_.alpha);
    const double gp = constants::gamma_0 * constants::mu_0 / (1.0 + alpha * alpha);
    const double two_K1_over_mu0Ms =
        2.0 * double(cfg_.K1) / (constants::mu_0 * double(cfg_.Ms));
    const double gamma0_hbar_over_2eMsd =
        constants::gamma_0 * constants::hbar /
        (2.0 * constants::e_charge * double(cfg_.Ms) * double(cfg_.d_free));
    const double lam2 = double(cfg_.Lambda) * double(cfg_.Lambda);
    const double dt = double(dt_);

    GReal* d_m = static_cast<GReal*>(d_m_);
    const double* d_J = static_cast<const double*>(d_J_);
    const double* d_T = static_cast<const double*>(d_T_);
    int* d_active = static_cast<int*>(d_active_);
    int* d_reason = static_cast<int*>(d_reason_);
    long long* d_scount = static_cast<long long*>(d_scount_);
    const unsigned* d_rngid = static_cast<const unsigned*>(d_rngid_);

    for (int s = 0; s < n_steps; ++s) {
        macrospin_step_kernel<<<nblocks(R_), TPB>>>(
            d_m, d_J, d_T, R_,
            double(cfg_.Ms), alpha, gp, double(cfg_.V),
            two_K1_over_mu0Ms, double(cfg_.easy.x), double(cfg_.easy.y), double(cfg_.easy.z),
            double(cfg_.H_ext.x), double(cfg_.H_ext.y), double(cfg_.H_ext.z),
            gamma0_hbar_over_2eMsd,
            double(cfg_.P), lam2, double(cfg_.beta),
            double(cfg_.p.x), double(cfg_.p.y), double(cfg_.p.z),
            dt, seed_,
            d_active, d_reason, d_scount, d_rngid,
            retire_on_ ? 1 : 0, mz_switch_, mz_reset_);
        CUDA_CHECK(cudaGetLastError());
        ++step_index_;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<double> BatchedMacrospinGPU::get_state() const {
    std::vector<GReal> buf(size_t(R_) * 3);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_m_, buf.size() * sizeof(GReal),
                          cudaMemcpyDeviceToHost));
    std::vector<double> out(size_t(R_) * 3);
    for (size_t i = 0; i < out.size(); ++i) out[i] = double(buf[i]);
    return out;
}

std::vector<double> BatchedMacrospinGPU::get_mz() const {
    std::vector<double> s = get_state();
    std::vector<double> mz(R_);
    for (int r = 0; r < R_; ++r) mz[r] = s[r*3+2];
    return mz;
}

// --- Phase 2.4: retire / refill -------------------------------------------
void BatchedMacrospinGPU::enable_retire(double mz_switch, double mz_reset) {
    retire_on_  = true;
    mz_switch_  = mz_switch;
    mz_reset_   = mz_reset;
}

std::vector<int> BatchedMacrospinGPU::get_active() const {
    std::vector<int> v(R_);
    CUDA_CHECK(cudaMemcpy(v.data(), d_active_, size_t(R_)*sizeof(int), cudaMemcpyDeviceToHost));
    return v;
}
std::vector<int> BatchedMacrospinGPU::get_reason() const {
    std::vector<int> v(R_);
    CUDA_CHECK(cudaMemcpy(v.data(), d_reason_, size_t(R_)*sizeof(int), cudaMemcpyDeviceToHost));
    return v;
}
std::vector<long> BatchedMacrospinGPU::get_stepcount() const {
    std::vector<long long> t(R_);
    CUDA_CHECK(cudaMemcpy(t.data(), d_scount_, size_t(R_)*sizeof(long long), cudaMemcpyDeviceToHost));
    return std::vector<long>(t.begin(), t.end());
}

int BatchedMacrospinGPU::num_active() const {
    std::vector<int> a = get_active();
    int n = 0; for (int x : a) n += x; return n;
}

void BatchedMacrospinGPU::refill(const std::vector<int>& slots,
                                 const std::vector<double>& state) {
    if (slots.empty()) return;
    const bool have_state = !state.empty();
    if (have_state && int(state.size()) != int(slots.size()) * 3)
        throw std::invalid_argument("BatchedMacrospinGPU::refill: state size != slots*3");

    // Pull the arrays we mutate, edit on host, push back (refill is an
    // occasional host-side bookkeeping op, not a per-step hot path).
    std::vector<GReal> m(size_t(R_)*3);
    std::vector<int> active(R_), reason(R_);
    std::vector<long long> scount(R_);
    std::vector<unsigned> rngid(R_);
    CUDA_CHECK(cudaMemcpy(m.data(),      d_m_,      m.size()*sizeof(GReal),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(active.data(), d_active_, size_t(R_)*sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(reason.data(), d_reason_, size_t(R_)*sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scount.data(), d_scount_, size_t(R_)*sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rngid.data(),  d_rngid_,  size_t(R_)*sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (size_t k = 0; k < slots.size(); ++k) {
        int r = slots[k];
        if (r < 0 || r >= R_)
            throw std::invalid_argument("BatchedMacrospinGPU::refill: slot out of range");
        double x, y, z;
        if (have_state) { x=state[k*3+0]; y=state[k*3+1]; z=state[k*3+2]; }
        else            { x=cfg_.easy.x;  y=cfg_.easy.y;  z=cfg_.easy.z;  }
        double n = std::sqrt(x*x+y*y+z*z); if (n>1e-30){x/=n;y/=n;z/=n;}
        m[r*3+0]=GReal(x); m[r*3+1]=GReal(y); m[r*3+2]=GReal(z);
        active[r] = 1; reason[r] = 0; scount[r] = 0;
        rngid[r] += 1;   // FRESH, non-repeating noise stream for the new trial
    }
    CUDA_CHECK(cudaMemcpy(d_m_,      m.data(),      m.size()*sizeof(GReal),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active_, active.data(), size_t(R_)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reason_, reason.data(), size_t(R_)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scount_, scount.data(), size_t(R_)*sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rngid_,  rngid.data(),  size_t(R_)*sizeof(unsigned), cudaMemcpyHostToDevice));
}

}  // namespace micromag

#endif // MICROMAG_CUDA
