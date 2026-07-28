// batched_macrospin_gpu.cu — Task 2 Phase 2.0: replica-batched macrospin.
// One thread per replica advances a full Depondt–Mertens step (predictor +
// corrector) for one single-cell macrospin. See the header for the physics.
//
// Phase 0 (multi-vendor seam): runtime/launch/RNG go through gpu_backend.hpp /
// gpu_rng.hpp instead of raw CUDA — a pure 1:1 substitution (bitwise identical).
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include <cmath>
#include <stdexcept>
#include <vector>

#include "micromag/batched_macrospin_gpu.hpp"
#include "micromag/gpu_backend.hpp"   // G0-1 runtime wrappers, G0-2 GPU_LAUNCH
#include "micromag/gpu_rng.hpp"       // G0-4 device Philox helper
#include "micromag/gpu_real.hpp"
#include "micromag/types.hpp"

namespace micromag {

namespace mg = micromag::gpu;

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
    if (sigma > 0.0)
        mg::philox_normal3(seed, sub, off, eta.x, eta.y, eta.z);

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

    d_m_      = mg::malloc(size_t(R_) * 3 * sizeof(GReal));
    d_J_      = mg::malloc(size_t(R_) * sizeof(double));
    d_T_      = mg::malloc(size_t(R_) * sizeof(double));
    d_active_ = mg::malloc(size_t(R_) * sizeof(int));
    d_reason_ = mg::malloc(size_t(R_) * sizeof(int));
    d_scount_ = mg::malloc(size_t(R_) * sizeof(long long));
    d_rngid_  = mg::malloc(size_t(R_) * sizeof(unsigned));

    // initial state m = +easy, J = 0, T = 0
    std::vector<GReal> m0(size_t(R_) * 3);
    for (int r = 0; r < R_; ++r) {
        m0[r*3+0] = static_cast<GReal>(cfg_.easy.x);
        m0[r*3+1] = static_cast<GReal>(cfg_.easy.y);
        m0[r*3+2] = static_cast<GReal>(cfg_.easy.z);
    }
    mg::memcpy(d_m_, m0.data(), m0.size() * sizeof(GReal), mg::MemcpyKind::H2D);
    mg::memset(d_J_, 0, size_t(R_) * sizeof(double));
    mg::memset(d_T_, 0, size_t(R_) * sizeof(double));
    // all replicas active, reason 0, step counter 0, RNG stream id 0
    std::vector<int> ones(R_, 1);
    mg::memcpy(d_active_, ones.data(), size_t(R_)*sizeof(int), mg::MemcpyKind::H2D);
    mg::memset(d_reason_, 0, size_t(R_) * sizeof(int));
    mg::memset(d_scount_, 0, size_t(R_) * sizeof(long long));
    mg::memset(d_rngid_,  0, size_t(R_) * sizeof(unsigned));
}

BatchedMacrospinGPU::~BatchedMacrospinGPU() {
    mg::free(d_m_);      mg::free(d_J_);      mg::free(d_T_);
    mg::free(d_active_); mg::free(d_reason_); mg::free(d_scount_); mg::free(d_rngid_);
}

void BatchedMacrospinGPU::set_J(const std::vector<double>& J) {
    if (int(J.size()) != R_)
        throw std::invalid_argument("BatchedMacrospinGPU::set_J: size != R");
    mg::memcpy(d_J_, J.data(), J.size() * sizeof(double), mg::MemcpyKind::H2D);
}

void BatchedMacrospinGPU::set_T(const std::vector<double>& T) {
    if (int(T.size()) != R_)
        throw std::invalid_argument("BatchedMacrospinGPU::set_T: size != R");
    mg::memcpy(d_T_, T.data(), T.size() * sizeof(double), mg::MemcpyKind::H2D);
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
    mg::memcpy(d_m_, buf.data(), buf.size() * sizeof(GReal), mg::MemcpyKind::H2D);
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
        GPU_LAUNCH(macrospin_step_kernel, nblocks(R_), TPB, 0, /*stream=*/0,
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
        mg::check_last("macrospin_step_kernel");
        ++step_index_;
    }
    mg::device_sync();
}

std::vector<double> BatchedMacrospinGPU::get_state() const {
    std::vector<GReal> buf(size_t(R_) * 3);
    mg::memcpy(buf.data(), d_m_, buf.size() * sizeof(GReal), mg::MemcpyKind::D2H);
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
    mg::memcpy(v.data(), d_active_, size_t(R_)*sizeof(int), mg::MemcpyKind::D2H);
    return v;
}
std::vector<int> BatchedMacrospinGPU::get_reason() const {
    std::vector<int> v(R_);
    mg::memcpy(v.data(), d_reason_, size_t(R_)*sizeof(int), mg::MemcpyKind::D2H);
    return v;
}
std::vector<long> BatchedMacrospinGPU::get_stepcount() const {
    std::vector<long long> t(R_);
    mg::memcpy(t.data(), d_scount_, size_t(R_)*sizeof(long long), mg::MemcpyKind::D2H);
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
    mg::memcpy(m.data(),      d_m_,      m.size()*sizeof(GReal),       mg::MemcpyKind::D2H);
    mg::memcpy(active.data(), d_active_, size_t(R_)*sizeof(int),       mg::MemcpyKind::D2H);
    mg::memcpy(reason.data(), d_reason_, size_t(R_)*sizeof(int),       mg::MemcpyKind::D2H);
    mg::memcpy(scount.data(), d_scount_, size_t(R_)*sizeof(long long), mg::MemcpyKind::D2H);
    mg::memcpy(rngid.data(),  d_rngid_,  size_t(R_)*sizeof(unsigned),  mg::MemcpyKind::D2H);

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
    mg::memcpy(d_m_,      m.data(),      m.size()*sizeof(GReal),       mg::MemcpyKind::H2D);
    mg::memcpy(d_active_, active.data(), size_t(R_)*sizeof(int),       mg::MemcpyKind::H2D);
    mg::memcpy(d_reason_, reason.data(), size_t(R_)*sizeof(int),       mg::MemcpyKind::H2D);
    mg::memcpy(d_scount_, scount.data(), size_t(R_)*sizeof(long long), mg::MemcpyKind::H2D);
    mg::memcpy(d_rngid_,  rngid.data(),  size_t(R_)*sizeof(unsigned),  mg::MemcpyKind::H2D);
}

}  // namespace micromag

#endif // MICROMAG_CUDA
