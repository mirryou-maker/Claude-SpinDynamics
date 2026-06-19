#pragma once
// spin_torque_gpu.hpp — GPU spin torque kernels (STT, SOT, Zhang-Li).
//
// GPU spin torques add to d_dm_out [1/s] (component-major [3×N]).
// They are applied AFTER launch_llg_torque() at each integrator stage:
//   H_eff → LLG torque → + spin torques → weighted accumulate → m update
//
// All classes take StructuredGrid at construction (for N, dx, dy, dz).
// Interface: ISpinTorqueGPU (pure virtual) + SpinTorqueSumGPU (compositor)
// Concrete: SlonczewskiSTTGPU, SpinOrbitTorqueGPU, ZhangLiSTTGPU
//
// Usage:
//   SlonczewskiSTTGPU stt(g, J, P, d, p_hat);
//   SpinTorqueSumGPU torques;
//   torques.add(stt);
//   integ.step(mat, demag, fields, torques);
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"
#include <vector>

namespace micromag {

// ---------------------------------------------------------------------------
// ISpinTorqueGPU — abstract interface for GPU spin torques.
//
// accumulate_gpu_ptr: adds dm/dt [1/s] to d_dm_out in-place.
// d_m and d_dm_out are component-major [3×N] on device.
// ---------------------------------------------------------------------------
class ISpinTorqueGPU {
public:
    virtual ~ISpinTorqueGPU() = default;

    virtual void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                                     double* d_dm_out) const = 0;
};

// ---------------------------------------------------------------------------
// SpinTorqueSumGPU — ordered compositor of ISpinTorqueGPU pointers.
// ---------------------------------------------------------------------------
class SpinTorqueSumGPU {
public:
    SpinTorqueSumGPU() = default;

    void add(ISpinTorqueGPU& t) { terms_.push_back(&t); }
    void clear()                { terms_.clear(); }
    std::size_t size() const    { return terms_.size(); }

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_dm_out) const {
        for (auto* t : terms_)
            t->accumulate_gpu_ptr(d_m, mat, d_dm_out);
    }

private:
    std::vector<ISpinTorqueGPU*> terms_;
};

// ---------------------------------------------------------------------------
// SlonczewskiSTTGPU — Slonczewski CPP spin-transfer torque
//
// τ = a_J [m×(m×p̂)] + b_J [m×p̂]
//   a_J = γ₀ħ J P / (2 e Ms d)   [1/s]
//   b_J = -β a_J
// ---------------------------------------------------------------------------
class SlonczewskiSTTGPU : public ISpinTorqueGPU {
public:
    // grid  : simulation grid (needed for N)
    // J     : current density [A/m²] (signed; >0 = e⁻ from FL to RL)
    // P     : spin polarisation [0,1]
    // d     : free-layer thickness [m]
    // p     : reference polarisation direction (normalised internally)
    // beta  : field-like/damping-like ratio (b_J = -β a_J)
    SlonczewskiSTTGPU(const StructuredGrid& grid,
                       Real J, Real P, Real d, Vec3 p, Real beta = 0.0);
    ~SlonczewskiSTTGPU();

    SlonczewskiSTTGPU(const SlonczewskiSTTGPU&)            = delete;
    SlonczewskiSTTGPU& operator=(const SlonczewskiSTTGPU&) = delete;

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_dm_out) const override;

    Real J()    const { return J_; }
    Real P()    const { return P_; }
    Real d()    const { return d_; }
    Real beta() const { return beta_; }
    Vec3 p()    const { return p_; }

    void set_J(Real J)       { J_ = J; }
    void set_P(Real P)       { P_ = P; }
    void set_beta(Real beta) { beta_ = beta; }

    double a_J(double Ms) const;

private:
    Index N_;
    Real J_, P_, d_, beta_;
    Vec3 p_;
    void* stream_ = nullptr;
};

// ---------------------------------------------------------------------------
// SpinOrbitTorqueGPU — spin Hall effect (SOT)
//
// τ = a_SOT [η_DL m×(m×σ̂) + η_FL (m×σ̂)]
//   a_SOT = γ₀ħ J_c θ_SH / (2 e Ms d_fm)   [1/s]
// ---------------------------------------------------------------------------
class SpinOrbitTorqueGPU : public ISpinTorqueGPU {
public:
    // grid    : simulation grid (needed for N)
    // J_c     : charge current density [A/m²] (signed)
    // theta_SH: spin Hall angle (signed; e.g. -0.07 for Ta, +0.12 for Pt)
    // d_fm    : FM thickness [m]
    // sigma   : spin-Hall polarisation direction (normalised internally)
    // eta_DL  : damping-like efficiency (default 1)
    // eta_FL  : field-like efficiency   (default 0)
    SpinOrbitTorqueGPU(const StructuredGrid& grid,
                        Real J_c, Real theta_SH, Real d_fm, Vec3 sigma,
                        Real eta_DL = 1.0, Real eta_FL = 0.0);
    ~SpinOrbitTorqueGPU();

    SpinOrbitTorqueGPU(const SpinOrbitTorqueGPU&)            = delete;
    SpinOrbitTorqueGPU& operator=(const SpinOrbitTorqueGPU&) = delete;

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_dm_out) const override;

    Real J_c()      const { return J_c_; }
    Real theta_SH() const { return theta_SH_; }
    Real d_fm()     const { return d_fm_; }
    Real eta_DL()   const { return eta_DL_; }
    Real eta_FL()   const { return eta_FL_; }
    Vec3 sigma()    const { return sigma_; }

    void set_J_c(Real J_c)     { J_c_ = J_c; }
    void set_theta_SH(Real th) { theta_SH_ = th; }
    void set_eta_DL(Real e)    { eta_DL_ = e; }
    void set_eta_FL(Real e)    { eta_FL_ = e; }

    double a_SOT(double Ms) const;

private:
    Index N_;
    Real J_c_, theta_SH_, d_fm_, eta_DL_, eta_FL_;
    Vec3 sigma_;
    void* stream_ = nullptr;
};

// ---------------------------------------------------------------------------
// ZhangLiSTTGPU — current-driven domain wall motion (CIP-STT)
//
// τ = u [(ĵ·∇)m − ξ m×(ĵ·∇)m]
//   u = P μ_B |J| / (e Ms)   [m/s]
// Gradient by finite differences (Neumann BC, 1st-order at boundaries).
// ---------------------------------------------------------------------------
class ZhangLiSTTGPU : public ISpinTorqueGPU {
public:
    // grid : simulation grid (for N, nx, ny, nz, dx, dy, dz)
    // J    : current density vector [A/m²], e.g. {1e12, 0, 0}
    // P    : spin polarisation [0,1]
    // xi   : non-adiabaticity parameter (typically 0.01–0.1)
    ZhangLiSTTGPU(const StructuredGrid& grid, Vec3 J, Real P, Real xi);
    ~ZhangLiSTTGPU();

    ZhangLiSTTGPU(const ZhangLiSTTGPU&)            = delete;
    ZhangLiSTTGPU& operator=(const ZhangLiSTTGPU&) = delete;

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_dm_out) const override;

    Vec3 J()   const { return J_; }
    Real P()   const { return P_; }
    Real xi()  const { return xi_; }

    void set_J(Vec3 J)   { J_  = J;  }
    void set_P(Real P)   { P_  = P;  }
    void set_xi(Real xi) { xi_ = xi; }

    double u(double Ms) const;

private:
    Index nx_, ny_, nz_;
    Real  dx_, dy_, dz_;
    Vec3  J_;
    Real  P_, xi_;
    void* stream_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
