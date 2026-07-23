#pragma once
// relax_gpu.hpp — GPU-side energy minimisation (relax / minimize).
// All convergence checking stays on GPU; only ONE scalar D2H per check.
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "demag_gpu_iface.hpp"
#include "effective_field_gpu_iface.hpp"
#include "exchange_gpu.hpp"
#include "field.hpp"
#include "field_kernels_gpu.hpp"
#include "gpu_real.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// RelaxGPU — damping-only LLG energy minimisation, fully on GPU.
//
// Algorithm: mumax3 Relax() equivalent
//   dm/dt = -γ'αμ₀ m×(m×H_eff)  (no precession term)
//   Fixed Euler step, convergence when max|m×H_eff| < threshold [A/m].
//
// Usage:
//   RelaxGPU relax(grid);
//   relax.upload(m0);
//   relax.run(mat, demag, exch, zeeman);  // blocks until converged
//   relax.download(m_out);
// ---------------------------------------------------------------------------
// Hoisted to namespace scope (see RK45IntegratorGPUOptions note) so the
// `Options opts = {}` default arguments below compile on GCC. The in-class
// `using` alias preserves RelaxGPU::Options for callers and Python bindings.
struct RelaxGPUOptions {
    Real alpha_relax   = 1.0;    // effective alpha during relaxation
    Real threshold     = 1.0;    // |m×H|_max [A/m]
    Real dt            = 1e-12;  // fixed step [s]
    int  max_steps     = 500'000;
    int  check_every   = 200;    // convergence check interval (GPU iterations)
    bool throw_on_max  = false;  // throw if max_steps reached without convergence
};

class RelaxGPU {
public:
    using Options = RelaxGPUOptions;

    explicit RelaxGPU(const StructuredGrid& grid);
    ~RelaxGPU();

    RelaxGPU(const RelaxGPU&)            = delete;
    RelaxGPU& operator=(const RelaxGPU&) = delete;

    void upload(const VectorField3D& m);
    void download(VectorField3D& m) const;

    // Fixed-field overload.
    int run(const Material& mat,
            IDemagGPU& demag,
            ExchangeFieldGPU& exch,
            ZeemanFieldGPU& zeeman,
            UniaxialAnisotropyFieldGPU* aniso = nullptr,
            Options opts = {});

    // FieldSumGPU overload — supports DMI, cubic anisotropy, etc.
    int run(const Material& mat, IDemagGPU& demag,
            FieldSumGPU& extra_fields, Options opts = {});

    double max_torque_now(const Material& mat,
                          IDemagGPU& demag,
                          ExchangeFieldGPU& exch,
                          ZeemanFieldGPU& zeeman,
                          UniaxialAnisotropyFieldGPU* aniso = nullptr);

    double max_torque_now(const Material& mat, IDemagGPU& demag,
                          FieldSumGPU& extra_fields);

    const StructuredGrid& grid() const { return *grid_; }

private:
    const StructuredGrid* grid_;
    size_t                N_;

    GReal*  d_m_  = nullptr;   // [3N] current magnetisation (GReal for P11 float32)
    GReal*  d_H_  = nullptr;   // [3N] effective field scratch
    double* d_max_= nullptr;   // [1] reduction scalar output (CUB); stays double
    double* d_percell_ = nullptr; // [N] per-cell scratch (|m×H|² for CUB Max)
    void*   d_cub_tmp_ = nullptr; // CUB DeviceReduce temp storage (lazily sized)
    size_t  cub_bytes_ = 0;
    void*   stream_= nullptr;

    // Per-cell |m×H|² then cub::DeviceReduce::Max → sqrt = max torque [A/m].
    // Assumes d_H_ already holds H_eff for d_m_src.
    double reduce_max_torque(const GReal* d_m_src);

    void compute_H_eff(const Material& mat,
                       IDemagGPU& demag,
                       ExchangeFieldGPU& exch,
                       ZeemanFieldGPU& zeeman,
                       UniaxialAnisotropyFieldGPU* aniso);

    void compute_H_eff(const Material& mat, IDemagGPU& demag,
                       FieldSumGPU& extra_fields);
};

// ---------------------------------------------------------------------------
// MinimizeGPU — steepest-descent energy minimisation on GPU.
//
// Uses the same damping-only step as RelaxGPU but with adaptive step size:
// grows dt on successful energy decrease, shrinks on energy increase.
// Requires GPU energy computation (dot-product reduction).
// ---------------------------------------------------------------------------
struct MinimizeGPUOptions {
    Real threshold   = 1.0;    // |m×H|_max [A/m]
    Real dt_init     = 1e-12;
    Real dt_max      = 1e-10;
    Real dt_min      = 1e-17;
    int  max_steps   = 200'000;
    int  check_every = 100;
    bool throw_on_max = false;
};

class MinimizeGPU {
public:
    using Options = MinimizeGPUOptions;

    explicit MinimizeGPU(const StructuredGrid& grid);
    ~MinimizeGPU();

    MinimizeGPU(const MinimizeGPU&)            = delete;
    MinimizeGPU& operator=(const MinimizeGPU&) = delete;

    void upload(const VectorField3D& m);
    void download(VectorField3D& m) const;

    // Optional per-cell Ms: weights the line-search energy per cell
    // (E = -mu0/2 sum Ms_i m.H dV). Without it the uniform mat.Ms is used.
    void set_Ms_field(const ScalarField3D& Ms);
    void clear_Ms_field();
    bool has_Ms_field() const { return d_Ms_w_ != nullptr; }

    // Fixed-field overload.
    int run(const Material& mat,
            IDemagGPU& demag,
            ExchangeFieldGPU& exch,
            ZeemanFieldGPU& zeeman,
            UniaxialAnisotropyFieldGPU* aniso = nullptr,
            Options opts = {});

    // FieldSumGPU overload.
    int run(const Material& mat, IDemagGPU& demag,
            FieldSumGPU& extra_fields, Options opts = {});

    const StructuredGrid& grid() const { return *grid_; }

private:
    const StructuredGrid* grid_;
    size_t                N_;

    GReal*  d_m_       = nullptr;   // [3N] current magnetisation
    GReal*  d_m_trial_ = nullptr;   // [3N] trial step
    GReal*  d_H_       = nullptr;   // [3N] effective field scratch
    double* d_max_     = nullptr;   // [1] reduction scalar output (CUB); stays double
    double* d_energy_  = nullptr;   // [N] per-cell scratch (m·H or |m×H|²); double
    double* d_Ms_w_    = nullptr;   // [N] optional per-cell Ms weight for the
                                    // line-search energy (null = uniform mat.Ms)
    void*   d_cub_tmp_ = nullptr;   // CUB DeviceReduce temp storage (lazily sized)
    size_t  cub_bytes_ = 0;
    void*   stream_= nullptr;

    // CUB reductions over d_energy_ (assume d_H_ holds H_eff for d_m_src).
    double reduce_max_torque(const GReal* d_m_src);   // Max → sqrt = |m×H|max [A/m]
    double reduce_mdotH_sum(const GReal* d_m_src);    // Sum of m·H (unscaled)

    double compute_energy(const Material& mat,
                          IDemagGPU& demag,
                          ExchangeFieldGPU& exch,
                          ZeemanFieldGPU& zeeman,
                          UniaxialAnisotropyFieldGPU* aniso);

    double compute_energy(const Material& mat, IDemagGPU& demag,
                          FieldSumGPU& extra_fields);

    void compute_H_eff_for(GReal* d_m_src,
                           const Material& mat,
                           IDemagGPU& demag,
                           ExchangeFieldGPU& exch,
                           ZeemanFieldGPU& zeeman,
                           UniaxialAnisotropyFieldGPU* aniso);

    void compute_H_eff_for(GReal* d_m_src, const Material& mat,
                           IDemagGPU& demag, FieldSumGPU& extra_fields);
};

}  // namespace micromag

#endif // MICROMAG_CUDA
