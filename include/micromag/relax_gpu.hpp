#pragma once
// relax_gpu.hpp — GPU-side energy minimisation (relax / minimize).
// All convergence checking stays on GPU; only ONE scalar D2H per check.
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "demag_gpu_iface.hpp"
#include "exchange_gpu.hpp"
#include "field.hpp"
#include "field_kernels_gpu.hpp"
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
class RelaxGPU {
public:
    struct Options {
        Real alpha_relax   = 1.0;    // effective alpha during relaxation
        Real threshold     = 1.0;    // |m×H|_max [A/m]
        Real dt            = 1e-12;  // fixed step [s]
        int  max_steps     = 500'000;
        int  check_every   = 200;    // convergence check interval (GPU iterations)
        bool throw_on_max  = false;  // throw if max_steps reached without convergence
    };

    explicit RelaxGPU(const StructuredGrid& grid);
    ~RelaxGPU();

    RelaxGPU(const RelaxGPU&)            = delete;
    RelaxGPU& operator=(const RelaxGPU&) = delete;

    void upload(const VectorField3D& m);
    void download(VectorField3D& m) const;

    // Run relaxation until convergence or max_steps.
    // Returns number of steps taken.
    int run(const Material& mat,
            IDemagGPU& demag,
            ExchangeFieldGPU& exch,
            ZeemanFieldGPU& zeeman,
            UniaxialAnisotropyFieldGPU* aniso = nullptr,
            Options opts = {});

    // Query max torque on current GPU state (useful for monitoring).
    double max_torque_now(const Material& mat,
                          IDemagGPU& demag,
                          ExchangeFieldGPU& exch,
                          ZeemanFieldGPU& zeeman,
                          UniaxialAnisotropyFieldGPU* aniso = nullptr);

    const StructuredGrid& grid() const { return *grid_; }

private:
    const StructuredGrid* grid_;
    size_t                N_;

    double* d_m_  = nullptr;   // [3N] current magnetisation
    double* d_H_  = nullptr;   // [3N] effective field scratch
    double* d_max_= nullptr;   // [1] max torque reduction buffer
    void*   stream_= nullptr;

    void compute_H_eff(const Material& mat,
                       IDemagGPU& demag,
                       ExchangeFieldGPU& exch,
                       ZeemanFieldGPU& zeeman,
                       UniaxialAnisotropyFieldGPU* aniso);
};

// ---------------------------------------------------------------------------
// MinimizeGPU — steepest-descent energy minimisation on GPU.
//
// Uses the same damping-only step as RelaxGPU but with adaptive step size:
// grows dt on successful energy decrease, shrinks on energy increase.
// Requires GPU energy computation (dot-product reduction).
// ---------------------------------------------------------------------------
class MinimizeGPU {
public:
    struct Options {
        Real threshold   = 1.0;    // |m×H|_max [A/m]
        Real dt_init     = 1e-12;
        Real dt_max      = 1e-10;
        Real dt_min      = 1e-17;
        int  max_steps   = 200'000;
        int  check_every = 100;
        bool throw_on_max = false;
    };

    explicit MinimizeGPU(const StructuredGrid& grid);
    ~MinimizeGPU();

    MinimizeGPU(const MinimizeGPU&)            = delete;
    MinimizeGPU& operator=(const MinimizeGPU&) = delete;

    void upload(const VectorField3D& m);
    void download(VectorField3D& m) const;

    int run(const Material& mat,
            IDemagGPU& demag,
            ExchangeFieldGPU& exch,
            ZeemanFieldGPU& zeeman,
            UniaxialAnisotropyFieldGPU* aniso = nullptr,
            Options opts = {});

    const StructuredGrid& grid() const { return *grid_; }

private:
    const StructuredGrid* grid_;
    size_t                N_;

    double* d_m_  = nullptr;
    double* d_m_trial_ = nullptr;
    double* d_H_  = nullptr;
    double* d_max_= nullptr;
    double* d_energy_ = nullptr;   // [N] per-cell energy for reduction
    void*   stream_= nullptr;

    double compute_energy(const Material& mat,
                          IDemagGPU& demag,
                          ExchangeFieldGPU& exch,
                          ZeemanFieldGPU& zeeman,
                          UniaxialAnisotropyFieldGPU* aniso);

    void compute_H_eff_for(double* d_m_src,
                           const Material& mat,
                           IDemagGPU& demag,
                           ExchangeFieldGPU& exch,
                           ZeemanFieldGPU& zeeman,
                           UniaxialAnisotropyFieldGPU* aniso);
};

}  // namespace micromag

#endif // MICROMAG_CUDA
