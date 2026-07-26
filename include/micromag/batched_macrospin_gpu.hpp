// batched_macrospin_gpu.hpp — Task 2 Phase 2.0: replica-batched macrospin.
//
// Scope (MVP): R INDEPENDENT single-cell macrospins advanced by ONE kernel
// launch per step (replica = thread). This is the NB30 / MTJ finite-T workload
// where the per-replica Python loop leaves the GPU ~26% utilised. The batched
// path advances all R replicas in one launch → the throughput win that Task 2
// exists to deliver, before the deeper GPUMagState/field-class refactor
// (Phase 2.1+). See CLAUDE-SD_FINITE_TEMP_ROADMAP.md Task 2, Phase 2.0.
//
// Physics per replica (identical to the CPU/GPU reference paths):
//   * uniaxial anisotropy  H_ani = (2K1/μ₀Ms)(m·û)û          (uniform K1, û)
//   * Zeeman               H_ext                              (uniform, A/m)
//   * Slonczewski STT      τ = a_J[m×(m×p̂)] + b_J[m×p̂],
//                          a_J = γ₀ħJ/(2 e Ms d)·ε(m·p̂),  b_J = −β a_J,
//                          ε = P·Λ²/((Λ²+1)+(Λ²−1)(m·p̂))     (PER-REPLICA J)
//   * thermal (FDT)        σ = √(2α k_B T/(μ₀² Ms γ₀ V dt))   (PER-REPLICA T)
// Integrator: Depondt–Mertens rotation (|m|=1 exact) with the STT folded into
// the rotation axis ω_stt = −a_J(m×p̂) − b_J p̂ (so ω_stt×m reproduces τ).
// Thermal noise: device Philox keyed by (seed, subsequence=replica, offset=step)
// → per-replica independent, bitwise reproducible, same realisation in the
// predictor and corrector (Stratonovich).
//
// Requires MICROMAG_CUDA=1.
#pragma once

#include <vector>

#include "micromag/material.hpp"
#include "micromag/types.hpp"

namespace micromag {

// Uniform (shared across replicas) macrospin configuration. Only J and T vary
// per replica (the NB30 sweep axes) — set via set_J()/set_T().
struct BatchedMacrospinConfig {
    Real Ms    = 580e3;   // saturation magnetisation [A/m]
    Real alpha = 0.02;    // Gilbert damping
    Real V     = 1e-24;   // cell volume [m³] (thermal + energetics)

    Real K1    = 0.5e6;               // uniaxial anisotropy [J/m³]
    Vec3 easy  = {0, 0, 1};           // easy axis (normalised internally)

    Vec3 H_ext = {0, 0, 0};           // Zeeman field [A/m]

    // Slonczewski STT (J is per-replica; these are shared):
    Real d_free = 1e-9;               // free-layer thickness [m]
    Real P      = 0.5;                // spin polarisation
    Real Lambda = 1.0;                // angular asymmetry (ε = P/2 at Λ=1)
    Real beta   = 0.0;                // field-like ratio (b_J = −β a_J)
    Vec3 p      = {0, 0, 1};          // reference-layer polarisation (normalised)
};

class BatchedMacrospinGPU {
public:
    // R replicas, fixed step dt, RNG seed. Initial state m = +easy for all.
    BatchedMacrospinGPU(int R, const BatchedMacrospinConfig& cfg,
                        Real dt, unsigned seed = 12345u);
    ~BatchedMacrospinGPU();

    BatchedMacrospinGPU(const BatchedMacrospinGPU&) = delete;
    BatchedMacrospinGPU& operator=(const BatchedMacrospinGPU&) = delete;

    // Per-replica current density J [A/m²] (length R) and temperature T [K]
    // (length R). Both may be updated between run() calls.
    void set_J(const std::vector<double>& J);
    void set_T(const std::vector<double>& T);

    // Overwrite the state: flat [R*3], replica-major m[r*3+c]. Normalised
    // per replica on upload.
    void set_state(const std::vector<double>& m);

    // Advance ALL R replicas by n_steps (one kernel launch per step, zero PCIe).
    void run(int n_steps);

    // Download the state: flat [R*3], m[r*3+c].
    std::vector<double> get_state() const;
    // Convenience: m_z of each replica, length R.
    std::vector<double> get_mz() const;

    int  R()          const { return R_; }
    long step_index() const { return step_index_; }

private:
    int      R_;
    Real     dt_;
    unsigned seed_;
    long     step_index_ = 0;
    BatchedMacrospinConfig cfg_;

    // device buffers
    void*  d_m_  = nullptr;   // GReal [R*3]
    void*  d_J_  = nullptr;   // double [R]
    void*  d_T_  = nullptr;   // double [R]
};

}  // namespace micromag
