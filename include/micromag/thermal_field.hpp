#pragma once

#include <memory>
#include <random>

#include "effective_field.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// ThermalField — Langevin stochastic thermal noise for the SLLG equation.
//
// Adds a per-cell Gaussian random field drawn from N(0, σ²) to H_eff:
//
//   σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V Δt))
//
// Usage inside a Heun step (fixed Δt):
//   1. thermal.resample(mat)               // generate new η^n — ONCE per step
//   2. heff.compute(m, mat, H);            // deterministic H_eff
//      thermal.accumulate(m, mat, H);      // H += H_th(η^n)  — predictor
//   3. heff.compute(m_pred, mat, H);
//      thermal.accumulate(m_pred, mat, H); // H += H_th(η^n)  — corrector
//      (same η^n: Stratonovich convention)
//
// NOTE: ThermalField is NOT compatible with RK45 (adaptive Δt changes σ).
//       Use HeunIntegrator with a fixed Δt.
// ---------------------------------------------------------------------------
class ThermalField : public IEffectiveField {
public:
    // grid : determines buffer size
    // T_K  : temperature in Kelvin
    // dt   : integration time step [s] — must match HeunIntegrator::dt()
    // seed : RNG seed (default 42 for reproducibility)
    ThermalField(const StructuredGrid& grid, Real T_K, Real dt,
                 unsigned seed = 42);

    // Setters (call before the next resample)
    void set_temperature(Real T_K);
    void set_dt(Real dt);

    // Generate a fresh set of noise vectors η × σ.
    // Call exactly ONCE at the start of each Heun step (before predictor).
    void resample(const Material& mat);

    // IEffectiveField -------------------------------------------------------
    // Adds the stored noise vectors to H_out (does NOT modify noise_).
    // Safe to call multiple times per step (same noise reused).
    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;
    const char* name() const override { return "ThermalField"; }
    // -----------------------------------------------------------------------

    Real temperature() const { return T_K_; }
    Real dt()          const { return dt_;  }

    // σ = sqrt(2α k_B T / (μ₀ Ms γ₀ V Δt))  [A/m]
    static Real sigma(Real T_K, Real dt,
                      const Material& mat, const StructuredGrid& grid);

private:
    Real T_K_;
    Real dt_;

    std::mt19937 rng_;
    std::normal_distribution<Real> dist_{0.0, 1.0};

    // Pre-allocated noise buffer; filled by resample(), read by accumulate()
    std::unique_ptr<VectorField3D> noise_;
};

}  // namespace micromag
