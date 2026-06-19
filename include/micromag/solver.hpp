#pragma once

#include "field.hpp"
#include "material.hpp"
#include "effective_field.hpp"
#include "spin_torque.hpp"

// Forward declarations
namespace micromag { class MaterialField3D; }

namespace micromag {

// ---------------------------------------------------------------------------
// max_torque  — max over all cells of |m × H_eff| [A/m]
//
// This is the precession-rate proxy used as a convergence criterion for
// Relax().  When max_torque < threshold, the system is at a local energy
// minimum.
// ---------------------------------------------------------------------------
Real max_torque(const VectorField3D& m,
                const Material& mat,
                const EffectiveFieldSum& heff);

// ---------------------------------------------------------------------------
// relax()  — energy minimisation via damping-only LLG
//
// Equivalent to mumax3 Relax(): runs LLG with precession disabled
// (DoPrecess = false) using a high effective damping (alpha_relax = 1.0)
// until max |m × H_eff| drops below `threshold` [A/m].
//
// Uses fixed-step Euler in the damping-only direction:
//   dm = -γ'αμ₀ m×(m×H) dt   (no precession term)
//
// Returns the number of steps taken.  Throws if max_steps is reached
// without convergence.
// ---------------------------------------------------------------------------
struct RelaxOptions {
    Real alpha_relax = 1.0;         // effective alpha during relaxation
    Real threshold   = 1.0;         // |m×H|_max [A/m], ~1 A/m ≈ mumax3 default
    Real dt          = 1e-12;       // fixed step [s]  (larger ok since no precession)
    int  max_steps   = 500'000;     // bail-out guard
    bool throw_on_max_steps = false;// if false: silently return when max_steps hit
};

int relax(VectorField3D& m,
          const Material& mat,
          const EffectiveFieldSum& heff,
          RelaxOptions opts = {},
          const MaterialField3D* matf  = nullptr,
          const SpinTorqueSum*   stt   = nullptr);

// ---------------------------------------------------------------------------
// minimize()  — steepest-descent energy minimisation
//
// Equivalent to mumax3 Minimize(): moves m along the steepest energy-descent
// direction (the damping-like torque direction) using a fixed step, then
// accepts the move only if energy decreases (backtracking line search).
//
// Converges when max |m × H_eff| < threshold [A/m].
// Returns the number of steps taken.
// ---------------------------------------------------------------------------
struct MinimizeOptions {
    Real threshold   = 1.0;         // |m×H|_max [A/m]
    Real dt_init     = 1e-12;       // initial trial step [s]
    Real dt_max      = 1e-10;       // maximum step
    Real dt_min      = 1e-17;       // minimum step
    int  max_steps   = 200'000;
    bool throw_on_max_steps = false;
};

int minimize(VectorField3D& m,
             const Material& mat,
             const EffectiveFieldSum& heff,
             MinimizeOptions opts     = {},
             const MaterialField3D*  matf = nullptr);

}  // namespace micromag
