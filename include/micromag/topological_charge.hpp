#pragma once

#include "field.hpp"
#include "grid.hpp"
#include "types.hpp"

namespace micromag {

// Topological charge (skyrmion number) for 2D spin textures.
//
// q_cell = m · (∂m/∂x × ∂m/∂y)   (discrete central differences, Neumann BC)
// Q      = (1/4π) Σ_cells q_cell * dx*dy
//
// For a uniform film (nz layers): Q is summed over all z-layers.
// Perfect Néel/Bloch skyrmion → Q ≈ ±1 (for nz=1).
//
// Returns: {total_Q, per-cell density field}
// density[i] = q_cell (without 1/4π or dA — raw lattice density)
// so that Q = (dx*dy / 4π) * Σ density[i].
std::pair<Real, ScalarField3D>
topological_charge(const VectorField3D& m);

// Convenience: return only total Q.
Real topological_charge_Q(const VectorField3D& m);

// Return only the per-cell density field q(ix,iy,iz) = m·(dm_dx × dm_dy).
ScalarField3D topological_charge_density(const VectorField3D& m);

}  // namespace micromag
