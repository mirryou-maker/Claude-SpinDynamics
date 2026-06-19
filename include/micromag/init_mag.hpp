#pragma once

#include "field.hpp"
#include "grid.hpp"
#include "types.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// Initial magnetization state generators
// All spatial coordinates are box-centred (origin = geometric centre).
// Returned VectorField3D has (approximately) unit-length m at each cell.
// ---------------------------------------------------------------------------

// Uniform state: every cell = dir (normalised internally).
VectorField3D uniform_mag(const StructuredGrid& g, Vec3 dir);

// Néel skyrmion — radial in-plane texture, out-of-plane core.
//   Profile: θ(ρ) = 2*atan(r/ρ) for ρ>0, π at ρ=0.
//   mx = sin(θ)*cos(charge*φ), my = sin(θ)*sin(charge*φ), mz = pol*cos(θ)
//   At core (ρ=0): mz = -pol (core pointing antiparallel to far-field).
//   r      : skyrmion radius [m]
//   charge : topological charge ±1
//   pol    : polarity (+1 = far-field along +z, core along -z)
//   cx, cy : offset of skyrmion centre from box centre [m]
VectorField3D neel_skyrmion(const StructuredGrid& g, Real r,
                             int charge = 1, int pol = 1,
                             Real cx = 0.0, Real cy = 0.0);

// Bloch skyrmion — tangential in-plane texture, out-of-plane core.
// Same profile as Néel but in-plane component rotated 90°.
VectorField3D bloch_skyrmion(const StructuredGrid& g, Real r,
                              int charge = 1, int pol = 1,
                              Real cx = 0.0, Real cy = 0.0);

// Two-domain state with a sharp domain wall.
//   m1 : direction for cells where coord <= 0  (normalised internally)
//   m2 : direction for cells where coord > 0   (normalised internally)
//   axis: 'x' | 'y' | 'z' — wall normal axis
VectorField3D two_domain(const StructuredGrid& g, Vec3 m1, Vec3 m2,
                          char axis = 'x');

// Circular vortex state (in-plane closure domain + out-of-plane core).
// θ(ρ) = π/2*(1 - exp(-ρ²/r_c²))  where r_c = 2*max(dx,dy).
//   circ : +1 = counterclockwise, -1 = clockwise
//   pol  : core polarity (+1 = mz>0 at core, -1 = mz<0)
VectorField3D vortex_state(const StructuredGrid& g, int circ = 1, int pol = 1);

// Random unit vectors (reproducible via seed).
VectorField3D random_mag(const StructuredGrid& g, unsigned seed = 42);

}  // namespace micromag
