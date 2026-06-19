#pragma once

#include "field.hpp"
#include "grid.hpp"
#include "types.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// Skyrmion / magnetic bubble tracking utilities
//
// All positions are returned in box-centred coordinates [m]:
//   origin = geometric centre of the simulation box
//   x_bc = (ix + 0.5)*dx - nx*dx/2
// ---------------------------------------------------------------------------

// Find the position of the skyrmion/vortex core by locating the cell with
// extremal mz.
//   find_max=false → minimum mz (pol=+1 skyrmion: mz_core = -1)
//   find_max=true  → maximum mz (pol=-1 skyrmion: mz_core = +1)
// Returns (cx, cy) in box-centred metres.
std::pair<Real, Real> skyrmion_corepos(const VectorField3D& m, bool find_max = false);

// Topological-charge-density-weighted centroid.
// Returns (cx, cy) in box-centred metres.
// More accurate than skyrmion_corepos for off-centre or multi-skyrmion states;
// robust against discretisation noise.
std::pair<Real, Real> bubble_pos(const VectorField3D& m);

// Count skyrmions via connected-component analysis of the topological charge
// density map.  Each connected region with |Q_component| >= threshold is
// counted as one skyrmion.  threshold=0.5 (default) correctly handles
// individually well-defined skyrmions.
int skyrmion_count(const VectorField3D& m, Real threshold = Real{0.5});

}  // namespace micromag
