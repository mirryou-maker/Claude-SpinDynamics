#pragma once

#include <string>
#include "field.hpp"
#include "grid.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// OVF 2.0 file I/O  (mumax3 / OOMMF compatible)
//
// Supports:
//   OVF2 text   (# Begin: Data Text)
//   OVF2 binary 8 (# Begin: Data Binary 8)  — preferred, lossless
//
// Data ordering: x varies fastest (matches our linear_index convention).
// ---------------------------------------------------------------------------

enum class OVFFormat {
    Text,      // ASCII text (portable, large files)
    Binary4,   // IEEE 754 float  (4 bytes/component) — mumax3 OVF1 compatible, half the size
    Binary8,   // IEEE 754 double (8 bytes/component) — matches mumax3 default, lossless
};

// Save a VectorField3D to an OVF 2.0 file.
// title: optional descriptive title stored in the header.
void save_ovf(const std::string& filename,
              const VectorField3D& m,
              const std::string& title  = "m",
              OVFFormat fmt             = OVFFormat::Binary8);

// Load an OVF file (OVF1 or OVF2, text or binary) into a VectorField3D.
// The grid is created from the header metadata; the caller can verify
// dimensions against their own grid afterwards.
VectorField3D load_ovf(const std::string& filename);

// Read only the grid dimensions from an OVF header (no data loaded).
StructuredGrid load_ovf_grid(const std::string& filename);

// Fill an existing VectorField3D from an OVF file.
// Throws if the file grid dimensions don't match m.grid().
void load_ovf_into(const std::string& filename, VectorField3D& m);

}  // namespace micromag
