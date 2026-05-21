#pragma once

#include <string>
#include "field.hpp"

namespace micromag {

// Write a VectorField3D as a VTK legacy ASCII file (.vtk).
// ParaView reads this directly. Cell centers are emitted as POINT_DATA
// at half-cell offsets (mumax/OOMMF convention).
//
// The 'field_name' becomes the VECTORS array name in ParaView.
void write_vtk_legacy(const std::string& filename,
                      const VectorField3D& field,
                      const std::string& field_name = "m");

}  // namespace micromag
