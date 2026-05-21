#include "micromag/vtk_writer.hpp"

#include <fstream>
#include <stdexcept>

namespace micromag {

void write_vtk_legacy(const std::string& filename,
                      const VectorField3D& field,
                      const std::string& field_name) {
    std::ofstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    const auto& g = field.grid();
    const Index n = g.size();

    f << "# vtk DataFile Version 3.0\n";
    f << "Micromag output\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << g.nx() << " " << g.ny() << " " << g.nz() << "\n";
    f << "ORIGIN " << g.dx() * 0.5 << " " << g.dy() * 0.5 << " " << g.dz() * 0.5 << "\n";
    f << "SPACING " << g.dx() << " " << g.dy() << " " << g.dz() << "\n";
    f << "POINT_DATA " << n << "\n";
    f << "VECTORS " << field_name << " double\n";

    for (Index idx = 0; idx < n; ++idx) {
        const Vec3& v = field[idx];
        f << v.x << " " << v.y << " " << v.z << "\n";
    }
}

}  // namespace micromag
