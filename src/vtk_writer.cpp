#include "micromag/vtk_writer.hpp"

#include <cstdio>
#include <stdexcept>

namespace micromag {

void write_vtk_legacy(const std::string& filename,
                      const VectorField3D& field,
                      const std::string& field_name) {
    // C stdio (FILE*) rather than std::ofstream: the C++ filebuf flush/close
    // deadlocks in the CUDA-linked build (see save_ovf in ovf_io.cpp).
    FILE* f = std::fopen(filename.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    const auto& g = field.grid();
    const Index n = g.size();

    std::fputs("# vtk DataFile Version 3.0\n", f);
    std::fputs("Micromag output\n", f);
    std::fputs("ASCII\n", f);
    std::fputs("DATASET STRUCTURED_POINTS\n", f);
    std::fprintf(f, "DIMENSIONS %lld %lld %lld\n",
                 (long long)g.nx(), (long long)g.ny(), (long long)g.nz());
    std::fprintf(f, "ORIGIN %g %g %g\n",
                 (double)(g.dx() * 0.5), (double)(g.dy() * 0.5), (double)(g.dz() * 0.5));
    std::fprintf(f, "SPACING %g %g %g\n",
                 (double)g.dx(), (double)g.dy(), (double)g.dz());
    std::fprintf(f, "POINT_DATA %lld\n", (long long)n);
    std::fprintf(f, "VECTORS %s double\n", field_name.c_str());

    for (Index idx = 0; idx < n; ++idx) {
        const Vec3& v = field[idx];
        std::fprintf(f, "%g %g %g\n", (double)v.x, (double)v.y, (double)v.z);
    }
    std::fclose(f);
}

}  // namespace micromag
