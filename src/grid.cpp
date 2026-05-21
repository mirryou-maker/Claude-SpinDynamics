#include "micromag/grid.hpp"

#include <stdexcept>

namespace micromag {

StructuredGrid::StructuredGrid(Index nx, Index ny, Index nz,
                               Real dx, Real dy, Real dz)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
    if (dx <= 0 || dy <= 0 || dz <= 0) {
        throw std::invalid_argument("Cell sizes must be positive");
    }
}

}  // namespace micromag
