#pragma once

#include <cstdint>
#include <vector>
#include "field.hpp"
#include "geom_mask.hpp"
#include "material.hpp"

namespace micromag {

// Forward declaration
class MaterialField3D;

// RegionMap — per-cell integer region IDs (0–255).
// Mirrors mumax3's region system: DefRegion(id, shape), SetRegion(id, value).
// Region 0 is the background (default for all cells).
//
// Typical usage:
//   RegionMap regions(grid);
//   regions.def_region(1, circle(grid, 50e-9));   // disk = region 1
//   regions.def_region(2, rect(grid, 100e-9, 20e-9)); // bar = region 2
//   regions.set_magnetization(1, m, {0,0,1});
//   regions.set_material(2, matf, Material::cobalt());
class RegionMap {
public:
    explicit RegionMap(const StructuredGrid& g, uint8_t default_id = 0);

    uint8_t& operator[](Index i) { return regions_[static_cast<size_t>(i)]; }
    uint8_t  operator[](Index i) const { return regions_[static_cast<size_t>(i)]; }

    uint8_t& at(Index ix, Index iy, Index iz);
    uint8_t  at(Index ix, Index iy, Index iz) const;

    // Assign region ID to all cells where mask > 0.5 (last call wins).
    void def_region(uint8_t id, const GeomMask& mask);

    // Return a GeomMask (binary 0/1) for all cells with the given region ID.
    GeomMask region_mask(uint8_t id) const;

    // Set magnetization in all cells of a region to val (normalised internally).
    void set_magnetization(uint8_t id, VectorField3D& m, Vec3 val) const;

    // Copy mat into every cell of a region in a MaterialField3D.
    void set_material(uint8_t id, MaterialField3D& matf, const Material& mat) const;

    const StructuredGrid& grid() const { return *grid_; }
    Index size() const { return static_cast<Index>(regions_.size()); }

private:
    const StructuredGrid* grid_;
    std::vector<uint8_t> regions_;
};

}  // namespace micromag
