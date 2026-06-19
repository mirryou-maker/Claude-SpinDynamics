#include "micromag/region_map.hpp"
#include "micromag/material_field.hpp"

namespace micromag {

RegionMap::RegionMap(const StructuredGrid& g, uint8_t default_id)
    : grid_(&g),
      regions_(static_cast<size_t>(g.size()), default_id) {}

uint8_t& RegionMap::at(Index ix, Index iy, Index iz) {
    return regions_[static_cast<size_t>(grid_->linear_index(ix, iy, iz))];
}

uint8_t RegionMap::at(Index ix, Index iy, Index iz) const {
    return regions_[static_cast<size_t>(grid_->linear_index(ix, iy, iz))];
}

void RegionMap::def_region(uint8_t id, const GeomMask& mask) {
    for (Index i = 0; i < size(); ++i)
        if (mask[i] > Real{0.5})
            regions_[static_cast<size_t>(i)] = id;
}

GeomMask RegionMap::region_mask(uint8_t id) const {
    GeomMask mask(*grid_);
    for (Index i = 0; i < size(); ++i)
        mask[i] = (regions_[static_cast<size_t>(i)] == id) ? Real{1} : Real{0};
    return mask;
}

void RegionMap::set_magnetization(uint8_t id, VectorField3D& m, Vec3 val) const {
    const Real n = val.norm();
    const Vec3 v = (n > 1e-30) ? (val / n) : Vec3{0, 0, 1};
    for (Index i = 0; i < size(); ++i)
        if (regions_[static_cast<size_t>(i)] == id)
            m[i] = v;
}

void RegionMap::set_material(uint8_t id, MaterialField3D& matf,
                              const Material& mat) const {
    for (Index i = 0; i < size(); ++i) {
        if (regions_[static_cast<size_t>(i)] != id) continue;
        matf.Ms_field()[i]        = mat.Ms;
        matf.A_field()[i]         = mat.A_exchange;
        matf.K_field()[i]         = mat.K_uniaxial;
        matf.alpha_field()[i]     = mat.alpha;
        matf.easy_axis_field()[i] = mat.easy_axis;
    }
}

}  // namespace micromag
