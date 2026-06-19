#pragma once

// surface_anisotropy.hpp — Interface / surface anisotropy field
//
// Models the perpendicular magnetic anisotropy arising from broken symmetry
// at FM/HM interfaces (e.g. Co/Pt, Co/MgO).  Equivalent to mumax3's Ks
// parameter.
//
// Physics (effective field form):
//   E_s  = -Ks * (m · n_hat)^2   [J/m²] per surface
//   H_s  = (2 Ks / (mu_0 Ms t_cell)) * (m · n_hat) * n_hat   [A/m]
//
// where t_cell is the cell thickness along n_hat and Ks [J/m²] is the
// surface anisotropy constant.  Ks > 0 gives an easy-axis along n_hat.
//
// The field is applied only to *surface cells* — cells that have at least
// one nearest neighbour outside the geometry (mask < 0.5) along n_hat.
// Interior cells are unaffected.  If no mask is attached, all top / bottom
// layers (iz = 0 and iz = nz-1 for the default n_hat = z) are treated as
// surface cells.
//
// Usage:
//   SurfaceAnisotropyField sa(Ks = 1.2e-3, n_hat = {0,0,1});
//   sa.set_mask(&disk_mask);     // optional: restricts to geometry
//   heff.add(std::make_shared<SurfaceAnisotropyField>(sa));

#include "effective_field.hpp"
#include "geom_mask.hpp"
#include "types.hpp"

namespace micromag {

class SurfaceAnisotropyField : public IEffectiveField {
public:
    // Ks    : surface anisotropy constant [J/m²].  Ks > 0 = PMA (easy perp).
    // n_hat : surface normal (normalised internally; default = z-axis).
    // Both surfaces (top and bottom along n_hat) are treated simultaneously.
    explicit SurfaceAnisotropyField(Real Ks, Vec3 n_hat = {0.0, 0.0, 1.0});

    // Attach a geometry mask.  Cells with mask < 0.5 are vacuum;
    // surface cells = inside cells with at least one vacuum neighbour along n_hat.
    // If nullptr (default), only the outermost layers along n_hat are treated.
    void set_mask(const GeomMask* mask)  { mask_ = mask; }
    void clear_mask()                    { mask_ = nullptr; }
    const GeomMask* mask()         const { return mask_; }

    // Properties
    Real  Ks()    const { return Ks_; }
    Vec3  n_hat() const { return n_;  }
    void  set_Ks(Real Ks)       { Ks_ = Ks; }
    void  set_n_hat(Vec3 n);    // normalises internally

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "SurfaceAnisotropy"; }

private:
    Real          Ks_;
    Vec3          n_;       // unit normal (always normalised)
    const GeomMask* mask_{nullptr};

    // Returns the cell thickness along n_hat for the given grid [m].
    Real cell_thickness(const StructuredGrid& g) const;

    // Returns true if the cell (ix,iy,iz) is a surface cell.
    bool is_surface_cell(const StructuredGrid& g,
                         Index ix, Index iy, Index iz) const;
};

}  // namespace micromag
