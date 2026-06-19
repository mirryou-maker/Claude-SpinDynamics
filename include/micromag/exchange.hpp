#pragma once

#include "effective_field.hpp"
#include <cstdint>
#include <unordered_map>

namespace micromag {

// Forward declarations — include geom_mask.hpp / material_field.hpp for the
// full definitions.
class GeomMask;
class MaterialField3D;
class RegionMap;

// Heisenberg exchange via 6-point Laplacian:
//   E = A ∫ |∇m|² dV,   H = (2A / μ₀ Ms) ∇²m
//
// When a GeomMask is attached (set_mask), cells with mask < 0.5 are treated
// as Neumann boundaries: no exchange flux crosses the geometry boundary.
// Cells with mask == 0 contribute nothing to H_out (skipped entirely).
//
// When a MaterialField3D is attached (set_material_field), per-cell Ms and
// A_exchange override the uniform `mat` argument:
//   H_i  = (2 / μ₀ Ms_i) Σ_neighbours (A_ij / d²)(m_j - m_i)
//   A_ij = 2 A_i A_j / (A_i + A_j)   (harmonic mean — region-boundary stiffness)
class ExchangeField : public IEffectiveField {
public:
    explicit ExchangeField(BoundaryCondition bc = BoundaryCondition::Neumann)
        : bc_(bc) {}

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m, const Material& mat) const override;

    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "Exchange"; }

    BoundaryCondition boundary() const { return bc_; }
    void set_boundary(BoundaryCondition bc) { bc_ = bc; }

    // Attach a geometry mask. mask=nullptr disables masking (default).
    // Caller must keep the mask alive for the lifetime of this field.
    void set_mask(const GeomMask* mask) { mask_ = mask; }
    void clear_mask() { mask_ = nullptr; }
    const GeomMask* mask() const { return mask_; }

    // Attach per-cell material parameters. nullptr disables it (default),
    // falling back to the uniform `mat` passed to accumulate()/energy().
    // Caller must keep the field alive for the lifetime of this object.
    void set_material_field(const MaterialField3D* matf) { matf_ = matf; }
    void clear_material_field() { matf_ = nullptr; }
    const MaterialField3D* material_field() const { return matf_; }

    // ---------------------------------------------------------------------------
    // Per-region-pair exchange coupling (mumax3 SetInterExchange analog).
    //
    // Attach a RegionMap so cell region IDs are accessible during field
    // accumulation. Then call set_inter_exchange(ri, rj, A_IEC) for each
    // region-pair boundary that needs an explicit coupling constant [J/m].
    //
    // At a bond between cells in regions ri and rj (ri ≠ rj):
    //   • If set_inter_exchange(ri, rj, A) was called → use A directly.
    //   • Otherwise → harmonic mean of the two cells' A_exchange values
    //     (MaterialField3D or uniform mat.A_exchange).
    //
    // Setting A_IEC = 0 disables exchange across that boundary pair.
    // The coupling is stored symmetrically: set_inter_exchange(i,j,A) also
    // sets (j,i,A).
    // ---------------------------------------------------------------------------
    void set_region_map(const RegionMap* rm) { rmap_ = rm; }
    void clear_region_map() { rmap_ = nullptr; }
    const RegionMap* region_map() const { return rmap_; }

    void set_inter_exchange(uint8_t ri, uint8_t rj, Real A_IEC);
    Real inter_exchange(uint8_t ri, uint8_t rj) const;
    void clear_inter_exchange() { inter_A_.clear(); }

private:
    BoundaryCondition bc_;
    const GeomMask*        mask_{nullptr};
    const MaterialField3D* matf_{nullptr};
    const RegionMap*       rmap_{nullptr};

    // key = ri*256+rj (symmetric: always ri<=rj stored).
    std::unordered_map<uint32_t, Real> inter_A_;

    static constexpr uint32_t inter_key(uint8_t ri, uint8_t rj) noexcept {
        uint8_t lo = ri < rj ? ri : rj;
        uint8_t hi = ri < rj ? rj : ri;
        return static_cast<uint32_t>(lo) * 256u + hi;
    }
    // Returns -1 if not set.
    Real lookup_inter(uint8_t ri, uint8_t rj) const {
        if (inter_A_.empty()) return Real{-1};
        auto it = inter_A_.find(inter_key(ri, rj));
        return (it != inter_A_.end()) ? it->second : Real{-1};
    }
};

}  // namespace micromag
