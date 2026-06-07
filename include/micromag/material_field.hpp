#pragma once

#include "field.hpp"
#include "material.hpp"

namespace micromag {

// MaterialField3D — per-cell material parameters on a StructuredGrid.
//
// Stores Ms, A_exchange, K_uniaxial, alpha as ScalarField3D and easy_axis as
// VectorField3D (same memory layout / linear_index as VectorField3D).
//
// Effective-field and integrator classes accept an optional
// `const MaterialField3D*` (set_material_field) — when attached, per-cell
// values override the uniform `Material` passed to accumulate()/step().
//
// Usage:
//   MaterialField3D matf(grid, Material::permalloy());
//   matf.K_field().at(i, j, k) = 5e5;       // vary K per cell
//   exch.set_material_field(&matf);
class MaterialField3D {
public:
    // Construct with every cell initialised to `uniform`.
    explicit MaterialField3D(const StructuredGrid& grid, const Material& uniform = {});

    const StructuredGrid& grid() const { return Ms_.grid(); }
    Index size() const { return Ms_.size(); }

    // Fill every cell with the given uniform Material.
    void set_uniform(const Material& mat);

    // Per-cell accessors (linear index).
    Real Ms(Index idx)        const { return Ms_[idx]; }
    Real A_exchange(Index idx) const { return A_[idx]; }
    Real K_uniaxial(Index idx) const { return K_[idx]; }
    Real alpha(Index idx)     const { return alpha_[idx]; }
    const Vec3& easy_axis(Index idx) const { return easy_axis_[idx]; }

    // Assemble a Material struct for one cell (convenience; copies values).
    Material at(Index i, Index j, Index k) const;
    Material operator[](Index linear) const;

    // Direct access to component fields, e.g. for bulk numpy I/O or
    // procedural generation (voronoi_grains, gradients, ...).
    ScalarField3D&       Ms_field()        { return Ms_; }
    const ScalarField3D& Ms_field()  const { return Ms_; }
    ScalarField3D&       A_field()         { return A_; }
    const ScalarField3D& A_field()   const { return A_; }
    ScalarField3D&       K_field()         { return K_; }
    const ScalarField3D& K_field()   const { return K_; }
    ScalarField3D&       alpha_field()       { return alpha_; }
    const ScalarField3D& alpha_field() const { return alpha_; }
    VectorField3D&       easy_axis_field()       { return easy_axis_; }
    const VectorField3D& easy_axis_field() const { return easy_axis_; }

private:
    ScalarField3D Ms_, A_, K_, alpha_;
    VectorField3D easy_axis_;
};

// voronoi_grains — randomized polycrystalline grain structure
// (mumax3 "Voronoi Tessellation" example / random-anisotropy model).
//
// Scatters `n_grains` random seed points through the grid volume, assigns
// each cell to its nearest seed (Voronoi tessellation), then gives every
// grain its own randomized uniaxial anisotropy:
//   K_uniaxial = max(0, base.K_uniaxial + N(0, sigma_K))   (Gaussian, clipped)
//   easy_axis  = uniformly-random unit vector               (random anisotropy)
// Ms, A_exchange and alpha are taken uniformly from `base`.
//
// sigma_K = 0 keeps every grain's K_uniaxial equal to base.K_uniaxial — only
// the easy-axis orientation varies grain to grain.
MaterialField3D voronoi_grains(const StructuredGrid& grid,
                               int n_grains,
                               const Material& base,
                               Real sigma_K = 0.0,
                               unsigned seed = 42);

}  // namespace micromag
