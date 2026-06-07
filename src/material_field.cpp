#include "micromag/material_field.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace micromag {

MaterialField3D::MaterialField3D(const StructuredGrid& grid, const Material& uniform)
    : Ms_(grid), A_(grid), K_(grid), alpha_(grid), easy_axis_(grid) {
    set_uniform(uniform);
}

void MaterialField3D::set_uniform(const Material& mat) {
    Ms_.set_uniform(mat.Ms);
    A_.set_uniform(mat.A_exchange);
    K_.set_uniform(mat.K_uniaxial);
    alpha_.set_uniform(mat.alpha);
    easy_axis_.set_uniform(mat.easy_axis);
}

Material MaterialField3D::at(Index i, Index j, Index k) const {
    const Index idx = grid().linear_index(i, j, k);
    return (*this)[idx];
}

Material MaterialField3D::operator[](Index linear) const {
    return Material{Ms_[linear], A_[linear], K_[linear], easy_axis_[linear], alpha_[linear]};
}

// ---------------------------------------------------------------------------
// voronoi_grains
// ---------------------------------------------------------------------------

namespace {

// Uniformly-random unit vector (Marsaglia: normalize an isotropic Gaussian).
Vec3 random_unit_vector(std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Vec3 v;
    Real n;
    do {
        v = {nd(rng), nd(rng), nd(rng)};
        n = v.norm();
    } while (n < 1e-12);
    return v / n;
}

}  // namespace

MaterialField3D voronoi_grains(const StructuredGrid& grid,
                               int n_grains,
                               const Material& base,
                               Real sigma_K,
                               unsigned seed) {
    if (n_grains < 1)
        throw std::invalid_argument("voronoi_grains: n_grains must be >= 1");

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(0.0, grid.nx() * grid.dx());
    std::uniform_real_distribution<double> uy(0.0, grid.ny() * grid.dy());
    std::uniform_real_distribution<double> uz(0.0, grid.nz() * grid.dz());
    std::normal_distribution<double>       nK(0.0, sigma_K);

    const auto n = static_cast<std::size_t>(n_grains);

    // Random grain-centre seed points, scattered through the physical volume.
    std::vector<Vec3> seeds(n);
    for (auto& s : seeds) s = Vec3{ux(rng), uy(rng), uz(rng)};

    // Per-grain randomized uniaxial anisotropy: Gaussian K about base.K_uniaxial
    // (clipped to >= 0) and a uniformly-random easy-axis orientation
    // (random-anisotropy model — mumax3 "Voronoi Tessellation" example).
    std::vector<Real> grain_K(n);
    std::vector<Vec3> grain_axis(n);
    for (std::size_t g = 0; g < n; ++g) {
        grain_K[g]    = std::max<Real>(0.0, base.K_uniaxial + nK(rng));
        grain_axis[g] = random_unit_vector(rng);
    }

    MaterialField3D matf(grid, base);
    for (Index k = 0; k < grid.nz(); ++k)
    for (Index j = 0; j < grid.ny(); ++j)
    for (Index i = 0; i < grid.nx(); ++i) {
        const Vec3 p{(i + 0.5) * grid.dx(), (j + 0.5) * grid.dy(), (k + 0.5) * grid.dz()};

        std::size_t nearest    = 0;
        Real        best_dist2 = (p - seeds[0]).norm_squared();
        for (std::size_t g = 1; g < n; ++g) {
            const Real d2 = (p - seeds[g]).norm_squared();
            if (d2 < best_dist2) { best_dist2 = d2; nearest = g; }
        }

        const Index idx = grid.linear_index(i, j, k);
        matf.K_field()[idx]         = grain_K[nearest];
        matf.easy_axis_field()[idx] = grain_axis[nearest];
    }

    return matf;
}

}  // namespace micromag
