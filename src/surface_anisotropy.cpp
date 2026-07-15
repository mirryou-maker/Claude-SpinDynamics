#include "micromag/surface_anisotropy.hpp"
#include "micromag/types.hpp"
#include <cmath>

namespace micromag {

namespace {
    constexpr Real mu0 = constants::mu_0;
}

// ---------------------------------------------------------------------------
SurfaceAnisotropyField::SurfaceAnisotropyField(Real Ks, Vec3 n_hat)
    : Ks_(Ks)
{
    set_n_hat(n_hat);
}

void SurfaceAnisotropyField::set_n_hat(Vec3 n)
{
    const Real len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    n_ = (len > 0) ? Vec3{n.x/len, n.y/len, n.z/len} : Vec3{0,0,1};
}

// ---------------------------------------------------------------------------
// cell_thickness: projection of cell dimensions onto n_hat [m]
// For n_hat = z this is dz; for n_hat = x it is dx, etc.
Real SurfaceAnisotropyField::cell_thickness(const StructuredGrid& g) const
{
    return std::abs(n_.x) * g.dx()
         + std::abs(n_.y) * g.dy()
         + std::abs(n_.z) * g.dz();
}

// ---------------------------------------------------------------------------
// is_surface_cell: true if cell has at least one neighbour along ±n_hat
// that is either outside the grid or outside the geometry mask.
bool SurfaceAnisotropyField::is_surface_cell(const StructuredGrid& g,
                                              Index ix, Index iy, Index iz) const
{
    // Detect the dominant direction of n_hat
    const Real ax = std::abs(n_.x), ay = std::abs(n_.y), az = std::abs(n_.z);

    if (mask_ != nullptr) {
        // Check all six neighbours; surface = inside cell next to a vacuum cell.
        auto outside = [&](Index ii, Index jj, Index kk) -> bool {
            if (ii < 0 || ii >= g.nx() || jj < 0 || jj >= g.ny() ||
                kk < 0 || kk >= g.nz())
                return true;   // off-grid => vacuum
            return (*mask_)[g.linear_index(ii, jj, kk)] < Real{0.5};
        };
        // Only check ±n̂ directions (dominant axis)
        if (az >= ay && az >= ax) {
            return outside(ix, iy, iz - 1) || outside(ix, iy, iz + 1);
        } else if (ay >= ax) {
            return outside(ix, iy - 1, iz) || outside(ix, iy + 1, iz);
        } else {
            return outside(ix - 1, iy, iz) || outside(ix + 1, iy, iz);
        }
    } else {
        // No mask: outermost layer(s) along n_hat direction
        if (az >= ay && az >= ax) {
            return (iz == 0) || (iz == g.nz() - 1);
        } else if (ay >= ax) {
            return (iy == 0) || (iy == g.ny() - 1);
        } else {
            return (ix == 0) || (ix == g.nx() - 1);
        }
    }
}

// ---------------------------------------------------------------------------
void SurfaceAnisotropyField::accumulate(const VectorField3D& m,
                                         const Material& mat,
                                         VectorField3D& H_out) const
{
    const auto& g  = m.grid();
    const Real t   = cell_thickness(g);
    const Real Ms  = mat.Ms;
    if (Ms == Real{0} || t == Real{0}) return;

    // H_s = (2 Ks / (mu0 * Ms * t)) * (m · n_hat) * n_hat
    const Real prefac = Real{2} * Ks_ / (mu0 * Ms * t);

    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    const Index Ntot = nx * ny * nz;
    #pragma omp parallel for schedule(static) if(Ntot > 4096)
    for (Index idx = 0; idx < Ntot; ++idx) {
        const Index ix   = idx % nx;
        const Index trow = idx / nx;
        const Index iy   = trow % ny;
        const Index iz   = trow / ny;
        if (!is_surface_cell(g, ix, iy, iz)) continue;
        if (mask_ && (*mask_)[idx] < Real{0.5}) continue;

        const Vec3& mi = m[idx];
        const Real  mn = mi.x * n_.x + mi.y * n_.y + mi.z * n_.z;
        H_out[idx] += n_ * (prefac * mn);
    }
}

// ---------------------------------------------------------------------------
Real SurfaceAnisotropyField::energy(const VectorField3D& m,
                                     [[maybe_unused]] const Material& mat) const
{
    const auto& g  = m.grid();
    const Real  t  = cell_thickness(g);
    const Real  dV = g.dx() * g.dy() * g.dz();
    Real E = Real{0};

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        if (!is_surface_cell(g, ix, iy, iz)) continue;
        if (mask_ && (*mask_)[g.linear_index(ix,iy,iz)] < Real{0.5}) continue;

        const Vec3& mi = m[g.linear_index(ix, iy, iz)];
        const Real  mn = mi.x * n_.x + mi.y * n_.y + mi.z * n_.z;
        // E_s per cell = -Ks * (m·n)^2 * (dV / t)  [dV/t = surface area element]
        E += -Ks_ * mn * mn * (dV / t);
    }
    return E;
}

}  // namespace micromag
