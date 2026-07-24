#include "micromag/dmi.hpp"
#include "micromag/detail/grad_helpers.hpp"
#include "micromag/grid.hpp"

namespace micromag {

using detail::grad_x;
using detail::grad_y;
using detail::grad_z;

// ---------------------------------------------------------------------------
// BulkDMIField
// ---------------------------------------------------------------------------

void BulkDMIField::accumulate(const VectorField3D& m,
                               const Material& mat,
                               VectorField3D& H_out) const {
    if (D_ == 0.0) return;

    const StructuredGrid& g = m.grid();
    // Free-boundary DMI condition (Rohart-Thiaville / mumax3 default):
    //   dm/dn = -(D/2A) (n_hat x m)  at a missing neighbour
    // (sign follows THIS code's field H=+2D/(mu0Ms) curl m, i.e. energy
    //  e=-D m.curl m; note the BULK D sign convention is therefore opposite
    //  to mumax3's Dbulk: D_CS = -D_mumax3).
    // With the BC ghost m* = m_c + s d (D/2A) gamma the boundary gradient is
    //   g_BC = g_onesided/2 + (D/4A) gamma,
    // and replacing the exchange field's Neumann ghost by m* adds
    //   dH_exch = s (D / mu0 Ms d) gamma      (s = +1 max face, -1 min face);
    // the exchange class knows nothing about D, so that correction lives here.
    // The prefactor and this exchange-ghost term carry 1/Ms, which is per-cell
    // when a MaterialField3D is attached (see set_material_field).
    const bool bc = !open_bc_ && mat.A_exchange > Real{0};
    const Real quarterDoverA = bc ? D_ / (Real{4} * mat.A_exchange) : Real{0};

    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    const Real dx = g.dx(), dy = g.dy(), dz = g.dz();
    const Index Ntot = nx * ny * nz;
    #pragma omp parallel for schedule(static) if(Ntot > 4096)
    for (Index idx = 0; idx < Ntot; ++idx) {
        const Index ix   = idx % nx;
        const Index trow = idx / nx;
        const Index iy   = trow % ny;
        const Index iz   = trow / ny;
        const Vec3 mc = m[idx];
        const Real Ms_c = matf_ ? matf_->Ms(idx) : mat.Ms;
        const Real prefac  = 2.0 * D_ / (constants::mu_0 * Ms_c);
        const Real bc_exch = bc ? D_ / (constants::mu_0 * Ms_c) : Real{0};
        Vec3 dH{0, 0, 0};
        Vec3 gx, gy, gz;
        if (!bc) {
            gx = grad_x(m, g, ix, iy, iz);
            gy = grad_y(m, g, ix, iy, iz);
            gz = grad_z(m, g, ix, iy, iz);
        } else {
            // A "missing" neighbour is either outside the grid or a vacuum
            // (GeomMask, |m| ~ 0) cell — both get the free-boundary condition,
            // matching mumax3's zero-Msat-neighbour treatment. When BOTH faces
            // of an axis are free (size-1 dimension or isolated cell) the
            // ghost-based central difference gives gradient = 2 (D/4A) gamma
            // and the exchange-ghost corrections cancel (this is what produces
            // the nonzero dm/dz of a single-layer bulk-DMI film in mumax3).
            if (mc.dot(mc) < Real{0.25}) continue;   // vacuum cell itself
            auto vac = [&](Index j) { const Vec3& v = m[j]; return v.dot(v) < Real{0.25}; };
            auto axis = [&](Index i, Index n, Real d, Index stride, const Vec3& gam) {
                const bool lo = (i > 0)     && !vac(idx - stride);
                const bool hi = (i < n - 1) && !vac(idx + stride);
                if (lo && hi) return (m[idx + stride] - m[idx - stride]) / (Real{2} * d);
                if (hi) { dH += gam * (-bc_exch / d);
                          return (m[idx + stride] - mc) * (Real{0.5} / d) + gam * quarterDoverA; }
                if (lo) { dH += gam * ( bc_exch / d);
                          return (mc - m[idx - stride]) * (Real{0.5} / d) + gam * quarterDoverA; }
                return gam * (Real{2} * quarterDoverA);
            };
            gx = axis(ix, nx, dx, Index{1},  Vec3{0, mc.z, -mc.y});   // gamma_x = -(x_hat x m)  [CS bulk energy = -D m.curl m]
            gy = axis(iy, ny, dy, nx,        Vec3{-mc.z, 0, mc.x});   // gamma_y = -(y_hat x m)
            gz = axis(iz, nz, dz, nx * ny,   Vec3{mc.y, -mc.x, 0});   // gamma_z = -(z_hat x m)
        }

        // curl m = (∂mz/∂y - ∂my/∂z, ∂mx/∂z - ∂mz/∂x, ∂my/∂x - ∂mx/∂y)
        Vec3 curl_m{ gy.z - gz.y, gz.x - gx.z, gx.y - gy.x };
        H_out[idx] += curl_m * prefac + dH;
    }
}

Real BulkDMIField::energy(const VectorField3D& m, [[maybe_unused]] const Material& mat) const {
    if (D_ == 0.0) return 0.0;

    const StructuredGrid& g = m.grid();
    const Real dV = g.cell_volume();
    Real E = 0.0;

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 gz = grad_z(m, g, ix, iy, iz);
        Vec3 curl_m{ gy.z - gz.y, gz.x - gx.z, gx.y - gy.x };
        E += m[idx].dot(curl_m);
    }
    return D_ * E * dV;
}

ScalarField3D BulkDMIField::energy_density(const VectorField3D& m,
                                            [[maybe_unused]] const Material& mat) const {
    ScalarField3D edens(m.grid());
    if (D_ == 0.0) return edens;

    const StructuredGrid& g = m.grid();
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 gz = grad_z(m, g, ix, iy, iz);
        Vec3 curl_m{ gy.z - gz.y, gz.x - gx.z, gx.y - gy.x };
        edens[idx] = D_ * m[idx].dot(curl_m);
    }
    return edens;
}

// ---------------------------------------------------------------------------
// InterfacialDMIField
// ---------------------------------------------------------------------------

void InterfacialDMIField::accumulate(const VectorField3D& m,
                                      const Material& mat,
                                      VectorField3D& H_out) const {
    if (D_ == 0.0) return;

    const StructuredGrid& g = m.grid();
    // Free-boundary DMI condition (Rohart-Thiaville / mumax3 default):
    //   dm/dn = -(D/2A) (z_hat x n_hat) x m  at missing in-plane neighbours
    // (sign follows THIS code's energy convention e = D[mz div(m) - m.grad(mz)];
    //  verified numerically against mumax3 relax() edge canting)
    // (z faces have n parallel to z_hat and contribute nothing). See
    // BulkDMIField::accumulate for the g_BC / dH_exch decomposition; without
    // this a uniform PMA+DMI film is a spurious equilibrium (no edge canting
    // sin(theta) ~ D / 2 sqrt(AK)). The 1/Ms prefactor and the exchange-ghost
    // term are per-cell when a MaterialField3D is attached.
    const bool bc = !open_bc_ && mat.A_exchange > Real{0};
    const Real quarterDoverA = bc ? D_ / (Real{4} * mat.A_exchange) : Real{0};

    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    const Real dx = g.dx(), dy = g.dy();
    const Index Ntot = nx * ny * nz;
    #pragma omp parallel for schedule(static) if(Ntot > 4096)
    for (Index idx = 0; idx < Ntot; ++idx) {
        const Index ix   = idx % nx;
        const Index trow = idx / nx;
        const Index iy   = trow % ny;
        const Index iz   = trow / ny;
        const Vec3 mc = m[idx];
        const Real Ms_c = matf_ ? matf_->Ms(idx) : mat.Ms;
        const Real prefac  = 2.0 * D_ / (constants::mu_0 * Ms_c);
        const Real bc_exch = bc ? D_ / (constants::mu_0 * Ms_c) : Real{0};
        Vec3 dH{0, 0, 0};
        Vec3 gx, gy;
        if (!bc) {
            gx = grad_x(m, g, ix, iy, iz);
            gy = grad_y(m, g, ix, iy, iz);
        } else {
            // Missing neighbour = outside grid OR vacuum (GeomMask, |m| ~ 0)
            // cell; both get the free-boundary condition (mumax3 zero-Msat
            // treatment). Both-faces-free: gradient = 2 (D/4A) gamma.
            if (mc.dot(mc) < Real{0.25}) continue;   // vacuum cell itself
            auto vac = [&](Index j) { const Vec3& v = m[j]; return v.dot(v) < Real{0.25}; };
            auto axis = [&](Index i, Index n, Real d, Index stride, const Vec3& gam) {
                const bool lo = (i > 0)     && !vac(idx - stride);
                const bool hi = (i < n - 1) && !vac(idx + stride);
                if (lo && hi) return (m[idx + stride] - m[idx - stride]) / (Real{2} * d);
                if (hi) { dH += gam * (-bc_exch / d);
                          return (m[idx + stride] - mc) * (Real{0.5} / d) + gam * quarterDoverA; }
                if (lo) { dH += gam * ( bc_exch / d);
                          return (mc - m[idx - stride]) * (Real{0.5} / d) + gam * quarterDoverA; }
                return gam * (Real{2} * quarterDoverA);
            };
            gx = axis(ix, nx, dx, Index{1}, Vec3{-mc.z, 0, mc.x});   // gamma_x = -(z_hat x x_hat) x m
            gy = axis(iy, ny, dy, nx,       Vec3{0, -mc.z, mc.y});   // gamma_y = -(z_hat x y_hat) x m
        }

        Vec3 H_dmi{ gx.z, gy.z, -(gx.x + gy.y) };
        H_out[idx] += H_dmi * prefac + dH;
    }
}

Real InterfacialDMIField::energy(const VectorField3D& m,
                                  [[maybe_unused]] const Material& mat) const {
    if (D_ == 0.0) return 0.0;

    const StructuredGrid& g = m.grid();
    const Real dV = g.cell_volume();
    Real E = 0.0;

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 mi = m[idx];
        Real div_xy = gx.x + gy.y;
        Real m_dot_grad_mz = mi.x * gx.z + mi.y * gy.z;
        E += mi.z * div_xy - m_dot_grad_mz;
    }
    return D_ * E * dV;
}

ScalarField3D InterfacialDMIField::energy_density(const VectorField3D& m,
                                                   [[maybe_unused]] const Material& mat) const {
    ScalarField3D edens(m.grid());
    if (D_ == 0.0) return edens;

    const StructuredGrid& g = m.grid();
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Index idx = g.linear_index(ix, iy, iz);
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 mi = m[idx];
        Real div_xy = gx.x + gy.y;
        Real m_dot_grad_mz = mi.x * gx.z + mi.y * gy.z;
        edens[idx] = D_ * (mi.z * div_xy - m_dot_grad_mz);
    }
    return edens;
}

}  // namespace micromag
