#include "micromag/exchange.hpp"

namespace micromag {

namespace {

// Fetch neighbor; Neumann: return self (→ zero Laplacian contribution);
// Periodic: wrap with sign-safe modulo.
inline Vec3 get_neighbor(const VectorField3D& m, const StructuredGrid& g,
                          Index i, Index j, Index k,
                          Index di, Index dj, Index dk,
                          BoundaryCondition bc) {
    Index ni = i + di, nj = j + dj, nk = k + dk;
    const bool out = (ni < 0 || ni >= g.nx() ||
                      nj < 0 || nj >= g.ny() ||
                      nk < 0 || nk >= g.nz());
    if (!out) return m.at(ni, nj, nk);
    if (bc == BoundaryCondition::Neumann) return m.at(i, j, k);
    // Periodic — sign-safe modulo
    ni = (ni % g.nx() + g.nx()) % g.nx();
    nj = (nj % g.ny() + g.ny()) % g.ny();
    nk = (nk % g.nz() + g.nz()) % g.nz();
    return m.at(ni, nj, nk);
}

}  // namespace

void ExchangeField::accumulate(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    if (mat.A_exchange == 0) return;

    const StructuredGrid& g = m.grid();
    const Real idx2 = 1.0 / (g.dx() * g.dx());
    const Real idy2 = 1.0 / (g.dy() * g.dy());
    const Real idz2 = 1.0 / (g.dz() * g.dz());
    const Real pre  = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);

    for (Index k = 0; k < g.nz(); ++k)
    for (Index j = 0; j < g.ny(); ++j)
    for (Index i = 0; i < g.nx(); ++i) {
        Vec3 mc = m.at(i, j, k);
        Vec3 lap =
            (get_neighbor(m, g, i,j,k, +1,0,0, bc_) - mc) * idx2 +
            (get_neighbor(m, g, i,j,k, -1,0,0, bc_) - mc) * idx2 +
            (get_neighbor(m, g, i,j,k, 0,+1,0, bc_) - mc) * idy2 +
            (get_neighbor(m, g, i,j,k, 0,-1,0, bc_) - mc) * idy2 +
            (get_neighbor(m, g, i,j,k, 0,0,+1, bc_) - mc) * idz2 +
            (get_neighbor(m, g, i,j,k, 0,0,-1, bc_) - mc) * idz2;
        H_out.at(i, j, k) += lap * pre;
    }
}

Real ExchangeField::energy(const VectorField3D& m,
                            const Material& mat) const {
    if (mat.A_exchange == 0) return 0;

    const StructuredGrid& g = m.grid();
    const Real V    = g.cell_volume();
    const Real idx2 = 1.0 / (g.dx() * g.dx());
    const Real idy2 = 1.0 / (g.dy() * g.dy());
    const Real idz2 = 1.0 / (g.dz() * g.dz());
    Real sum = 0;

    // Iterate only +x, +y, +z bonds to avoid double-counting.
    for (Index k = 0; k < g.nz(); ++k)
    for (Index j = 0; j < g.ny(); ++j)
    for (Index i = 0; i < g.nx(); ++i) {
        Vec3 mc = m.at(i, j, k);
        auto add_bond = [&](Index ni, Index nj, Index nk, Real ih2) {
            Vec3 d = m.at(ni, nj, nk) - mc;
            sum += d.norm_squared() * ih2;
        };

        if (i + 1 < g.nx())                              add_bond(i+1, j,   k,   idx2);
        else if (bc_ == BoundaryCondition::Periodic)      add_bond(0,   j,   k,   idx2);

        if (j + 1 < g.ny())                              add_bond(i,   j+1, k,   idy2);
        else if (bc_ == BoundaryCondition::Periodic)      add_bond(i,   0,   k,   idy2);

        if (k + 1 < g.nz())                              add_bond(i,   j,   k+1, idz2);
        else if (bc_ == BoundaryCondition::Periodic)      add_bond(i,   j,   0,   idz2);
    }
    return mat.A_exchange * sum * V;
}

}  // namespace micromag
