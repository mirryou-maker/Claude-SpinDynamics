#include "micromag/exchange.hpp"
#include "micromag/geom_mask.hpp"
#include "micromag/material_field.hpp"
#include "micromag/region_map.hpp"
#include "micromag/detail/grad_helpers.hpp"

namespace micromag {

namespace {

// Linear index of the neighbor to use for the bond from (i,j,k) in direction
// (di,dj,dk), with Neumann fallback for both grid AND mask boundaries.
// If mask != nullptr and the in-bounds neighbor has mask < 0.5, the centre
// index is returned (zero exchange flux at the geometry interface).
inline Index neighbor_index(const StructuredGrid& g,
                            Index i, Index j, Index k,
                            Index di, Index dj, Index dk,
                            BoundaryCondition bc,
                            const GeomMask* mask) {
    Index ni = i + di, nj = j + dj, nk = k + dk;
    const bool out = (ni < 0 || ni >= g.nx() ||
                      nj < 0 || nj >= g.ny() ||
                      nk < 0 || nk >= g.nz());
    if (!out) {
        if (mask && (*mask)(ni, nj, nk) < Real{0.5})
            return g.linear_index(i, j, k);
        return g.linear_index(ni, nj, nk);
    }
    if (bc == BoundaryCondition::Neumann) return g.linear_index(i, j, k);
    // Periodic — sign-safe modulo
    ni = (ni % g.nx() + g.nx()) % g.nx();
    nj = (nj % g.ny() + g.ny()) % g.ny();
    nk = (nk % g.nz() + g.nz()) % g.nz();
    return g.linear_index(ni, nj, nk);
}

// Region-boundary exchange stiffness (harmonic mean — Oxs/mumax3 convention).
inline Real harmonic_mean(Real a, Real b) {
    const Real s = a + b;
    return (s > 0) ? (Real{2} * a * b / s) : Real{0};
}

}  // namespace

// ---------------------------------------------------------------------------
// set_inter_exchange / inter_exchange / clear_inter_exchange
// ---------------------------------------------------------------------------
void ExchangeField::set_inter_exchange(uint8_t ri, uint8_t rj, Real A_IEC) {
    inter_A_[inter_key(ri, rj)] = A_IEC;
}

Real ExchangeField::inter_exchange(uint8_t ri, uint8_t rj) const {
    auto it = inter_A_.find(inter_key(ri, rj));
    return (it != inter_A_.end()) ? it->second : Real{-1};
}

// ---------------------------------------------------------------------------
// accumulate
// ---------------------------------------------------------------------------
void ExchangeField::accumulate(const VectorField3D& m,
                                const Material& mat,
                                VectorField3D& H_out) const {
    if (!matf_ && mat.A_exchange == 0 && !rmap_) return;

    const StructuredGrid& g = m.grid();
    const Real idx2 = 1.0 / (g.dx() * g.dx());
    const Real idy2 = 1.0 / (g.dy() * g.dy());
    const Real idz2 = 1.0 / (g.dz() * g.dz());
    const Real pre_uniform = 2.0 * mat.A_exchange / (constants::mu_0 * mat.Ms);

    // Flattened single loop (x-fastest: ic = i + nx*(j + ny*k)) so OpenMP
    // parallelises across all cells regardless of thin-z grid shape. Each
    // iteration writes only its own H_out[ic] → race-free.
    const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
    const Index Ntot = nx * ny * nz;
    #pragma omp parallel for schedule(static) if(Ntot > 4096)
    for (Index ic = 0; ic < Ntot; ++ic) {
        const Index i    = ic % nx;
        const Index trow = ic / nx;
        const Index j    = trow % ny;
        const Index k    = trow / ny;
        // Skip cells outside the geometry (mask == 0 → inactive)
        if (mask_ && (*mask_)(i, j, k) < Real{0.5}) continue;

        const Vec3  mc = m[ic];

        const Index in_px = neighbor_index(g, i,j,k, +1,0,0, bc_, mask_);
        const Index in_mx = neighbor_index(g, i,j,k, -1,0,0, bc_, mask_);
        const Index in_py = neighbor_index(g, i,j,k, 0,+1,0, bc_, mask_);
        const Index in_my = neighbor_index(g, i,j,k, 0,-1,0, bc_, mask_);
        const Index in_pz = neighbor_index(g, i,j,k, 0,0,+1, bc_, mask_);
        const Index in_mz = neighbor_index(g, i,j,k, 0,0,-1, bc_, mask_);

        const bool need_per_bond = (matf_ != nullptr) || (rmap_ != nullptr && !inter_A_.empty());

        if (!need_per_bond) {
            Vec3 lap = (m[in_px] - mc) * idx2 + (m[in_mx] - mc) * idx2 +
                       (m[in_py] - mc) * idy2 + (m[in_my] - mc) * idy2 +
                       (m[in_pz] - mc) * idz2 + (m[in_mz] - mc) * idz2;
            H_out[ic] += lap * pre_uniform;
            continue;
        }

        // Per-bond path: per-cell material or inter-region coupling
        const Real Ms_c = matf_ ? matf_->Ms(ic) : mat.Ms;
        if (Ms_c <= 0) continue;
        const Real A_c   = matf_ ? matf_->A_exchange(ic) : mat.A_exchange;
        const Real pre_c = 2.0 / (constants::mu_0 * Ms_c);

        // Returns A_ij for the bond (ic → in), respecting inter-region table.
        const uint8_t rid_c = rmap_ ? (*rmap_)[ic] : uint8_t{0};
        auto bond_A = [&](Index in) -> Real {
            if (rmap_) {
                uint8_t rid_n = (*rmap_)[in];
                if (rid_n != rid_c) {
                    Real iec = lookup_inter(rid_c, rid_n);
                    if (!std::isnan(iec)) return iec;
                }
            }
            const Real A_n = matf_ ? matf_->A_exchange(in) : mat.A_exchange;
            return harmonic_mean(A_c, A_n);
        };

        auto bond = [&](Index in, Real ih2) -> Vec3 {
            return (m[in] - mc) * (bond_A(in) * ih2);
        };

        Vec3 acc = bond(in_px, idx2) + bond(in_mx, idx2)
                 + bond(in_py, idy2) + bond(in_my, idy2)
                 + bond(in_pz, idz2) + bond(in_mz, idz2);
        H_out[ic] += acc * pre_c;
    }
}

Real ExchangeField::energy(const VectorField3D& m,
                            const Material& mat) const {
    if (!matf_ && mat.A_exchange == 0) return 0;

    const StructuredGrid& g = m.grid();
    const Real V    = g.cell_volume();
    const Real idx2 = 1.0 / (g.dx() * g.dx());
    const Real idy2 = 1.0 / (g.dy() * g.dy());
    const Real idz2 = 1.0 / (g.dz() * g.dz());
    Real sum = 0;

    // Iterate only +x, +y, +z bonds to avoid double-counting.
    // Skip bonds where either endpoint is outside the geometry.
    for (Index k = 0; k < g.nz(); ++k)
    for (Index j = 0; j < g.ny(); ++j)
    for (Index i = 0; i < g.nx(); ++i) {
        if (mask_ && (*mask_)(i, j, k) < Real{0.5}) continue;

        const Index ic = g.linear_index(i, j, k);
        const Vec3  mc = m[ic];
        const Real  A_c = matf_ ? matf_->A_exchange(ic) : mat.A_exchange;

        auto add_bond = [&](Index ni, Index nj, Index nk, Real ih2) {
            if (mask_ && (*mask_)(ni, nj, nk) < Real{0.5}) return;
            const Index in = g.linear_index(ni, nj, nk);
            const Real  A_b = matf_ ? harmonic_mean(A_c, matf_->A_exchange(in)) : A_c;
            Vec3 d = m[in] - mc;
            sum += A_b * d.norm_squared() * ih2;
        };

        if (i + 1 < g.nx())                              add_bond(i+1, j,   k,   idx2);
        else if (bc_ == BoundaryCondition::Periodic)      add_bond(0,   j,   k,   idx2);

        if (j + 1 < g.ny())                              add_bond(i,   j+1, k,   idy2);
        else if (bc_ == BoundaryCondition::Periodic)      add_bond(i,   0,   k,   idy2);

        if (k + 1 < g.nz())                              add_bond(i,   j,   k+1, idz2);
        else if (bc_ == BoundaryCondition::Periodic)      add_bond(i,   j,   0,   idz2);
    }
    return sum * V;
}

ScalarField3D ExchangeField::energy_density(const VectorField3D& m,
                                             const Material& mat) const {
    ScalarField3D edens(m.grid());
    if (!matf_ && mat.A_exchange == 0) return edens;

    const StructuredGrid& g = m.grid();
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        if (mask_ && (*mask_)(ix, iy, iz) < Real{0.5}) continue;
        const Index idx = g.linear_index(ix, iy, iz);
        const Real A = matf_ ? matf_->A_exchange(idx) : mat.A_exchange;
        using detail::grad_x; using detail::grad_y; using detail::grad_z;
        Vec3 gx = grad_x(m, g, ix, iy, iz);
        Vec3 gy = grad_y(m, g, ix, iy, iz);
        Vec3 gz = grad_z(m, g, ix, iy, iz);
        // e = A*(|∂m/∂x|² + |∂m/∂y|² + |∂m/∂z|²) [J/m³]
        edens[idx] = A * (gx.dot(gx) + gy.dot(gy) + gz.dot(gz));
    }
    return edens;
}

}  // namespace micromag
