#include "micromag/skyrmion_tools.hpp"
#include "micromag/topological_charge.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <queue>
#include <vector>

namespace micromag {

// ---------------------------------------------------------------------------
// skyrmion_corepos
// ---------------------------------------------------------------------------
std::pair<Real, Real> skyrmion_corepos(const VectorField3D& m, bool find_max) {
    const StructuredGrid& g = m.grid();
    const Real Lx_half = Real{0.5} * Real(g.nx()) * g.dx();
    const Real Ly_half = Real{0.5} * Real(g.ny()) * g.dy();

    Real best_mz = find_max ? -std::numeric_limits<Real>::max()
                             :  std::numeric_limits<Real>::max();
    Real cx = 0.0, cy = 0.0;

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Real mz = m[g.linear_index(ix, iy, iz)].z;
        const bool better = find_max ? (mz > best_mz) : (mz < best_mz);
        if (better) {
            best_mz = mz;
            cx = (Real(ix) + Real{0.5}) * g.dx() - Lx_half;
            cy = (Real(iy) + Real{0.5}) * g.dy() - Ly_half;
        }
    }
    return {cx, cy};
}

// ---------------------------------------------------------------------------
// bubble_pos — topological-charge-density weighted centroid
// ---------------------------------------------------------------------------
std::pair<Real, Real> bubble_pos(const VectorField3D& m) {
    const ScalarField3D dens = topological_charge_density(m);
    const StructuredGrid& g  = m.grid();
    const Real Lx_half = Real{0.5} * Real(g.nx()) * g.dx();
    const Real Ly_half = Real{0.5} * Real(g.ny()) * g.dy();

    Real sum_w = 0.0, sum_wx = 0.0, sum_wy = 0.0;
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Real w = std::abs(dens[g.linear_index(ix, iy, iz)]);
        const Real x = (Real(ix) + Real{0.5}) * g.dx() - Lx_half;
        const Real y = (Real(iy) + Real{0.5}) * g.dy() - Ly_half;
        sum_w  += w;
        sum_wx += w * x;
        sum_wy += w * y;
    }
    if (sum_w < 1e-30) return {0.0, 0.0};
    return {sum_wx / sum_w, sum_wy / sum_w};
}

// ---------------------------------------------------------------------------
// skyrmion_count — connected-component analysis on Q-density map
// ---------------------------------------------------------------------------
int skyrmion_count(const VectorField3D& m, Real threshold) {
    const ScalarField3D dens = topological_charge_density(m);
    const StructuredGrid& g  = m.grid();
    const Index gx = g.nx(), gy = g.ny();
    const Real factor = g.dx() * g.dy() / (4.0 * std::numbers::pi);

    // Collapse z: per-xy topological charge contribution
    const size_t Nxy = static_cast<size_t>(gx * gy);
    std::vector<Real> q_xy(Nxy, 0.0);
    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < gy; ++iy)
    for (Index ix = 0; ix < gx; ++ix)
        q_xy[static_cast<size_t>(ix + gx*iy)] +=
            dens[g.linear_index(ix, iy, iz)] * factor;

    // Active-cell threshold: 5% of peak |q_xy|
    Real q_max = 0.0;
    for (Real q : q_xy) q_max = std::max(q_max, std::abs(q));
    if (q_max < 1e-20) return 0;
    const Real cell_min = q_max * Real{0.05};

    // BFS connected components
    std::vector<bool> visited(Nxy, false);
    int count = 0;

    for (Index sy = 0; sy < gy; ++sy)
    for (Index sx = 0; sx < gx; ++sx) {
        const size_t flat0 = static_cast<size_t>(sx + gx*sy);
        if (visited[flat0] || std::abs(q_xy[flat0]) <= cell_min) continue;

        std::queue<std::pair<Index,Index>> bfs;
        bfs.push({sx, sy});
        visited[flat0] = true;
        Real Q_comp = 0.0;

        while (!bfs.empty()) {
            auto [px, py] = bfs.front(); bfs.pop();
            Q_comp += q_xy[static_cast<size_t>(px + gx*py)];

            const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (const auto& d : dirs) {
                const Index nbx = px + d[0], nby = py + d[1];
                if (nbx < 0 || nbx >= gx || nby < 0 || nby >= gy) continue;
                const size_t nbflat = static_cast<size_t>(nbx + gx*nby);
                if (visited[nbflat] || std::abs(q_xy[nbflat]) <= cell_min) continue;
                visited[nbflat] = true;
                bfs.push({nbx, nby});
            }
        }
        if (std::abs(Q_comp) >= threshold) ++count;
    }
    return count;
}

}  // namespace micromag
