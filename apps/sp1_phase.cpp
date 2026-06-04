// sp1_phase.cpp — µMAG SP#1 Phase Diagram
//
// Scans square Permalloy element sizes from 60 to 300 nm.
// For each size: relaxes both uniform (+x) and vortex initial conditions,
// compares total energies, and identifies which state is the ground state.
//
// Finds the critical vortex-nucleation size L_c where E_vortex crosses E_uniform.
//
// Material:  Permalloy (Ms=800 kA/m, A=13 pJ/m, K=0, α=0.5)
// Geometry:  L×L×10 nm square, 5 nm cubic cells
// Physics:   Exchange + Demag only (no Zeeman field)
// Reference: Cowburn et al., PRL 83, 1042 (1999) — Py squares, L_c ≈ 220 nm

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "micromag/demag.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/exchange.hpp"
#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/integrator.hpp"
#include "micromag/material.hpp"
#include "micromag/spin_torque.hpp"
#include "micromag/types.hpp"

using namespace micromag;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static Vec3 mean_m(const VectorField3D& m) {
    Vec3 s{0,0,0};
    for (Index i = 0; i < m.size(); ++i) {
        s.x += m[i].x; s.y += m[i].y; s.z += m[i].z;
    }
    const double N = static_cast<double>(m.size());
    return {s.x/N, s.y/N, s.z/N};
}

// Max pointwise torque |m × H| for convergence check
static double max_torque(const VectorField3D& m,
                          const Material& mat,
                          EffectiveFieldSum& heff) {
    VectorField3D H(m.grid());
    heff.compute(m, mat, H);
    double mx = 0.0;
    for (Index i = 0; i < m.size(); ++i) {
        Vec3 t = m[i].cross(H[i]);
        double v = std::sqrt(t.x*t.x + t.y*t.y + t.z*t.z);
        if (v > mx) mx = v;
    }
    return mx;
}

// Relax with RK45; return final total energy [J]
static double relax_energy(VectorField3D& m,
                             const Material& mat,
                             EffectiveFieldSum& heff,
                             double tol_Am = 500.0,  // converge tighter for energy accuracy
                             double t_max  = 3e-9)
{
    RK45Integrator integ;
    double t = 0.0;
    int steps = 0;

    while (t < t_max) {
        double dt = integ.step(m, mat, heff);
        t += dt;
        ++steps;
        if (steps % 100 == 0) {
            if (max_torque(m, mat, heff) < tol_Am) break;
        }
    }
    return heff.total_energy(m, mat);
}

// ---------------------------------------------------------------------------
// Single size: returns (E_uniform, E_vortex, |<m>|_unif, |<m>|_vortex)
// ---------------------------------------------------------------------------
static void run_size(int n_cells,           // cells per side (L/5nm)
                      const Material& mat,
                      double cell_nm,        // cell size [m]
                      double& E_unif,
                      double& E_vort,
                      double& mag_unif,
                      double& mag_vort,
                      double& wall_s)
{
    const double d = cell_nm;
    const int nz = 2;  // 10 nm thick = 2 × 5 nm cells
    StructuredGrid g(n_cells, n_cells, nz, d, d, d);

    auto t0 = Clock::now();

    // Build shared field stack (reuse heavy DemagField twice)
    auto demag   = std::make_shared<DemagField>(g);
    auto exch_u  = std::make_shared<ExchangeField>();
    auto exch_v  = std::make_shared<ExchangeField>();

    EffectiveFieldSum heff_u, heff_v;
    heff_u.add(exch_u);  heff_u.add(demag);
    heff_v.add(exch_v);  heff_v.add(demag);

    // --- Uniform init ---
    VectorField3D m_u(g);
    m_u.set_uniform({1.0, 0.05, 0.0});
    m_u.normalize();
    E_unif = relax_energy(m_u, mat, heff_u);
    Vec3 avg_u = mean_m(m_u);
    mag_unif = std::sqrt(avg_u.x*avg_u.x + avg_u.y*avg_u.y + avg_u.z*avg_u.z);

    // --- Vortex init ---
    VectorField3D m_v(g);
    m_v.set_vortex(n_cells * d * 0.5, n_cells * d * 0.5, 5e-9);
    m_v.normalize();
    E_vort = relax_energy(m_v, mat, heff_v);
    Vec3 avg_v = mean_m(m_v);
    mag_vort = std::sqrt(avg_v.x*avg_v.x + avg_v.y*avg_v.y + avg_v.z*avg_v.z);

    wall_s = elapsed_s(t0);
}

// ---------------------------------------------------------------------------
int main() {
    Material mat = Material::permalloy();
    mat.alpha = 0.5;

    const double lex = std::sqrt(2.0 * mat.A_exchange /
                                  (constants::mu_0 * mat.Ms * mat.Ms));
    const double cell_m = 5e-9;

    std::cout << "=== SP#1 Phase Diagram: Vortex Nucleation in Permalloy Squares ===\n"
              << "Ms=" << mat.Ms/1e3 << " kA/m  A=" << mat.A_exchange*1e12 << " pJ/m"
              << "  α=" << mat.alpha
              << "  l_ex=" << lex*1e9 << " nm\n"
              << "Geometry: L×L×10 nm, 5 nm cells, Exchange + Demag\n\n";

    // Print table header
    std::cout << std::left
              << std::setw(8)  << "L (nm)"
              << std::setw(8)  << "L/l_ex"
              << std::setw(8)  << "Cells"
              << std::setw(14) << "E_unif (aJ)"
              << std::setw(14) << "E_vortex (aJ)"
              << std::setw(12) << "ΔE/E_u (%)"
              << std::setw(12) << "|<m>|_u"
              << std::setw(12) << "|<m>|_v"
              << std::setw(14) << "Ground state"
              << "\n"
              << std::string(102, '-') << "\n";

    // Sizes to scan [nm]
    const double sizes[] = {60, 80, 100, 110, 120, 130, 140, 150,
                              160, 180, 200, 250, 300};
    const int N = static_cast<int>(sizeof(sizes)/sizeof(sizes[0]));

    double L_cross_lo = 0, L_cross_hi = 0;  // brackets for critical size
    double prev_dE_frac = 0;
    bool crossed = false;

    for (int k = 0; k < N; ++k) {
        const double L_nm = sizes[k];
        const int n_cells = static_cast<int>(std::round(L_nm * 1e-9 / cell_m));

        double E_u, E_v, mag_u, mag_v, wall;
        run_size(n_cells, mat, cell_m, E_u, E_v, mag_u, mag_v, wall);

        const double dE_frac = (E_v - E_u) / std::abs(E_u) * 100.0;  // % of E_uniform
        const bool vortex_wins = (E_v < E_u);
        const char* ground = vortex_wins ? "Vortex ←" : "S-state";

        std::cout << std::fixed << std::left
                  << std::setw(8)  << std::setprecision(0) << L_nm
                  << std::setw(8)  << std::setprecision(1) << L_nm/( lex*1e9)
                  << std::setw(8)  << n_cells*n_cells*2
                  << std::setw(14) << std::setprecision(4) << E_u * 1e18
                  << std::setw(14) << std::setprecision(4) << E_v * 1e18
                  << std::setw(12) << std::setprecision(1) << dE_frac
                  << std::setw(12) << std::setprecision(3) << mag_u
                  << std::setw(12) << std::setprecision(3) << mag_v
                  << std::setw(14) << ground
                  << "  (" << std::setprecision(1) << wall << " s)\n";

        // Detect crossing (sign change in ΔE)
        if (k > 0 && !crossed && (dE_frac > 0) != (prev_dE_frac > 0)) {
            L_cross_lo = sizes[k-1];
            L_cross_hi = L_nm;
            crossed = true;
        }
        prev_dE_frac = dE_frac;
    }

    std::cout << std::string(102, '-') << "\n\n";

    if (crossed) {
        // Linear interpolation to find L_c
        // E_v(L) - E_u(L) = 0  ↔  ΔE/E_u = 0
        std::cout << "Critical size L_c (vortex nucleation): "
                  << L_cross_lo << " – " << L_cross_hi << " nm\n"
                  << "(linear interpolation: ~"
                  << 0.5*(L_cross_lo + L_cross_hi) << " nm)\n\n";
    } else {
        std::cout << "No crossover found in scanned range — "
                  << "extend scan or check convergence.\n\n";
    }

    std::cout << "Interpretation:\n"
              << "  S-state region:  L < L_c  (exchange energy dominates, single-domain cheaper)\n"
              << "  Vortex region:   L > L_c  (demag energy of closure domains offsets vortex core cost)\n"
              << "  Bistability:     both states can be metastable near L_c\n";

    return 0;
}
