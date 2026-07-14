// sp1_thickness.cpp -- SP#1: critical size L_c vs element thickness
//
// For each thickness t = 5, 10, 20, 40 nm:
//   Scans element sizes L to bracket the vortex-nucleation critical size L_c(t).
//   Compares E_vortex vs E_S-state; finds crossing by linear interpolation.
//
// Extracts scaling exponent beta from  L_c ~ t^beta
// Reference: Cowburn et al., PRL 83, 1042 (1999) -- found beta ~ 1/3 experimentally.
//
// Material: Permalloy, Ms=800 kA/m, A=13 pJ/m, alpha=0.5
// Cells:    5 nm cubic throughout (t=5nm -> 1 cell in z; t=40nm -> 8 cells in z)

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

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

static double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static double max_torque(const VectorField3D& m,
                          const Material& mat,
                          EffectiveFieldSum& heff) {
    VectorField3D H(m.grid());
    heff.compute(m, mat, H);
    double mx = 0.0;
    for (Index i = 0; i < m.size(); ++i) {
        Vec3 t = m[i].cross(H[i]);
        mx = std::max(mx, std::sqrt(t.x*t.x + t.y*t.y + t.z*t.z));
    }
    return mx;
}

static double relax_energy(VectorField3D& m, const Material& mat,
                             EffectiveFieldSum& heff,
                             double tol_Am = 500.0, double t_max = 3e-9) {
    RK45Integrator integ;
    double t = 0.0; int steps = 0;
    while (t < t_max) {
        t += integ.step(m, mat, heff);
        if (++steps % 100 == 0 && max_torque(m, mat, heff) < tol_Am) break;
    }
    return heff.total_energy(m, mat);
}

// Returns DeltaE/E_uniform (negative = vortex wins)
static double delta_E_frac(int nx, int nz, double d, const Material& mat) {
    StructuredGrid g(nx, nx, nz, d, d, d);

    auto demag  = std::make_shared<DemagField>(g);
    auto exch_u = std::make_shared<ExchangeField>();
    auto exch_v = std::make_shared<ExchangeField>();
    EffectiveFieldSum heff_u, heff_v;
    heff_u.add(exch_u); heff_u.add(demag);
    heff_v.add(exch_v); heff_v.add(demag);

    VectorField3D mu(g); mu.set_uniform({1.0, 0.05, 0.0}); mu.normalize();
    double Eu = relax_energy(mu, mat, heff_u);

    VectorField3D mv(g); mv.set_vortex(nx*d*0.5, nx*d*0.5, 5e-9); mv.normalize();
    double Ev = relax_energy(mv, mat, heff_v);

    return (Ev - Eu) / std::abs(Eu);
}

// Scan sizes to bracket L_c; return interpolated L_c [nm].
// L_prev tracks the previous size explicitly (avoids UB pointer arithmetic).
static double find_Lc(int nz, double d_m, const Material& mat,
                       const std::vector<double>& sizes_nm,
                       std::ostream& log) {
    const double d = d_m;
    double dE_prev = 1.0;   // start positive (S-state wins)
    double L_prev  = 0.0;

    for (double L_nm : sizes_nm) {
        int nx = static_cast<int>(std::round(L_nm * 1e-9 / d));
        auto t0 = Clock::now();
        double dE = delta_E_frac(nx, nz, d, mat);
        double w  = elapsed_s(t0);

        log << "    L=" << std::setw(5) << std::setprecision(0) << std::fixed << L_nm
            << " nm  (" << std::setw(5) << nx*nx*nz << " cells)"
            << "  DeltaE/E_u=" << std::setw(7) << std::setprecision(1) << std::fixed
            << dE*100 << "%"
            << "  " << (dE < 0 ? "Vortex" : "S-state")
            << "  (" << std::setprecision(1) << w << " s)\n";

        if (dE < 0 && dE_prev >= 0 && L_prev > 0) {
            // Crossed from S-state to Vortex between L_prev and L_nm
            double frac = dE_prev / (dE_prev - dE);   // 0..1, linear interpolation
            return L_prev + frac * (L_nm - L_prev);
        }
        dE_prev = dE;
        L_prev  = L_nm;
    }
    if (dE_prev < 0) {
        log << "  (vortex already wins at smallest scanned size -- L_c < "
            << sizes_nm.front() << " nm)\n";
        return sizes_nm.front();
    }
    log << "  (S-state still wins at largest scanned size -- L_c > "
        << sizes_nm.back() << " nm)\n";
    return sizes_nm.back();
}

// ---------------------------------------------------------------------------
int main() {
    Material mat = Material::permalloy();
    mat.alpha = 0.5;

    const double lex = std::sqrt(2.0 * mat.A_exchange /
                                  (constants::mu_0 * mat.Ms * mat.Ms));
    const double d = 5e-9;  // 5 nm cells throughout

    std::cout << "=== SP#1: Critical Size L_c vs Thickness ===\n"
              << "Permalloy: Ms=" << mat.Ms/1e3 << " kA/m  A=" << mat.A_exchange*1e12
              << " pJ/m  alpha=" << mat.alpha << "  l_ex=" << lex*1e9 << " nm\n"
              << "Exchange + Demag, 5 nm cells, alpha=0.5\n\n";

    // (thickness_nm, nz, size_scan_nm[])
    struct ThickCase {
        double t_nm;
        int    nz;
        std::vector<double> sizes_nm;
    };

    const std::vector<ThickCase> cases = {
        // t=5 nm (1 cell): quasi-2D -- L_c expected >200 nm
        {  5, 1, { 60, 100, 140, 180, 220, 250, 260, 280, 300 }},
        // t=10 nm (2 cells): L_c ~ 115 nm confirmed by sp1_phase
        { 10, 2, { 80, 100, 110, 120, 130, 150 }},
        // t=20 nm (4 cells): scan from smaller sizes -- L_c may be < 100 nm
        { 20, 4, { 50, 60, 70, 80, 90, 100, 120, 150 }},
        // t=40 nm (8 cells): start even smaller
        { 40, 8, { 50, 60, 70, 80, 90, 100, 120, 150 }},
    };

    // Store results for scaling fit
    std::vector<double> t_vals, Lc_vals;

    for (const auto& c : cases) {
        std::cout << "--- t = " << c.t_nm << " nm  (nz=" << c.nz
                  << ", t/l_ex=" << std::setprecision(1) << c.t_nm/( lex*1e9)
                  << ") ---\n";

        auto t0_thick = Clock::now();
        double Lc = find_Lc(c.nz, d, mat, c.sizes_nm, std::cout);
        double wall_total = elapsed_s(t0_thick);

        std::cout << "  -> L_c ~ " << std::setprecision(1) << Lc
                  << " nm  (L_c/l_ex = " << Lc/(lex*1e9)
                  << ")  total: " << wall_total << " s\n\n";

        t_vals.push_back(c.t_nm);
        Lc_vals.push_back(Lc);
    }

    // -------------------------------------------------------------------
    // Summary table + scaling fit  L_c = a * t^beta
    // -------------------------------------------------------------------
    std::cout << "=== Summary: L_c vs thickness ===\n\n";
    std::cout << std::setw(10) << "t (nm)"
              << std::setw(12) << "t/l_ex"
              << std::setw(12) << "L_c (nm)"
              << std::setw(12) << "L_c/l_ex"
              << "\n"
              << std::string(46, '-') << "\n";
    for (size_t i = 0; i < t_vals.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(10) << t_vals[i]
                  << std::setw(12) << t_vals[i]/(lex*1e9)
                  << std::setw(12) << Lc_vals[i]
                  << std::setw(12) << Lc_vals[i]/(lex*1e9)
                  << "\n";
    }
    std::cout << std::string(46, '-') << "\n\n";

    // Power-law fit: log(L_c) = beta*log(t) + log(a)
    // Least-squares on log-log data
    if (t_vals.size() >= 2) {
        double sum_x=0, sum_y=0, sum_xx=0, sum_xy=0;
        int n = static_cast<int>(t_vals.size());
        for (int i = 0; i < n; ++i) {
            double lx = std::log(t_vals[i]);
            double ly = std::log(Lc_vals[i]);
            sum_x  += lx;
            sum_y  += ly;
            sum_xx += lx*lx;
            sum_xy += lx*ly;
        }
        double beta = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x);
        double loga = (sum_y - beta*sum_x) / n;
        double a    = std::exp(loga);

        std::cout << "Power-law fit:  L_c = " << std::setprecision(1) << a
                  << " x t^" << std::setprecision(3) << beta << " nm\n"
                  << "(Cowburn 1999 experiment: beta ~ 0.33-0.5)\n\n";

        // Predicted values
        std::cout << "Fit check:\n";
        for (int i = 0; i < n; ++i) {
            double Lc_fit = a * std::pow(t_vals[i], beta);
            std::cout << "  t=" << t_vals[i] << " nm:  measured L_c="
                      << std::setprecision(1) << Lc_vals[i]
                      << " nm,  fit=" << Lc_fit << " nm\n";
        }
    }

    return 0;
}
