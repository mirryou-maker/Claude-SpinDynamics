// sp3.cpp — µMAG Standard Problem #3: Hysteresis loop
//
// Computes the magnetisation hysteresis loop for a square Permalloy element:
//   Geometry   : 1 µm × 1 µm × 20 nm  (10 nm cells → 100×100×2 = 20 K cells)
//   Protocol   : Sweep H_x from +150 mT → −150 mT in 5 mT steps.
//                At each field: RK45 relax until |m×H_eff|_max < 1 kA/m or t > 2 ns.
//                Warm start: each field begins from the previous field's final state.
//   Expected   : Vortex nucleation at H_nuc ≈ −15 to −30 mT (10 nm cells)
//                Vortex annihilation (reversal) at H_ann ≈ −60 to −90 mT
//
// Note: 10 nm cells give l_ex ≈ 1.8 dx — captures hysteresis qualitatively.
//       For quantitative accuracy (µMAG SP#3 reference) use 5 nm cells.
//
// Run: .\build\windows-msvc\bin\Release\sp3.exe

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "micromag/demag.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/exchange.hpp"
#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/integrator.hpp"
#include "micromag/material.hpp"
#include "micromag/types.hpp"
#include "micromag/vtk_writer.hpp"
#include "micromag/zeeman.hpp"

using namespace micromag;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static Vec3 mean_m(const VectorField3D& m) {
    Vec3 s{0, 0, 0};
    for (Index i = 0; i < m.size(); ++i) {
        s.x += m[i].x; s.y += m[i].y; s.z += m[i].z;
    }
    const double N = static_cast<double>(m.size());
    return {s.x/N, s.y/N, s.z/N};
}

// Max pointwise |m × H_eff| — convergence diagnostic (A/m)
static double max_torque(const VectorField3D& m, const Material& mat,
                          EffectiveFieldSum& heff)
{
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

// Relax to equilibrium with RK45 adaptive stepping.
// Returns number of steps taken.
static int relax(VectorField3D& m, const Material& mat, EffectiveFieldSum& heff,
                  double tol_Am = 1e3, double t_max = 2e-9)
{
    RK45Integrator::Options opts;
    opts.dt_init = 5e-14;
    opts.dt_max  = 5e-12;   // 5 ps max step — stable for α=0.5 Permalloy
    RK45Integrator integ(opts);

    double t = 0.0;
    int steps = 0;
    while (t < t_max) {
        double dt = integ.step(m, mat, heff);
        t += dt;
        ++steps;
        if (steps % 50 == 0 && max_torque(m, mat, heff) < tol_Am) break;
    }
    return steps;
}

// State classification
static const char* classify(double mx, double mag) {
    if (mag > 0.80) return mx > 0 ? "Sat+x" : "Sat-x";
    if (mag < 0.30) return "Vortex";
    return "Multi";
}

// ---------------------------------------------------------------------------
int main() {
    const double mu_0     = constants::mu_0;
    const double mT_to_Am = 1e-3 / mu_0;   // 1 mT → H in A/m

    // Grid: 1 µm × 1 µm × 20 nm, 10 nm cells → 100×100×2 = 20 K cells
    const StructuredGrid grid(100, 100, 2, 10e-9, 10e-9, 10e-9);

    Material mat = Material::permalloy();
    mat.alpha = 0.5;   // overdamped — fast convergence to local minimum

    const double lex = std::sqrt(2.0 * mat.A_exchange / (mu_0 * mat.Ms * mat.Ms));

    std::cout << "=== µMAG Standard Problem #3: Hysteresis Loop ===\n"
              << "Permalloy: Ms=" << std::fixed << std::setprecision(0)
              << mat.Ms/1e3 << " kA/m  "
              << "A=" << mat.A_exchange*1e12 << " pJ/m  "
              << std::setprecision(2) << "alpha=" << mat.alpha << "\n"
              << "l_ex = " << lex*1e9 << " nm  "
              << "(dx = " << grid.dx()*1e9 << " nm = "
              << std::setprecision(1) << grid.dx()/lex << " l_ex)\n"
              << "Grid: " << grid.nx() << "x" << grid.ny() << "x" << grid.nz()
              << " = " << grid.size() << " cells  ("
              << grid.nx()*grid.dx()*1e9 << "x"
              << grid.ny()*grid.dy()*1e9 << "x"
              << grid.nz()*grid.dz()*1e9 << " nm)\n\n";

    // -----------------------------------------------------------------------
    // Build effective field (DemagField is expensive — construct once)
    // -----------------------------------------------------------------------
    std::cout << "Constructing DemagField... " << std::flush;
    auto t_ctor = Clock::now();
    auto demag_ptr = std::make_shared<DemagField>(grid);
    std::cout << elapsed_ms(t_ctor) << " ms\n\n";

    auto zeeman_ptr = std::make_shared<ZeemanField>();
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ExchangeField>());
    heff.add(demag_ptr);
    heff.add(zeeman_ptr);

    // -----------------------------------------------------------------------
    // Initial state: uniform +x (small tilt to break perfect symmetry)
    // -----------------------------------------------------------------------
    VectorField3D m(grid);
    m.set_uniform({1.0, 0.02, 0.01});
    m.normalize();

    // Pre-relax at +150 mT to establish clean starting state
    const double H_start_mT = 150.0;
    std::cout << "Pre-relaxing at H = +" << H_start_mT << " mT... " << std::flush;
    zeeman_ptr->set_H_ext({H_start_mT * mT_to_Am, 0.0, 0.0});
    auto t_pre = Clock::now();
    int n_pre = relax(m, mat, heff, 500.0, 3e-9);   // tighter tol for start
    std::cout << elapsed_ms(t_pre) << " ms  (" << n_pre << " steps)\n\n";

    // -----------------------------------------------------------------------
    // Field sweep: +150 → −150 mT in 5 mT steps
    // -----------------------------------------------------------------------
    const double H_max_mT  =  150.0;
    const double H_min_mT  = -150.0;
    const double H_step_mT =   -5.0;
    const int    n_fields   = static_cast<int>(
        std::round((H_min_mT - H_max_mT) / H_step_mT)) + 1;

    // Table header
    const int W = 8;
    std::cout << std::setw(W) << "H(mT)"
              << std::setw(W) << "<mx>"
              << std::setw(W) << "<my>"
              << std::setw(W) << "|<m>|"
              << std::setw(8) << "State"
              << std::setw(7) << "steps"
              << "\n" << std::string(55, '-') << "\n";

    struct Row { double H_mT, mx, my, mag; };
    std::vector<Row> rows;
    rows.reserve(n_fields);

    double prev_mag = 1.0;
    double prev_mx  = 1.0;
    double H_nuc    = 1e9;   // sentinel: not found yet (first |<m>| drop)
    double H_sw     = 1e9;   // sentinel: <mx> crossing zero
    double H_ann    = 1e9;   // sentinel: reversal complete

    for (double H_mT = H_max_mT; H_mT >= H_min_mT - 0.5; H_mT += H_step_mT) {
        zeeman_ptr->set_H_ext({H_mT * mT_to_Am, 0.0, 0.0});

        int nsteps = relax(m, mat, heff, 1e3, 2e-9);

        Vec3 avg = mean_m(m);
        double mag = std::sqrt(avg.x*avg.x + avg.y*avg.y + avg.z*avg.z);
        const char* state = classify(avg.x, mag);

        // Non-uniform nucleation: first step where |<m>| drops below 0.70
        // (indicating non-uniform reversal onset; coarse 10 nm grid gives partial vortex).
        if (H_nuc > 1e8 && prev_mag > 0.75 && mag < 0.70)
            H_nuc = H_mT;

        // Switching field: <mx> first crosses zero (positive → negative)
        if (H_sw > 1e8 && prev_mx > 0 && avg.x < 0)
            H_sw = H_mT;

        // Reversal complete: |<m>| recovers with mx < −0.70
        if (H_sw < 1e8 && H_ann > 1e8 && avg.x < -0.70)
            H_ann = H_mT;

        // Marker for events
        const char* marker = "";
        if (std::abs(H_mT - H_nuc) < 0.1) marker = "  <-- nucleation";
        if (std::abs(H_mT - H_sw)  < 0.1) marker = "  <-- mx=0 (switching)";
        if (std::abs(H_mT - H_ann) < 0.1) marker = "  <-- reversal complete";

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(W)   << H_mT
                  << std::setprecision(4)
                  << std::setw(W)   << avg.x
                  << std::setw(W)   << avg.y
                  << std::setw(W)   << mag
                  << std::setw(8)   << state
                  << std::setw(7)   << nsteps
                  << marker << "\n";

        rows.push_back({H_mT, avg.x, avg.y, mag});
        prev_mag = mag;
        prev_mx  = avg.x;
    }

    // -----------------------------------------------------------------------
    // Results summary
    // -----------------------------------------------------------------------
    std::cout << std::string(55, '-') << "\n\n";
    std::cout << "=== Results ===\n" << std::setprecision(1);
    if (H_nuc < 1e8)
        std::cout << "  Non-uniform nucleation  H_nuc = " << H_nuc << " mT\n";
    else
        std::cout << "  Non-uniform nucleation: not detected\n";
    if (H_sw < 1e8)
        std::cout << "  Switching field        H_sw  = " << H_sw  << " mT"
                  << "  (<mx> crosses zero)\n";
    else
        std::cout << "  Switching field: not detected\n";
    if (H_ann < 1e8)
        std::cout << "  Reversal complete      H_ann = " << H_ann << " mT\n";
    else
        std::cout << "  Reversal complete: not detected\n";
    if (H_nuc < 1e8 && H_ann < 1e8)
        std::cout << "  Hysteresis window:  DeltaH = "
                  << std::abs(H_ann - H_nuc) << " mT\n";
    std::cout << "\n  Note: 10 nm cells (1.8 l_ex) under-resolve vortex core.\n"
              << "        Use 5 nm cells for quantitative µMAG SP#3 accuracy.\n";

    // Save CSV for notebook plotting
    std::ofstream csv("sp3_hysteresis.csv");
    csv << "H_mT,mx,my,mag\n";
    for (auto& r : rows)
        csv << std::fixed << std::setprecision(4)
            << r.H_mT << "," << r.mx << "," << r.my << "," << r.mag << "\n";
    csv.close();
    std::cout << "\nData saved: sp3_hysteresis.csv\n";

    // Save VTK at final state (after reversal) for visualisation
    write_vtk_legacy("sp3_final.vtu", m, "m");
    std::cout << "VTK saved:  sp3_final.vtu\n";

    return 0;
}
