// sp1.cpp — µMAG Standard Problem #1
//
// Finds equilibrium magnetization states in thin Permalloy squares.
// Only Exchange + Demag fields (no applied Zeeman).
// High damping α=0.5 for fast convergence to local energy minima.
//
// Two element sizes bracket the critical vortex-nucleation length:
//   Small: 200×200×10 nm  →  S-state / flower expected
//   Large: 500×500×10 nm  →  Vortex expected
//
// Each size tested with two initial conditions:
//   A) Uniform +x (saturated)
//   B) Explicit in-plane vortex
//
// Permalloy parameters (NIST µMAG standard):
//   Ms = 860 kA/m,  A = 13 pJ/m,  K = 0,  α = 0.5
//
// Exchange length: l_ex = sqrt(2A / (µ₀ Ms²)) ≈ 5.3 nm
//   → Small element: 200 nm ≈ 38 l_ex  (borderline, S-state likely)
//   → Large element: 500 nm ≈ 94 l_ex  (vortex expected)
//
// Run: .\build\windows-msvc\bin\Release\sp1.exe

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
#include "micromag/vtk_writer.hpp"

using namespace micromag;
using Clock = std::chrono::steady_clock;

static double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Volume-averaged magnetization
static Vec3 mean_m(const VectorField3D& m) {
    Vec3 s{0, 0, 0};
    for (Index i = 0; i < m.size(); ++i) {
        s.x += m[i].x; s.y += m[i].y; s.z += m[i].z;
    }
    const double N = static_cast<double>(m.size());
    return {s.x / N, s.y / N, s.z / N};
}

// Max pointwise |m × H_eff| (LLG torque) — convergence diagnostic
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

// Relax to equilibrium with RK45.
// Stops when max torque < tol_Am (A/m) OR t_sim > t_max.
static void relax(VectorField3D& m, const Material& mat,
                   EffectiveFieldSum& heff,
                   double tol_Am = 1e3,    // stop when |m×H| < 1 kA/m
                   double t_max  = 5e-9,   // 5 ns hard limit
                   const char* vtk_out = nullptr)
{
    RK45Integrator integ;
    double t = 0.0;
    int steps = 0;

    while (t < t_max) {
        double dt = integ.step(m, mat, heff);
        t += dt;
        ++steps;

        if (steps % 200 == 0) {
            double tau = max_torque(m, mat, heff);
            if (tau < tol_Am) break;
        }
    }

    if (vtk_out) write_vtk_legacy(vtk_out, m, "m");
}

// ---------------------------------------------------------------------------
static void run_case(const char* tag,
                      const StructuredGrid& grid,
                      const Material& mat,
                      bool use_vortex)
{
    VectorField3D m(grid);
    if (use_vortex) {
        // Counter-clockwise in-plane vortex centred in element
        m.set_vortex(grid.nx() * grid.dx() * 0.5,
                     grid.ny() * grid.dy() * 0.5,
                     5e-9);
    } else {
        m.set_uniform({1.0, 0.05, 0.0});   // slight tilt off +x to break symmetry
    }
    m.normalize();

    EffectiveFieldSum heff;
    heff.add(std::make_shared<ExchangeField>());
    heff.add(std::make_shared<DemagField>(grid));

    std::string vtk_name = std::string("sp1_") + tag + ".vtu";

    auto t0 = Clock::now();
    relax(m, mat, heff, 1e3, 5e-9, vtk_name.c_str());
    double wall = elapsed_s(t0);

    Vec3 avg = mean_m(m);
    double mag2 = avg.x*avg.x + avg.y*avg.y + avg.z*avg.z;
    double mag  = std::sqrt(mag2);
    double E    = heff.total_energy(m, mat);

    // State classification: |<m>|>0.85 → single-domain, <0.15 → vortex
    const char* state = (mag > 0.85) ? "Single-domain (S/flower)" :
                        (mag < 0.15) ? "Vortex" : "Multi-domain";

    std::cout << "  [" << tag << "]\n"
              << "    <mx,my,mz> = ("
              << std::setprecision(4) << std::fixed
              << avg.x << ", " << avg.y << ", " << avg.z << ")\n"
              << "    |<m>|      = " << mag << "\n"
              << "    E_total    = " << std::scientific << E << " J\n"
              << "    state      → " << state << "\n"
              << "    wall time  = " << std::fixed << wall << " s\n\n";
}

// ---------------------------------------------------------------------------
int main() {
    // Permalloy (NIST standard) with overdamped α for fast convergence
    Material mat = Material::permalloy();
    mat.alpha = 0.5;

    // Exchange length: l_ex = sqrt(2A / (µ₀Ms²))
    const double lex = std::sqrt(2.0 * mat.A_exchange /
                                  (constants::mu_0 * mat.Ms * mat.Ms));

    std::cout << "=== µMAG Standard Problem #1 ===\n"
              << "Permalloy: Ms=" << mat.Ms/1e3 << " kA/m  "
              << "A=" << mat.A_exchange*1e12 << " pJ/m  "
              << "α=" << mat.alpha << "\n"
              << "Exchange length l_ex = " << lex*1e9 << " nm\n\n";

    // -----------------------------------------------------------------------
    // Case 1 : Small square — 200×200×10 nm, 5nm cells → 40×40×2 = 3200 cells
    //   200 nm ≈ " << 200e-9/lex << " l_ex  →  S-state likely
    // -----------------------------------------------------------------------
    {
        const int nx=40, ny=40, nz=2;
        const double d=5e-9;
        StructuredGrid g(nx, ny, nz, d, d, d);

        std::cout << "--- Case 1: " << nx*d*1e9 << "×" << ny*d*1e9
                  << "×" << nz*d*1e9 << " nm  (" << nx*ny*nz
                  << " cells, " << (200e-9/lex) << " l_ex) ---\n";

        run_case("small_uniform", g, mat, false);
        run_case("small_vortex",  g, mat, true);
    }

    // -----------------------------------------------------------------------
    // Case 2 : Large square — 500×500×10 nm, 5nm cells → 100×100×2 = 20000 cells
    //   500 nm ≈ " << 500e-9/lex << " l_ex  →  Vortex likely
    // -----------------------------------------------------------------------
    {
        const int nx=100, ny=100, nz=2;
        const double d=5e-9;
        StructuredGrid g(nx, ny, nz, d, d, d);

        std::cout << "--- Case 2: " << nx*d*1e9 << "×" << ny*d*1e9
                  << "×" << nz*d*1e9 << " nm  (" << nx*ny*nz
                  << " cells, " << (500e-9/lex) << " l_ex) ---\n";

        run_case("large_uniform", g, mat, false);
        run_case("large_vortex",  g, mat, true);
    }

    std::cout << "VTK files: sp1_small_uniform.vtu, sp1_small_vortex.vtu,\n"
              << "           sp1_large_uniform.vtu, sp1_large_vortex.vtu\n";
    return 0;
}
