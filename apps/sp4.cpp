// µMAG Standard Problem #4  —  Permalloy 500 × 125 × 3 nm
//
// Reference: https://www.ctcms.nist.gov/~rdm/mumag.org.html
//   Apply μ₀H = 25 mT at 170° from +x in the xy-plane to a Permalloy film
//   initially saturated along +x.  Integrate LLG (α = 0.02) and record the
//   time evolution of the average magnetisation until the sample switches to
//   the –x direction.
//
//   Key reference values (µMAG, cell size 2.5 × 2.5 × 3 nm):
//     Switching time  t_sw ≈ 0.10 – 0.15 ns  (when <mx> crosses zero)
//     Final state     <mx> ≈ –0.9862,  <my> ≈ 0.0,  <mz> ≈ 0.0
//
// Build:  cmake --build build/windows-msvc --config Release
// Run:    .\build\windows-msvc\bin\Release\sp4.exe
// Output: sp4_result.csv  (t_ns, mx, my, mz)

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/exchange.hpp"
#include "micromag/demag.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;

int main() {
    // -----------------------------------------------------------------------
    // Geometry: 200 × 50 × 1 cells, 2.5 × 2.5 × 3 nm each
    // → physical size 500 × 125 × 3 nm
    // -----------------------------------------------------------------------
    constexpr Index Nx = 200, Ny = 50, Nz = 1;
    constexpr Real  dx = 2.5e-9, dy = 2.5e-9, dz = 3.0e-9;
    StructuredGrid grid(Nx, Ny, Nz, dx, dy, dz);

    // -----------------------------------------------------------------------
    // Material: Permalloy  (µMAG SP#4 values)
    //   Ms = 800 kA/m,  A = 13 pJ/m,  K = 0,  α = 0.02
    // -----------------------------------------------------------------------
    Material mat    = Material::permalloy();
    // permalloy() already sets: Ms=800kA/m, A_exchange=13pJ/m, K=0, alpha=0.02

    // -----------------------------------------------------------------------
    // Initial state: uniform +x with a small +y tilt (breaks symmetry)
    // -----------------------------------------------------------------------
    VectorField3D m(grid);
    constexpr Real tilt = 0.5 * constants::pi / 180.0;   // 0.5°
    for (Index i = 0; i < m.size(); ++i)
        m[i] = { std::cos(tilt), std::sin(tilt), 0.0 };

    // -----------------------------------------------------------------------
    // Applied field: μ₀H = 25 mT at 170° from +x  (Field A, µMAG SP#4)
    //   Hx = –(25 mT / μ₀) × cos(10°)  ≈ –19 591 A/m
    //   Hy = +(25 mT / μ₀) × sin(10°)  ≈  +3 453 A/m
    // -----------------------------------------------------------------------
    constexpr Real B_mT    = 25e-3;
    constexpr Real theta   = 170.0 * constants::pi / 180.0;
    const Vec3 H_app = {
        B_mT * std::cos(theta) / constants::mu_0,
        B_mT * std::sin(theta) / constants::mu_0,
        0.0
    };

    // -----------------------------------------------------------------------
    // Effective field: Zeeman + Exchange + Demag
    // -----------------------------------------------------------------------
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(H_app));
    heff.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));
    heff.add(std::make_shared<DemagField>(grid));

    // -----------------------------------------------------------------------
    // RK4 integrator
    // -----------------------------------------------------------------------
    constexpr Real dt       = 5e-14;     // 0.05 ps  (stable for Py with α=0.02)
    constexpr Index N_steps = 20000;     // 1 ns total
    constexpr Index out_every = 500;     // print every 25 ps

    RK4Integrator rk4(dt);

    // -----------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------
    std::ofstream csv("sp4_result.csv");
    csv << "t_ns,mx,my,mz\n";

    std::cout << "=== µMAG Standard Problem #4 ===\n";
    std::cout << "Grid : " << Nx << " × " << Ny << " × " << Nz
              << " (" << grid.size() << " cells)\n";
    std::cout << "Cell : " << dx*1e9 << " × " << dy*1e9 << " × "
              << dz*1e9 << " nm\n";
    std::cout << "Mat  : Ms=" << mat.Ms/1e3 << " kA/m"
              << "  A=" << mat.A_exchange*1e12 << " pJ/m"
              << "  α=" << mat.alpha << "\n";
    std::cout << "Field: μ₀H = " << B_mT*1e3 << " mT @ 170°  ("
              << "Hx=" << H_app.x/1e3 << " kA/m, "
              << "Hy=" << H_app.y/1e3 << " kA/m)\n";
    std::cout << "dt   : " << dt*1e12 << " ps  |  "
              << N_steps << " steps = " << N_steps*dt*1e9 << " ns\n\n";

    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::setw(8)  << "t [ns]"
              << std::setw(11) << "<mx>"
              << std::setw(11) << "<my>"
              << std::setw(11) << "<mz>" << "\n";
    std::cout << std::string(41, '-') << "\n";

    bool switched = false;

    for (Index step = 0; step <= N_steps; ++step) {
        // ---- Output every out_every steps ----
        if (step % out_every == 0) {
            Real smx = 0, smy = 0, smz = 0;
            for (Index i = 0; i < m.size(); ++i) {
                smx += m[i].x;  smy += m[i].y;  smz += m[i].z;
            }
            const Real N = static_cast<Real>(m.size());
            smx /= N;  smy /= N;  smz /= N;

            const Real t_ns = step * dt * 1e9;
            std::cout << std::setw(8) << t_ns
                      << std::setw(11) << smx
                      << std::setw(11) << smy
                      << std::setw(11) << smz << "\n";
            csv << t_ns << "," << smx << "," << smy << "," << smz << "\n";

            if (!switched && smx < 0.0) {
                switched = true;
                std::cout << "  >>> Switching detected at t = "
                          << t_ns << " ns  (<mx> crossed zero)\n";
            }
        }

        if (step < N_steps) rk4.step(m, mat, heff);
    }

    // ---- Final averages ----
    Real smx = 0, smy = 0, smz = 0;
    for (Index i = 0; i < m.size(); ++i) {
        smx += m[i].x;  smy += m[i].y;  smz += m[i].z;
    }
    const Real N = static_cast<Real>(m.size());
    smx /= N;  smy /= N;  smz /= N;

    std::cout << "\n=== Final state ===\n";
    std::cout << "  <mx> = " << smx << "  (µMAG ref ≈ –0.9862)\n";
    std::cout << "  <my> = " << smy << "  (µMAG ref ≈  0.0)\n";
    std::cout << "  <mz> = " << smz << "  (µMAG ref ≈  0.0)\n";
    if (switched) {
        const Real err_mx = std::abs(smx - (-0.9862));
        std::cout << "\n  |<mx> – (–0.9862)| = " << err_mx;
        if (err_mx < 0.02)
            std::cout << "  ✓  within 2% of µMAG reference\n";
        else
            std::cout << "  (run longer or refine grid for better agreement)\n";
    } else {
        std::cout << "\n  Switching NOT observed — try longer integration time.\n";
    }

    std::cout << "\nCSV written to sp4_result.csv\n";
    return 0;
}
