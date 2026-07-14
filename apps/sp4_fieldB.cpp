// uMAG Standard Problem #4  --  Field B  (190deg)
// Same geometry/material as sp4.cpp, field angle changed to 190deg.
// In Field B the Hy component is negative (Hy = 25mT x sin(190deg) ~ -3.45 kA/m)
// vs Field A where Hy > 0.  Both cases show switching but via different paths.
//
// uMAG reference (Field B, 2.5x2.5x3 nm mesh):
//   Switching time  t_sw ~ 0.10-0.15 ns
//   Final state     <mx> ~ -0.9839

#include <cmath>
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
    constexpr Index Nx = 200, Ny = 50, Nz = 1;
    constexpr Real  dx = 2.5e-9, dy = 2.5e-9, dz = 3.0e-9;
    StructuredGrid grid(Nx, Ny, Nz, dx, dy, dz);

    Material mat = Material::permalloy();

    // ---- Field B: 190deg ----
    constexpr Real B_mT    = 25e-3;
    constexpr Real theta   = 190.0 * constants::pi / 180.0;   // Field B
    const Vec3 H_app = {
        B_mT * std::cos(theta) / constants::mu_0,
        B_mT * std::sin(theta) / constants::mu_0,
        0.0
    };

    // Initial state
    constexpr Real tilt = 0.5 * constants::pi / 180.0;
    VectorField3D m(grid);
    for (Index i = 0; i < m.size(); ++i)
        m[i] = { std::cos(tilt), std::sin(tilt), 0.0 };

    // Effective field
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(H_app));
    heff.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));
    heff.add(std::make_shared<DemagField>(grid));

    // RK45 adaptive integrator
    RK45Integrator::Options opts;
    opts.rtol    = 1e-4;
    opts.atol    = 1e-6;
    opts.dt_init = 5e-14;
    opts.dt_max  = 5e-12;
    RK45Integrator rk45(opts);

    std::ofstream csv("sp4_fieldB_result.csv");
    csv << "t_ns,mx,my,mz\n";

    constexpr Real t_end    = 1e-9;
    constexpr Real out_dt   = 25e-12;   // print every 25 ps

    std::cout << "=== uMAG SP#4  Field B (190deg) ===\n";
    std::cout << "Hx=" << H_app.x/1e3 << " kA/m  Hy=" << H_app.y/1e3 << " kA/m\n\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::setw(8)  << "t [ns]"
              << std::setw(11) << "<mx>"
              << std::setw(11) << "<my>"
              << std::setw(11) << "<mz>" << "\n";
    std::cout << std::string(41, '-') << "\n";

    Real t = 0.0, next_out = 0.0;
    bool switched = false;

    while (t < t_end) {
        // Output at regular intervals
        if (t >= next_out - 1e-15) {
            Real smx=0, smy=0, smz=0;
            for (Index i=0; i<m.size(); ++i) { smx+=m[i].x; smy+=m[i].y; smz+=m[i].z; }
            const Real N = static_cast<Real>(m.size());
            smx/=N; smy/=N; smz/=N;
            const Real t_ns = t * 1e9;
            std::cout << std::setw(8) << t_ns
                      << std::setw(11) << smx
                      << std::setw(11) << smy
                      << std::setw(11) << smz << "\n";
            csv << t_ns << "," << smx << "," << smy << "," << smz << "\n";
            if (!switched && smx < 0.0) {
                switched = true;
                std::cout << "  >>> Switch at t = " << t_ns << " ns\n";
            }
            next_out += out_dt;
        }
        t += rk45.step(m, mat, heff);
    }

    // Final state
    Real smx=0, smy=0, smz=0;
    for (Index i=0; i<m.size(); ++i) { smx+=m[i].x; smy+=m[i].y; smz+=m[i].z; }
    const Real N=static_cast<Real>(m.size());
    smx/=N; smy/=N; smz/=N;

    std::cout << "\n=== Final state ===\n";
    std::cout << "  <mx> = " << smx << "  (uMAG ref ~ -0.9839)\n";
    std::cout << "  <my> = " << smy << "\n";
    std::cout << "  <mz> = " << smz << "\n";
    std::cout << "  steps: accepted=" << rk45.n_accepted()
              << "  rejected=" << rk45.n_rejected() << "\n";
    std::cout << "CSV -> sp4_fieldB_result.csv\n";
    return 0;
}
