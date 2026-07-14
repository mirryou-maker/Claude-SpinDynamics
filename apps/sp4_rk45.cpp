// uMAG Standard Problem #4  --  RK45 adaptive integrator
// Side-by-side performance comparison with fixed-step RK4.
//
// Run:  .\build\windows-msvc\bin\Release\sp4_rk45.exe

#include <chrono>
#include <cmath>
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
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
static EffectiveFieldSum make_heff(const StructuredGrid& grid, Vec3 H_app) {
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(H_app));
    heff.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));
    heff.add(std::make_shared<DemagField>(grid));
    return heff;
}

static Vec3 avg_m(const VectorField3D& m) {
    Real sx=0,sy=0,sz=0;
    for (Index i=0;i<m.size();++i){sx+=m[i].x;sy+=m[i].y;sz+=m[i].z;}
    const Real n=static_cast<Real>(m.size());
    return {sx/n,sy/n,sz/n};
}

// ---------------------------------------------------------------------------
int main() {
    // SP#4 geometry
    const StructuredGrid grid(200, 50, 1, 2.5e-9, 2.5e-9, 3.0e-9);
    const Material mat = Material::permalloy();

    constexpr Real B_mT  = 25e-3;
    constexpr Real theta = 170.0 * constants::pi / 180.0;
    const Vec3 H_app = {
        B_mT * std::cos(theta) / constants::mu_0,
        B_mT * std::sin(theta) / constants::mu_0,
        0.0
    };

    constexpr Real t_end     = 1e-9;     // 1 ns
    constexpr Real dt_rk4    = 5e-14;    // 0.05 ps  (fixed)
    constexpr Real tilt      = 0.5 * constants::pi / 180.0;

    auto make_m = [&]() {
        VectorField3D m(grid);
        for (Index i=0;i<m.size();++i)
            m[i] = {std::cos(tilt), std::sin(tilt), 0.0};
        return m;
    };

    // -----------------------------------------------------------------------
    // Run 1: fixed-step RK4
    // -----------------------------------------------------------------------
    std::cout << "=== RK4 (fixed dt=" << dt_rk4*1e12 << " ps) ===\n";
    {
        auto m    = make_m();
        auto heff = make_heff(grid, H_app);
        RK4Integrator rk4(dt_rk4);

        auto t0 = Clock::now();
        Real t = 0.0;
        Index steps = 0;
        while (t < t_end) { rk4.step(m, mat, heff); t += dt_rk4; ++steps; }
        const double wall = std::chrono::duration<double>(Clock::now()-t0).count();

        Vec3 a = avg_m(m);
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  steps      : " << steps << "\n";
        std::cout << "  wall time  : " << wall << " s\n";
        std::cout << "  final <mx> : " << a.x << "  (ref -0.9862)\n";
        std::cout << "  final <my> : " << a.y << "\n";
    }

    // -----------------------------------------------------------------------
    // Run 2: adaptive RK45
    // -----------------------------------------------------------------------
    std::cout << "\n=== RK45 (adaptive, rtol=1e-4, atol=1e-6) ===\n";
    {
        auto m    = make_m();
        auto heff = make_heff(grid, H_app);

        RK45Integrator::Options opts;
        opts.rtol    = 1e-4;
        opts.atol    = 1e-6;
        opts.dt_init = dt_rk4;
        opts.dt_max  = 5e-12;   // 5 ps max
        RK45Integrator rk45(opts);

        auto t0 = Clock::now();
        Real t = 0.0;
        while (t < t_end) { t += rk45.step(m, mat, heff); }
        const double wall = std::chrono::duration<double>(Clock::now()-t0).count();

        Vec3 a = avg_m(m);
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  accepted   : " << rk45.n_accepted() << "\n";
        std::cout << "  rejected   : " << rk45.n_rejected() << "\n";
        std::cout << "  total      : " << rk45.n_accepted()+rk45.n_rejected() << "\n";
        std::cout << "  wall time  : " << wall << " s\n";
        std::cout << "  final <mx> : " << a.x << "  (ref -0.9862)\n";
        std::cout << "  final <my> : " << a.y << "\n";
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\n";
    return 0;
}
