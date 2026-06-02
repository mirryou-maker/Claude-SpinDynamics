// µMAG Standard Problem #4  —  T = 300 K (stochastic LLG)
//
// Runs the SP#4 Field A scenario with Langevin thermal noise at T=300K.
// Uses HeunIntegrator (fixed Δt) — RK45 cannot be used with thermal noise
// because σ ∝ 1/√Δt: changing Δt changes the physics.
//
// Comparison:
//   T=0   (deterministic): HeunIntegrator, no ThermalField
//   T=300K (stochastic):   HeunIntegrator + ThermalField, N_real realizations
//
// σ at T=300K, dt=0.1ps, V=(2.5nm×2.5nm×3nm):
//   σ ≈ 705 A/m  (0.09% of Ms=800kA/m) — tiny perturbation on deterministic dynamics.
//
// Run:  .\build\windows-msvc\bin\Release\sp4_thermal.exe

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/exchange.hpp"
#include "micromag/demag.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;

// ---------------------------------------------------------------------------
struct RunResult {
    double t_switch_ns = -1.0;  // ns; -1 = no switch observed
    double mx_final = 0;
    double my_final = 0;
    int    n_steps  = 0;
};

static Vec3 avg_m(const VectorField3D& m) {
    Real sx=0,sy=0,sz=0;
    for (Index i=0;i<m.size();++i){sx+=m[i].x;sy+=m[i].y;sz+=m[i].z;}
    const Real N=static_cast<Real>(m.size());
    return {sx/N,sy/N,sz/N};
}

static RunResult run_sp4(Real T_K, unsigned seed,
                          const StructuredGrid& grid,
                          const Material& mat,
                          const Vec3& H_app,
                          Real dt, int N_steps)
{
    // Build effective fields
    EffectiveFieldSum heff;
    heff.add(std::make_shared<ZeemanField>(H_app));
    heff.add(std::make_shared<ExchangeField>(BoundaryCondition::Neumann));
    heff.add(std::make_shared<DemagField>(grid));

    // Initial state
    constexpr Real tilt = 0.5 * constants::pi / 180.0;
    VectorField3D m(grid);
    for (Index i=0;i<m.size();++i) m[i] = {std::cos(tilt), std::sin(tilt), 0.0};

    HeunIntegrator heun(dt);

    RunResult res;
    bool switched = false;

    if (T_K > 0.0) {
        ThermalField thermal(grid, T_K, dt, seed);
        for (int step = 0; step < N_steps; ++step) {
            heun.step(m, mat, heff, &thermal);
            ++res.n_steps;
            if (!switched) {
                Vec3 a = avg_m(m);
                if (a.x < 0.0) {
                    switched = true;
                    res.t_switch_ns = step * dt * 1e9;
                }
            }
        }
    } else {
        for (int step = 0; step < N_steps; ++step) {
            heun.step(m, mat, heff);   // T=0: no thermal
            ++res.n_steps;
            if (!switched) {
                Vec3 a = avg_m(m);
                if (a.x < 0.0) {
                    switched = true;
                    res.t_switch_ns = step * dt * 1e9;
                }
            }
        }
    }

    Vec3 a = avg_m(m);
    res.mx_final = a.x;
    res.my_final = a.y;
    return res;
}

// ---------------------------------------------------------------------------
int main() {
    // SP#4 geometry
    constexpr Real dx=2.5e-9, dy=2.5e-9, dz=3.0e-9;
    const StructuredGrid grid(200, 50, 1, dx, dy, dz);

    Material mat = Material::permalloy();   // Ms=800kA/m, A=13pJ/m, α=0.02

    constexpr Real B_mT = 25e-3;
    constexpr Real theta = 170.0 * constants::pi / 180.0;
    const Vec3 H_app = {
        B_mT*std::cos(theta)/constants::mu_0,
        B_mT*std::sin(theta)/constants::mu_0,
        0.0
    };

    constexpr Real dt      = 1e-13;    // 0.1 ps
    constexpr int  N_steps = 10000;    // 1 ns
    constexpr int  N_real  = 3;        // realizations (balance speed/stats)

    // Report thermal noise magnitude
    const Real V   = dx*dy*dz;
    const Real sig = ThermalField::sigma(300.0, dt, mat, grid);
    std::cout << "=== SP#4  T=300K  (stochastic LLG) ===\n";
    std::cout << "Grid : 200×50×1, " << dx*1e9 << "×" << dy*1e9 << "×" << dz*1e9 << " nm\n";
    std::cout << "dt   : " << dt*1e12 << " ps,  N_steps=" << N_steps
              << " (" << N_steps*dt*1e9 << " ns)\n";
    std::cout << "σ(300K) = " << sig << " A/m"
              << "  (σ/Ms = " << sig/mat.Ms*100 << "% — small thermal perturbation)\n\n";

    // --- T=0 reference ---
    std::cout << "--- T=0 reference (deterministic Heun, seed irrelevant) ---\n";
    auto t0 = std::chrono::steady_clock::now();
    RunResult ref = run_sp4(0.0, 0, grid, mat, H_app, dt, N_steps);
    double wall0 = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::cout << "  t_switch = ";
    if (ref.t_switch_ns >= 0) std::cout << ref.t_switch_ns << " ns";
    else                       std::cout << "(no switch)";
    std::cout << "   <mx>_final=" << std::fixed << std::setprecision(5) << ref.mx_final
              << "  (" << wall0 << "s)\n\n";

    // --- T=300K realizations ---
    std::cout << "--- T=300K (" << N_real << " realizations) ---\n";
    std::cout << std::setw(8) << "seed"
              << std::setw(14) << "t_switch[ns]"
              << std::setw(12) << "<mx>_final"
              << std::setw(12) << "<my>_final" << "\n";
    std::cout << std::string(46,'-') << "\n";

    double sum_tsw=0, sum_mx=0; int n_sw=0;
    std::vector<double> tsw_list;

    for (int r = 0; r < N_real; ++r) {
        auto tr0 = std::chrono::steady_clock::now();
        RunResult res = run_sp4(300.0, static_cast<unsigned>(r*1013+7),
                                grid, mat, H_app, dt, N_steps);
        double wall = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-tr0).count();

        std::cout << std::setw(8) << r
                  << std::setw(14);
        if (res.t_switch_ns >= 0) {
            std::cout << res.t_switch_ns;
            sum_tsw += res.t_switch_ns; ++n_sw;
            tsw_list.push_back(res.t_switch_ns);
        } else {
            std::cout << "(no switch)";
        }
        std::cout << std::setprecision(5)
                  << std::setw(12) << res.mx_final
                  << std::setw(12) << res.my_final
                  << "  (" << std::setprecision(1) << wall << "s)\n";
        sum_mx += res.mx_final;
    }

    // Summary
    std::cout << "\n=== Summary ===\n";
    std::cout << "T=0 reference : t_switch=" << ref.t_switch_ns << " ns"
              << "  <mx>=" << ref.mx_final << "\n";
    if (n_sw > 0) {
        double mean_tsw = sum_tsw/n_sw;
        double mean_mx  = sum_mx/N_real;

        double var=0;
        for (double t : tsw_list) var += (t-mean_tsw)*(t-mean_tsw);
        double std_tsw = (n_sw>1) ? std::sqrt(var/(n_sw-1)) : 0.0;

        std::cout << "T=300K (" << N_real << " real.) : "
                  << "mean t_sw=" << std::setprecision(4) << mean_tsw
                  << " ± " << std_tsw << " ns"
                  << "  mean <mx>=" << mean_mx << "\n";
        std::cout << "\nΔt_sw = " << mean_tsw - ref.t_switch_ns << " ns  ("
                  << (mean_tsw/ref.t_switch_ns-1)*100 << "% change from T=0)\n";
    }
    std::cout << "\nNote: σ/Ms ≈ 0.09% → deterministic dynamics dominates.\n";
    std::cout << "Thermal effects are a small stochastic perturbation on the\n";
    std::cout << "switching trajectory, causing ~ns-scale jitter in t_switch.\n";
    return 0;
}
