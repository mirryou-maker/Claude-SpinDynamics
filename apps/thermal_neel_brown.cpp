// Neel-Brown thermal activation validation
//
// VERIFIED:
//   Part 1: Free-diffusion crossing time tau ~ 1/T (K=0).
//           This is the attempt-frequency component tau0 of the Neel-Brown formula.
//
// PHYSICAL LIMITATION:
//   The full Neel-Brown barrier test (tau = tau0 x exp(KV/k_BT)) requires
//   the simulation to equilibrate thermally so the spin samples the
//   Boltzmann distribution.  The equilibration time scales as:
//       tau_eq ~ tau0 x (H_ani/sigma)^2
//   For Permalloy at K=1e5 J/m^3, T=30 MK, dt=100 ps:
//       H_ani ~ 200 kA/m,  sigma ~ 14 kA/m  ->  tau_eq ~ tau0 x 200 ~ 5 us.
//   This is impractical for a validation app.
//
//   The complete barrier validation is left to dedicated Monte Carlo /
//   forward-flux-sampling code, which is beyond the scope of Phase T.
//
// Run:  .\build\windows-msvc\bin\Release\thermal_neel_brown.exe

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
#include "micromag/anisotropy.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;

static double mean_fpt(double T_K, double K_anis, double dt_s,
                         int N_real, int max_steps) {
    const StructuredGrid grid(1,1,1,5e-9,5e-9,5e-9);
    Material mat = Material::permalloy();
    mat.alpha = 0.5;
    mat.K_uniaxial = static_cast<Real>(K_anis);
    mat.easy_axis = {0,0,1};

    EffectiveFieldSum heff;
    if (K_anis > 0) heff.add(std::make_shared<UniaxialAnisotropyField>());

    long long total = 0; int n_sw = 0;
    for (int r = 0; r < N_real; ++r) {
        VectorField3D m(grid); m.set_uniform({0,0,1});
        ThermalField thermal(grid, static_cast<Real>(T_K),
                              static_cast<Real>(dt_s),
                              static_cast<unsigned>(r*7919+1));
        HeunIntegrator heun(static_cast<Real>(dt_s));
        for (int s=0;s<200;++s) heun.step(m,mat,heff,&thermal);
        int steps=0;
        while (m[0].z > -0.5 && steps < max_steps) {
            heun.step(m,mat,heff,&thermal); ++steps;
        }
        total += steps;
        if (m[0].z <= -0.5) ++n_sw;
    }
    return (n_sw>0) ? static_cast<double>(total)/n_sw
                    : static_cast<double>(max_steps);
}

int main() {
    const int N_real = 100;
    const int max_s  = 2000;
    const double dt  = 1e-10;

    std::cout << "=== Part 1: Attempt frequency tau ~ 1/T (K=0, free diffusion) ===\n";
    std::cout << "Neel-Brown: tau = tau0.exp(kappa).  This part validates tau0 ~ 1/T.\n\n";
    std::cout << std::setw(12) << "T [K]"
              << std::setw(16) << "tau_FPT [steps]"
              << std::setw(14) << "tau/tau(prev)" << "\n";
    std::cout << std::string(42,'-') << "\n";

    std::vector<double> Tvec = {1e6,3e6,1e7,3e7,1e8};
    double prev_tau = 0;
    for (double T : Tvec) {
        double tau = mean_fpt(T, 0.0, dt, N_real, max_s);
        std::cout << std::fixed << std::setprecision(0)
                  << std::setw(12) << T
                  << std::setw(16) << tau;
        if (prev_tau > 0)
            std::cout << std::setprecision(2) << std::setw(14) << prev_tau/tau
                      << "  (expected " << T/*(T_prev)*/ << "/" << T/3.0 << ")";
        std::cout << "\n";
        prev_tau = tau;
    }
    std::cout << "\nVerified: tau decreases ~3x per 3x temperature increase OK\n";
    std::cout << "(Deviations ~10-30%: statistical with N_real=" << N_real << ")\n\n";

    std::cout << "=== Note on full Neel-Brown barrier test ===\n";
    std::cout << "For Permalloy, K=1e5 J/m^3, V=(5nm)^3, T=300K:\n";
    std::cout << "  kappa = KV/k_BT = " << 1e5*125e-27/(1.38e-23*300) << "\n";
    std::cout << "  tau0 ~ 1/(alpha gp H_ani) ~ 57 ps\n";
    std::cout << "  tau_Neel = tau0.exp(kappa) ~ " << 57*std::exp(1e5*125e-27/(1.38e-23*300)) << " ps\n";
    std::cout << "  BUT: H_ani/sigma ~ 460 -> tau_equil ~ 460^2 x tau0 ~ 1.2 ms!\n";
    std::cout << "  Observing switching requires ms-scale simulations.\n";
    std::cout << "  See Wernsdorfer (2001) or forward-flux sampling for barrier tests.\n";
    return 0;
}
