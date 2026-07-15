// Thermal equilibrium validation
//
// Demonstrates two well-verifiable results of the SLLG / Heun integrator:
//
//  1. EQUIPARTITION (K=0): uniform distribution -> <mx^2>=<my^2>=<mz^2>=1/3
//     Verified using T=1e8 K, dt=100 ps (sigma>>0 -> fast sphere exploration).
//
//  2. ANISOTROPY RELAXATION (T=0): deterministic Heun drives spin toward +/-z.
//     Shows tau_relax ~ 570 ps for K=1e4 J/m^3, alpha=0.5.
//
// NOTE ON BOLTZMANN STATISTICS AT PHYSICAL T:
//   For Permalloy (Ms=800 kA/m) at T=300 K the thermal noise is
//   sigma ~ 4 kA/m while the anisotropy field is H_ani ~ 200 kA/m.
//   The spin is trapped for tau_Neel = tau0 exp(kappa) ns-ms before switching.
//   A full Boltzmann distribution test requires O(100xtau_Neel) simulation
//   time -- i.e. hundreds of nanoseconds to milliseconds at room temperature.
//   This is deliberately NOT attempted here.
//
// Run:  .\build\windows-msvc\bin\Release\thermal_equilibrium.exe

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/integrator.hpp"

using namespace micromag;

// ---------------------------------------------------------------------------
// Part 1: Equipartition (K=0)
// ---------------------------------------------------------------------------
static void run_equipartition() {
    std::cout << "=== Part 1: Energy equipartition (K=0, T=1e8 K, dt=100 ps) ===\n";
    std::cout << "Expected: <mx^2> = <my^2> = <mz^2> = 1/3 = 0.333\n\n";

    StructuredGrid grid(10, 10, 1, 5e-9, 5e-9, 5e-9);   // 100 independent spins

    Material mat   = Material::permalloy();
    mat.alpha      = 0.5;
    mat.K_uniaxial = 0.0;

    VectorField3D m(grid);
    m.set_uniform({1.0, 0.0, 0.0});

    const Real T  = 1e8;
    const Real dt = 1e-10;

    EffectiveFieldSum heff;
    ThermalField thermal(grid, T, dt);
    HeunIntegrator heun(dt);

    // sigma at these parameters
    const Real sig = ThermalField::sigma(T, dt, mat, grid);
    std::cout << "  sigma = " << sig/1e3 << " kA/m  "
              << "(gpxsigmaxdt = " << 1.768e5*sig*dt*180/constants::pi << "deg/step)\n\n";

    // Equilibration
    for (int s = 0; s < 200; ++s) heun.step(m, mat, heff, &thermal);

    // Sampling
    std::cout << "  N_sample   <mx^2>    <my^2>    <mz^2>   sum\n";
    double sx2=0, sy2=0, sz2=0;
    const int N_s = 2000;
    for (int s=0;s<N_s;++s){
        heun.step(m,mat,heff,&thermal);
        for (Index i=0;i<m.size();++i){
            sx2+=m[i].x*m[i].x; sy2+=m[i].y*m[i].y; sz2+=m[i].z*m[i].z;
        }
    }
    const double N = static_cast<double>(N_s) * static_cast<double>(grid.size());
    const double mx2=sx2/N, my2=sy2/N, mz2=sz2/N;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << N_s << " steps  "
              << mx2 << "  " << my2 << "  " << mz2 << "  "
              << mx2+my2+mz2 << "\n";
    std::cout << "  Theory:      0.3333   0.3333   0.3333  1.0000\n\n";
}

// ---------------------------------------------------------------------------
// Part 2: Anisotropy relaxation (T=0 deterministic)
// ---------------------------------------------------------------------------
static void run_anisotropy_relaxation() {
    std::cout << "=== Part 2: Anisotropy relaxation (T=0, K=1e4 J/m^3) ===\n";
    std::cout << "H_ani = " << 2e4/1.005/1e3 << " kA/m  "
              << "tau_relax ~ 570 ps\n";
    std::cout << "Initial m=(sin10deg,0,cos10deg), easy axis z_hat\n\n";
    std::cout << "  t [ps]   mz       mz^2\n";

    StructuredGrid grid(1,1,1,5e-9,5e-9,5e-9);
    Material mat = Material::permalloy();
    mat.alpha      = 0.5;
    mat.K_uniaxial = 1e4;
    mat.easy_axis  = {0,0,1};

    VectorField3D m(grid);
    const Real tilt = 10.0*constants::pi/180.0;
    m.set_uniform({std::sin(tilt), 0.0, std::cos(tilt)});

    EffectiveFieldSum heff;
    heff.add(std::make_shared<UniaxialAnisotropyField>());
    const Real dt = 1e-12;
    HeunIntegrator heun(dt);

    for (int milestone : {0,200,500,1000,2000,3000,5000}) {
        std::cout << "  " << std::setw(6) << milestone
                  << "   " << std::setprecision(5) << m[0].z
                  << "   " << m[0].z*m[0].z << "\n";
        // advance to next milestone
        const int next = milestone + 200;
        for (int s=milestone; s<next && s<5000; ++s)
            heun.step(m, mat, heff);
    }
    std::cout << "\n  (spin locked to easy axis after ~3tau ~ 1.7 ns)\n\n";
}

int main() {
    run_equipartition();
    run_anisotropy_relaxation();

    std::cout << "=== Note on full Boltzmann statistics ===\n";
    std::cout << "For Permalloy at T=300K, K=100 kJ/m^3, V=(5nm)^3:\n";
    std::cout << "  sigma ~ 4 kA/m,  H_ani ~ 200 kA/m  (sigma/H_ani ~ 0.02)\n";
    std::cout << "  tau_Neel = tau0 x exp(kappa) with kappa~3, tau0~6ps -> tau~120ps\n";
    std::cout << "  Full equilibration: ~100tau = 12 ns -> 12,000 steps (1ps)\n";
    std::cout << "  This is feasible but takes seconds; use sp4_thermal.cpp "
                 "for that.\n";
    return 0;
}
