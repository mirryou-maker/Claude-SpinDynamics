#include <iomanip>
#include <iostream>
#include <memory>

#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/spin_torque.hpp"
#include "micromag/integrator.hpp"
#include "micromag/vtk_writer.hpp"

// ---------------------------------------------------------------
// Macrospin helper: print one-line status
// ---------------------------------------------------------------
static void print_row(int step, double dt,
                      const micromag::VectorField3D& m,
                      const micromag::EffectiveFieldSum& heff,
                      const micromag::Material& mat) {
    micromag::Vec3 mv = m.at(0, 0, 0);
    std::cout << std::left
              << std::setw(8)  << step
              << std::setw(12) << step * dt * 1e12
              << std::setw(10) << std::setprecision(4) << mv.x
              << std::setw(10) << mv.y
              << std::setw(10) << mv.z
              << "\n";
}

int main() {
    using namespace micromag;

    // Single cell — macrospin approximation
    StructuredGrid g(1, 1, 1, 2e-9, 2e-9, 2e-9);
    Material mat = Material::cobalt();
    mat.alpha      = 0.02;
    mat.K_uniaxial = 4.5e5;   // out-of-plane easy axis along z

    EffectiveFieldSum heff;
    heff.add(std::make_shared<UniaxialAnisotropyField>());

    const Real dt = 5e-14;
    RK4Integrator rk4(dt);

    // ------------------------------------------------------------------
    // Demo 1: STT switching
    // m starts near +z (easy axis). Large positive J with p = -z:
    //   a_J = γ₀ħJP / (2eMsd) > 0  → antidamping w.r.t. p = -z
    //   After enough steps, m crosses equator and lands on -z.
    // ------------------------------------------------------------------
    {
        VectorField3D m(g);
        Real eps = 0.02;
        m.set_uniform({eps, 0, std::sqrt(1 - eps*eps)});

        // p = +z: same as initial m.
        // a_J > 0 → antidamping w.r.t. +z → m escapes to -z (P→AP switching).
        SpinTorqueSum stt_sum;
        stt_sum.add(std::make_shared<SlonczewskiSTT>(
            4e12,   // J [A/m²]
            0.60,   // P
            2e-9,   // d [m]
            Vec3{0, 0, 1},   // p = +z  (P→AP: m switches to -z)
            0.0));  // no field-like term

        std::cout << "=== Demo 1: STT P->AP switching (p = +z, J = 4e12 A/m^2) ===\n";
        std::cout << std::left
                  << std::setw(8) << "step" << std::setw(12) << "t [ps]"
                  << std::setw(10) << "m_x" << std::setw(10) << "m_y"
                  << std::setw(10) << "m_z" << "\n";

        for (int step = 0; step <= 6000; ++step) {
            if (step % 500 == 0) print_row(step, dt, m, heff, mat);
            if (step < 6000) rk4.step(m, mat, heff, &stt_sum);
        }
        write_vtk_legacy("stt_final.vtk", m, "m");
        std::cout << "\n";
    }

    // ------------------------------------------------------------------
    // Demo 2: SOT-driven dynamics
    // m starts at +z. x-current in Pt with θ_SH = 0.12 → σ = +y.
    // DL-SOT pushes m away from +z; with a moderate out-of-plane field
    // the system finds a new equilibrium (or steady-state precession).
    // ------------------------------------------------------------------
    {
        VectorField3D m(g);
        m.set_uniform({0, 0, 1});

        EffectiveFieldSum heff2;
        heff2.add(std::make_shared<UniaxialAnisotropyField>());
        heff2.add(std::make_shared<ZeemanField>(Vec3{0, 0, 2e5}));

        SpinTorqueSum sot_sum;
        sot_sum.add(std::make_shared<SpinOrbitTorque>(
            1e12,   // J_c [A/m²]
            0.12,   // θ_SH (Pt)
            2e-9,   // d_FM [m]
            Vec3{0, 1, 0},   // σ = ŷ
            1.0,    // η_DL
            0.1));  // η_FL

        std::cout << "=== Demo 2: SOT dynamics (J_c = 1e12, θ_SH = 0.12, σ = ŷ) ===\n";
        std::cout << std::left
                  << std::setw(8) << "step" << std::setw(12) << "t [ps]"
                  << std::setw(10) << "m_x" << std::setw(10) << "m_y"
                  << std::setw(10) << "m_z" << "\n";

        for (int step = 0; step <= 4000; ++step) {
            if (step % 400 == 0) print_row(step, dt, m, heff2, mat);
            if (step < 4000) rk4.step(m, mat, heff2, &sot_sum);
        }
        write_vtk_legacy("sot_final.vtk", m, "m");
        std::cout << "\n";
    }

    std::cout << "Wrote stt_final.vtk and sot_final.vtk\n";
    return 0;
}
