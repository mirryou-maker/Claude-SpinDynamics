#pragma once

#include "types.hpp"

namespace micromag {

struct Material {
    Real Ms{8e5};             // Saturation magnetization [A/m]
    Real A_exchange{1.3e-11}; // Exchange stiffness [J/m]
    Real K_uniaxial{0};       // First uniaxial anisotropy K1 [J/m³]
    Vec3 easy_axis{0, 0, 1};  // Easy-axis direction (normalized at use)
    Real alpha{0.02};         // Gilbert damping
    Real Ku2{0};              // Second uniaxial K2: ΔE = -Ku2*(m·û)^4 [J/m³]

    static Material permalloy();
    static Material cobalt();
    static Material iron();
};

}  // namespace micromag
