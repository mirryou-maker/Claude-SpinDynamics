#pragma once

#include <cmath>
#include <cstddef>

namespace micromag {

// Floating-point precision used throughout.
// Change to float for single-precision builds.
using Real = double;

// Signed integer for indices and counts (supports negative offsets in stencils).
using Index = std::ptrdiff_t;

// 3D vector with basic arithmetic. POD-like, trivially copyable.
struct Vec3 {
    Real x{0}, y{0}, z{0};

    constexpr Vec3() = default;
    constexpr Vec3(Real xv, Real yv, Real zv) : x(xv), y(yv), z(zv) {}

    constexpr Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    constexpr Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    constexpr Vec3 operator*(Real s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(Real s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3& operator*=(Real s) { x *= s; y *= s; z *= s; return *this; }
    Vec3& operator/=(Real s) { x /= s; y /= s; z /= s; return *this; }

    constexpr Real dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    constexpr Vec3 cross(const Vec3& v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }
    constexpr Real norm_squared() const { return dot(*this); }
    Real norm() const { return std::sqrt(norm_squared()); }
};

constexpr Vec3 operator*(Real s, const Vec3& v) { return v * s; }

namespace constants {
inline constexpr Real pi       = 3.14159265358979323846;
inline constexpr Real mu_0     = 4.0 * pi * 1e-7;        // [T·m/A]
inline constexpr Real hbar     = 1.054571817e-34;         // ħ [J·s]
inline constexpr Real e_charge = 1.602176634e-19;         // e [C]
inline constexpr Real gamma_0  = 1.760859630e11;          // |γ| [rad/(T·s)]
inline constexpr Real k_B      = 1.380649e-23;             // Boltzmann [J/K]
}  // namespace constants

}  // namespace micromag
