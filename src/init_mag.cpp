#include "micromag/init_mag.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

static constexpr double kPi   = std::numbers::pi;
static constexpr double kPi_2 = std::numbers::pi / 2.0;

namespace micromag {

VectorField3D uniform_mag(const StructuredGrid& g, Vec3 dir) {
    VectorField3D m(g);
    const Real n = dir.norm();
    const Vec3 v = (n > 1e-30) ? (dir / n) : Vec3{0, 0, 1};
    m.set_uniform(v);
    return m;
}

// Skyrmion θ profile: 2*atan(r/rho), goes from π at rho=0 to 0 at rho→∞
static inline Real sky_theta(Real rho, Real r) {
    if (rho < 1e-30) return kPi;
    return 2.0 * std::atan(r / rho);
}

VectorField3D neel_skyrmion(const StructuredGrid& g, Real r,
                             int charge, int pol,
                             Real cx, Real cy) {
    VectorField3D m(g);
    const Real box_cx = 0.5 * g.nx() * g.dx();
    const Real box_cy = 0.5 * g.ny() * g.dy();

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + 0.5) * g.dx() - box_cx - cx;
        const Real y = (static_cast<Real>(iy) + 0.5) * g.dy() - box_cy - cy;
        const Real rho = std::sqrt(x*x + y*y);
        const Real phi = std::atan2(y, x);
        const Real theta = sky_theta(rho, r);
        const Real sin_t = std::sin(theta);
        // Néel: in-plane component is radial (φ_in = charge*φ)
        const Real phi_in = static_cast<Real>(charge) * phi;
        // mz core = pol (θ=π at centre → cos=-1, so use -pol·cos θ);
        // pol=+1 → core up, pol=-1 → core down (mumax3 convention).
        m[g.linear_index(ix, iy, iz)] = {
            sin_t * std::cos(phi_in),
            sin_t * std::sin(phi_in),
            -static_cast<Real>(pol) * std::cos(theta)
        };
    }
    return m;
}

VectorField3D bloch_skyrmion(const StructuredGrid& g, Real r,
                              int charge, int pol,
                              Real cx, Real cy) {
    VectorField3D m(g);
    const Real box_cx = 0.5 * g.nx() * g.dx();
    const Real box_cy = 0.5 * g.ny() * g.dy();

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + 0.5) * g.dx() - box_cx - cx;
        const Real y = (static_cast<Real>(iy) + 0.5) * g.dy() - box_cy - cy;
        const Real rho = std::sqrt(x*x + y*y);
        const Real phi = std::atan2(y, x);
        const Real theta = sky_theta(rho, r);
        const Real sin_t = std::sin(theta);
        // Bloch: in-plane component is tangential (φ_in = charge*φ + π/2)
        const Real phi_in = static_cast<Real>(charge) * phi + kPi_2;
        // mz core = pol (mumax3 convention: pol=+1 → core up, pol=-1 → core down)
        m[g.linear_index(ix, iy, iz)] = {
            sin_t * std::cos(phi_in),
            sin_t * std::sin(phi_in),
            -static_cast<Real>(pol) * std::cos(theta)
        };
    }
    return m;
}

VectorField3D two_domain(const StructuredGrid& g, Vec3 m1, Vec3 m2,
                          char axis) {
    VectorField3D m(g);
    const Real n1 = m1.norm(), n2 = m2.norm();
    const Vec3 v1 = (n1 > 1e-30) ? (m1 / n1) : Vec3{0, 0, 1};
    const Vec3 v2 = (n2 > 1e-30) ? (m2 / n2) : Vec3{0, 0, 1};

    const Real cx = 0.5 * g.nx() * g.dx();
    const Real cy = 0.5 * g.ny() * g.dy();
    const Real cz = 0.5 * g.nz() * g.dz();

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        bool side1;
        if (axis == 'y') {
            Real y = (static_cast<Real>(iy) + 0.5) * g.dy() - cy;
            side1 = (y <= 0);
        } else if (axis == 'z') {
            Real z = (static_cast<Real>(iz) + 0.5) * g.dz() - cz;
            side1 = (z <= 0);
        } else {
            Real x = (static_cast<Real>(ix) + 0.5) * g.dx() - cx;
            side1 = (x <= 0);
        }
        m[g.linear_index(ix, iy, iz)] = side1 ? v1 : v2;
    }
    return m;
}

VectorField3D vortex_state(const StructuredGrid& g, int circ, int pol) {
    VectorField3D m(g);
    const Real box_cx = 0.5 * g.nx() * g.dx();
    const Real box_cy = 0.5 * g.ny() * g.dy();
    // Core radius: 2 cells
    const Real r_c = 2.0 * std::max(g.dx(), g.dy());

    for (Index iz = 0; iz < g.nz(); ++iz)
    for (Index iy = 0; iy < g.ny(); ++iy)
    for (Index ix = 0; ix < g.nx(); ++ix) {
        const Real x = (static_cast<Real>(ix) + 0.5) * g.dx() - box_cx;
        const Real y = (static_cast<Real>(iy) + 0.5) * g.dy() - box_cy;
        const Real rho = std::sqrt(x*x + y*y);
        const Real phi = std::atan2(y, x);

        // θ(ρ): 0 at core → π/2 far from core (in-plane closure)
        const Real theta = kPi_2 * (1.0 - std::exp(-(rho*rho) / (r_c*r_c)));
        const Real sin_t = std::sin(theta);
        // Tangential in-plane direction for vortex: circ*(φ + π/2)
        const Real phi_v = static_cast<Real>(circ) * phi + kPi_2;
        m[g.linear_index(ix, iy, iz)] = {
            sin_t * std::cos(phi_v),
            sin_t * std::sin(phi_v),
            static_cast<Real>(pol) * std::cos(theta)
        };
    }
    return m;
}

VectorField3D random_mag(const StructuredGrid& g, unsigned seed) {
    VectorField3D m(g);
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    for (Index i = 0; i < m.size(); ++i) {
        Vec3 v;
        Real n;
        do {
            v = { static_cast<Real>(nd(rng)),
                  static_cast<Real>(nd(rng)),
                  static_cast<Real>(nd(rng)) };
            n = v.norm();
        } while (n < 1e-12);
        m[i] = v / n;
    }
    return m;
}

}  // namespace micromag
