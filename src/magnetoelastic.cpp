#include "micromag/magnetoelastic.hpp"
#include "micromag/types.hpp"

namespace micromag {

namespace {
    constexpr Real mu0 = constants::mu_0;
}

MagnetoelasticField::MagnetoelasticField(Real B1, Real B2)
    : B1_(B1), B2_(B2) {}

void MagnetoelasticField::set_strain(Real exx, Real eyy, Real ezz,
                                      Real exy, Real exz, Real eyz)
{
    exx_ = exx; eyy_ = eyy; ezz_ = ezz;
    exy_ = exy; exz_ = exz; eyz_ = eyz;
}

// ---------------------------------------------------------------------------
void MagnetoelasticField::accumulate(const VectorField3D& m,
                                      const Material& mat,
                                      VectorField3D& H_out) const
{
    const Real Ms = mat.Ms;
    if (Ms == Real{0}) return;
    // prefac = -2/(µ₀ Ms)
    const Real prefac = Real{-2} / (mu0 * Ms);
    const Real p1 = prefac * B1_;
    const Real p2 = prefac * B2_;

    const auto& g = m.grid();
    const Index N = g.nx() * g.ny() * g.nz();

    for (Index i = 0; i < N; ++i) {
        const Vec3& mi = m[i];
        const Real mx = mi.x, my = mi.y, mz = mi.z;

        H_out[i] += Vec3{
            p1 * mx * exx_ + p2 * (my * exy_ + mz * exz_),
            p1 * my * eyy_ + p2 * (mx * exy_ + mz * eyz_),
            p1 * mz * ezz_ + p2 * (mx * exz_ + my * eyz_)
        };
    }
}

// ---------------------------------------------------------------------------
Real MagnetoelasticField::energy(const VectorField3D& m,
                                  const Material& mat) const
{
    const auto& g  = m.grid();
    const Real  dV = g.dx() * g.dy() * g.dz();
    const Index N  = g.nx() * g.ny() * g.nz();
    Real E = Real{0};

    for (Index i = 0; i < N; ++i) {
        const Vec3& mi = m[i];
        const Real mx = mi.x, my = mi.y, mz = mi.z;
        // e_me = B1*(mx²*exx + my²*eyy + mz²*ezz)
        //      + 2*B2*(mx*my*exy + my*mz*eyz + mx*mz*exz)
        E += B1_ * (mx*mx*exx_ + my*my*eyy_ + mz*mz*ezz_)
           + Real{2} * B2_ * (mx*my*exy_ + my*mz*eyz_ + mx*mz*exz_);
    }
    return E * dV;
}

}  // namespace micromag
