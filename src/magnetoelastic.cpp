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
    const Real prefac = Real{-2} / (mu0 * Ms);
    const Real p1 = prefac * B1_;
    const Real p2 = prefac * B2_;

    const auto& g = m.grid();
    const Index N = g.nx() * g.ny() * g.nz();

    #pragma omp parallel for schedule(static) if(N > 4096)
    for (Index i = 0; i < N; ++i) {
        const Vec3& mi = m[i];
        const Real mx = mi.x, my = mi.y, mz = mi.z;

        // Read per-cell strain if field is set; fall back to scalar
        const Real exx_v = f_exx_ ? (*f_exx_)[i] : exx_;
        const Real eyy_v = f_eyy_ ? (*f_eyy_)[i] : eyy_;
        const Real ezz_v = f_ezz_ ? (*f_ezz_)[i] : ezz_;
        const Real exy_v = f_exy_ ? (*f_exy_)[i] : exy_;
        const Real exz_v = f_exz_ ? (*f_exz_)[i] : exz_;
        const Real eyz_v = f_eyz_ ? (*f_eyz_)[i] : eyz_;

        H_out[i] += Vec3{
            p1 * mx * exx_v + p2 * (my * exy_v + mz * exz_v),
            p1 * my * eyy_v + p2 * (mx * exy_v + mz * eyz_v),
            p1 * mz * ezz_v + p2 * (mx * exz_v + my * eyz_v)
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

        const Real exx_v = f_exx_ ? (*f_exx_)[i] : exx_;
        const Real eyy_v = f_eyy_ ? (*f_eyy_)[i] : eyy_;
        const Real ezz_v = f_ezz_ ? (*f_ezz_)[i] : ezz_;
        const Real exy_v = f_exy_ ? (*f_exy_)[i] : exy_;
        const Real exz_v = f_exz_ ? (*f_exz_)[i] : exz_;
        const Real eyz_v = f_eyz_ ? (*f_eyz_)[i] : eyz_;

        E += B1_ * (mx*mx*exx_v + my*my*eyy_v + mz*mz*ezz_v)
           + Real{2} * B2_ * (mx*my*exy_v + my*mz*eyz_v + mx*mz*exz_v);
    }
    return E * dV;
}

}  // namespace micromag
