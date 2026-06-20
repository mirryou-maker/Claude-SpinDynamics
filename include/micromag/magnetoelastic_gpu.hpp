#pragma once

// magnetoelastic_gpu.hpp — GPU magnetoelastic (magnetostrictive) field
//
// Drop-in GPU replacement for MagnetoelasticField (uniform strain only).
// CUDA kernel applies the B1/B2 formula to all N cells in parallel.
//
// Per-cell strain fields are not supported in this GPU version (use the
// CPU MagnetoelasticField with set_exx_field() for spatially varying strain).

#ifdef MICROMAG_CUDA

#include "effective_field.hpp"
#include "effective_field_gpu_iface.hpp"
#include "field.hpp"
#include "types.hpp"

namespace micromag {

class MagnetoelasticFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    explicit MagnetoelasticFieldGPU(Real B1, Real B2,
                                     const StructuredGrid& grid);
    ~MagnetoelasticFieldGPU();

    MagnetoelasticFieldGPU(const MagnetoelasticFieldGPU&)            = delete;
    MagnetoelasticFieldGPU& operator=(const MagnetoelasticFieldGPU&) = delete;

    // Uniform strain tensor setters (same as CPU MagnetoelasticField)
    void set_strain(Real exx, Real eyy = Real{0}, Real ezz = Real{0},
                    Real exy = Real{0}, Real exz = Real{0}, Real eyz = Real{0});

    void set_exx(Real v) { exx_ = v; }
    void set_eyy(Real v) { eyy_ = v; }
    void set_ezz(Real v) { ezz_ = v; }
    void set_exy(Real v) { exy_ = v; }
    void set_exz(Real v) { exz_ = v; }
    void set_eyz(Real v) { eyz_ = v; }

    Real exx() const { return exx_; }
    Real eyy() const { return eyy_; }
    Real ezz() const { return ezz_; }
    Real exy() const { return exy_; }
    Real exz() const { return exz_; }
    Real eyz() const { return eyz_; }

    Real B1() const      { return B1_; }
    Real B2() const      { return B2_; }
    void set_B1(Real b)  { B1_ = b; }
    void set_B2(Real b)  { B2_ = b; }

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "MagnetoelasticFieldGPU"; }

    void accumulate_gpu_ptr(const GReal* d_m, const Material& mat,
                             GReal* d_H_out) const override;

    void set_stream(void* s) { stream_ = s; }

private:
    Real   B1_, B2_;
    Real   exx_{0}, eyy_{0}, ezz_{0};
    Real   exy_{0}, exz_{0}, eyz_{0};
    size_t N_;

    void*  d_m_scratch_ = nullptr;
    void*  d_H_scratch_ = nullptr;
    void*  stream_       = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
