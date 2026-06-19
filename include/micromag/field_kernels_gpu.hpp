#pragma once

// field_kernels_gpu.hpp — G2 + Phase E: GPU Zeeman, Uniaxial-Anisotropy,
// and Cubic-Anisotropy fields.
//
// All classes implement IEffectiveField (drop-in for CPU equivalents) and
// expose an accumulate_gpu_ptr() path for the G6 full-GPU LLG pipeline.
//
// Memory layout (GPU buffers): [3×N] component-major
//   buf[c*N + idx],  idx = ix + nx*(iy + ny*iz),  c ∈ {0,1,2}
// Same convention as ExchangeFieldGPU and DemagFieldGPU::d_M_compact_.

#ifdef MICROMAG_CUDA

#include "anisotropy.hpp"
#include "cubic_anisotropy.hpp"
#include "effective_field.hpp"
#include "effective_field_gpu_iface.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "material_field.hpp"
#include "types.hpp"
#include "zeeman.hpp"

namespace micromag {

// ---------------------------------------------------------------------------
// ZeemanFieldGPU
// ---------------------------------------------------------------------------
class ZeemanFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    ZeemanFieldGPU(const StructuredGrid& grid, const Vec3& H_ext = {0, 0, 0});
    ~ZeemanFieldGPU();

    ZeemanFieldGPU(const ZeemanFieldGPU&)            = delete;
    ZeemanFieldGPU& operator=(const ZeemanFieldGPU&) = delete;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;
    const char* name() const override { return "ZeemanFieldGPU"; }

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;

    Vec3 H_ext() const            { return H_ext_; }
    void set_H_ext(const Vec3& H) { H_ext_ = H; }
    void set_stream(void* s)      { stream_ = s; }

private:
    size_t N_;
    Vec3   H_ext_;
    void*  stream_ = nullptr;
};

// ---------------------------------------------------------------------------
// UniaxialAnisotropyFieldGPU
// ---------------------------------------------------------------------------
class UniaxialAnisotropyFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    explicit UniaxialAnisotropyFieldGPU(const StructuredGrid& grid);
    ~UniaxialAnisotropyFieldGPU();

    UniaxialAnisotropyFieldGPU(const UniaxialAnisotropyFieldGPU&)            = delete;
    UniaxialAnisotropyFieldGPU& operator=(const UniaxialAnisotropyFieldGPU&) = delete;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;
    const char* name() const override { return "UniaxialAnisotropyFieldGPU"; }

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;
    void set_stream(void* s) { stream_ = s; }

    // Per-cell material: uploads K_uniaxial, easy_axis, Ms from MaterialField3D.
    // Once set, per-cell mode is active; call clear_material_field() to revert.
    void set_material_field(const MaterialField3D& matf);
    void clear_material_field();
    bool has_material_field() const { return d_K_field_ != nullptr; }

private:
    size_t N_;
    void*   d_m_scratch_ = nullptr;
    void*   d_H_scratch_ = nullptr;
    void*   stream_ = nullptr;

    // Per-cell buffers (null = uniform mode)
    double* d_K_field_    = nullptr;  // double[N]   — K_uniaxial per cell
    double* d_axis_field_ = nullptr;  // double[3*N] — easy_axis per cell (component-major)
    double* d_Ms_field_   = nullptr;  // double[N]   — Ms per cell
};

// ---------------------------------------------------------------------------
// CubicAnisotropyFieldGPU
//
// GPU drop-in for CubicAnisotropyField.
// e = Kc1*(a1²a2² + a2²a3² + a3²a1²) + Kc2*(a1²a2²a3²), ai = m·ci
// c3 = c1×c2 computed at construction.
// ---------------------------------------------------------------------------
class CubicAnisotropyFieldGPU : public IEffectiveField, public IEffectiveFieldGPU {
public:
    CubicAnisotropyFieldGPU(const StructuredGrid& grid,
                             Real Kc1 = 0, Real Kc2 = 0,
                             Vec3 c1 = {1,0,0}, Vec3 c2 = {0,1,0});
    ~CubicAnisotropyFieldGPU();

    CubicAnisotropyFieldGPU(const CubicAnisotropyFieldGPU&)            = delete;
    CubicAnisotropyFieldGPU& operator=(const CubicAnisotropyFieldGPU&) = delete;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override;
    Real energy(const VectorField3D& m, const Material& mat) const override;
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;
    const char* name() const override { return "CubicAnisotropyFieldGPU"; }

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;
    void set_stream(void* s) { stream_ = s; }

    Real Kc1() const { return Kc1_; }
    Real Kc2() const { return Kc2_; }
    Vec3 c1()  const { return c1_; }
    Vec3 c2()  const { return c2_; }
    Vec3 c3()  const { return c3_; }
    void set_Kc1(Real k) { Kc1_ = k; }
    void set_Kc2(Real k) { Kc2_ = k; }
    void set_axes(Vec3 c1, Vec3 c2) {
        c1_ = c1; c2_ = c2;
        c3_ = c1_.cross(c2_);
    }

private:
    size_t N_;
    Real   Kc1_, Kc2_;
    Vec3   c1_, c2_, c3_;
    void*  d_m_scratch_ = nullptr;
    void*  d_H_scratch_ = nullptr;
    void*  stream_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
