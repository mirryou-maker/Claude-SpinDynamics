#pragma once

// surface_anisotropy_gpu.hpp — GPU surface/interface anisotropy field.
// Drop-in GPU replacement for SurfaceAnisotropyField.
// Precomputes a surface-cell boolean mask at construction; CUDA kernel
// applies H_s only to surface cells.

#ifdef MICROMAG_CUDA

#include "effective_field.hpp"
#include "field.hpp"
#include "geom_mask.hpp"
#include "grid.hpp"
#include "types.hpp"

namespace micromag {

class SurfaceAnisotropyFieldGPU : public IEffectiveField {
public:
    explicit SurfaceAnisotropyFieldGPU(Real Ks, const StructuredGrid& grid,
                                        Vec3 n_hat = {0.0, 0.0, 1.0});
    ~SurfaceAnisotropyFieldGPU();

    SurfaceAnisotropyFieldGPU(const SurfaceAnisotropyFieldGPU&)            = delete;
    SurfaceAnisotropyFieldGPU& operator=(const SurfaceAnisotropyFieldGPU&) = delete;

    void set_mask(const GeomMask& mask) { geom_mask_ = &mask; rebuild_surface_mask(); }
    void clear_mask()                   { geom_mask_ = nullptr; rebuild_surface_mask(); }

    Real Ks()    const { return Ks_; }
    Vec3 n_hat() const { return n_; }
    void set_Ks(Real Ks) { Ks_ = Ks; }
    void set_n_hat(Vec3 n);

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    const char* name() const override { return "SurfaceAnisotropyFieldGPU"; }

    void accumulate_gpu_ptr(const double* d_m, const Material& mat,
                             double* d_H_out) const;

    void set_stream(void* s) { stream_ = s; }

private:
    Real   Ks_;
    Vec3   n_;
    Index  nx_, ny_, nz_;
    Real   dx_, dy_, dz_;
    Real   t_cell_;      // cell thickness along n_hat [m]
    size_t N_;

    const GeomMask* geom_mask_{nullptr};

    void*  d_is_surface_{nullptr};
    void*  d_m_scratch_  = nullptr;
    void*  d_H_scratch_  = nullptr;
    void*  stream_        = nullptr;

    void rebuild_surface_mask();
    bool is_surface_cell(Index ix, Index iy, Index iz) const;
    Real compute_t_cell() const;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
