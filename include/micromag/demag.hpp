#pragma once

#include <complex>
#include <vector>

#include "effective_field.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "material.hpp"
#include "types.hpp"
#include "fftw_plan_handle.hpp"

namespace micromag {

// Forward declaration — include material_field.hpp for the full definition.
class MaterialField3D;

// FFT-based demagnetization field using the Newell (1993) analytical tensor.
//
// H_demag = -N * M  (SI, H in A/m)
// where N is the 3x3 symmetric demagnetization tensor computed via zero-padded
// FFT convolution.  Kernel is precomputed once at construction; runtime cost
// is 6 forward FFTs + 6 pointwise products + 3 inverse FFTs per step.
//
// When a MaterialField3D is attached (set_material_field), the per-cell Ms
// is used to build M = Ms_i * m_i before the FFT (mumax3 "Regions" style
// spatially-varying saturation magnetisation).
class DemagField : public IEffectiveField {
public:
    explicit DemagField(const StructuredGrid& grid);
    ~DemagField();

    // Non-copyable (owns FFTW plans).
    DemagField(const DemagField&)            = delete;
    DemagField& operator=(const DemagField&) = delete;

    void accumulate(const VectorField3D& m,
                    const Material& mat,
                    VectorField3D& H_out) const override;

    Real energy(const VectorField3D& m,
                const Material& mat) const override;

    // Per-cell: e_i = -μ₀/2 Ms_i (m_i·H_demag_i) [J/m³]
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override;

    const char* name() const override { return "DemagField"; }

    // Attach per-cell Ms. nullptr disables it (default), falling back to the
    // uniform `mat.Ms` passed to accumulate()/energy().
    // Caller must keep the field alive for the lifetime of this object.
    void set_material_field(const MaterialField3D* matf) { matf_ = matf; }
    void clear_material_field() { matf_ = nullptr; }
    const MaterialField3D* material_field() const { return matf_; }

private:
    const MaterialField3D* matf_{nullptr};

    // Grid geometry (stored for convenience).
    Index nx_, ny_, nz_;   // real-space dimensions
    Real  dx_, dy_, dz_;
    Index pad_nx_, pad_ny_, pad_nz_;   // zero-padded FFT dimensions
    Index fft_nx_;                      // pad_nx_/2+1 (r2c output)

    // Precomputed demag kernel in frequency space.
    // 6 independent components: xx, yy, zz, xy, xz, yz.
    // Size: pad_nz_ * pad_ny_ * fft_nx_
    std::vector<std::complex<double>> K_xx_, K_yy_, K_zz_;
    std::vector<std::complex<double>> K_xy_, K_xz_, K_yz_;

    // Scratch arrays for forward/inverse transforms (mutable for const accumulate).
    mutable std::vector<double>              r_buf_;    // real input  (padded, single component)
    mutable std::vector<std::complex<double>> c_buf_;   // complex FFT (single component — precompute only)

    // Batch buffers for runtime accumulate(): all 3 components packed contiguously.
    // r_buf_3_: layout [Mx_padded | My_padded | Mz_padded] (or [Hx|Hy|Hz] on IFFT output)
    // c_buf_3_: layout [Mx_f | My_f | Mz_f]               (frequency domain)
    mutable std::vector<double>              r_buf_3_;
    mutable std::vector<std::complex<double>> c_buf_3_;

    // FFTW plans (forward r2c and inverse c2r).
    // plan_fwd_ / plan_inv_: single-component, used by precompute_kernel().
    // plan_fwd_3_ / plan_inv_3_: 3-component batch, used by accumulate() (faster).
    FFTWPlan plan_fwd_;
    FFTWPlan plan_inv_;
    FFTWPlan plan_fwd_3_;   // fftw_plan_many: 3 r2c in one call
    FFTWPlan plan_inv_3_;   // fftw_plan_many: 3 c2r in one call

    // Helpers -----------------------------------------------------------------
    void precompute_kernel();

public:
    // Newell (1993) analytical formulas — public so GPU backend can reuse them.
    static double newell_f(double x, double y, double z);
    static double newell_g(double x, double y, double z);
    static double nxx(double x, double y, double z, double dx, double dy, double dz);
    static double nxy(double x, double y, double z, double dx, double dy, double dz);

private:
};

}  // namespace micromag
