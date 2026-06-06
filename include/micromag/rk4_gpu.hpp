#pragma once

// rk4_gpu.hpp — G4 (LLG torque) + G5 (RK4 stage kernels)
//
// G4: llg_torque kernel — dm/dt = -γ'μ₀[(m×H) + α m×(m×H)]
//   γ' = γ₀/(1+α²),  m ∈ S²,  H in A/m,  dm/dt in 1/s
//
// G5: RK4 stage helper kernels — assembled into full steps by G6.
//   rk4_stage:      m_stage = m0 + scale × ki     (m_stage for next eval)
//   rk4_accumulate: k_acc  += weight × ki          (weighted running sum)
//   rk4_finalize:   m_new   = m0 + dt × k_acc      (final Butcher sum)
//   normalize:      m[idx] /= |m[idx]|              (unit-sphere clamp)
//
// All buffers [3×N] component-major: buf[c*N + idx]
// All kernels launched on the stream passed by the caller.
// Kernels are SETTING (not adding) for torque; ADDING for accumulate.
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "types.hpp"   // constants::gamma_0, constants::mu_0

namespace micromag {

// ---------------------------------------------------------------------------
// G4: LLG torque — SETS d_ki from d_m and d_H
//
// dm/dt = -γ'μ₀ [(m×H) + α m×(m×H)],    γ' = γ₀μ₀/(1+α²)
// ---------------------------------------------------------------------------
void launch_llg_torque(double*       d_ki,   // [3×N] OUTPUT (set, not add)
                        const double* d_m,    // [3×N]
                        const double* d_H,    // [3×N]
                        double        alpha,
                        int           N,
                        void*         stream);

// ---------------------------------------------------------------------------
// G5: RK4 stage kernels
// ---------------------------------------------------------------------------

// m_out[i] = m0[i] + scale × ki[i]   (all 3N elements)
void launch_rk4_stage(double*       m_out,
                       const double* m0,
                       const double* ki,
                       double        scale,
                       int           N,
                       void*         stream);

// k_acc[i] += weight × ki[i]   (all 3N elements)
void launch_rk4_accumulate(double*       k_acc,
                             const double* ki,
                             double        weight,
                             int           N,
                             void*         stream);

// m_new[i] = m0[i] + dt × k_acc[i]   (all 3N elements)
void launch_rk4_finalize(double*       m_new,
                          const double* m0,
                          const double* k_acc,
                          double        dt,
                          int           N,
                          void*         stream);

// |m[idx]| = 1  (per cell; N = number of cells, not 3N)
void launch_normalize(double* m,
                       int     N,
                       void*   stream);

// ---------------------------------------------------------------------------
// DOPRI5 stage kernels (used by RK45IntegratorGPU)
// All m/k buffers are [3×N] component-major.
// ---------------------------------------------------------------------------

// Stage 3:  m_s = m0 + h*(3/40 k1 + 9/40 k2)
void launch_dopri5_stage3(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, int N, void* stream);

// Stage 4:  m_s = m0 + h*(44/45 k1 - 56/15 k2 + 32/9 k3)
void launch_dopri5_stage4(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, const double* k3,
                           int N, void* stream);

// Stage 5:  m_s = m0 + h*(19372/6561 k1 - 25360/2187 k2 + 64448/6561 k3 - 212/729 k4)
void launch_dopri5_stage5(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, const double* k3,
                           const double* k4, int N, void* stream);

// Stage 6:  m_s = m0 + h*(9017/3168 k1 - 355/33 k2 + 46732/5247 k3 + 49/176 k4 - 5103/18656 k5)
void launch_dopri5_stage6(double* m_s, const double* m0, double h,
                           const double* k1, const double* k2, const double* k3,
                           const double* k4, const double* k5, int N, void* stream);

// 5th-order solution:  m5 = m0 + h*(35/384 k1 + 500/1113 k3 + 125/192 k4 - 2187/6784 k5 + 11/84 k6)
void launch_dopri5_m5(double* m5, const double* m0, double h,
                       const double* k1, const double* k3, const double* k4,
                       const double* k5, const double* k6, int N, void* stream);

// Error estimate:  err = h*(71/57600 k1 - 71/16695 k3 + ... - 1/40 k7)
void launch_dopri5_err(double* err, double h,
                        const double* k1, const double* k3, const double* k4,
                        const double* k5, const double* k6, const double* k7,
                        int N, void* stream);

// RMS error norm.  d_sum must be a device-allocated double.
// Returns sqrt(sum((e/sc)^2) / 3N) as host scalar.
double launch_dopri5_err_norm(double* d_sum,
                               const double* d_err, const double* d_m,
                               const double* d_m5,
                               double rtol, double atol, int N, void* stream);

// ---------------------------------------------------------------------------
// Heun-specific helpers (also used in G8 HeunIntegratorGPU)
// ---------------------------------------------------------------------------

// dst[i] += src[i]  — flat 3N,  used to add thermal noise to d_H
void launch_add_3N(double*       dst,
                    const double* src,
                    int           N,
                    void*         stream);

// m[i] += dt_half * (k1[i] + k2[i])  — flat 3N  (Heun corrector)
void launch_heun_corrector(double*       m,
                             const double* k1,
                             const double* k2,
                             double        dt_half,
                             int           N,
                             void*         stream);

}  // namespace micromag

#endif // MICROMAG_CUDA
