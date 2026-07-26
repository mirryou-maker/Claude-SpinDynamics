// batched_llg_gpu.hpp — Task 2 Phase 2.1: multi-cell replica-batched LLG.
//
// Generalises BatchedMacrospinGPU (N=1) to an N-cell grid with the LOCAL
// effective fields — exchange (6-point Laplacian, Neumann BC), uniaxial
// anisotropy, Zeeman — plus Slonczewski STT and FDT thermal noise, integrated
// with Depondt–Mertens rotation (|m|=1 exact). This establishes the Task 2
// replica layout and the batched local-field kernels; the batched demag (cuFFT
// batch=R) is Phase 2.2. See CLAUDE-SD_FINITE_TEMP_ROADMAP.md Task 2, Phase 2.1.
//
// Layout: replica-OUTERMOST, component-major within a replica:
//     m[r*3N + c*N + idx],  idx = ix + nx*(iy + ny*iz),  c in {0,1,2}
// (r-outer matches the roadmap; within-replica c*N+idx matches GPUMagState so a
//  single replica reproduces the DepondtMertensGPU path bitwise.)
//
// Per replica: independent J and T (the sweep axes). Material/geometry shared.
//
// Requires MICROMAG_CUDA=1.
#pragma once

#include <vector>

#include "micromag/grid.hpp"
#include "micromag/types.hpp"

namespace micromag {

class BatchedDemagGPU;   // Phase 2.2 (defined under MICROMAG_CUDA)

struct BatchedLLGConfig {
    Real Ms    = 580e3;    // saturation magnetisation [A/m]
    Real alpha = 0.02;     // Gilbert damping
    Real A     = 1.3e-11;  // exchange stiffness [J/m]

    Real K1    = 0.0;               // uniaxial anisotropy [J/m³]
    Vec3 easy  = {0, 0, 1};         // easy axis (normalised internally)

    Vec3 H_ext = {0, 0, 0};         // Zeeman field [A/m]

    // Slonczewski STT (J per-replica; these shared). d_free is the FM thickness.
    Real d_free = 1e-9;
    Real P      = 0.5;
    Real Lambda = 1.0;
    Real beta   = 0.0;
    Vec3 p      = {0, 0, 1};
};

class BatchedLLGGPU {
public:
    // R replicas of `grid`, fixed step dt, RNG seed. Initial m = +easy.
    BatchedLLGGPU(int R, const StructuredGrid& grid,
                  const BatchedLLGConfig& cfg, Real dt, unsigned seed = 12345u);
    ~BatchedLLGGPU();

    BatchedLLGGPU(const BatchedLLGGPU&) = delete;
    BatchedLLGGPU& operator=(const BatchedLLGGPU&) = delete;

    void set_J(const std::vector<double>& J);   // length R  [A/m²]
    void set_T(const std::vector<double>& T);   // length R  [K]

    // Phase 2.2: enable replica-batched demag (cuFFT batch=R, shared Newell
    // kernel). Off by default. Call before run(); the demag object lives for
    // the integrator's lifetime.
    void enable_demag();
    bool demag_enabled() const { return demag_ != nullptr; }

    // Overwrite state: flat [R*3N], m[r*3N + c*N + idx]; normalised per cell.
    void set_state(const std::vector<double>& m);
    // Set every cell of every replica to the same direction.
    void set_uniform(double mx, double my, double mz);

    // Advance ALL R replicas by n_steps (multi-kernel Depondt step, zero PCIe).
    void run(int n_steps);

    // Download state: flat [R*3N], m[r*3N + c*N + idx].
    std::vector<double> get_state() const;
    // Per-replica spatially-averaged magnetisation, flat [R*3] (mx,my,mz).
    std::vector<double> get_avg_m() const;

    int  R()          const { return R_; }
    int  N()          const { return N_; }
    long step_index() const { return step_index_; }

private:
    void substep_(unsigned long long noise_offset);

    int      R_, nx_, ny_, nz_, N_;
    double   dx_, dy_, dz_;
    Real     dt_;
    unsigned seed_;
    long     step_index_ = 0;
    BatchedLLGConfig cfg_;

    void* d_m_  = nullptr;   // GReal [R*3N]
    void* d_m0_ = nullptr;   // GReal [R*3N] saved mⁿ
    void* d_H_  = nullptr;   // GReal [R*3N]
    void* d_w1_ = nullptr;   // GReal [R*3N] ω1/ω̄
    void* d_w2_ = nullptr;   // GReal [R*3N] ω2
    void* d_J_  = nullptr;   // double [R]
    void* d_T_  = nullptr;   // double [R]

    BatchedDemagGPU* demag_ = nullptr;   // Phase 2.2 (owned; null = disabled)
};

}  // namespace micromag
