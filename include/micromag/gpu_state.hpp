#pragma once

// gpu_state.hpp — G3: GPU-resident magnetization state for full-GPU LLG pipeline.
//
// GPUMagState owns five [3×N] double buffers on GPU:
//   d_m_     — current magnetization (uploaded once; updated each step)
//   d_H_     — accumulated H_eff (zeroed at start of each RK4 stage)
//   d_m0_    — RK4 stage start (saved before each step)
//   d_ki_    — current stage dm/dt (written by LLG-torque kernel, G4)
//   d_k_acc_ — weighted sum ∑ w_i k_i  (RK4 accumulator, G5)
//
// Memory layout: [3×N] component-major, buf[c*N + idx]
//   idx = ix + nx*(iy + ny*iz)  (x-fastest), c ∈ {0=Mx, 1=My, 2=Mz}
// Identical to ExchangeFieldGPU / DemagFieldGPU::d_M_compact_.
//
// All GPU operations run on the internal CUDA stream (stream_).
// G6 (RK4IntegratorGPU) will assign the same stream to field kernels so
// that the entire LLG step executes with no inter-stream synchronisation.
//
// Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include "field.hpp"
#include "grid.hpp"
#include "types.hpp"

namespace micromag {

class GPUMagState {
public:
    explicit GPUMagState(const StructuredGrid& grid);
    ~GPUMagState();

    // Non-copyable (owns GPU resources)
    GPUMagState(const GPUMagState&)            = delete;
    GPUMagState& operator=(const GPUMagState&) = delete;

    // ------------------------------------------------------------------
    // CPU ↔ GPU transfers (synchronous — stream is sync'd before return)
    // ------------------------------------------------------------------

    // Pack VectorField3D → d_m_ (initial upload or periodic re-sync)
    void upload(const VectorField3D& m);

    // d_m_ → VectorField3D (periodic monitoring; every N steps in G6)
    void download(VectorField3D& m) const;

    // d_H_ → VectorField3D (for monitoring / testing)
    void download_H(VectorField3D& H) const;

    // Snapshot buffers → VectorField3D (used in tests and debugging)
    void download_m0(VectorField3D& m)    const;
    void download_ki(VectorField3D& k)    const;
    void download_k_acc(VectorField3D& k) const;

    // Upload H into d_H_ (same stream — used by G4/G5 tests to avoid multi-stream sync)
    void upload_H(const VectorField3D& H);

    // ------------------------------------------------------------------
    // Async GPU bookkeeping (on stream_; no CPU sync — caller owns sync)
    // ------------------------------------------------------------------

    void zero_H();       // d_H_     = 0  (before accumulating fields)
    void zero_k_acc();   // d_k_acc_ = 0  (before first RK4 stage)
    void save_m0();      // d_m0_    = d_m_  (device-to-device, start of step)
    void sync() const;   // cudaStreamSynchronize — explicit CPU barrier

    // ------------------------------------------------------------------
    // Raw GPU pointer accessors (passed to CUDA kernels in G4/G5/G6)
    // ------------------------------------------------------------------
    double*       d_m()     { return reinterpret_cast<double*>(d_m_);     }
    const double* d_m()     const { return reinterpret_cast<const double*>(d_m_); }
    double*       d_H()     { return reinterpret_cast<double*>(d_H_);     }
    double*       d_m0()    { return reinterpret_cast<double*>(d_m0_);    }
    double*       d_ki()    { return reinterpret_cast<double*>(d_ki_);    }
    double*       d_k_acc() { return reinterpret_cast<double*>(d_k_acc_); }

    // CUDA stream (void* to avoid cufft/cuda_runtime headers here)
    void*  stream() const { return stream_; }
    size_t N()      const { return N_; }

private:
    size_t N_;   // nx * ny * nz

    void* d_m_     = nullptr;   // double[3×N]
    void* d_H_     = nullptr;   // double[3×N]
    void* d_m0_    = nullptr;   // double[3×N]
    void* d_ki_    = nullptr;   // double[3×N]
    void* d_k_acc_ = nullptr;   // double[3×N]

    // Pinned host staging buffer — avoids intermediate host copy on DMA
    double* h_staging_ = nullptr;   // double[3×N]

    void* stream_ = nullptr;
};

}  // namespace micromag

#endif // MICROMAG_CUDA
