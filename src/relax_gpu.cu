// relax_gpu.cu ??GPU energy minimisation (RelaxGPU + MinimizeGPU).
//
// All stepping and convergence checking stay on GPU.
// Only ONE scalar D2H per convergence check (max torque or energy).
//
// Memory layout: [3횞N] component-major, x-fastest.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
#include "micromag/cuda_sync_debug.hpp"
#include <cub/device/device_reduce.cuh>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>   // memcpy

#include "micromag/gpu_real.hpp"
#include "micromag/relax_gpu.hpp"
#include "micromag/types.hpp"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
            throw std::runtime_error(std::string("CUDA(relax): ")      \
                                   + cudaGetErrorString(_e));           \
    } while (0)

namespace micromag {

// ===========================================================================
// CUB DeviceReduce helpers — one robust, single-call GPU reduction over N
// elements (replaces the previous shared-memory block reductions, which also
// silently lost accuracy for N > 65536 in the two-pass energy sum). Temp
// storage is sized on first use and cached in (d_tmp, cap).
// ===========================================================================
static void cub_max(void*& d_tmp, size_t& cap,
                    const double* d_in, double* d_out, int N, cudaStream_t s) {
    size_t need = 0;
    CUDA_CHECK(cub::DeviceReduce::Max(nullptr, need, d_in, d_out, N, s));
    if (need > cap) {
        if (d_tmp) cudaFree(d_tmp);
        CUDA_CHECK(cudaMalloc(&d_tmp, need));
        cap = need;
    }
    size_t bytes = need;
    CUDA_CHECK(cub::DeviceReduce::Max(d_tmp, bytes, d_in, d_out, N, s));
}

static void cub_sum(void*& d_tmp, size_t& cap,
                    const double* d_in, double* d_out, int N, cudaStream_t s) {
    size_t need = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, need, d_in, d_out, N, s));
    if (need > cap) {
        if (d_tmp) cudaFree(d_tmp);
        CUDA_CHECK(cudaMalloc(&d_tmp, need));
        cap = need;
    }
    size_t bytes = need;
    CUDA_CHECK(cub::DeviceReduce::Sum(d_tmp, bytes, d_in, d_out, N, s));
}

// ===========================================================================
// Damping-only Euler step:
//   dm/dt = -款'慣關? m횞(m횞H)   (no precession)
//   m_new = m + dt * dm/dt,   then normalise
// ===========================================================================
__global__ static void damping_euler_kernel(
    GReal* __restrict__ m,        // [3N] updated in-place
    const GReal* __restrict__ H,  // [3N] effective field
    int N, double gp_alpha, double dt)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const double mx = static_cast<double>(m[0*N+idx]),
                 my = static_cast<double>(m[1*N+idx]),
                 mz = static_cast<double>(m[2*N+idx]);
    const double Hx = static_cast<double>(H[0*N+idx]),
                 Hy = static_cast<double>(H[1*N+idx]),
                 Hz = static_cast<double>(H[2*N+idx]);

    // mxH = m 횞 H
    const double mxHx = my*Hz - mz*Hy;
    const double mxHy = mz*Hx - mx*Hz;
    const double mxHz = mx*Hy - my*Hx;

    // m횞(m횞H)
    const double mxmxHx = my*mxHz - mz*mxHy;
    const double mxmxHy = mz*mxHx - mx*mxHz;
    const double mxmxHz = mx*mxHy - my*mxHx;

    // dm = -款'慣關? m횞(m횞H) 횞 dt
    double nx = mx - gp_alpha * mxmxHx * dt;
    double ny = my - gp_alpha * mxmxHy * dt;
    double nz = mz - gp_alpha * mxmxHz * dt;

    // Normalise
    const double inv_n = 1.0 / sqrt(nx*nx + ny*ny + nz*nz);
    m[0*N+idx] = static_cast<GReal>(nx * inv_n);
    m[1*N+idx] = static_cast<GReal>(ny * inv_n);
    m[2*N+idx] = static_cast<GReal>(nz * inv_n);
}

// ===========================================================================
// Per-cell squared torque  |m×H|²  → tsq_out[N]  (reduced by cub::DeviceReduce::Max)
// ===========================================================================
__global__ static void torque_sq_kernel(
    double* __restrict__       tsq_out,  // [N] stays double for precise reduction
    const GReal* __restrict__  m,
    const GReal* __restrict__  H,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double mx = static_cast<double>(m[0*N+idx]),
                 my = static_cast<double>(m[1*N+idx]),
                 mz = static_cast<double>(m[2*N+idx]);
    const double Hx = static_cast<double>(H[0*N+idx]),
                 Hy = static_cast<double>(H[1*N+idx]),
                 Hz = static_cast<double>(H[2*N+idx]);
    const double tx = my*Hz - mz*Hy;
    const double ty = mz*Hx - mx*Hz;
    const double tz = mx*Hy - my*Hx;
    tsq_out[idx] = tx*tx + ty*ty + tz*tz;
}

// ===========================================================================
// Per-cell energy: e[i] = -mu0/2 * Ms * m[i]쨌H[i] * dV  (exchange + demag)
// We store just m쨌H (energy density / (mu0*Ms*dV/2))
// For total energy, the caller sums and scales.
// Actually, let's store m쨌H (dimensionless dot product), and scale later.
// ===========================================================================
__global__ static void mdot_H_kernel(
    double* __restrict__       e_out,  // [N] m쨌H per cell ??stays double for reduction
    const GReal* __restrict__  m,
    const GReal* __restrict__  H,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    e_out[idx] = static_cast<double>(m[0*N+idx])*static_cast<double>(H[0*N+idx])
               + static_cast<double>(m[1*N+idx])*static_cast<double>(H[1*N+idx])
               + static_cast<double>(m[2*N+idx])*static_cast<double>(H[2*N+idx]);
}

// ===========================================================================
// copy [3N] GPU buffer ??another (for trial step in minimize)
// ===========================================================================
__global__ static void copy3N_kernel(GReal* dst, const GReal* src, int N3) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N3) dst[i] = src[i];
}

// ===========================================================================
// Shared allocation helper
// ===========================================================================
static void relax_alloc(size_t N, GReal*& d_m, GReal*& d_H,
                         double*& d_max, void*& stream,
                         GReal** d_trial = nullptr,
                         double** d_energy = nullptr) {
    CUDA_CHECK(cudaMalloc(&d_m,    3 * N * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_H,    3 * N * sizeof(GReal)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(double)));
    if (d_trial)  CUDA_CHECK(cudaMalloc(d_trial, 3 * N * sizeof(GReal)));
    if (d_energy) CUDA_CHECK(cudaMalloc(d_energy, N * sizeof(double)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream = static_cast<void*>(s);
}

static void relax_free(GReal* d_m, GReal* d_H, double* d_max, void* stream, GReal* d_trial = nullptr, double* d_energy = nullptr) {
    cudaFree(d_m); cudaFree(d_H); cudaFree(d_max);
    if (d_trial)  cudaFree(d_trial);
    if (d_energy) cudaFree(d_energy);
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

static void pack3N(const VectorField3D& m, size_t N, std::vector<GReal>& buf) {
    buf.resize(3 * N);
    for (Index i = 0; i < static_cast<Index>(N); ++i) {
        buf[i]       = static_cast<GReal>(m[i].x);
        buf[N   + i] = static_cast<GReal>(m[i].y);
        buf[2*N + i] = static_cast<GReal>(m[i].z);
    }
}

static void unpack3N(const std::vector<GReal>& buf, size_t N, VectorField3D& m) {
    for (Index i = 0; i < static_cast<Index>(N); ++i) {
        m[i].x = static_cast<double>(buf[i]);
        m[i].y = static_cast<double>(buf[N + i]);
        m[i].z = static_cast<double>(buf[2*N + i]);
    }
}

// ===========================================================================
// RelaxGPU
// ===========================================================================
RelaxGPU::RelaxGPU(const StructuredGrid& grid)
    : grid_(&grid), N_(static_cast<size_t>(grid.size())) {
    // d_percell_ ([N] torque² scratch) allocated via the d_energy slot.
    relax_alloc(N_, d_m_, d_H_, d_max_, stream_, nullptr, &d_percell_);
}

RelaxGPU::~RelaxGPU() {
    relax_free(d_m_, d_H_, d_max_, stream_, nullptr, d_percell_);
    if (d_cub_tmp_) cudaFree(d_cub_tmp_);
}

// Per-cell |m×H|² → CUB Max → sqrt. d_H_ must already hold H_eff for d_m_src.
double RelaxGPU::reduce_max_torque(const GReal* d_m_src) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;
    torque_sq_kernel<<<grd, blk, 0, s>>>(d_percell_, d_m_src, d_H_, N);
    MICROMAG_KERNEL_CHECK();
    cub_max(d_cub_tmp_, cub_bytes_, d_percell_, d_max_, N, s);
    double h_max;
    CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return std::sqrt(h_max);
}

void RelaxGPU::upload(const VectorField3D& m) {
    std::vector<GReal> buf;
    pack3N(m, N_, buf);
    CUDA_CHECK(cudaMemcpy(d_m_, buf.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
}

void RelaxGPU::download(VectorField3D& m) const {
    std::vector<GReal> buf(3 * N_);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_m_, 3*N_*sizeof(GReal), cudaMemcpyDeviceToHost));
    unpack3N(buf, N_, m);
}

void RelaxGPU::compute_H_eff(const Material& mat,
                               IDemagGPU& demag,
                               ExchangeFieldGPU& exch,
                               ZeemanFieldGPU& zeeman,
                               UniaxialAnisotropyFieldGPU* aniso) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    // Single-stream: redirect all fields to RelaxGPU's own stream so stream
    // ordering serialises the memset and every field kernel without barriers.
    demag.set_stream(stream_);
    exch.set_stream(stream_);
    zeeman.set_stream(stream_);
    if (aniso) aniso->set_stream(stream_);
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    demag.accumulate_gpu_ptr(d_m_, mat, d_H_);
    exch.accumulate_gpu_ptr(d_m_, mat, d_H_);
    zeeman.accumulate_gpu_ptr(d_m_, mat, d_H_);
    if (aniso)
        aniso->accumulate_gpu_ptr(d_m_, mat, d_H_);
}

double RelaxGPU::max_torque_now(const Material& mat,
                                 IDemagGPU& demag,
                                 ExchangeFieldGPU& exch,
                                 ZeemanFieldGPU& zeeman,
                                 UniaxialAnisotropyFieldGPU* aniso) {
    compute_H_eff(mat, demag, exch, zeeman, aniso);
    return reduce_max_torque(d_m_);
}

int RelaxGPU::run(const Material& mat,
                   IDemagGPU& demag,
                   ExchangeFieldGPU& exch,
                   ZeemanFieldGPU& zeeman,
                   UniaxialAnisotropyFieldGPU* aniso,
                   Options opts) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;

    const double gp_alpha = constants::gamma_0 * constants::mu_0 * opts.alpha_relax
                            / (1.0 + opts.alpha_relax * opts.alpha_relax);
    const double dt = opts.dt;

    for (int step = 0; step < opts.max_steps; ++step) {
        // Always recompute H_eff (required for correct physics every step).
        compute_H_eff(mat, demag, exch, zeeman, aniso);

        // Convergence check: D2H only one scalar every check_every steps.
        if (step % opts.check_every == 0) {
            if (reduce_max_torque(d_m_) < opts.threshold)
                return step;
        }

        // Damping-only Euler step
        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_,
            d_H_,
            N, gp_alpha, dt);
        MICROMAG_KERNEL_CHECK();
    }

    CUDA_CHECK(cudaStreamSynchronize(s));

    if (opts.throw_on_max)
        throw std::runtime_error(
            "RelaxGPU::run(): max_steps reached without convergence");
    return opts.max_steps;
}

// ===========================================================================
// MinimizeGPU
// ===========================================================================
MinimizeGPU::MinimizeGPU(const StructuredGrid& grid)
    : grid_(&grid), N_(static_cast<size_t>(grid.size())) {
    relax_alloc(N_, d_m_, d_H_, d_max_, stream_, &d_m_trial_, &d_energy_);
}

MinimizeGPU::~MinimizeGPU() {
    relax_free(d_m_, d_H_, d_max_, stream_, d_m_trial_, d_energy_);
    if (d_cub_tmp_) cudaFree(d_cub_tmp_);
}

// Per-cell |m×H|² → CUB Max → sqrt. d_H_ must already hold H_eff for d_m_src.
double MinimizeGPU::reduce_max_torque(const GReal* d_m_src) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;
    torque_sq_kernel<<<grd, blk, 0, s>>>(d_energy_, d_m_src, d_H_, N);
    MICROMAG_KERNEL_CHECK();
    cub_max(d_cub_tmp_, cub_bytes_, d_energy_, d_max_, N, s);
    double h_max;
    CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return std::sqrt(h_max);
}

// Per-cell m·H → CUB Sum (one scalar). d_H_ must already hold H_eff for d_m_src.
double MinimizeGPU::reduce_mdotH_sum(const GReal* d_m_src) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;
    mdot_H_kernel<<<grd, blk, 0, s>>>(d_energy_, d_m_src, d_H_, N);
    MICROMAG_KERNEL_CHECK();
    cub_sum(d_cub_tmp_, cub_bytes_, d_energy_, d_max_, N, s);
    double h_sum;
    CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return h_sum;
}

void MinimizeGPU::upload(const VectorField3D& m) {
    std::vector<GReal> buf;
    pack3N(m, N_, buf);
    CUDA_CHECK(cudaMemcpy(d_m_, buf.data(), 3*N_*sizeof(GReal), cudaMemcpyHostToDevice));
}

void MinimizeGPU::download(VectorField3D& m) const {
    std::vector<GReal> buf(3 * N_);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_m_, 3*N_*sizeof(GReal), cudaMemcpyDeviceToHost));
    unpack3N(buf, N_, m);
}

void MinimizeGPU::compute_H_eff_for(GReal* d_m_src,
                                     const Material& mat,
                                     IDemagGPU& demag,
                                     ExchangeFieldGPU& exch,
                                     ZeemanFieldGPU& zeeman,
                                     UniaxialAnisotropyFieldGPU* aniso) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    demag.set_stream(stream_);
    exch.set_stream(stream_);
    zeeman.set_stream(stream_);
    if (aniso) aniso->set_stream(stream_);
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    demag.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    exch.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    zeeman.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    if (aniso)
        aniso->accumulate_gpu_ptr(d_m_src, mat, d_H_);
}

double MinimizeGPU::compute_energy(const Material& mat,
                                    IDemagGPU& demag,
                                    ExchangeFieldGPU& exch,
                                    ZeemanFieldGPU& zeeman,
                                    UniaxialAnisotropyFieldGPU* aniso) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;

    // H_eff for current d_m_
    compute_H_eff_for(d_m_, mat, demag, exch, zeeman, aniso);

    // Per-cell m쨌H
    const double h_sum = reduce_mdotH_sum(d_m_);

    // E = -mu0/2 * Ms * sum(m.H) * dV   (standard LLG energy sign)
    const double dV = grid_->cell_volume();
    return -constants::mu_0 * 0.5 * mat.Ms * h_sum * dV;
}

int MinimizeGPU::run(const Material& mat,
                      IDemagGPU& demag,
                      ExchangeFieldGPU& exch,
                      ZeemanFieldGPU& zeeman,
                      UniaxialAnisotropyFieldGPU* aniso,
                      Options opts) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;

    const double alpha = 1.0;
    const double gp_alpha = constants::gamma_0 * constants::mu_0 * alpha / 2.0;
    double dt = opts.dt_init;

    for (int step = 0; step < opts.max_steps; ++step) {
        // H_eff for current m
        compute_H_eff_for(d_m_, mat, demag, exch, zeeman, aniso);

        // Convergence check every check_every steps
        if (step % opts.check_every == 0) {
            if (reduce_max_torque(d_m_) < opts.threshold)
                return step;
        }

        // Energy before trial step
        const double E0 = reduce_mdotH_sum(d_m_);

        // Trial step: copy m ??m_trial, advance m_trial
        copy3N_kernel<<<(3*N+blk-1)/blk, blk, 0, s>>>(
            d_m_trial_,
            d_m_,
            3*N);
        MICROMAG_KERNEL_CHECK();
        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_trial_,
            d_H_,
            N, gp_alpha, dt);
        MICROMAG_KERNEL_CHECK();

        // Energy after trial (need H_eff for m_trial)
        compute_H_eff_for(d_m_trial_, mat, demag, exch, zeeman, aniso);
        const double E1 = reduce_mdotH_sum(d_m_trial_);

        if (E1 < E0) {
            // Accept: swap m ??m_trial
            std::swap(d_m_, d_m_trial_);
            dt = std::min(opts.dt_max, dt * 1.2);
        } else {
            dt *= 0.5;
            if (dt < opts.dt_min) {
                std::swap(d_m_, d_m_trial_);
                dt = opts.dt_init;
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(s));

    if (opts.throw_on_max)
        throw std::runtime_error(
            "MinimizeGPU::run(): max_steps reached without convergence");
    return opts.max_steps;
}

// ===========================================================================
// RelaxGPU ??FieldSumGPU overloads
// ===========================================================================
void RelaxGPU::compute_H_eff(const Material& mat, IDemagGPU& demag,
                               FieldSumGPU& extra_fields) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    demag.set_stream(stream_);
    extra_fields.set_stream(stream_);
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    extra_fields.accumulate_gpu_ptr(d_m_, mat, d_H_);
    demag.accumulate_gpu_ptr(d_m_, mat, d_H_);
}

double RelaxGPU::max_torque_now(const Material& mat, IDemagGPU& demag,
                                 FieldSumGPU& extra_fields) {
    compute_H_eff(mat, demag, extra_fields);
    return reduce_max_torque(d_m_);
}

int RelaxGPU::run(const Material& mat, IDemagGPU& demag,
                   FieldSumGPU& extra_fields, Options opts) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;

    const double gp_alpha = constants::gamma_0 * constants::mu_0 * opts.alpha_relax
                            / (1.0 + opts.alpha_relax * opts.alpha_relax);
    const double dt = opts.dt;

    for (int step = 0; step < opts.max_steps; ++step) {
        compute_H_eff(mat, demag, extra_fields);

        if (step % opts.check_every == 0) {
            if (reduce_max_torque(d_m_) < opts.threshold)
                return step;
        }

        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_,
            d_H_,
            N, gp_alpha, dt);
        MICROMAG_KERNEL_CHECK();
    }

    CUDA_CHECK(cudaStreamSynchronize(s));

    if (opts.throw_on_max)
        throw std::runtime_error(
            "RelaxGPU::run(): max_steps reached without convergence");
    return opts.max_steps;
}

// ===========================================================================
// MinimizeGPU ??FieldSumGPU overloads
// ===========================================================================
void MinimizeGPU::compute_H_eff_for(GReal* d_m_src, const Material& mat,
                                     IDemagGPU& demag, FieldSumGPU& extra_fields) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    demag.set_stream(stream_);
    extra_fields.set_stream(stream_);
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    extra_fields.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    demag.accumulate_gpu_ptr(d_m_src, mat, d_H_);
}

double MinimizeGPU::compute_energy(const Material& mat, IDemagGPU& demag,
                                    FieldSumGPU& extra_fields) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;

    compute_H_eff_for(d_m_, mat, demag, extra_fields);
    const double h_sum = reduce_mdotH_sum(d_m_);

    // E = -mu0/2 * Ms * sum(m.H) * dV   (standard LLG energy sign)
    const double dV = grid_->cell_volume();
    return -constants::mu_0 * 0.5 * mat.Ms * h_sum * dV;
}

int MinimizeGPU::run(const Material& mat, IDemagGPU& demag,
                      FieldSumGPU& extra_fields, Options opts) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;

    const double alpha = 1.0;
    const double gp_alpha = constants::gamma_0 * constants::mu_0 * alpha / 2.0;
    double dt = opts.dt_init;

    for (int step = 0; step < opts.max_steps; ++step) {
        compute_H_eff_for(d_m_, mat, demag, extra_fields);

        if (step % opts.check_every == 0) {
            if (reduce_max_torque(d_m_) < opts.threshold)
                return step;
        }

        const double E0 = reduce_mdotH_sum(d_m_);

        copy3N_kernel<<<(3*N+blk-1)/blk, blk, 0, s>>>(
            d_m_trial_,
            d_m_,
            3*N);
        MICROMAG_KERNEL_CHECK();
        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_trial_,
            d_H_,
            N, gp_alpha, dt);
        MICROMAG_KERNEL_CHECK();

        compute_H_eff_for(d_m_trial_, mat, demag, extra_fields);
        const double E1 = reduce_mdotH_sum(d_m_trial_);

        if (E1 < E0) {
            std::swap(d_m_, d_m_trial_);
            dt = std::min(opts.dt_max, dt * 1.2);
        } else {
            dt *= 0.5;
            if (dt < opts.dt_min) {
                std::swap(d_m_, d_m_trial_);
                dt = opts.dt_init;
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(s));

    if (opts.throw_on_max)
        throw std::runtime_error(
            "MinimizeGPU::run(): max_steps reached without convergence");
    return opts.max_steps;
}

}  // namespace micromag

#endif // MICROMAG_CUDA



