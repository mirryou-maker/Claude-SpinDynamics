// relax_gpu.cu ??GPU energy minimisation (RelaxGPU + MinimizeGPU).
//
// All stepping and convergence checking stay on GPU.
// Only ONE scalar D2H per convergence check (max torque or energy).
//
// Memory layout: [3횞N] component-major, x-fastest.

#ifdef MICROMAG_CUDA

#include <cuda_runtime.h>
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
// Device utility: atomicMax for double (non-negative values only).
// IEEE bit pattern of non-negative doubles is ordered like uint64.
// ===========================================================================
__device__ static void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long  val_ull  = __double_as_longlong(val);
    unsigned long long  old_ull  = *addr_ull;
    while (val_ull > old_ull) {
        unsigned long long assumed = old_ull;
        old_ull = atomicCAS(addr_ull, assumed, val_ull);
    }
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
// max-torque reduction: writes max |m횞H|짼 (atomicMax) to d_max[0].
// Caller must set d_max[0] = 0 before launch.
// ===========================================================================
__global__ static void max_torque_kernel(
    double* __restrict__       d_max,  // [1] output ??stays double for precise reduction
    const GReal* __restrict__  m,
    const GReal* __restrict__  H,
    int N)
{
    __shared__ double smem[256];
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid   = threadIdx.x;

    double local_max = 0.0;
    if (idx < N) {
        const double mx = static_cast<double>(m[0*N+idx]),
                     my = static_cast<double>(m[1*N+idx]),
                     mz = static_cast<double>(m[2*N+idx]);
        const double Hx = static_cast<double>(H[0*N+idx]),
                     Hy = static_cast<double>(H[1*N+idx]),
                     Hz = static_cast<double>(H[2*N+idx]);
        const double tx = my*Hz - mz*Hy;
        const double ty = mz*Hx - mx*Hz;
        const double tz = mx*Hy - my*Hx;
        local_max = tx*tx + ty*ty + tz*tz;
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmax(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    if (tid == 0) atomicMaxDouble(d_max, smem[0]);
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
// Block-sum reduction (for energy): writes partial sums to out[blockIdx.x].
// ===========================================================================
__global__ static void block_sum_kernel(
    double* __restrict__ out,
    const double* __restrict__ in,
    int N)
{
    __shared__ double smem[256];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    smem[tid] = (idx < N) ? in[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
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
    relax_alloc(N_, d_m_, d_H_, d_max_, stream_);
}

RelaxGPU::~RelaxGPU() {
    relax_free(d_m_, d_H_, d_max_, stream_);
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
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    demag.accumulate_gpu_ptr(d_m_, mat, d_H_);
    exch.accumulate_gpu_ptr(d_m_, mat, d_H_);
    zeeman.accumulate_gpu_ptr(d_m_, mat, d_H_);
    if (aniso) aniso->accumulate_gpu_ptr(d_m_, mat, d_H_);
}

double RelaxGPU::max_torque_now(const Material& mat,
                                 IDemagGPU& demag,
                                 ExchangeFieldGPU& exch,
                                 ZeemanFieldGPU& zeeman,
                                 UniaxialAnisotropyFieldGPU* aniso) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    compute_H_eff(mat, demag, exch, zeeman, aniso);
    const double zero = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(d_max_, &zero, sizeof(double), cudaMemcpyHostToDevice, s));
    const int blk = 256, grd = static_cast<int>((N_+blk-1)/blk);
    max_torque_kernel<<<grd, blk, 0, s>>>(d_max_,
        d_m_,
        d_H_,
        static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
    double h_max;
    CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return std::sqrt(h_max);
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
            const double zero = 0.0;
            CUDA_CHECK(cudaMemcpyAsync(d_max_, &zero, sizeof(double), cudaMemcpyHostToDevice, s));
            max_torque_kernel<<<grd, blk, 0, s>>>(d_max_,
                d_m_,
                d_H_,
                N);
            CUDA_CHECK(cudaGetLastError());
            double h_max;
            CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (std::sqrt(h_max) < opts.threshold)
                return step;
        }

        // Damping-only Euler step
        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_,
            d_H_,
            N, gp_alpha, dt);
        CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    demag.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    exch.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    zeeman.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    if (aniso) aniso->accumulate_gpu_ptr(d_m_src, mat, d_H_);
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
    mdot_H_kernel<<<grd, blk, 0, s>>>(d_energy_,
        d_m_,
        d_H_,
        N);
    CUDA_CHECK(cudaGetLastError());

    // Block-level sum reduction: d_energy ??d_max (reuse as scratch)
    // First pass: grd blocks ??grd partial sums stored in first grd cells
    block_sum_kernel<<<grd, blk, 0, s>>>(d_max_, d_energy_, N);
    CUDA_CHECK(cudaGetLastError());

    // Second pass: sum the grd partial sums (grd ??256 for N ??65536)
    // For larger N: multi-pass or download partial sums
    const int grd2 = (grd + blk - 1) / blk;
    block_sum_kernel<<<grd2, blk, 0, s>>>(d_energy_, d_max_, grd);
    CUDA_CHECK(cudaGetLastError());

    double h_sum;
    CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_energy_, sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    // E = -關?/2 * Ms * sum(m쨌H) * dV   (standard LLG energy sign)
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
            const double zero = 0.0;
            CUDA_CHECK(cudaMemcpyAsync(d_max_, &zero, sizeof(double), cudaMemcpyHostToDevice, s));
            max_torque_kernel<<<grd, blk, 0, s>>>(d_max_,
                d_m_,
                d_H_,
                N);
            CUDA_CHECK(cudaGetLastError());
            double h_max;
            CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (std::sqrt(h_max) < opts.threshold)
                return step;
        }

        // Energy before trial step
        mdot_H_kernel<<<grd, blk, 0, s>>>(d_energy_,
            d_m_,
            d_H_,
            N);
        block_sum_kernel<<<grd, blk, 0, s>>>(d_max_, d_energy_, N);
        const int grd2 = (grd + blk - 1) / blk;
        block_sum_kernel<<<grd2, blk, 0, s>>>(d_energy_, d_max_, grd);
        double E0;
        CUDA_CHECK(cudaMemcpyAsync(&E0, d_energy_, sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        // Trial step: copy m ??m_trial, advance m_trial
        copy3N_kernel<<<(3*N+blk-1)/blk, blk, 0, s>>>(
            d_m_trial_,
            d_m_,
            3*N);
        CUDA_CHECK(cudaGetLastError());
        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_trial_,
            d_H_,
            N, gp_alpha, dt);
        CUDA_CHECK(cudaGetLastError());

        // Energy after trial (need H_eff for m_trial)
        compute_H_eff_for(d_m_trial_, mat, demag, exch, zeeman, aniso);
        mdot_H_kernel<<<grd, blk, 0, s>>>(d_energy_,
            d_m_trial_,
            d_H_,
            N);
        block_sum_kernel<<<grd, blk, 0, s>>>(d_max_, d_energy_, N);
        block_sum_kernel<<<grd2, blk, 0, s>>>(d_energy_, d_max_, grd);
        double E1;
        CUDA_CHECK(cudaMemcpyAsync(&E1, d_energy_, sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

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
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    extra_fields.accumulate_gpu_ptr(d_m_, mat, d_H_);
    CUDA_CHECK(cudaDeviceSynchronize());
    demag.accumulate_gpu_ptr(d_m_, mat, d_H_);
}

double RelaxGPU::max_torque_now(const Material& mat, IDemagGPU& demag,
                                 FieldSumGPU& extra_fields) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    compute_H_eff(mat, demag, extra_fields);
    const double zero = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(d_max_, &zero, sizeof(double), cudaMemcpyHostToDevice, s));
    const int blk = 256, grd = static_cast<int>((N_+blk-1)/blk);
    max_torque_kernel<<<grd, blk, 0, s>>>(d_max_,
        d_m_,
        d_H_,
        static_cast<int>(N_));
    CUDA_CHECK(cudaGetLastError());
    double h_max;
    CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return std::sqrt(h_max);
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
            const double zero = 0.0;
            CUDA_CHECK(cudaMemcpyAsync(d_max_, &zero, sizeof(double), cudaMemcpyHostToDevice, s));
            max_torque_kernel<<<grd, blk, 0, s>>>(d_max_,
                d_m_,
                d_H_,
                N);
            CUDA_CHECK(cudaGetLastError());
            double h_max;
            CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (std::sqrt(h_max) < opts.threshold)
                return step;
        }

        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_,
            d_H_,
            N, gp_alpha, dt);
        CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaMemsetAsync(d_H_, 0, 3*N_*sizeof(GReal), s));
    extra_fields.accumulate_gpu_ptr(d_m_src, mat, d_H_);
    CUDA_CHECK(cudaDeviceSynchronize());
    demag.accumulate_gpu_ptr(d_m_src, mat, d_H_);
}

double MinimizeGPU::compute_energy(const Material& mat, IDemagGPU& demag,
                                    FieldSumGPU& extra_fields) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int N = static_cast<int>(N_);
    const int blk = 256, grd = (N + blk - 1) / blk;

    compute_H_eff_for(d_m_, mat, demag, extra_fields);
    mdot_H_kernel<<<grd, blk, 0, s>>>(d_energy_,
        d_m_,
        d_H_,
        N);
    CUDA_CHECK(cudaGetLastError());
    block_sum_kernel<<<grd, blk, 0, s>>>(d_max_, d_energy_, N);
    CUDA_CHECK(cudaGetLastError());
    const int grd2 = (grd + blk - 1) / blk;
    block_sum_kernel<<<grd2, blk, 0, s>>>(d_energy_, d_max_, grd);
    CUDA_CHECK(cudaGetLastError());
    double h_sum;
    CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_energy_, sizeof(double), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
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
            const double zero = 0.0;
            CUDA_CHECK(cudaMemcpyAsync(d_max_, &zero, sizeof(double), cudaMemcpyHostToDevice, s));
            max_torque_kernel<<<grd, blk, 0, s>>>(d_max_,
                d_m_,
                d_H_,
                N);
            CUDA_CHECK(cudaGetLastError());
            double h_max;
            CUDA_CHECK(cudaMemcpyAsync(&h_max, d_max_, sizeof(double), cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (std::sqrt(h_max) < opts.threshold)
                return step;
        }

        mdot_H_kernel<<<grd, blk, 0, s>>>(d_energy_,
            d_m_,
            d_H_,
            N);
        block_sum_kernel<<<grd, blk, 0, s>>>(d_max_, d_energy_, N);
        const int grd2 = (grd + blk - 1) / blk;
        block_sum_kernel<<<grd2, blk, 0, s>>>(d_energy_, d_max_, grd);
        double E0;
        CUDA_CHECK(cudaMemcpyAsync(&E0, d_energy_, sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        copy3N_kernel<<<(3*N+blk-1)/blk, blk, 0, s>>>(
            d_m_trial_,
            d_m_,
            3*N);
        CUDA_CHECK(cudaGetLastError());
        damping_euler_kernel<<<grd, blk, 0, s>>>(
            d_m_trial_,
            d_H_,
            N, gp_alpha, dt);
        CUDA_CHECK(cudaGetLastError());

        compute_H_eff_for(d_m_trial_, mat, demag, extra_fields);
        mdot_H_kernel<<<grd, blk, 0, s>>>(d_energy_,
            d_m_trial_,
            d_H_,
            N);
        block_sum_kernel<<<grd, blk, 0, s>>>(d_max_, d_energy_, N);
        block_sum_kernel<<<grd2, blk, 0, s>>>(d_energy_, d_max_, grd);
        double E1;
        CUDA_CHECK(cudaMemcpyAsync(&E1, d_energy_, sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

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



