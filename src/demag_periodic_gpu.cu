// DemagFieldPeriodicGPU: cuFFT periodic-BC demag without zero-padding.
// Kernel precomputed on CPU (periodic Newell image sum), uploaded once.
// Pipeline per step: pack ??H2D ??R2C batch ??MAC ??C2R batch ??scale ??D2H ??unpack.

#ifdef MICROMAG_CUDA

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include "micromag/gpu_real.hpp"
#include "micromag/demag_periodic_gpu.hpp"
#include "micromag/types.hpp"

// ---------------------------------------------------------------------------
// Error-check macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult _r = (call); \
    if (_r != CUFFT_SUCCESS) \
        throw std::runtime_error("cuFFT error " + std::to_string((int)_r)); \
} while(0)

namespace micromag {

// ---------------------------------------------------------------------------
// Helpers: cast stored void* handles to cuFFT handle
// ---------------------------------------------------------------------------
static inline cufftHandle&  handle(int& h) { return *reinterpret_cast<cufftHandle*>(&h); }
static inline cufftHandle   handle(const int& h) { return *reinterpret_cast<const cufftHandle*>(&h); }

// ===========================================================================
// GPU kernels (file scope so they can be called from member functions)
// ===========================================================================

// Pointwise symmetric demag MAC in frequency domain:
//   HF_x = K_xx*MF_x + K_xy*MF_y + K_xz*MF_z
//   HF_y = K_xy*MF_x + K_yy*MF_y + K_yz*MF_z
//   HF_z = K_xz*MF_x + K_yz*MF_y + K_zz*MF_z
// All arrays interleaved as [x-slice | y-slice | z-slice] of length cplx_sz.
__global__ static void mac_periodic(
    GREAL_CUFFT_COMPLEX* __restrict__ HF_all,
    const GREAL_CUFFT_COMPLEX* __restrict__ Kxx,
    const GREAL_CUFFT_COMPLEX* __restrict__ Kxy,
    const GREAL_CUFFT_COMPLEX* __restrict__ Kxz,
    const GREAL_CUFFT_COMPLEX* __restrict__ Kyy,
    const GREAL_CUFFT_COMPLEX* __restrict__ Kyz,
    const GREAL_CUFFT_COMPLEX* __restrict__ Kzz,
    const GREAL_CUFFT_COMPLEX* __restrict__ MF_all,
    size_t cplx_sz)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= cplx_sz) return;
    const auto Mx = MF_all[i], My = MF_all[i+cplx_sz], Mz = MF_all[i+2*cplx_sz];
    HF_all[i]           = GREAL_CUFFT_CADD(GREAL_CUFFT_CADD(GREAL_CUFFT_CMUL(Kxx[i],Mx), GREAL_CUFFT_CMUL(Kxy[i],My)), GREAL_CUFFT_CMUL(Kxz[i],Mz));
    HF_all[i+cplx_sz]   = GREAL_CUFFT_CADD(GREAL_CUFFT_CADD(GREAL_CUFFT_CMUL(Kxy[i],Mx), GREAL_CUFFT_CMUL(Kyy[i],My)), GREAL_CUFFT_CMUL(Kyz[i],Mz));
    HF_all[i+2*cplx_sz] = GREAL_CUFFT_CADD(GREAL_CUFFT_CADD(GREAL_CUFFT_CMUL(Kxz[i],Mx), GREAL_CUFFT_CMUL(Kyz[i],My)), GREAL_CUFFT_CMUL(Kzz[i],Mz));
}

// Scale in-place: dst[i] *= scale
__global__ static void scale_all3(GReal* __restrict__ dst, double scale, size_t n3) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n3) dst[i] = static_cast<GReal>(static_cast<double>(dst[i]) * scale);
}

// Add: dst[i] += src[i]
__global__ static void add_all3(
    GReal* __restrict__       dst,
    const GReal* __restrict__ src,
    size_t n3)
{
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n3) dst[i] += src[i];
}

// ===========================================================================
// CPU Newell helpers (periodic image sum ??identical to demag_periodic.cpp)
// ===========================================================================
static double nf(double x, double y, double z) {
    x=std::abs(x); y=std::abs(y); z=std::abs(z);
    const double x2=x*x,y2=y*y,z2=z*z, r=std::sqrt(x2+y2+z2);
    if(r==0.0) return 0.0;
    double v=0.0;
    const double dxz=std::sqrt(x2+z2); if(dxz>0) v+=y*(z2-x2)*0.5*std::asinh(y/dxz);
    const double dxy=std::sqrt(x2+y2); if(dxy>0) v+=z*(y2-x2)*0.5*std::asinh(z/dxy);
    if(x>0) v-=x*y*z*std::atan(y*z/(x*r));
    v+=(2*x2-y2-z2)*r/6.0; return v;
}
static double ng(double x, double y, double z) {
    z=std::abs(z);
    const double x2=x*x,y2=y*y,z2=z*z, r=std::sqrt(x2+y2+z2);
    if(r==0.0) return 0.0;
    double v=0.0;
    const double dxy=std::sqrt(x2+y2); if(dxy>0) v+=x*y*z*std::asinh(z/dxy);
    const double dyz=std::sqrt(y2+z2); if(dyz>0) v+=y*(3*z2-y2)/6*std::asinh(x/dyz);
    const double dxz=std::sqrt(x2+z2); if(dxz>0) v+=x*(3*z2-x2)/6*std::asinh(y/dxz);
    if(z>0) v-=z*z2/6*std::atan(x*y/(z*r));
    if(y>0) v-=z*y2*0.5*std::atan(x*z/(y*r));
    if(x>0) v-=z*x2*0.5*std::atan(y*z/(x*r));
    v-=x*y*r/3.0; return v;
}

static double nxx_p(double x,double y,double z,double dx,double dy,double dz) {
    int nx=static_cast<int>(std::round(x/dx));
    int ny=static_cast<int>(std::round(y/dy));
    int nz=static_cast<int>(std::round(z/dz));
    double s=0;
    for(int a:{0,1})for(int b:{0,1})for(int c:{0,1})
    for(int d:{0,1})for(int e:{0,1})for(int g:{0,1}){
        int sgn=((a+b+c+d+e+g)%2==0)?1:-1;
        s+=sgn*nf((nx+a-d)*dx,(ny+b-e)*dy,(nz+c-g)*dz);
    }
    return s/(4*constants::pi*dx*dy*dz);
}
static double nxy_p(double x,double y,double z,double dx,double dy,double dz) {
    int nx=static_cast<int>(std::round(x/dx));
    int ny=static_cast<int>(std::round(y/dy));
    int nz=static_cast<int>(std::round(z/dz));
    double s=0;
    for(int a:{0,1})for(int b:{0,1})for(int c:{0,1})
    for(int d:{0,1})for(int e:{0,1})for(int g:{0,1}){
        int sgn=((a+b+c+d+e+g)%2==0)?1:-1;
        s+=sgn*ng((nx+a-d)*dx,(ny+b-e)*dy,(nz+c-g)*dz);
    }
    return s/(4*constants::pi*dx*dy*dz);
}

// ===========================================================================
// Constructor
// ===========================================================================
DemagFieldPeriodicGPU::DemagFieldPeriodicGPU(const StructuredGrid& grid, int n_rep)
    : nx_(grid.nx()), ny_(grid.ny()), nz_(grid.nz()),
      dx_(grid.dx()), dy_(grid.dy()), dz_(grid.dz()),
      fft_nx_(nx_/2+1),
      real_sz_(static_cast<size_t>(nx_)*ny_*nz_),
      cplx_sz_(static_cast<size_t>(fft_nx_)*ny_*nz_),
      n_rep_(n_rep)
{
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    stream_ = s;

    // 3-component batch buffers
    CUDA_CHECK(cudaMallocAsync(&d_M_all_,  3*real_sz_*sizeof(GReal),             s));
    CUDA_CHECK(cudaMallocAsync(&d_MF_all_, 3*cplx_sz_*sizeof(GREAL_CUFFT_COMPLEX), s));

    // Kernel components (frequency-domain)
    CUDA_CHECK(cudaMallocAsync(&d_K_xx_, cplx_sz_*sizeof(GREAL_CUFFT_COMPLEX), s));
    CUDA_CHECK(cudaMallocAsync(&d_K_yy_, cplx_sz_*sizeof(GREAL_CUFFT_COMPLEX), s));
    CUDA_CHECK(cudaMallocAsync(&d_K_zz_, cplx_sz_*sizeof(GREAL_CUFFT_COMPLEX), s));
    CUDA_CHECK(cudaMallocAsync(&d_K_xy_, cplx_sz_*sizeof(GREAL_CUFFT_COMPLEX), s));
    CUDA_CHECK(cudaMallocAsync(&d_K_xz_, cplx_sz_*sizeof(GREAL_CUFFT_COMPLEX), s));
    CUDA_CHECK(cudaMallocAsync(&d_K_yz_, cplx_sz_*sizeof(GREAL_CUFFT_COMPLEX), s));

    // Pinned host staging
    CUDA_CHECK(cudaMallocHost(&h_M_pinned_, 3*real_sz_*sizeof(GReal)));
    CUDA_CHECK(cudaMallocHost(&h_H_pinned_, 3*real_sz_*sizeof(GReal)));

    // cuFFT plans ??(nz, ny, nx) in C row-major (x fastest after transpose)
    const int dims[3] = {(int)nz_, (int)ny_, (int)nx_};

    // Single-transform plan for kernel precompute
    CUFFT_CHECK(cufftCreate(reinterpret_cast<cufftHandle*>(&plan_fwd_single_)));
    CUFFT_CHECK(cufftSetStream(handle(plan_fwd_single_), s));
    size_t ws1=0;
    CUFFT_CHECK(cufftMakePlanMany(handle(plan_fwd_single_),3,const_cast<int*>(dims),
        nullptr,1,(int)real_sz_, nullptr,1,(int)cplx_sz_, GREAL_CUFFT_TYPE,1,&ws1));

    // Batch=3 forward plan
    CUFFT_CHECK(cufftCreate(reinterpret_cast<cufftHandle*>(&plan_fwd_batch_)));
    CUFFT_CHECK(cufftSetStream(handle(plan_fwd_batch_), s));
    size_t wsf=0;
    CUFFT_CHECK(cufftMakePlanMany(handle(plan_fwd_batch_),3,const_cast<int*>(dims),
        nullptr,1,(int)real_sz_, nullptr,1,(int)cplx_sz_, GREAL_CUFFT_TYPE,3,&wsf));

    // Batch=3 inverse plan
    CUFFT_CHECK(cufftCreate(reinterpret_cast<cufftHandle*>(&plan_inv_batch_)));
    CUFFT_CHECK(cufftSetStream(handle(plan_inv_batch_), s));
    size_t wsi=0;
    CUFFT_CHECK(cufftMakePlanMany(handle(plan_inv_batch_),3,const_cast<int*>(dims),
        nullptr,1,(int)cplx_sz_, nullptr,1,(int)real_sz_, GREAL_CUFFT_ITYPE,3,&wsi));

    CUDA_CHECK(cudaStreamSynchronize(s));
    precompute_kernel();
}

// ===========================================================================
// Destructor
// ===========================================================================
DemagFieldPeriodicGPU::~DemagFieldPeriodicGPU() {
    auto s = static_cast<cudaStream_t>(stream_);
    auto destroy = [&](int& hh) {
        if (hh) { cufftDestroy(handle(hh)); hh=0; }
    };
    destroy(plan_fwd_single_);
    destroy(plan_fwd_batch_);
    destroy(plan_inv_batch_);
    if (d_M_all_)  { cudaFreeAsync(d_M_all_,  s); d_M_all_=nullptr; }
    if (d_MF_all_) { cudaFreeAsync(d_MF_all_, s); d_MF_all_=nullptr; }
    if (d_K_xx_)   { cudaFreeAsync(d_K_xx_, s); d_K_xx_=nullptr; }
    if (d_K_yy_)   { cudaFreeAsync(d_K_yy_, s); d_K_yy_=nullptr; }
    if (d_K_zz_)   { cudaFreeAsync(d_K_zz_, s); d_K_zz_=nullptr; }
    if (d_K_xy_)   { cudaFreeAsync(d_K_xy_, s); d_K_xy_=nullptr; }
    if (d_K_xz_)   { cudaFreeAsync(d_K_xz_, s); d_K_xz_=nullptr; }
    if (d_K_yz_)   { cudaFreeAsync(d_K_yz_, s); d_K_yz_=nullptr; }
    if (h_M_pinned_) { cudaFreeHost(h_M_pinned_); h_M_pinned_=nullptr; }
    if (h_H_pinned_) { cudaFreeHost(h_H_pinned_); h_H_pinned_=nullptr; }
    if (stream_owned_ && stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
}

// ===========================================================================
// set_stream -- P2: redirect batch FFT plans to a shared integrator stream.
// ===========================================================================
void DemagFieldPeriodicGPU::set_stream(void* s) {
    if (stream_ == s) return;
    if (stream_owned_ && stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    }
    stream_ = s;
    stream_owned_ = false;
    cufftSetStream(handle(plan_fwd_batch_),  static_cast<cudaStream_t>(s));
    cufftSetStream(handle(plan_inv_batch_),  static_cast<cudaStream_t>(s));
}

// ===========================================================================
// precompute_kernel ??CPU periodic Newell sum ??H2D ??cuFFT ??zero k=0
// ===========================================================================
void DemagFieldPeriodicGPU::precompute_kernel() {
    auto s = static_cast<cudaStream_t>(stream_);
    const double Lx=nx_*dx_, Ly=ny_*dy_, Lz=nz_*dz_;

    // Host staging for one kernel component at a time (always double for accuracy)
    std::vector<double> h_k_dbl(real_sz_);
    std::vector<GReal>  h_k(real_sz_);

    // Temp device buffer for one real kernel component (GReal for plan compatibility)
    GReal* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, real_sz_*sizeof(GReal)));

    auto fill_and_fft = [&](void* d_Kdest, auto kern) {
        // Image-sum Newell kernel on CPU in double, then cast to GReal
        for (Index kz=0; kz<nz_; ++kz)
        for (Index ky=0; ky<ny_; ++ky)
        for (Index kx=0; kx<nx_; ++kx) {
            double x0 = (kx<=nx_/2) ? kx*dx_ : (kx-nx_)*dx_;
            double y0 = (ky<=ny_/2) ? ky*dy_ : (ky-ny_)*dy_;
            double z0 = (kz<=nz_/2) ? kz*dz_ : (kz-nz_)*dz_;
            double val=0.0;
            for(int n1=-n_rep_;n1<=n_rep_;++n1)
            for(int n2=-n_rep_;n2<=n_rep_;++n2)
            for(int n3=-n_rep_;n3<=n_rep_;++n3)
                val += kern(x0+n1*Lx, y0+n2*Ly, z0+n3*Lz);
            h_k[kx + nx_*(ky + ny_*kz)] = static_cast<GReal>(val);
        }
        CUDA_CHECK(cudaMemcpy(d_tmp, h_k.data(), real_sz_*sizeof(GReal), cudaMemcpyHostToDevice));
        // FFT d_tmp -> d_Kdest (using single-shot plan, type matches GREAL_CUFFT_TYPE)
        CUFFT_CHECK(GREAL_CUFFT_EXEC_FWD(handle(plan_fwd_single_),
            d_tmp, static_cast<GREAL_CUFFT_COMPLEX*>(d_Kdest)));
        CUDA_CHECK(cudaDeviceSynchronize());
        // Zero k=0 mode (uniform M -> no periodic demag field)
        CUDA_CHECK(cudaMemset(d_Kdest, 0, sizeof(GREAL_CUFFT_COMPLEX)));
    };

    fill_and_fft(d_K_xx_, [&](double x,double y,double z){ return nxx_p(x,y,z,dx_,dy_,dz_); });
    fill_and_fft(d_K_yy_, [&](double x,double y,double z){ return nxx_p(y,x,z,dy_,dx_,dz_); });
    fill_and_fft(d_K_zz_, [&](double x,double y,double z){ return nxx_p(z,y,x,dz_,dy_,dx_); });
    fill_and_fft(d_K_xy_, [&](double x,double y,double z){ return nxy_p(x,y,z,dx_,dy_,dz_); });
    fill_and_fft(d_K_xz_, [&](double x,double y,double z){ return nxy_p(x,z,y,dx_,dz_,dy_); });
    fill_and_fft(d_K_yz_, [&](double x,double y,double z){ return nxy_p(y,z,x,dy_,dz_,dx_); });

    cudaFree(d_tmp);
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ===========================================================================
// GPU-pointer path: d_m [3횞N component-major, on-GPU] ??d_H_out [3횞N, on-GPU]
// ===========================================================================
void DemagFieldPeriodicGPU::accumulate_gpu_ptr(const GReal* d_m,
                                                 const Material& mat,
                                                 GReal* d_H_out) const {
    auto s = static_cast<cudaStream_t>(stream_);
    const size_t N3 = 3*real_sz_;
    constexpr int BLK = 256;

    // 1. Copy d_m ??d_M_all_ (both component-major 3횞N)
    CUDA_CHECK(cudaMemcpyAsync(d_M_all_, d_m, N3*sizeof(GReal), cudaMemcpyDeviceToDevice, s));

    // 2. Batch R2C FFT: d_M_all_ ??d_MF_all_
    CUFFT_CHECK(GREAL_CUFFT_EXEC_FWD(handle(plan_fwd_batch_),
        reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_),
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_)));

    // 3. Pointwise MAC
    const int gcx = (int)((cplx_sz_+BLK-1)/BLK);
    mac_periodic<<<gcx,BLK,0,s>>>(
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_xx_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_xy_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_xz_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_yy_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_yz_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_zz_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_MF_all_),
        cplx_sz_);
    CUDA_CHECK(cudaGetLastError());

    // 4. Batch C2R IFFT: d_HF_all_ ??d_H_all_
    CUFFT_CHECK(GREAL_CUFFT_EXEC_INV(handle(plan_inv_batch_),
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_),
        reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_)));

    // 5. Scale: H_demag = -1/N * IFFT(K쨌MF)
    // M was packed as Ms*m, so the factor is just -1/N (not -Ms/N).
    const double scale = -1.0 / static_cast<double>(real_sz_);
    scale_all3<<<(int)((N3+BLK-1)/BLK), BLK, 0, s>>>(
        reinterpret_cast<GReal*>(d_M_all_), scale, N3);
    CUDA_CHECK(cudaGetLastError());

    // 6. d_H_out += d_H_all_
    add_all3<<<(int)((N3+BLK-1)/BLK), BLK, 0, s>>>(
        d_H_out, reinterpret_cast<const GReal*>(d_M_all_), N3);
    CUDA_CHECK(cudaGetLastError());

    // Standalone mode: sync so caller on a different stream sees the writes.
    // Shared-stream mode: stream ordering on the integrator stream suffices.
    if (stream_owned_) { CUDA_CHECK(cudaStreamSynchronize(s)); }
}

// ===========================================================================
// CPU-path: pack m ??GPU ??compute ??download ??unpack to H_out
// ===========================================================================
void DemagFieldPeriodicGPU::accumulate(const VectorField3D& m,
                                         const Material& mat,
                                         VectorField3D& H_out) const {
    auto s = static_cast<cudaStream_t>(stream_);
    const size_t N  = real_sz_;
    const size_t N3 = 3*N;

    // Pack m ??h_M_pinned_ (component-major: Mx|My|Mz, each N doubles)
    for (size_t i=0; i<N; ++i) {
        const Vec3& v = m[static_cast<Index>(i)];
        h_M_pinned_[i]       = static_cast<GReal>(mat.Ms * v.x);
        h_M_pinned_[i+N]     = static_cast<GReal>(mat.Ms * v.y);
        h_M_pinned_[i+2*N]   = static_cast<GReal>(mat.Ms * v.z);
    }

    // H2D
    CUDA_CHECK(cudaMemcpyAsync(d_M_all_, h_M_pinned_,
        N3*sizeof(GReal), cudaMemcpyHostToDevice, s));


    // GPU compute via ptr path (copies d_M_all_ to itself ??no-op copy)
    // Avoid double-copy: call the inner pipeline directly
    constexpr int BLK = 256;
    const int gcx = (int)((cplx_sz_+BLK-1)/BLK);

    CUFFT_CHECK(GREAL_CUFFT_EXEC_FWD(handle(plan_fwd_batch_),
        reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_),
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_)));

    mac_periodic<<<gcx,BLK,0,s>>>(
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_xx_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_xy_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_xz_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_yy_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_yz_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_K_zz_),
        reinterpret_cast<const GREAL_CUFFT_COMPLEX*>(d_MF_all_),
        cplx_sz_);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(GREAL_CUFFT_EXEC_INV(handle(plan_inv_batch_),
        reinterpret_cast<GREAL_CUFFT_COMPLEX*>(d_MF_all_),
        reinterpret_cast<GREAL_CUFFT_REAL*>(d_M_all_)));

    const double scale = -1.0 / static_cast<double>(N);
    scale_all3<<<(int)((N3+BLK-1)/BLK), BLK, 0, s>>>(
        reinterpret_cast<GReal*>(d_M_all_), scale, N3);
    CUDA_CHECK(cudaGetLastError());

    // D2H
    CUDA_CHECK(cudaMemcpyAsync(h_H_pinned_, d_M_all_,
        N3*sizeof(GReal), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    // Unpack: add H_demag to H_out (component-major ??Vec3 cell-major)
    for (size_t i=0; i<N; ++i) {
        H_out[static_cast<Index>(i)].x += static_cast<double>(h_H_pinned_[i]);
        H_out[static_cast<Index>(i)].y += static_cast<double>(h_H_pinned_[i+N]);
        H_out[static_cast<Index>(i)].z += static_cast<double>(h_H_pinned_[i+2*N]);
    }
}

// ===========================================================================
// energy / energy_density  (compute via accumulate, then integrate)
// ===========================================================================
Real DemagFieldPeriodicGPU::energy(const VectorField3D& m,
                                    const Material& mat) const {
    VectorField3D H(m.grid());
    for (Index i=0; i<H.size(); ++i) H[i]={0,0,0};
    accumulate(m, mat, H);
    Real E=0.0;
    const Real dV = m.grid().cell_volume();
    for (Index i=0; i<m.size(); ++i)
        E -= constants::mu_0 * mat.Ms * m[i].dot(H[i]) * dV;
    return 0.5*E;
}

ScalarField3D DemagFieldPeriodicGPU::energy_density(const VectorField3D& m,
                                                      const Material& mat) const {
    VectorField3D H(m.grid());
    for (Index i=0; i<H.size(); ++i) H[i]={0,0,0};
    accumulate(m, mat, H);
    ScalarField3D ed(m.grid());
    for (Index i=0; i<m.size(); ++i)
        ed[i] = -0.5 * constants::mu_0 * mat.Ms * m[i].dot(H[i]);
    return ed;
}

} // namespace micromag

#endif // MICROMAG_CUDA




