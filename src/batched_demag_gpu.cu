// batched_demag_gpu.cu — Task 2 Phase 2.2: replica-batched demag (cuFFT batch=R).
// Shared Newell kernel tensor (precomputed once) + batched FFT/MAC/IFFT over R
// replicas. Physics matches DemagField (H_demag = −N·M). Requires MICROMAG_CUDA=1.

#ifdef MICROMAG_CUDA

#include <algorithm>
#include <vector>
#include <stdexcept>

#include "micromag/batched_demag_gpu.hpp"   // pulls gpu_fft.hpp (GpuFftManyRC, cufft types)
#include "micromag/demag.hpp"       // DemagField::nxx / nxy (public static)
#include "micromag/gpu_backend.hpp" // G0-1 runtime wrappers, G0-2 GPU_LAUNCH
#include "micromag/gpu_real.hpp"

namespace micromag {

namespace mg = micromag::gpu;

namespace {

constexpr int TPB = 256;
inline int nblocks(long n){ return int((n+TPB-1)/TPB); }

// scatter Ms·m (component-major, [R·3N]) into zero-padded [R·3·real_sz].
__global__ void scatter_kernel(double* __restrict__ M, const GReal* __restrict__ m,
                               int R, int nx, int ny, int nz,
                               int pnx, int pny, long real_sz, double Ms)
{
    const long N = long(nx)*ny*nz;
    long t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= R*N) return;
    const int r = int(t / N);
    const long idx = t - long(r)*N;
    const int iz = int(idx / (long(nx)*ny));
    const int iy = int((idx / nx) % ny);
    const int ix = int(idx % nx);
    const long pidx = ix + long(pnx)*(iy + long(pny)*iz);
    const long mb = long(r)*3*N;
    const long Mb = long(r)*3*real_sz;
    M[Mb + 0*real_sz + pidx] = Ms * double(m[mb + 0*N + idx]);
    M[Mb + 1*real_sz + pidx] = Ms * double(m[mb + 1*N + idx]);
    M[Mb + 2*real_sz + pidx] = Ms * double(m[mb + 2*N + idx]);
}

// HF = −(K·MF), shared real kernel K broadcast over replicas.
__global__ void mac_kernel(cufftDoubleComplex* __restrict__ MF,
                           const double* __restrict__ Kxx, const double* __restrict__ Kyy,
                           const double* __restrict__ Kzz, const double* __restrict__ Kxy,
                           const double* __restrict__ Kxz, const double* __restrict__ Kyz,
                           int R, long cplx)
{
    long t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= R*cplx) return;
    const int r = int(t / cplx);
    const long i = t - long(r)*cplx;
    const long b = long(r)*3*cplx;
    const cufftDoubleComplex Mx=MF[b+0*cplx+i], My=MF[b+1*cplx+i], Mz=MF[b+2*cplx+i];
    const double kxx=Kxx[i],kyy=Kyy[i],kzz=Kzz[i],kxy=Kxy[i],kxz=Kxz[i],kyz=Kyz[i];
    MF[b+0*cplx+i].x = -(kxx*Mx.x + kxy*My.x + kxz*Mz.x);
    MF[b+0*cplx+i].y = -(kxx*Mx.y + kxy*My.y + kxz*Mz.y);
    MF[b+1*cplx+i].x = -(kxy*Mx.x + kyy*My.x + kyz*Mz.x);
    MF[b+1*cplx+i].y = -(kxy*Mx.y + kyy*My.y + kyz*Mz.y);
    MF[b+2*cplx+i].x = -(kxz*Mx.x + kyz*My.x + kzz*Mz.x);
    MF[b+2*cplx+i].y = -(kxz*Mx.y + kyz*My.y + kzz*Mz.y);
}

// H[R·3N] += norm · M_inv[R·3·real_sz]  (extract unpadded region)
__global__ void extract_add_kernel(GReal* __restrict__ H, const double* __restrict__ M,
                                   int R, int nx, int ny, int nz,
                                   int pnx, int pny, long real_sz, double norm)
{
    const long N = long(nx)*ny*nz;
    long t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= R*N) return;
    const int r = int(t / N);
    const long idx = t - long(r)*N;
    const int iz = int(idx / (long(nx)*ny));
    const int iy = int((idx / nx) % ny);
    const int ix = int(idx % nx);
    const long pidx = ix + long(pnx)*(iy + long(pny)*iz);
    const long mb = long(r)*3*N;
    const long Mb = long(r)*3*real_sz;
    H[mb+0*N+idx] = static_cast<GReal>(double(H[mb+0*N+idx]) + norm*M[Mb+0*real_sz+pidx]);
    H[mb+1*N+idx] = static_cast<GReal>(double(H[mb+1*N+idx]) + norm*M[Mb+1*real_sz+pidx]);
    H[mb+2*N+idx] = static_cast<GReal>(double(H[mb+2*N+idx]) + norm*M[Mb+2*real_sz+pidx]);
}

}  // namespace

// ---------------------------------------------------------------------------
BatchedDemagGPU::BatchedDemagGPU(const StructuredGrid& grid, int R)
    : R_(R), nx_(int(grid.nx())), ny_(int(grid.ny())), nz_(int(grid.nz()))
{
    if (R <= 0) throw std::invalid_argument("BatchedDemagGPU: R must be > 0");
    N_ = long(nx_)*ny_*nz_;
    pad_nx_ = 2*nx_; pad_ny_ = 2*ny_; pad_nz_ = (nz_==1)?1:2*nz_;
    fft_nx_ = pad_nx_/2 + 1;
    real_sz_ = size_t(pad_nx_)*pad_ny_*pad_nz_;
    cplx_sz_ = size_t(fft_nx_)*pad_ny_*pad_nz_;
    norm_    = 1.0 / double(real_sz_);

    d_Kxx_ = mg::malloc(cplx_sz_*sizeof(double));
    d_Kyy_ = mg::malloc(cplx_sz_*sizeof(double));
    d_Kzz_ = mg::malloc(cplx_sz_*sizeof(double));
    d_Kxy_ = mg::malloc(cplx_sz_*sizeof(double));
    d_Kxz_ = mg::malloc(cplx_sz_*sizeof(double));
    d_Kyz_ = mg::malloc(cplx_sz_*sizeof(double));
    d_M_   = mg::malloc(size_t(R_)*3*real_sz_*sizeof(double));
    d_MF_  = mg::malloc(size_t(R_)*3*cplx_sz_*sizeof(mg::fft_complex_t));

    // batched fwd(D2Z)+inv(Z2D) plans, batch = 3R (G0-3 FFT seam). n in FFT
    // order = {pad_nz, pad_ny, pad_nx}.
    int rank = (pad_nz_==1) ? 2 : 3;
    int n3[3] = {pad_nz_, pad_ny_, pad_nx_};
    int* n = (rank==2) ? (n3+1) : n3;
    fft_ = new mg::GpuFftManyRC(rank, n, int(real_sz_), int(cplx_sz_), 3*R_);

    precompute_kernel_(grid.dx(), grid.dy(), grid.dz());
}

BatchedDemagGPU::~BatchedDemagGPU() {
    delete fft_; delete fft1_;
    mg::free(d_Kxx_); mg::free(d_Kyy_); mg::free(d_Kzz_);
    mg::free(d_Kxy_); mg::free(d_Kxz_); mg::free(d_Kyz_);
    mg::free(d_M_);   mg::free(d_MF_);
}

void BatchedDemagGPU::precompute_kernel_(double dx, double dy, double dz) {
    // Build the 6 real-space padded Newell components on host (same fill as
    // DemagField::precompute_kernel), FFT once, keep the (real) transform.
    std::vector<double> rbuf(real_sz_);
    mg::fft_complex_t* d_c = static_cast<mg::fft_complex_t*>(mg::malloc(cplx_sz_*sizeof(mg::fft_complex_t)));
    double* d_r = static_cast<double*>(mg::malloc(real_sz_*sizeof(double)));
    int rank = (pad_nz_==1) ? 2 : 3;
    int n3[3] = {pad_nz_, pad_ny_, pad_nx_};
    int* n = (rank==2) ? (n3+1) : n3;
    fft1_ = new mg::GpuFftManyRC(rank, n, int(real_sz_), int(cplx_sz_), 1);

    auto put = [&](int px,int py,int pz,double v){
        if(px<0)px+=pad_nx_; if(py<0)py+=pad_ny_; if(pz<0)pz+=pad_nz_;
        rbuf[size_t(px)+pad_nx_*(size_t(py)+pad_ny_*size_t(pz))]=v; };

    auto fill_diag = [&](void* d_K, auto fn){
        std::fill(rbuf.begin(), rbuf.end(), 0.0);
        for(int kz=0;kz<nz_;++kz)for(int ky=0;ky<ny_;++ky)for(int kx=0;kx<nx_;++kx){
            double v=fn(kx*dx,ky*dy,kz*dz);
            put(kx,ky,kz,v);
            if(kx>0)put(-kx,ky,kz,v); if(ky>0)put(kx,-ky,kz,v); if(kz>0)put(kx,ky,-kz,v);
            if(kx>0&&ky>0)put(-kx,-ky,kz,v); if(kx>0&&kz>0)put(-kx,ky,-kz,v);
            if(ky>0&&kz>0)put(kx,-ky,-kz,v);
            if(kx>0&&ky>0&&kz>0)put(-kx,-ky,-kz,v);
        }
        mg::memcpy(d_r, rbuf.data(), real_sz_*sizeof(double), mg::MemcpyKind::H2D);
        fft1_->exec_fwd(d_r, d_c);
        // keep real part (kernel FFT is real by symmetry)
        std::vector<mg::fft_complex_t> hc(cplx_sz_);
        mg::memcpy(hc.data(), d_c, cplx_sz_*sizeof(mg::fft_complex_t), mg::MemcpyKind::D2H);
        std::vector<double> kr(cplx_sz_);
        for(size_t i=0;i<cplx_sz_;++i) kr[i]=hc[i].x;
        mg::memcpy(d_K, kr.data(), cplx_sz_*sizeof(double), mg::MemcpyKind::H2D);
    };
    auto fill_off = [&](void* d_K,int sx,int sy,int sz,auto fn){
        std::fill(rbuf.begin(), rbuf.end(), 0.0);
        for(int kz=0;kz<nz_;++kz)for(int ky=0;ky<ny_;++ky)for(int kx=0;kx<nx_;++kx){
            double v=fn(kx*dx,ky*dy,kz*dz);
            for(int ix:{0,1})for(int iy:{0,1})for(int iz:{0,1}){
                if(ix&&kx==0)continue; if(iy&&ky==0)continue; if(iz&&kz==0)continue;
                double s=(ix?(double)sx:1.0)*(iy?(double)sy:1.0)*(iz?(double)sz:1.0);
                put(ix?-kx:kx, iy?-ky:ky, iz?-kz:kz, s*v);
            }
        }
        mg::memcpy(d_r, rbuf.data(), real_sz_*sizeof(double), mg::MemcpyKind::H2D);
        fft1_->exec_fwd(d_r, d_c);
        std::vector<mg::fft_complex_t> hc(cplx_sz_);
        mg::memcpy(hc.data(), d_c, cplx_sz_*sizeof(mg::fft_complex_t), mg::MemcpyKind::D2H);
        std::vector<double> kr(cplx_sz_);
        for(size_t i=0;i<cplx_sz_;++i) kr[i]=hc[i].x;
        mg::memcpy(d_K, kr.data(), cplx_sz_*sizeof(double), mg::MemcpyKind::H2D);
    };

    fill_diag(d_Kxx_, [&](double x,double y,double z){return DemagField::nxx(x,y,z,dx,dy,dz);});
    fill_diag(d_Kyy_, [&](double x,double y,double z){return DemagField::nxx(y,x,z,dy,dx,dz);});
    fill_diag(d_Kzz_, [&](double x,double y,double z){return DemagField::nxx(z,y,x,dz,dy,dx);});
    fill_off(d_Kxy_, -1,-1,+1, [&](double x,double y,double z){return DemagField::nxy(x,y,z,dx,dy,dz);});
    fill_off(d_Kxz_, -1,+1,-1, [&](double x,double y,double z){return DemagField::nxy(x,z,y,dx,dz,dy);});
    fill_off(d_Kyz_, +1,-1,-1, [&](double x,double y,double z){return DemagField::nxy(y,z,x,dy,dz,dx);});

    mg::free(d_r); mg::free(d_c);
}

void BatchedDemagGPU::set_stream(void* stream) {
    fft_->set_stream(reinterpret_cast<mg::stream_t>(stream));
}

void BatchedDemagGPU::accumulate_add(const GReal* d_m, GReal* d_H, double Ms, void* stream) {
    auto s = reinterpret_cast<mg::stream_t>(stream);
    double* M = static_cast<double*>(d_M_);
    mg::fft_complex_t* MF = static_cast<mg::fft_complex_t*>(d_MF_);
    const long RN = long(R_)*N_;

    mg::memset_async(M, 0, size_t(R_)*3*real_sz_*sizeof(double), s);
    GPU_LAUNCH(scatter_kernel, nblocks(RN), TPB, 0, s, M, d_m, R_, nx_, ny_, nz_,
        pad_nx_, pad_ny_, long(real_sz_), Ms);
    fft_->exec_fwd(M, MF);
    const long RC = long(R_)*long(cplx_sz_);
    GPU_LAUNCH(mac_kernel, nblocks(RC), TPB, 0, s, MF,
        static_cast<double*>(d_Kxx_), static_cast<double*>(d_Kyy_), static_cast<double*>(d_Kzz_),
        static_cast<double*>(d_Kxy_), static_cast<double*>(d_Kxz_), static_cast<double*>(d_Kyz_),
        R_, long(cplx_sz_));
    fft_->exec_inv(MF, M);
    GPU_LAUNCH(extract_add_kernel, nblocks(RN), TPB, 0, s, d_H, M, R_, nx_, ny_, nz_,
        pad_nx_, pad_ny_, long(real_sz_), norm_);
    mg::check_last("batched_demag accumulate");
}

}  // namespace micromag

#endif // MICROMAG_CUDA
