// VkFFT vs cuFFT feasibility spike: time a R2C transform of the demag size
// (pad_nx x pad_ny, batch=3, float32) with both, to decide if VkFFT is worth a
// full demag integration.  Throwaway benchmark — not part of the build.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>

#define VKFFT_BACKEND 1            // CUDA
#include "VkFFT/vkFFT.h"

#define CK(x)  do{ cudaError_t e=(x); if(e){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));return 1;} }while(0)
#define CKF(x) do{ cufftResult e=(x); if(e){printf("cuFFT err %s @%d: %d\n",#x,__LINE__,(int)e);return 1;} }while(0)
#define CKV(x) do{ VkFFTResult e=(x); if(e){printf("VkFFT err %s @%d: %d\n",#x,__LINE__,(int)e);return 1;} }while(0)

int main(int argc, char** argv) {
    int NX = (argc>1)?atoi(argv[1]):2048;   // grid; padded = 2*NX x 2*NY
    int NY = (argc>2)?atoi(argv[2]):512;
    const int padx = 2*NX, pady = 2*NY;     // single-layer film -> 2-D
    const int batch = 3;
    const int iters = 200;
    const size_t real_n = (size_t)padx*pady;
    const size_t cplx_n = (size_t)(padx/2+1)*pady;
    printf("transform: %dx%d R2C, batch=%d, f32  (demag of %dx%dx1 grid)\n", padx, pady, batch, NX, NY);

    CK(cudaSetDevice(0));
    cuInit(0);
    CUdevice cuDev; cuDeviceGet(&cuDev, 0);
    CUcontext cuCtx; cuCtxGetCurrent(&cuCtx);
    if(!cuCtx){ cuDevicePrimaryCtxRetain(&cuCtx,cuDev); cuCtxSetCurrent(cuCtx); }

    // ---- cuFFT R2C batched ----
    float* d_real;  cufftComplex* d_cplx;
    CK(cudaMalloc(&d_real, batch*real_n*sizeof(float)));
    CK(cudaMalloc(&d_cplx, batch*cplx_n*sizeof(cufftComplex)));
    CK(cudaMemset(d_real, 0, batch*real_n*sizeof(float)));
    cufftHandle plan; int n[2]={pady,padx};
    CKF(cufftPlanMany(&plan,2,n,nullptr,1,(int)real_n,nullptr,1,(int)cplx_n,CUFFT_R2C,batch));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for(int i=0;i<20;i++) cufftExecR2C(plan,d_real,d_cplx);     // warmup
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for(int i=0;i<iters;i++) cufftExecR2C(plan,d_real,d_cplx);
    cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    float ms_cufft; cudaEventElapsedTime(&ms_cufft,e0,e1); ms_cufft/=iters;
    printf("  cuFFT  R2C batch=3: %.4f ms/transform\n", ms_cufft);

    // ---- VkFFT R2C batched (CUDA backend) ----
    // VkFFT R2C in-place: buffer must hold the complex output (padx/2+1)*2 reals per row.
    uint64_t bufBytes = (uint64_t)batch*cplx_n*sizeof(cufftComplex);
    float* d_vk; CK(cudaMalloc(&d_vk, bufBytes)); CK(cudaMemset(d_vk,0,bufBytes));
    VkFFTConfiguration conf = {};
    conf.FFTdim = 2; conf.size[0]=padx; conf.size[1]=pady;
    conf.numberBatches = batch;
    conf.performR2C = 1;
    conf.device = &cuDev;
    conf.bufferSize = &bufBytes;
    conf.buffer = (void**)&d_vk;
    VkFFTApplication app = {};
    CKV(initializeVkFFT(&app, conf));
    VkFFTLaunchParams lp = {};
    lp.buffer = (void**)&d_vk;
    for(int i=0;i<20;i++) CKV(VkFFTAppend(&app,-1,&lp));        // warmup (forward)
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for(int i=0;i<iters;i++) VkFFTAppend(&app,-1,&lp);
    cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    float ms_vk; cudaEventElapsedTime(&ms_vk,e0,e1); ms_vk/=iters;
    printf("  VkFFT  R2C batch=3: %.4f ms/transform\n", ms_vk);

    printf("  -> VkFFT is %.2fx %s than cuFFT\n", ms_cufft/ms_vk, (ms_vk<ms_cufft)?"FASTER":"slower");
    deleteVkFFT(&app); cufftDestroy(plan); cudaFree(d_real); cudaFree(d_cplx); cudaFree(d_vk);
    return 0;
}
