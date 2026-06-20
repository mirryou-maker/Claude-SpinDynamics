// Extended VkFFT vs cuFFT feasibility spike.
// Tests R2C demag-sized transforms (batch=3, f32) across 2D and 3D sizes,
// including non-power-of-2 sizes found in actual µMAG benchmarks.
//
// Compile (run from benchmarks/perf/vkfft_spike/):
//   set VCPKG=C:\vcpkg\installed\x64-windows\include
//   set CL=%VCPKG%\.. (not needed, we pass -I directly)
//   nvcc spike2.cu -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe" ^
//        -I"C:\vcpkg\installed\x64-windows\include" ^
//        -lcufft -lnvrtc -lcuda -O2 --std=c++17 -Xcompiler "/wd4819 /wd4244" ^
//        -o spike2.exe
//
// Run:  spike2.exe
//
// Sizes tested:
//   2D pow2:     4096x1024, 2048x512, 1024x256 (perf sweep - all pow2)
//   2D non-pow2: 400x128   (SP#4 200x64x1)
//                200x100   (SP#1 100x50x1)
//                200x200   (SP#3 100x100x2 XY plane)
//                2000x1000 (large non-pow2)
//   3D pow2 Nz: 512x256x8, 256x128x16  (Nz pow2)
//   3D non-pow2 Nz: 200x200x4  (SP#3 actual: nz=2 -> pad_nz=4, 200 not pow2)
//                   400x200x10  (Nz=10 non-pow2)
//                   400x200x6   (Nz=6 non-pow2)
//                   512x256x10  (large, non-pow2 Nz only)
//                   512x256x6   (Nz=6 non-pow2)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>

#define VKFFT_BACKEND 1
#include "VkFFT/vkFFT.h"

#define CK(x)  do{ cudaError_t _e=(x); if(_e){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(_e));return;} }while(0)
#define CKF(x) do{ cufftResult _e=(x); if(_e){printf("cuFFT err %s @%d: %d\n",#x,__LINE__,(int)_e);return;} }while(0)
#define CKV(x) do{ VkFFTResult _e=(x); if(_e){printf("VkFFT err %s @%d: %d (see VkFFT.h)\n",#x,__LINE__,(int)_e);return;} }while(0)

struct TestCase {
    const char* label;
    int padx, pady, padz;  // actual padded FFT dimensions (pad_nz=1 for thin film)
};

// Helper: is a number a power of 2?
static bool ispow2(int n) { return n > 0 && (n & (n-1)) == 0; }
static const char* pow2tag(int x, int y, int z) {
    bool px=ispow2(x), py=ispow2(y), pz=ispow2(z)||z==1;
    if(px&&py&&pz) return " [all-pow2]";
    if(!px&&!py) return " [xy non-pow2]";
    if(!px) return " [x non-pow2]";
    if(!py) return " [y non-pow2]";
    if(!pz) return " [z non-pow2]";
    return "";
}

static void bench_size(CUdevice cuDev, int padx, int pady, int padz,
                       const char* label, int iters=200) {
    const int batch = 3;
    const int dim = (padz == 1) ? 2 : 3;
    // R2C: x is innermost (fastest), so output x-size is padx/2+1
    const size_t real_n  = (size_t)padx * pady * padz;
    const size_t cplx_n  = (size_t)(padx/2+1) * pady * padz;

    printf("\n  %s  %dx%dx%d  dim=%dD  batch=%d%s\n",
           label, padx, pady, padz, dim, batch, pow2tag(padx,pady,padz));
    printf("  real_n=%zu  cplx_n=%zu  mem=%.1f MB\n",
           real_n, cplx_n,
           (batch*real_n*sizeof(float) + batch*cplx_n*sizeof(float)*2) / 1e6);

    // Allocate cuFFT buffers
    float* d_real = nullptr; float* d_cplx = nullptr;
    if(cudaMalloc(&d_real, batch*real_n*sizeof(float)) != cudaSuccess ||
       cudaMalloc(&d_cplx, batch*cplx_n*2*sizeof(float)) != cudaSuccess) {
        printf("  [SKIP] cudaMalloc failed (OOM?)\n");
        if(d_real) cudaFree(d_real);
        if(d_cplx) cudaFree(d_cplx);
        return;
    }
    cudaMemset(d_real, 0, batch*real_n*sizeof(float));

    // cuFFT plan
    cufftHandle plan;
    {
        int n[3]; int rank;
        // cuFFT uses C order: slowest-first, so [padz, pady, padx]
        if(dim == 2) { n[0]=pady; n[1]=padx; rank=2; }
        else         { n[0]=padz; n[1]=pady; n[2]=padx; rank=3; }
        cufftResult cr = cufftPlanMany(&plan, rank, n,
            nullptr, 1, (int)real_n,
            nullptr, 1, (int)cplx_n,
            CUFFT_R2C, batch);
        if(cr != CUFFT_SUCCESS) {
            printf("  cuFFT plan FAILED (%d) — size may be unsupported\n", (int)cr);
            cudaFree(d_real); cudaFree(d_cplx);
            return;
        }
    }

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Warmup cuFFT
    for(int i=0;i<20;i++)
        cufftExecR2C(plan, d_real, (cufftComplex*)d_cplx);
    cudaDeviceSynchronize();

    // Time cuFFT
    cudaEventRecord(e0);
    for(int i=0;i<iters;i++)
        cufftExecR2C(plan, d_real, (cufftComplex*)d_cplx);
    cudaEventRecord(e1);
    cudaDeviceSynchronize();
    float ms_cufft; cudaEventElapsedTime(&ms_cufft, e0, e1); ms_cufft /= iters;
    printf("    cuFFT  R2C: %7.4f ms\n", ms_cufft);

    cufftDestroy(plan);

    // VkFFT plan
    uint64_t bufBytes = (uint64_t)batch * cplx_n * 2 * sizeof(float);
    float* d_vk = nullptr;
    if(cudaMalloc(&d_vk, bufBytes) != cudaSuccess) {
        printf("  [SKIP] VkFFT cudaMalloc failed\n");
        cudaFree(d_real); cudaFree(d_cplx);
        return;
    }
    cudaMemset(d_vk, 0, bufBytes);

    VkFFTConfiguration conf = {};
    conf.FFTdim      = (uint64_t)dim;
    conf.size[0]     = (uint64_t)padx;
    conf.size[1]     = (uint64_t)pady;
    if(dim == 3) conf.size[2] = (uint64_t)padz;
    conf.numberBatches = (uint64_t)batch;
    conf.performR2C  = 1;
    conf.device      = &cuDev;
    conf.bufferSize  = &bufBytes;
    conf.buffer      = (void**)&d_vk;

    VkFFTApplication app = {};
    VkFFTResult vr = initializeVkFFT(&app, conf);
    if(vr != VKFFT_SUCCESS) {
        printf("    VkFFT  init FAILED (%d) — size unsupported\n", (int)vr);
        cudaFree(d_real); cudaFree(d_cplx); cudaFree(d_vk);
        return;
    }

    VkFFTLaunchParams lp = {};
    lp.buffer = (void**)&d_vk;

    // Warmup VkFFT
    for(int i=0;i<20;i++) VkFFTAppend(&app, -1, &lp);
    cudaDeviceSynchronize();

    // Time VkFFT
    cudaEventRecord(e0);
    for(int i=0;i<iters;i++) VkFFTAppend(&app, -1, &lp);
    cudaEventRecord(e1);
    cudaDeviceSynchronize();
    float ms_vk; cudaEventElapsedTime(&ms_vk, e0, e1); ms_vk /= iters;
    printf("    VkFFT  R2C: %7.4f ms  ->  VkFFT is %.2fx %s\n",
           ms_vk, ms_cufft / ms_vk,
           (ms_vk < ms_cufft) ? "FASTER" : "slower");

    deleteVkFFT(&app);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_real); cudaFree(d_cplx); cudaFree(d_vk);
}

int main() {
    cudaSetDevice(0);
    cuInit(0);
    CUdevice cuDev; cuDeviceGet(&cuDev, 0);
    CUcontext cuCtx; cuCtxGetCurrent(&cuCtx);
    if(!cuCtx){ cuDevicePrimaryCtxRetain(&cuCtx, cuDev); cuCtxSetCurrent(cuCtx); }

    char name[256]; cuDeviceGetName(name, 256, cuDev);
    int sm_major, sm_minor;
    cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDev);
    cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDev);
    printf("GPU: %s (sm_%d%d)\n", name, sm_major, sm_minor);
    printf("VkFFT version: %d\n\n", (int)VkFFTGetVersion());

    // ──────────────────────────────────────────────────────────────────
    // 2D thin-film cases  (pad_nz = 1, FFT is purely 2D)
    // ──────────────────────────────────────────────────────────────────
    printf("═══ 2D THIN-FILM CASES (pad_nz=1) ══════════════════════════════\n");

    // Power-of-2 baseline (from original spike)
    bench_size(cuDev, 4096, 1024, 1, "2D pow2 [large]     ");
    bench_size(cuDev, 2048,  512, 1, "2D pow2 [medium]    ");
    bench_size(cuDev, 1024,  256, 1, "2D pow2 [small]     ");

    // Non-power-of-2 — actual µMAG benchmark grids
    // SP#4: 200x64x1 grid  -> demag pad: 400x128x1
    bench_size(cuDev,  400,  128, 1, "2D SP#4 [200x64x1]  ");
    // SP#1: 100x50x1 grid  -> demag pad: 200x100x1
    bench_size(cuDev,  200,  100, 1, "2D SP#1 [100x50x1]  ");
    // SP#3 XY: 100x100x2   -> demag pad: 200x200 per layer
    bench_size(cuDev,  200,  200, 1, "2D SP#3 [100x100,xy]");
    // Large non-pow2 (1000x500 grid)
    bench_size(cuDev, 2000, 1000, 1, "2D non-pow2 [large] ");
    // Medium non-pow2 (300x150)
    bench_size(cuDev,  600,  300, 1, "2D non-pow2 [300x150]");

    // ──────────────────────────────────────────────────────────────────
    // 3D cases with power-of-2 Nz (baseline for 3D)
    // ──────────────────────────────────────────────────────────────────
    printf("\n═══ 3D CASES — POWER-OF-2 Nz ════════════════════════════════════\n");
    bench_size(cuDev,  512, 256,  8, "3D pow2 [256x128x4] ");
    bench_size(cuDev,  256, 128, 16, "3D pow2 [128x64x8]  ");
    bench_size(cuDev,  512, 256,  4, "3D pow2 [256x128x2] ");

    // ──────────────────────────────────────────────────────────────────
    // 3D cases with NON-power-of-2 Nz  (user request)
    // ──────────────────────────────────────────────────────────────────
    printf("\n═══ 3D CASES — NON-POWER-OF-2 Nz ═══════════════════════════════\n");
    // SP#3 actual: 100x100x2 grid  -> pad: 200x200x4 (Nz pow2, but 200 non-pow2)
    bench_size(cuDev,  200, 200,  4, "3D SP#3 [100x100x2] ");
    // Nz=5 -> pad_nz=10 (non-pow2)
    bench_size(cuDev,  400, 200, 10, "3D [200x100x5]      ");
    // Nz=3 -> pad_nz=6 (non-pow2)
    bench_size(cuDev,  400, 200,  6, "3D [200x100x3]      ");
    // Nz=7 -> pad_nz=14 (non-pow2)
    bench_size(cuDev,  400, 200, 14, "3D [200x100x7]      ");
    // Large: pow2 XY, non-pow2 Nz
    bench_size(cuDev,  512, 256, 10, "3D pow2-XY Nz=10    ");
    bench_size(cuDev,  512, 256,  6, "3D pow2-XY Nz=6     ");
    bench_size(cuDev,  512, 256, 14, "3D pow2-XY Nz=14    ");
    // All-non-pow2: 200x200x10
    bench_size(cuDev,  200, 200, 10, "3D all-non-pow2 x10 ");
    bench_size(cuDev,  200, 200,  6, "3D all-non-pow2 x6  ");
    // Larger 3D non-pow2 Nz
    bench_size(cuDev, 1024, 512, 10, "3D large Nz=10      ");
    bench_size(cuDev, 1024, 512,  6, "3D large Nz=6       ");

    printf("\nDone.\n");
    return 0;
}
