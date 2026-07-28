// Minimal HIP-over-CUDA validation of the Phase 0 seam runtime + RNG arms.
// Compiled with: HIP_PLATFORM=nvidia hipcc -DMICROMAG_CUDA=1
//                -DMICROMAG_GPU_BACKEND_HIP=1 -I<repo>/include ...
// Exercises gpu_backend.hpp (mg::malloc/memcpy/device_sync/GPU_LAUNCH) and
// gpu_rng.hpp (gpu::philox_normal3) through the HIP arm.
#include <cstdio>
#include <cmath>
#include <vector>

#include "micromag/gpu_backend.hpp"
#include "micromag/gpu_rng.hpp"

namespace mg = micromag::gpu;

__global__ void fill_normals(double* out, int n, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double e0, e1, e2;
    mg::philox_normal3(seed, (unsigned long long)i, 0ull, e0, e1, e2);
    out[3*i+0] = e0; out[3*i+1] = e1; out[3*i+2] = e2;
}

int main() {
    const int R = 4096;
    double* d = static_cast<double*>(mg::malloc(size_t(R)*3*sizeof(double)));
    GPU_LAUNCH(fill_normals, (R+255)/256, 256, 0, 0, d, R, 12345u);
    mg::check_last("fill_normals");
    mg::device_sync();
    std::vector<double> h(size_t(R)*3);
    mg::memcpy(h.data(), d, h.size()*sizeof(double), mg::MemcpyKind::D2H);
    mg::free(d);
    // sample mean / variance of N(0,1) draws
    double m = 0, v = 0;
    for (double x : h) m += x;
    m /= h.size();
    for (double x : h) v += (x-m)*(x-m);
    v /= h.size();
    printf("HIP-over-CUDA seam OK: N=%zu draws  mean=%.4f (~0)  var=%.4f (~1)\n",
           h.size(), m, v);
    printf("first 3: %.4f %.4f %.4f\n", h[0], h[1], h[2]);
    return (std::fabs(m) < 0.05 && std::fabs(v-1.0) < 0.1) ? 0 : 1;
}
