#include "micromag/gpu_probe.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <string>

namespace micromag::gpu_probe {
namespace {

__global__ void noop_kernel() {}

// Variadic: __CUDA_ARCH_LIST__ expands with commas ("750,800,...,1200").
#define MM_STR2(...) #__VA_ARGS__
#define MM_STR(...) MM_STR2(__VA_ARGS__)

// nvcc (>= 11.5) defines __CUDA_ARCH_LIST__ as the comma-separated list of
// architectures this TU is compiled for — i.e. exactly what is embedded in
// the fatbinary (e.g. "890,1200").
const char* arch_list() {
#ifdef __CUDA_ARCH_LIST__
    return MM_STR(__CUDA_ARCH_LIST__);
#else
    return "unknown";
#endif
}

struct ProbeResult {
    bool        ok;
    std::string diag;
};

ProbeResult run_probe() {
    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err != cudaSuccess || ndev == 0) {
        return {false, std::string("No usable CUDA device: ") +
                       (err == cudaSuccess ? "0 devices" : cudaGetErrorString(err))};
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    std::ostringstream os;
    os << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor
       << "); build kernel archs: " << arch_list();

    noop_kernel<<<1, 1>>>();
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        os << " -> INCOMPATIBLE: " << cudaGetErrorString(err)
           << ". This package's kernels do not cover cc " << prop.major << "."
           << prop.minor << " — rebuild with -DMICROMAG_CUDA_ARCHS=release "
           << "(or an arch list containing " << prop.major << prop.minor << ").";
        return {false, os.str()};
    }
    os << " -> OK";
    return {true, os.str()};
}

const ProbeResult& cached() {
    static const ProbeResult r = run_probe();   // thread-safe magic static
    return r;
}

}  // namespace

bool kernel_ok() { return cached().ok; }

std::string diagnostic() { return cached().diag; }

}  // namespace micromag::gpu_probe
