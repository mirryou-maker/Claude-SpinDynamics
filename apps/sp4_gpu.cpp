// SP#4 GPU benchmark (Phase 3 — Step 1 stub)
// Real implementation in Step 6 after demag_cuda.cu is complete.

#ifdef MICROMAG_CUDA
#include <iostream>
#include "micromag/demag_gpu.hpp"

int main() {
    std::cout << "SP#4 GPU benchmark\n";
    std::cout << "DemagFieldGPU stub compiled successfully.\n";
    std::cout << "Full implementation pending (Step 4).\n";
    return 0;
}

#else
#include <iostream>
int main() {
    std::cout << "Rebuild with -DMICROMAG_USE_CUDA=ON to enable GPU support.\n";
    return 1;
}
#endif
