#pragma once

#include <string>

namespace micromag::gpu_probe {

// True when a CUDA device is present AND this build's embedded kernel images
// can actually run on it (a no-op kernel is launched once and the result is
// cached). Distinguishes "compiled with CUDA" from "runnable here" — a
// single-arch package on a mismatched GPU fails HERE, at probe time, instead
// of deep inside the first DemagFieldGPU constructor
// (see claude-sd-gpu-arch-issue_JYCho.md).
bool kernel_ok();

// Human-readable compatibility report, e.g.
//   "GPU: NVIDIA GeForce RTX 3080 (cc 8.6); build kernel archs: 1200
//    -> INCOMPATIBLE: no kernel image is available ... (rebuild with
//    -DMICROMAG_CUDA_ARCHS=release ...)"
std::string diagnostic();

}  // namespace micromag::gpu_probe
