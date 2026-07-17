#pragma once

// Guarded main() for console apps: an uncaught exception (e.g. a CUDA error
// from a kernel-arch mismatch) must never become an abort()/0xC0000409
// fail-fast that closes the console with no message. Rename the real entry
// point to `static int run_main()` and add `MICROMAG_GUARDED_MAIN(run_main)`.
// (Motivated by claude-sd-gpu-arch-issue_JYCho.md: sp4_gpu.exe printed its CPU
// section then died silently at the first GPU call on non-sm_120 hardware.)

#include <cstdio>
#include <exception>

#define MICROMAG_GUARDED_MAIN(fn)                                          \
    int main() {                                                           \
        try {                                                              \
            return fn();                                                   \
        } catch (const std::exception& e) {                                \
            std::fflush(stdout);                                           \
            std::fprintf(stderr, "\nFATAL: %s\n", e.what());               \
            return 1;                                                      \
        } catch (...) {                                                    \
            std::fflush(stdout);                                           \
            std::fprintf(stderr, "\nFATAL: unknown exception\n");          \
            return 2;                                                      \
        }                                                                  \
    }
