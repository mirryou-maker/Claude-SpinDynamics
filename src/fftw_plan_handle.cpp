#include "micromag/fftw_plan_handle.hpp"

#include <fftw3.h>

namespace micromag {

void FFTWPlan::reset() noexcept {
    if (p_) {
        fftw_destroy_plan(reinterpret_cast<fftw_plan>(p_));
        p_ = nullptr;
    }
}

}  // namespace micromag
