#pragma once

// Forward declaration keeps <fftw3.h> out of public headers.
struct fftw_plan_s;

namespace micromag {

// Move-only RAII owner of an FFTW plan: fftw_destroy_plan on destruction,
// so plan cleanup is exception-safe and needs no manual destructor code.
// reset() is defined in src/fftw_plan_handle.cpp (the only place that needs
// the full fftw3.h).
class FFTWPlan {
public:
    FFTWPlan() noexcept = default;
    explicit FFTWPlan(fftw_plan_s* p) noexcept : p_(p) {}
    ~FFTWPlan() { reset(); }

    FFTWPlan(FFTWPlan&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    FFTWPlan& operator=(FFTWPlan&& o) noexcept {
        if (this != &o) { reset(); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }
    FFTWPlan(const FFTWPlan&)            = delete;
    FFTWPlan& operator=(const FFTWPlan&) = delete;

    void reset() noexcept;                                  // fftw_destroy_plan
    fftw_plan_s* get() const noexcept { return p_; }        // for fftw_execute
    explicit operator bool() const noexcept { return p_ != nullptr; }

private:
    fftw_plan_s* p_ = nullptr;
};

}  // namespace micromag
