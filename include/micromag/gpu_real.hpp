#pragma once

// gpu_real.hpp — P11: GPU scalar type selector.
//
// MICROMAG_FLOAT32=ON (cmake option): GPU kernels use float (25–50% bandwidth gain).
// Default (OFF): GPU kernels use double (full precision, current behaviour).
//
// CPU interface (upload/download) always stays double; conversion happens inside
// GPUMagState at the staging buffer boundary.
//
// cuFFT macro aliases allow demag_cuda.cu / demag_periodic_gpu.cu to swap
// D2Z/Z2D ↔ R2C/C2R plans with zero conditional-compilation boilerplate.

// ----- cuFFT type / plan aliases (preprocessor macros, global scope) -------
#ifdef MICROMAG_FLOAT32
#  define GREAL_CUFFT_TYPE     CUFFT_R2C
#  define GREAL_CUFFT_ITYPE    CUFFT_C2R
#  define GREAL_CUFFT_REAL     cufftReal
#  define GREAL_CUFFT_COMPLEX  cufftComplex
#  define GREAL_CUFFT_EXEC_FWD cufftExecR2C
#  define GREAL_CUFFT_EXEC_INV cufftExecC2R
// cuComplex arithmetic (float-complex)
#  define GREAL_CUFFT_CADD     cuCaddf
#  define GREAL_CUFFT_CMUL     cuCmulf
#else
#  define GREAL_CUFFT_TYPE     CUFFT_D2Z
#  define GREAL_CUFFT_ITYPE    CUFFT_Z2D
#  define GREAL_CUFFT_REAL     cufftDoubleReal
#  define GREAL_CUFFT_COMPLEX  cufftDoubleComplex
#  define GREAL_CUFFT_EXEC_FWD cufftExecD2Z
#  define GREAL_CUFFT_EXEC_INV cufftExecZ2D
// cuDoubleComplex arithmetic (double-complex)
#  define GREAL_CUFFT_CADD     cuCadd
#  define GREAL_CUFFT_CMUL     cuCmul
#endif

// ----- C++ type alias inside micromag namespace ----------------------------
namespace micromag {

#ifdef MICROMAG_FLOAT32
using GReal = float;
#else
using GReal = double;
#endif

} // namespace micromag
