cmake_minimum_required(VERSION 3.24)

project(micromag
    VERSION 0.1.0
    DESCRIPTION "Micromagnetic simulator"
    LANGUAGES CXX
)
# CUDA language is enabled later (conditionally) via enable_language(CUDA)

# C++ standard --------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type --------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# UTF-8 source encoding (avoid C4819 on Korean Windows) ---------------------
# Restrict to CXX only — nvcc does not accept /utf-8 as a direct flag
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# Position-independent code (needed for pybind11) ---------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MSVC runtime: force the dynamic CRT (/MD) for ALL targets, including nvcc host
# compilation. nvcc defaults its host compiler to the STATIC CRT (/MT -> LIBCMT),
# which mixes two C runtimes in the CUDA Python module — iostream/locale globals
# get duplicated with separate locks, deadlocking std::ofstream (save_ovf /
# write_vtk_legacy hung on the GPU build). CMP0091 (NEW since cmake 3.24) makes
# this apply to CUDA host code too, so there is a single /MD runtime.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Output directories --------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options -------------------------------------------------------------------
option(MICROMAG_BUILD_PYTHON "Build Python bindings"     ON)
option(MICROMAG_BUILD_TESTS  "Build unit tests"          ON)
option(MICROMAG_USE_CUDA     "Enable CUDA/cuFFT GPU demag backend" OFF)
option(MICROMAG_FLOAT32      "Use float32 on GPU (reduces VRAM, ~1.5x bandwidth)" OFF)
option(MICROMAG_VKFFT        "Replace cuFFT batch plans with VkFFT (faster for non-pow2 sizes)" OFF)
set(MICROMAG_VKFFT_PATH "" CACHE PATH "Root directory of VkFFT (containing vkFFT/vkFFT.h)")

# FFTW3 (required for demagnetisation field) --------------------------------
find_package(FFTW3 CONFIG REQUIRED)

# OpenMP — parallelise CPU per-cell loops (field accumulate, integrator
# stages, demag pack/unpack). Optional: pragmas are no-ops without it.
find_package(OpenMP)

# Core library --------------------------------------------------------------
add_library(micromag_core
    src/grid.cpp
    src/field.cpp
    src/vtk_writer.cpp
    src/material.cpp
    src/material_field.cpp
    src/effective_field.cpp
    src/zeeman.cpp
    src/anisotropy.cpp
    src/exchange.cpp
    src/integrator.cpp
    src/spin_torque.cpp
    src/demag.cpp
    src/demag_periodic.cpp
    src/rkky.cpp
    src/zeeman_spatial.cpp
    src/thermal_field.cpp
    src/geom_mask.cpp
    src/mfm.cpp
    src/solver.cpp
    src/dmi.cpp
    src/ovf_io.cpp
    src/cubic_anisotropy.cpp
    src/region_map.cpp
    src/init_mag.cpp
    src/topological_charge.cpp
    src/skyrmion_tools.cpp
    src/surface_anisotropy.cpp
    src/magnetoelastic.cpp
)
target_include_directories(micromag_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(micromag_core PUBLIC cxx_std_20)
target_link_libraries(micromag_core PUBLIC FFTW3::fftw3)
if(OpenMP_CXX_FOUND)
    target_link_libraries(micromag_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "  OpenMP:        enabled (CPU loop parallelism)")
else()
    message(STATUS "  OpenMP:        NOT found (CPU loops run serially)")
endif()

# Demo executables ----------------------------------------------------------
add_executable(hello_micromag apps/hello_micromag.cpp)
target_link_libraries(hello_micromag PRIVATE micromag_core)

add_executable(field_demo apps/field_demo.cpp)
target_link_libraries(field_demo PRIVATE micromag_core)

add_executable(llg_demo apps/llg_demo.cpp)
target_link_libraries(llg_demo PRIVATE micromag_core)

add_executable(stt_sot_demo apps/stt_sot_demo.cpp)
target_link_libraries(stt_sot_demo PRIVATE micromag_core)

add_executable(sp1 apps/sp1.cpp)
target_link_libraries(sp1 PRIVATE micromag_core)

add_executable(sp1_phase apps/sp1_phase.cpp)
target_link_libraries(sp1_phase PRIVATE micromag_core)

add_executable(sp1_thickness apps/sp1_thickness.cpp)
target_link_libraries(sp1_thickness PRIVATE micromag_core)

add_executable(sp4 apps/sp4.cpp)
target_link_libraries(sp4 PRIVATE micromag_core)

add_executable(sp4_rk45 apps/sp4_rk45.cpp)
target_link_libraries(sp4_rk45 PRIVATE micromag_core)

add_executable(sp4_fieldB apps/sp4_fieldB.cpp)
target_link_libraries(sp4_fieldB PRIVATE micromag_core)

add_executable(thermal_equilibrium apps/thermal_equilibrium.cpp)
target_link_libraries(thermal_equilibrium PRIVATE micromag_core)

add_executable(thermal_neel_brown apps/thermal_neel_brown.cpp)
target_link_libraries(thermal_neel_brown PRIVATE micromag_core)

add_executable(sp4_thermal apps/sp4_thermal.cpp)
target_link_libraries(sp4_thermal PRIVATE micromag_core)

add_executable(sp3 apps/sp3.cpp)
target_link_libraries(sp3 PRIVATE micromag_core)

# Bloch DW width validation (Priority 2: µMAG validation extension)
add_executable(bloch_dw apps/bloch_dw.cpp)
target_link_libraries(bloch_dw PRIVATE micromag_core)

# CUDA/cuFFT GPU backend (Phase 3) -----------------------------------------
if(MICROMAG_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cufft curand)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Separate CUDA library — keeps nvcc and MSVC compilation units isolated
    add_library(micromag_cuda
        src/demag_cuda.cu
        src/demag_periodic_gpu.cu   # J5: GPU periodic-BC demag (no zero-padding)
        src/dmi_gpu.cu              # K2: GPU BulkDMI + InterfacialDMI
        src/relax_gpu.cu            # P4: RelaxGPU + MinimizeGPU
        src/exchange_gpu.cu      # G1: GPU exchange field (6-point Laplacian)
        src/field_kernels_gpu.cu # G2: GPU Zeeman + Uniaxial-Anisotropy
        src/gpu_state.cu         # G3: GPU-resident magnetization state
        src/rk4_gpu.cu           # G4+G5: LLG torque + RK4 stage kernels
        src/rk4_integrator_gpu.cu   # G6: RK4IntegratorGPU (full-GPU LLG step)
        src/heun_integrator_gpu.cu  # G8: HeunIntegratorGPU (GPU SLLG, cuRAND)
        src/rk45_integrator_gpu.cu  # G9: RK45IntegratorGPU (adaptive DOPRI5)
        src/spin_torque_gpu.cu      # P3: GPU STT/SOT/Zhang-Li spin torque kernels
        src/magnetoelastic_gpu.cu       # Phase S: GPU magnetoelastic field
        src/surface_anisotropy_gpu.cu   # Phase S: GPU surface anisotropy field
        src/zeeman_spatial_gpu.cu       # ZeemanFieldSpatialGPU — per-cell spatial Zeeman
        src/rkky_gpu.cu                 # RKKYFieldGPU — interlayer RKKY coupling GPU
    )
    target_include_directories(micromag_cuda
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(micromag_cuda
        PUBLIC
            micromag_core
            CUDA::cufft
            CUDA::curand
    )
    set_target_properties(micromag_cuda PROPERTIES
        CUDA_ARCHITECTURES native          # auto-detect installed GPU arch
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RUNTIME_LIBRARY Shared        # dynamic CUDA RT — avoids LNK4098
    )
    # Suppress C4819 (unicode) warnings in CUDA host compilation on Korean Windows.
    # /Zc:preprocessor enables MSVC's conformant preprocessor (required for CUB/Thrust
    # CCCL headers — without it, 100+ template errors occur in cub/device/device_reduce.cuh).
    if(MSVC)
        target_compile_options(micromag_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MD>          # dynamic CRT (match /MD host)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
    # Propagate the MICROMAG_CUDA flag so headers can guard GPU code
    target_compile_definitions(micromag_cuda PUBLIC MICROMAG_CUDA=1)
    target_compile_definitions(micromag_core PUBLIC MICROMAG_CUDA=1)
    # P11: optional float32 GPU mode.
    # PUBLIC on micromag_core/micromag_cuda propagates MICROMAG_FLOAT32 to every
    # consumer that links them (including the _micromag Python module created
    # later), so no per-target definition is needed below.
    if(MICROMAG_FLOAT32)
        target_compile_definitions(micromag_cuda PUBLIC MICROMAG_FLOAT32=1)
        target_compile_definitions(micromag_core PUBLIC MICROMAG_FLOAT32=1)
    endif()

    # SP#4 GPU benchmark app
    add_executable(sp4_gpu apps/sp4_gpu.cpp)
    target_link_libraries(sp4_gpu PRIVATE micromag_cuda)

    # Large-grid GPU demag benchmark
    add_executable(benchmark_large apps/benchmark_large.cpp)
    target_link_libraries(benchmark_large PRIVATE micromag_cuda)

    # G7: Full-GPU LLG benchmark (CPU RK4 vs GPU RK4IntegratorGPU)
    add_executable(sp4_full_gpu_bench apps/sp4_full_gpu_bench.cpp)
    target_link_libraries(sp4_full_gpu_bench PRIVATE micromag_cuda)
    if(MSVC)
        target_link_options(sp4_full_gpu_bench PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()

    # SP#4 GPU full 1 ns validation
    add_executable(sp4_gpu_1ns apps/sp4_gpu_1ns.cpp)
    target_link_libraries(sp4_gpu_1ns PRIVATE micromag_cuda)
    if(MSVC)
        target_link_options(sp4_gpu_1ns PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()

    # G9: RK45IntegratorGPU adaptive benchmark vs CPU + fixed-step GPU
    add_executable(sp4_rk45_gpu apps/sp4_rk45_gpu.cpp)
    target_link_libraries(sp4_rk45_gpu PRIVATE micromag_cuda)
    if(MSVC)
        target_link_options(sp4_rk45_gpu PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()

    # Large-grid full LLG scaling benchmark (SP#4 → Medium → 2.5M cells)
    add_executable(llg_large_bench apps/llg_large_bench.cpp)
    target_link_libraries(llg_large_bench PRIVATE micromag_cuda)
    if(MSVC)
        target_link_options(llg_large_bench PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()

    # Standalone GPU demag phase profiler (set MICROMAG_DEMAG_PROFILE=1 at runtime)
    add_executable(demag_profile apps/demag_profile.cpp)
    target_link_libraries(demag_profile PRIVATE micromag_cuda)
    if(MSVC)
        target_link_options(demag_profile PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    if(MSVC)
        target_link_options(benchmark_large PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    # Suppress LNK4098 (LIBCMT vs MSVCRT mismatch — harmless CUDA/MSVC quirk)
    if(MSVC)
        target_link_options(sp4_gpu PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()

    # P3: optional VkFFT backend (header-only; replaces cuFFT batch plans in demag)
    if(MICROMAG_VKFFT)
        if(MICROMAG_VKFFT_PATH STREQUAL "")
            message(FATAL_ERROR "MICROMAG_VKFFT=ON requires MICROMAG_VKFFT_PATH to point to the VkFFT root directory")
        endif()
        target_include_directories(micromag_cuda PRIVATE "${MICROMAG_VKFFT_PATH}")
        target_compile_definitions(micromag_cuda PRIVATE MICROMAG_VKFFT=1)
        # VkFFT CUDA backend requires nvrtc + cuda driver libraries
        find_library(NVRTC_LIB nvrtc HINTS "${CUDAToolkit_LIBRARY_DIR}" PATH_SUFFIXES x64)
        find_library(CUDA_DRV_LIB cuda HINTS "${CUDAToolkit_LIBRARY_DIR}" PATH_SUFFIXES x64)
        if(NOT NVRTC_LIB OR NOT CUDA_DRV_LIB)
            message(FATAL_ERROR "MICROMAG_VKFFT=ON: could not find nvrtc or cuda driver library in ${CUDAToolkit_LIBRARY_DIR}")
        endif()
        target_link_libraries(micromag_cuda PRIVATE "${NVRTC_LIB}" "${CUDA_DRV_LIB}")
        message(STATUS "  VkFFT:         enabled (${MICROMAG_VKFFT_PATH})")
    else()
        message(STATUS "  VkFFT:         OFF (cuFFT batch)")
    endif()

    message(STATUS "  CUDA toolkit:  ${CUDAToolkit_VERSION}")
    message(STATUS "  cuFFT:         found")
endif()

# Python bindings -----------------------------------------------------------
if(MICROMAG_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_micromag python/bindings.cpp)
    target_link_libraries(_micromag PRIVATE micromag_core)

    # When building with CUDA, expose GPU classes in the Python module too
    if(MICROMAG_USE_CUDA)
        target_link_libraries(_micromag PRIVATE micromag_cuda)
        if(MSVC)
            target_link_options(_micromag PRIVATE /NODEFAULTLIB:LIBCMT)
        endif()
    endif()

    # Place the compiled module next to the Python package directory.
    # Explicitly set per-config dirs to prevent MSVC from appending Release/Debug.
    set_target_properties(_micromag PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY           ${CMAKE_BINARY_DIR}/python
        LIBRARY_OUTPUT_DIRECTORY_RELEASE   ${CMAKE_BINARY_DIR}/python
        LIBRARY_OUTPUT_DIRECTORY_DEBUG     ${CMAKE_BINARY_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY           ${CMAKE_BINARY_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY_RELEASE   ${CMAKE_BINARY_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY_DEBUG     ${CMAKE_BINARY_DIR}/python
    )

    # Copy the Python package next to the compiled module
    add_custom_command(TARGET _micromag POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python/micromag
            ${CMAKE_BINARY_DIR}/python/micromag
    )
endif()

# Tests ---------------------------------------------------------------------
if(MICROMAG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Status summary ------------------------------------------------------------
message(STATUS "")
message(STATUS "micromag ${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:        ${MICROMAG_BUILD_PYTHON}")
message(STATUS "  Tests:         ${MICROMAG_BUILD_TESTS}")
message(STATUS "  CUDA:          ${MICROMAG_USE_CUDA}")
message(STATUS "")
