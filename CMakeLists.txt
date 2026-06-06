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

# Output directories --------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options -------------------------------------------------------------------
option(MICROMAG_BUILD_PYTHON "Build Python bindings"     ON)
option(MICROMAG_BUILD_TESTS  "Build unit tests"          ON)
option(MICROMAG_USE_CUDA     "Enable CUDA/cuFFT GPU demag backend" OFF)

# FFTW3 (required for demagnetisation field) --------------------------------
find_package(FFTW3 CONFIG REQUIRED)

# Core library --------------------------------------------------------------
add_library(micromag_core
    src/grid.cpp
    src/field.cpp
    src/vtk_writer.cpp
    src/material.cpp
    src/effective_field.cpp
    src/zeeman.cpp
    src/anisotropy.cpp
    src/exchange.cpp
    src/integrator.cpp
    src/spin_torque.cpp
    src/demag.cpp
    src/demag_periodic.cpp
    src/rkky.cpp
    src/thermal_field.cpp
)
target_include_directories(micromag_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(micromag_core PUBLIC cxx_std_20)
target_link_libraries(micromag_core PUBLIC FFTW3::fftw3)

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

# CUDA/cuFFT GPU backend (Phase 3) -----------------------------------------
if(MICROMAG_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cufft curand)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Separate CUDA library — keeps nvcc and MSVC compilation units isolated
    add_library(micromag_cuda
        src/demag_cuda.cu
        src/exchange_gpu.cu      # G1: GPU exchange field (6-point Laplacian)
        src/field_kernels_gpu.cu # G2: GPU Zeeman + Uniaxial-Anisotropy
        src/gpu_state.cu         # G3: GPU-resident magnetization state
        src/rk4_gpu.cu           # G4+G5: LLG torque + RK4 stage kernels
        src/rk4_integrator_gpu.cu   # G6: RK4IntegratorGPU (full-GPU LLG step)
        src/heun_integrator_gpu.cu  # G8: HeunIntegratorGPU (GPU SLLG, cuRAND)
        src/rk45_integrator_gpu.cu  # G9: RK45IntegratorGPU (adaptive DOPRI5)
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
    # Suppress C4819 (unicode) warnings in CUDA host compilation on Korean Windows
    if(MSVC)
        target_compile_options(micromag_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    # Propagate the MICROMAG_CUDA flag so headers can guard GPU code
    target_compile_definitions(micromag_cuda PUBLIC MICROMAG_CUDA=1)
    target_compile_definitions(micromag_core PUBLIC MICROMAG_CUDA=1)

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
    if(MSVC)
        target_link_options(benchmark_large PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    # Suppress LNK4098 (LIBCMT vs MSVCRT mismatch — harmless CUDA/MSVC quirk)
    if(MSVC)
        target_link_options(sp4_gpu PRIVATE /NODEFAULTLIB:LIBCMT)
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
