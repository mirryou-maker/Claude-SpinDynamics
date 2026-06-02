cmake_minimum_required(VERSION 3.24)

project(micromag
    VERSION 0.1.0
    DESCRIPTION "Micromagnetic simulator"
    LANGUAGES CXX
)

# C++ standard --------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type --------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# UTF-8 source encoding (avoid C4819 on Korean Windows) ---------------------
if(MSVC)
    add_compile_options(/utf-8)
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
option(MICROMAG_USE_CUDA     "Enable CUDA backend (later phase)" OFF)

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

add_executable(sp4 apps/sp4.cpp)
target_link_libraries(sp4 PRIVATE micromag_core)

add_executable(sp4_rk45 apps/sp4_rk45.cpp)
target_link_libraries(sp4_rk45 PRIVATE micromag_core)

add_executable(sp4_fieldB apps/sp4_fieldB.cpp)
target_link_libraries(sp4_fieldB PRIVATE micromag_core)

# Python bindings -----------------------------------------------------------
if(MICROMAG_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_micromag python/bindings.cpp)
    target_link_libraries(_micromag PRIVATE micromag_core)

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
