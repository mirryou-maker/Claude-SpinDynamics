# Catch2 v3: prefer an installed config (vcpkg / system). If absent (e.g. a Linux
# box without catch2 installed), fetch and build it from source so the tests
# still build without a system package.
find_package(Catch2 3 CONFIG QUIET)
if(NOT Catch2_FOUND)
    # NOTE: on bleeding-edge GCC 15 (e.g. Ubuntu 26.04) Catch2's own Release LTO
    # can fail to link the test exes (undefined reference to __cxa_call_terminate).
    # The library + Python module build fine regardless; on Ubuntu 24.04 LTS
    # (GCC 13) the tests build and pass. See benchmarks/linux_cpu_parity.md.
    message(STATUS "  Catch2:        not found — fetching v3.5.2 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)  # for catch_discover_tests
endif()

add_executable(unit_tests
    test_grid.cpp
    test_field.cpp
    test_zeeman.cpp
    test_anisotropy.cpp
    test_exchange.cpp
    test_llg.cpp
    test_spin_torque.cpp
    test_demag.cpp
    test_demag_periodic.cpp
    test_rkky.cpp
    test_zeeman_spatial.cpp
    diag_step1.cpp
    test_rk45.cpp
    test_thermal.cpp
    test_geom.cpp
    test_mfm.cpp
    test_material_field.cpp
    test_dmi.cpp
    test_phase_e.cpp
    test_new_apis.cpp
    test_magnetoelastic.cpp
    test_surface_anisotropy.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        micromag_core
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(unit_tests)

# GPU unit tests — only built when MICROMAG_USE_CUDA=ON
if(MICROMAG_USE_CUDA)
    add_executable(unit_tests_gpu
        test_demag_gpu.cpp
        test_demag_periodic_gpu.cpp     # J5: DemagFieldPeriodicGPU tests
        test_exchange_gpu.cpp        # G1: GPU exchange field tests
        test_field_kernels_gpu.cpp  # G2: GPU Zeeman + Anisotropy tests
        test_gpu_state.cpp          # G3: GPUMagState tests
        test_rk4_gpu.cpp                # G4+G5: LLG torque + RK4 stage tests
        test_rk4_integrator_gpu.cpp     # G6: RK4IntegratorGPU accuracy tests
        test_heun_integrator_gpu.cpp    # G8: HeunIntegratorGPU (SLLG) tests
        test_dmi_gpu.cpp                # K2: BulkDMIFieldGPU + InterfacialDMIFieldGPU tests
        test_relax_gpu.cpp              # P4: RelaxGPU + MinimizeGPU tests
        test_field_sum_gpu.cpp          # P2: FieldSumGPU compositor tests
        test_spin_torque_gpu.cpp        # P3: GPU STT/SOT/Zhang-Li tests
        test_new_apis_gpu.cpp           # Priority 1-2: per-cell material, RKKYFieldGPU, ZeemanSpatialGPU
        test_magnetoelastic_gpu.cpp     # MagnetoelasticFieldGPU vs CPU (B1/B2)
        test_surface_anisotropy_gpu.cpp # SurfaceAnisotropyFieldGPU vs CPU (interface PMA)
    )
    target_link_libraries(unit_tests_gpu
        PRIVATE
            micromag_cuda           # transitively includes micromag_core
            Catch2::Catch2WithMain
    )
    # Suppress LNK4098 (same as sp4_gpu)
    if(MSVC)
        target_link_options(unit_tests_gpu PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    catch_discover_tests(unit_tests_gpu)
endif()
