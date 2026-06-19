find_package(Catch2 3 CONFIG REQUIRED)

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
