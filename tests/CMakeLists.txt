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
    diag_step1.cpp
    test_rk45.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        micromag_core
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(unit_tests)
