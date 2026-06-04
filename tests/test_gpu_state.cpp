// test_gpu_state.cpp — G3: GPUMagState unit tests
// Tag: [gpu_state][gpu]

#ifdef MICROMAG_CUDA

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "micromag/field.hpp"
#include "micromag/grid.hpp"
#include "micromag/gpu_state.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// T1: constructor smoke test
// ---------------------------------------------------------------------------
TEST_CASE("GPUMagState: constructor allocates buffers", "[gpu_state][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    REQUIRE(state.N() == 64u);
    REQUIRE(state.d_m()     != nullptr);
    REQUIRE(state.d_H()     != nullptr);
    REQUIRE(state.d_m0()    != nullptr);
    REQUIRE(state.d_ki()    != nullptr);
    REQUIRE(state.d_k_acc() != nullptr);
    REQUIRE(state.stream()  != nullptr);
}

// ---------------------------------------------------------------------------
// T2: upload → download roundtrip (exact double precision)
// ---------------------------------------------------------------------------
TEST_CASE("GPUMagState: upload/download roundtrip", "[gpu_state][gpu]") {
    StructuredGrid g(8, 6, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    // Fill with non-trivial values
    VectorField3D m_in(g);
    for (Index iz=0; iz<g.nz(); ++iz)
    for (Index iy=0; iy<g.ny(); ++iy)
    for (Index ix=0; ix<g.nx(); ++ix) {
        double phi = ix*0.3 + iy*0.2 + iz*0.1;
        m_in.at(ix,iy,iz) = {std::cos(phi), std::sin(phi), 0.0};
    }

    state.upload(m_in);

    VectorField3D m_out(g);
    for (Index i=0; i<g.size(); ++i) m_out[i] = {0,0,0};
    state.download(m_out);

    // Should recover exactly (no computation, pure data transfer)
    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(m_out[i].x, WithinAbs(m_in[i].x, 1e-15));
        REQUIRE_THAT(m_out[i].y, WithinAbs(m_in[i].y, 1e-15));
        REQUIRE_THAT(m_out[i].z, WithinAbs(m_in[i].z, 1e-15));
    }
}

// ---------------------------------------------------------------------------
// T3: zero_H → download_H gives zero field
// ---------------------------------------------------------------------------
TEST_CASE("GPUMagState: zero_H clears H buffer", "[gpu_state][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    // First upload some data as m (so d_m_ is non-zero), then zero d_H_
    VectorField3D m(g); m.set_uniform({1, 0, 0});
    state.upload(m);
    state.zero_H();
    state.sync();

    VectorField3D H(g);
    state.download_H(H);

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(H[i].x, WithinAbs(0.0, 1e-30));
        REQUIRE_THAT(H[i].y, WithinAbs(0.0, 1e-30));
        REQUIRE_THAT(H[i].z, WithinAbs(0.0, 1e-30));
    }
}

// ---------------------------------------------------------------------------
// T4: save_m0 → d_m0 matches d_m at call time; later upload doesn't affect d_m0
// ---------------------------------------------------------------------------
TEST_CASE("GPUMagState: save_m0 snapshots d_m", "[gpu_state][gpu]") {
    StructuredGrid g(6, 6, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    // Upload initial m (+z), save snapshot
    VectorField3D m_init(g); m_init.set_uniform({0, 0, 1});
    state.upload(m_init);
    state.save_m0();
    state.sync();

    // Overwrite d_m with +x
    VectorField3D m_new(g); m_new.set_uniform({1, 0, 0});
    state.upload(m_new);

    // d_m should now be +x
    VectorField3D m_check(g);
    state.download(m_check);
    REQUIRE_THAT(m_check[0].x, WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(m_check[0].z, WithinAbs(0.0, 1e-15));

    // d_m0 should still hold the +z snapshot
    VectorField3D m0_check(g);
    state.download_m0(m0_check);
    REQUIRE_THAT(m0_check[0].x, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(m0_check[0].y, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(m0_check[0].z, WithinAbs(1.0, 1e-15));
}

// ---------------------------------------------------------------------------
// T5: zero_k_acc → download_k_acc gives zero field
// ---------------------------------------------------------------------------
TEST_CASE("GPUMagState: zero_k_acc clears accumulator", "[gpu_state][gpu]") {
    StructuredGrid g(4, 4, 2, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    // Pre-fill d_k_acc via upload+save trick:
    //   upload non-zero m → d_m, then copy to k_acc indirectly.
    // Since we have no direct upload-to-k_acc, we first fill d_m then
    // copy to d_ki (which we can observe via download_m0 after save_m0).
    // Simplest: just call zero_k_acc on a freshly constructed state and
    // verify download_k_acc gives zeros (tests the memset path).
    state.zero_k_acc();
    state.sync();

    VectorField3D k(g);
    state.download_k_acc(k);

    for (Index i = 0; i < g.size(); ++i) {
        REQUIRE_THAT(k[i].x, WithinAbs(0.0, 1e-30));
        REQUIRE_THAT(k[i].y, WithinAbs(0.0, 1e-30));
        REQUIRE_THAT(k[i].z, WithinAbs(0.0, 1e-30));
    }
}

// ---------------------------------------------------------------------------
// T6: all pointers are distinct (no aliasing)
// ---------------------------------------------------------------------------
TEST_CASE("GPUMagState: GPU buffers are non-aliasing", "[gpu_state][gpu]") {
    StructuredGrid g(4, 4, 4, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    REQUIRE(state.d_m()     != state.d_H());
    REQUIRE(state.d_m()     != state.d_m0());
    REQUIRE(state.d_m()     != state.d_ki());
    REQUIRE(state.d_m()     != state.d_k_acc());
    REQUIRE(state.d_H()     != state.d_m0());
    REQUIRE(state.d_H()     != state.d_ki());
    REQUIRE(state.d_H()     != state.d_k_acc());
    REQUIRE(state.d_m0()    != state.d_ki());
    REQUIRE(state.d_m0()    != state.d_k_acc());
    REQUIRE(state.d_ki()    != state.d_k_acc());
}

// ---------------------------------------------------------------------------
// T7: large grid — construction succeeds, upload/download correct
// ---------------------------------------------------------------------------
TEST_CASE("GPUMagState: 200×200×5 large grid", "[gpu_state][gpu]") {
    StructuredGrid g(200, 200, 5, 5e-9, 5e-9, 5e-9);
    GPUMagState state(g);

    REQUIRE(state.N() == 200u*200u*5u);

    VectorField3D m(g); m.set_uniform({0.6, 0.8, 0.0});
    state.upload(m);
    VectorField3D m2(g);
    state.download(m2);

    REQUIRE_THAT(m2[0].x, WithinAbs(0.6, 1e-15));
    REQUIRE_THAT(m2[0].y, WithinAbs(0.8, 1e-15));
    REQUIRE_THAT(m2[0].z, WithinAbs(0.0, 1e-15));
}

#endif // MICROMAG_CUDA
