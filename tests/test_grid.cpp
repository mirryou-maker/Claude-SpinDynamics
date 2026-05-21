#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "micromag/grid.hpp"

using namespace micromag;
using Catch::Matchers::WithinAbs;

TEST_CASE("StructuredGrid basic dimensions", "[grid]") {
    StructuredGrid g(4, 5, 6, 1.0, 2.0, 3.0);
    REQUIRE(g.nx() == 4);
    REQUIRE(g.ny() == 5);
    REQUIRE(g.nz() == 6);
    REQUIRE(g.size() == 120);
    REQUIRE_THAT(g.cell_volume(), WithinAbs(6.0, 1e-12));
}

TEST_CASE("StructuredGrid cell centers", "[grid]") {
    StructuredGrid g(2, 2, 2, 1.0, 1.0, 1.0);
    auto c000 = g.cell_center(0, 0, 0);
    REQUIRE_THAT(c000.x, WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(c000.y, WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(c000.z, WithinAbs(0.5, 1e-12));

    auto c111 = g.cell_center(1, 1, 1);
    REQUIRE_THAT(c111.x, WithinAbs(1.5, 1e-12));
}

TEST_CASE("StructuredGrid linear indexing", "[grid]") {
    StructuredGrid g(3, 4, 5, 1.0, 1.0, 1.0);
    REQUIRE(g.linear_index(0, 0, 0) == 0);
    REQUIRE(g.linear_index(1, 0, 0) == 1);
    REQUIRE(g.linear_index(0, 1, 0) == 3);   // = nx
    REQUIRE(g.linear_index(0, 0, 1) == 12);  // = nx*ny
    REQUIRE(g.linear_index(2, 3, 4) == 2 + 3 * 3 + 4 * 3 * 4);
}

TEST_CASE("StructuredGrid rejects bad dimensions", "[grid]") {
    REQUIRE_THROWS_AS(StructuredGrid(0, 1, 1, 1, 1, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(StructuredGrid(1, 1, 1, -1, 1, 1), std::invalid_argument);
}
