#include "micromag/material.hpp"

namespace micromag {

Material Material::permalloy() {
    return {8.0e5, 1.3e-11, 0.0, {0, 0, 1}, 0.02};
}

Material Material::cobalt() {
    return {1.4e6, 3.0e-11, 4.5e5, {0, 0, 1}, 0.05};
}

Material Material::iron() {
    return {1.7e6, 2.1e-11, 4.8e4, {0, 0, 1}, 0.02};
}

}  // namespace micromag
