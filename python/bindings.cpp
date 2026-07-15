// _micromag module entry point. The actual bindings live in bind_*.cpp,
// one translation unit per domain (see bind_common.hpp).
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_core(py::module_& m);
void bind_fields(py::module_& m);
void bind_dynamics(py::module_& m);
void bind_gpu(py::module_& m);

PYBIND11_MODULE(_micromag, m) {
    m.doc() = "Micromag C++ core bindings";
    bind_core(m);
    bind_fields(m);
    bind_dynamics(m);
    bind_gpu(m);
}
