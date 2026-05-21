#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "micromag/types.hpp"
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/vtk_writer.hpp"

namespace py = pybind11;
using namespace micromag;

PYBIND11_MODULE(_micromag, m) {
    m.doc() = "Micromag C++ core bindings";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<Real, Real, Real>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("norm", &Vec3::norm)
        .def("dot", &Vec3::dot)
        .def("cross", &Vec3::cross)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    py::class_<StructuredGrid>(m, "StructuredGrid")
        .def(py::init<Index, Index, Index, Real, Real, Real>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("dx"), py::arg("dy"), py::arg("dz"))
        .def_property_readonly("nx", &StructuredGrid::nx)
        .def_property_readonly("ny", &StructuredGrid::ny)
        .def_property_readonly("nz", &StructuredGrid::nz)
        .def_property_readonly("dx", &StructuredGrid::dx)
        .def_property_readonly("dy", &StructuredGrid::dy)
        .def_property_readonly("dz", &StructuredGrid::dz)
        .def_property_readonly("size", &StructuredGrid::size)
        .def("cell_center", &StructuredGrid::cell_center);

    py::class_<VectorField3D>(m, "VectorField3D")
        .def(py::init<const StructuredGrid&>(), py::keep_alive<1, 2>())
        .def_property_readonly("grid", &VectorField3D::grid,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("size", &VectorField3D::size)
        .def("set_uniform", &VectorField3D::set_uniform)
        .def("set_vortex", &VectorField3D::set_vortex,
             py::arg("cx"), py::arg("cy"), py::arg("core_radius"))
        .def("normalize", &VectorField3D::normalize)
        .def("at", [](VectorField3D& f, Index i, Index j, Index k) {
            return f.at(i, j, k);
        });

    m.def("write_vtk_legacy", &write_vtk_legacy,
          py::arg("filename"), py::arg("field"), py::arg("field_name") = "m");
}
