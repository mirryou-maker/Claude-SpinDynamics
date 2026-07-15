// Core types: Vec3, StructuredGrid, VectorField3D, ScalarField3D, GeomMask, MFM.
#include "bind_common.hpp"

void bind_core(py::module_& m) {
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

    py::class_<StructuredGrid, std::shared_ptr<StructuredGrid>>(m, "StructuredGrid")
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
        })
        .def("component", &VectorField3D::component, py::arg("c"),
             "Extract component c (0=x, 1=y, 2=z) as ScalarField3D.")
        .def("shift_x", &VectorField3D::shift_x,
             py::arg("n"), py::arg("fill_m"),
             "Shift field n cells along x (n>0: right, n<0: left). "
             "Boundary filled with fill_m.")
        .def("shift_y", &VectorField3D::shift_y,
             py::arg("n"), py::arg("fill_m"),
             "Shift field n cells along y.")
        .def("zero_crossing_x", &VectorField3D::zero_crossing_x,
             py::arg("c"), py::arg("iy"), py::arg("iz"),
             "Find x-index of first sign change of component c in row (iy,iz). "
             "Returns -1 if none found.")
        .def("__getitem__", [](const VectorField3D& f, Index idx) { return f[idx]; })
        .def("__setitem__", [](VectorField3D& f, Index idx, const Vec3& v) { f[idx] = v; },
             py::arg("idx"), py::arg("v"),
             "Set one cell by linear index: field[idx] = Vec3(x, y, z).")
        .def("apply_mask", &VectorField3D::apply_mask, py::arg("mask"),
             "Multiply m by mask value per cell (m=0 where mask=0). In-place.")
        .def("crop",
             [](const VectorField3D& src,
                Index ix0, Index ix1,
                Index iy0, Index iy1,
                Index iz0, Index iz1) -> py::array_t<double> {
                 // Returns a (dnz, dny, dnx, 3) numpy array directly —
                 // avoids grid lifetime issues in Python.
                 const auto& g = src.grid();
                 const Index dnx = ix1 - ix0 + 1;
                 const Index dny = iy1 - iy0 + 1;
                 const Index dnz = iz1 - iz0 + 1;
                 py::array_t<double> arr({(Py_ssize_t)dnz, (Py_ssize_t)dny,
                                          (Py_ssize_t)dnx, (Py_ssize_t)3});
                 auto buf = arr.mutable_unchecked<4>();
                 for (Index iz = iz0; iz <= iz1; ++iz)
                 for (Index iy = iy0; iy <= iy1; ++iy)
                 for (Index ix = ix0; ix <= ix1; ++ix) {
                     Index src_idx = ix + g.nx()*(iy + g.ny()*iz);
                     const Vec3& v = src[src_idx];
                     buf(iz-iz0, iy-iy0, ix-ix0, 0) = v.x;
                     buf(iz-iz0, iy-iy0, ix-ix0, 1) = v.y;
                     buf(iz-iz0, iy-iy0, ix-ix0, 2) = v.z;
                 }
                 return arr;
             },
             py::arg("ix0"), py::arg("ix1"),
             py::arg("iy0"), py::arg("iy1"),
             py::arg("iz0"), py::arg("iz1"),
             "Crop sub-region [ix0..ix1]×[iy0..iy1]×[iz0..iz1] (inclusive).\n"
             "Returns (dnz, dny, dnx, 3) float64 numpy array directly.");

    // ------------------------------------------------------------------
    // ScalarField3D
    // ------------------------------------------------------------------
    py::class_<ScalarField3D>(m, "ScalarField3D")
        .def(py::init<const StructuredGrid&>(), py::keep_alive<1, 2>())
        .def_property_readonly("grid", &ScalarField3D::grid,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("size", &ScalarField3D::size)
        .def("set_uniform", &ScalarField3D::set_uniform, py::arg("v"))
        .def("at", [](ScalarField3D& f, Index i, Index j, Index k) -> Real& {
            return f.at(i, j, k);
        }, py::arg("i"), py::arg("j"), py::arg("k"),
           py::return_value_policy::reference_internal)
        .def("__getitem__", [](const ScalarField3D& f, Index idx) {
            return f[idx];
        })
        .def("__setitem__", [](ScalarField3D& f, Index idx, Real v) {
            f[idx] = v;
        });

    m.def("to_numpy_scalar",
          [](const ScalarField3D& f) -> py::array_t<double> {
              const auto& g = f.grid();
              const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
              py::array_t<double> arr({(Py_ssize_t)nz, (Py_ssize_t)ny, (Py_ssize_t)nx});
              auto buf = arr.mutable_unchecked<3>();
              for (Index iz = 0; iz < nz; ++iz)
              for (Index iy = 0; iy < ny; ++iy)
              for (Index ix = 0; ix < nx; ++ix)
                  buf(iz, iy, ix) = f[static_cast<Index>(ix + nx*(iy + ny*iz))];
              return arr;
          },
          py::arg("field"),
          "Copy ScalarField3D into a (nz, ny, nx) float64 numpy array.");

    m.def("write_vtk_legacy", &write_vtk_legacy,
          py::arg("filename"), py::arg("field"), py::arg("field_name") = "m");

    // ------------------------------------------------------------------
    // Phase B1: GeomMask — per-cell occupancy for geometry definition
    // ------------------------------------------------------------------

    py::class_<GeomMask>(m, "GeomMask")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             py::keep_alive<1, 2>(),
             "Per-cell occupancy mask [0,1]. 1=inside, 0=outside.")
        .def_property_readonly("grid", &GeomMask::grid,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("size", &GeomMask::size)
        .def("set_uniform", &GeomMask::set_uniform, py::arg("v"))
        .def("invert", &GeomMask::invert,
             "In-place invert: v → 1-v for each cell.")
        .def("__invert__",
             [](const GeomMask& src) {
                 GeomMask result(src.grid());
                 for (Index i = 0; i < src.size(); ++i)
                     result[i] = Real{1} - src[i];
                 return result;
             },
             "Return a new inverted mask (~mask): v → 1-v for each cell.")
        .def("at", [](GeomMask& mask, Index i, Index j, Index k) -> Real& {
            return mask(i, j, k);
        }, py::arg("i"), py::arg("j"), py::arg("k"),
           py::return_value_policy::reference_internal)
        .def("__getitem__", [](const GeomMask& mask, Index idx) {
            return mask[idx];
        })
        .def("__setitem__", [](GeomMask& mask, Index idx, Real v) {
            mask[idx] = v;
        })
        .def("to_numpy",
             [](const GeomMask& mask) -> py::array_t<double> {
                 const auto& g = mask.grid();
                 const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
                 py::array_t<double> arr({(Py_ssize_t)nz, (Py_ssize_t)ny,
                                          (Py_ssize_t)nx});
                 auto buf = arr.mutable_unchecked<3>();
                 for (Index iz = 0; iz < nz; ++iz)
                 for (Index iy = 0; iy < ny; ++iy)
                 for (Index ix = 0; ix < nx; ++ix)
                     buf(iz, iy, ix) = mask(ix, iy, iz);
                 return arr;
             },
             "Copy mask into a (nz, ny, nx) float64 numpy array.");

    m.def("union_",    &union_,    py::arg("a"), py::arg("b"),
          "Geometry union: max(a,b) per cell. Returns new GeomMask.");
    m.def("sub_",      &sub_,      py::arg("a"), py::arg("b"),
          "Geometry subtraction: max(a-b,0) per cell. Returns new GeomMask.");
    m.def("intersect_",&intersect_,py::arg("a"), py::arg("b"),
          "Geometry intersection: min(a,b) per cell. Returns new GeomMask.");

    // Primitive shape factories — centered at geometric centre of the grid
    m.def("ellipse",  &ellipse,  py::arg("grid"), py::arg("a"), py::arg("b"),
          "Ellipse in xy-plane: (x/a)^2+(y/b)^2<=1, extruded through z. "
          "Centred at box centre. Returns GeomMask.");
    m.def("circle",   &circle,   py::arg("grid"), py::arg("r"),
          "Circle in xy-plane: x^2+y^2<=r^2. Returns GeomMask.");
    m.def("rect",     &rect,     py::arg("grid"), py::arg("lx"), py::arg("ly"),
          "Rectangle in xy-plane: |x|<=lx/2, |y|<=ly/2. Returns GeomMask.");
    m.def("cylinder", &cylinder, py::arg("grid"), py::arg("r"), py::arg("h"),
          "Finite cylinder along z: x^2+y^2<=r^2, |z|<=h/2. Returns GeomMask.");

    // Phase E: additional shape factories
    m.def("square",  &square,  py::arg("grid"), py::arg("side"),
          "Square in xy-plane: side×side, centred at box centre. Returns GeomMask.");
    m.def("cuboid",  &cuboid,  py::arg("grid"), py::arg("lx"), py::arg("ly"), py::arg("lz"),
          "Rectangular box: |x|<=lx/2, |y|<=ly/2, |z|<=lz/2. Returns GeomMask.");
    m.def("sphere",  &sphere,  py::arg("grid"), py::arg("r"),
          "Sphere: x²+y²+z²<=r². Returns GeomMask.");
    m.def("ellipsoid", &ellipsoid, py::arg("grid"), py::arg("a"), py::arg("b"), py::arg("c"),
          "Ellipsoid: (x/a)²+(y/b)²+(z/c)²<=1. Semi-axes a,b,c along x,y,z. Returns GeomMask.");
    m.def("layer",   &layer,   py::arg("grid"), py::arg("n"),
          "Select single z-layer n (0-based). Returns GeomMask.");
    m.def("layers",  &layers,  py::arg("grid"), py::arg("n1"), py::arg("n2"),
          "Select z-layers n1..n2 inclusive. Returns GeomMask.");
    m.def("x_range", &x_range, py::arg("grid"), py::arg("x1"), py::arg("x2"),
          "Slab x1<=x<=x2 (box-centred coords). Returns GeomMask.");
    m.def("y_range", &y_range, py::arg("grid"), py::arg("y1"), py::arg("y2"),
          "Slab y1<=y<=y2 (box-centred coords). Returns GeomMask.");
    m.def("z_range", &z_range, py::arg("grid"), py::arg("z1"), py::arg("z2"),
          "Slab z1<=z<=z2 (box-centred coords). Returns GeomMask.");

    // Geometric transformations
    m.def("translate", &translate,
          py::arg("mask"), py::arg("shift_x"), py::arg("shift_y"),
          "Shift mask by (shift_x, shift_y) metres (rounded to nearest cell). "
          "Returns new GeomMask.");
    m.def("rotate", &rotate,
          py::arg("mask"), py::arg("theta"),
          "Rotate mask by theta radians (CCW) around box centre using bilinear "
          "interpolation. Returns new GeomMask.");

    // ------------------------------------------------------------------
    // Phase B2: MFM Imaging
    // ------------------------------------------------------------------

    py::enum_<TipMode>(m, "TipMode")
        .value("Monopole", TipMode::Monopole,
               "Monopole tip: signal ∝ Hz at lift height.")
        .value("Dipole",   TipMode::Dipole,
               "Dipole tip: signal ∝ ∂Hz/∂z at lift height (standard MFM).");

    py::class_<MFMImage>(m, "MFMImage")
        .def(py::init<const StructuredGrid&, Real, TipMode>(),
             py::arg("grid"), py::arg("lift_m"),
             py::arg("tip") = TipMode::Dipole,
             py::keep_alive<1, 2>(),
             "MFM image simulator. lift_m: tip height above sample [m]. "
             "tip: TipMode.Monopole or TipMode.Dipole (default).")
        .def("compute",
             [](const MFMImage& mfm,
                const VectorField3D& mv,
                const Material& mat) -> py::array_t<double> {
                 auto signal = mfm.compute(mv, mat);
                 const Index ny = mv.grid().ny();
                 const Index nxg = mv.grid().nx();
                 py::array_t<double> arr({(Py_ssize_t)ny, (Py_ssize_t)nxg});
                 auto buf = arr.mutable_unchecked<2>();
                 for (Index iy = 0; iy < ny; ++iy)
                 for (Index ix = 0; ix < nxg; ++ix)
                     buf(iy, ix) = signal[static_cast<std::size_t>(iy * nxg + ix)];
                 return arr;
             },
             py::arg("m"), py::arg("mat"),
             "Compute MFM signal. Returns (ny, nx) float64 numpy array.")
        .def_property_readonly("lift", &MFMImage::lift,
                               "Lift height above sample [m].")
        .def_property_readonly("tip",  &MFMImage::tip,
                               "Tip mode (Monopole or Dipole).");

}
