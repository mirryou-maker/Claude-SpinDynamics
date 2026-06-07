#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "micromag/types.hpp"
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/vtk_writer.hpp"
#include "micromag/material.hpp"
#include "micromag/effective_field.hpp"
#include "micromag/zeeman.hpp"
#include "micromag/anisotropy.hpp"
#include "micromag/exchange.hpp"
#include "micromag/demag.hpp"
#include "micromag/demag_periodic.hpp"
#include "micromag/rkky.hpp"
#include "micromag/zeeman_spatial.hpp"
#include "micromag/geom_mask.hpp"
#include "micromag/mfm.hpp"
#include "micromag/integrator.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/spin_torque.hpp"

#ifdef MICROMAG_CUDA
#include "micromag/demag_gpu.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/rk45_integrator_gpu.hpp"
#include "micromag/heun_integrator_gpu.hpp"
#endif

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
        .def("at", [](ScalarField3D& f, Index i, Index j, Index k) {
            return f.at(i, j, k);
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
                 const Index nx = mfm.lift() >= 0 ? mv.grid().nx() : mv.grid().nx();
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

    // ------------------------------------------------------------------
    // Phase 1b: Material + Effective fields
    // ------------------------------------------------------------------

    py::class_<Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("Ms",          &Material::Ms)
        .def_readwrite("A_exchange",  &Material::A_exchange)
        .def_readwrite("K_uniaxial",  &Material::K_uniaxial)
        .def_readwrite("easy_axis",   &Material::easy_axis)
        .def_readwrite("alpha",       &Material::alpha)
        .def_static("permalloy", &Material::permalloy)
        .def_static("cobalt",    &Material::cobalt)
        .def_static("iron",      &Material::iron);

    py::enum_<BoundaryCondition>(m, "BoundaryCondition")
        .value("Neumann",  BoundaryCondition::Neumann)
        .value("Periodic", BoundaryCondition::Periodic);

    py::class_<IEffectiveField, std::shared_ptr<IEffectiveField>>(m, "IEffectiveField")
        .def("accumulate", &IEffectiveField::accumulate)
        .def("energy",     &IEffectiveField::energy)
        .def_property_readonly("name", &IEffectiveField::name);

    py::class_<ZeemanField, IEffectiveField, std::shared_ptr<ZeemanField>>(m, "ZeemanField")
        .def(py::init<const Vec3&>(), py::arg("H_ext") = Vec3{0, 0, 0})
        .def_property("H_ext", &ZeemanField::H_ext, &ZeemanField::set_H_ext);

    py::class_<UniaxialAnisotropyField, IEffectiveField,
               std::shared_ptr<UniaxialAnisotropyField>>(m, "UniaxialAnisotropyField")
        .def(py::init<>());

    py::class_<ExchangeField, IEffectiveField, std::shared_ptr<ExchangeField>>(m, "ExchangeField")
        .def(py::init<BoundaryCondition>(), py::arg("bc") = BoundaryCondition::Neumann)
        .def_property("boundary", &ExchangeField::boundary, &ExchangeField::set_boundary)
        .def("set_mask",
             [](ExchangeField& f, const GeomMask& mask) { f.set_mask(&mask); },
             py::arg("mask"), py::keep_alive<1, 2>(),
             "Attach geometry mask: cells with mask<0.5 become Neumann boundaries.")
        .def("clear_mask", &ExchangeField::clear_mask,
             "Remove geometry mask (no masking).");

    py::class_<DemagField, IEffectiveField, std::shared_ptr<DemagField>>(m, "DemagField")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"))
        .def("accumulate", &DemagField::accumulate)
        .def("energy",     &DemagField::energy)
        .def_property_readonly("name", &DemagField::name);

    py::class_<DemagFieldPeriodic, IEffectiveField,
               std::shared_ptr<DemagFieldPeriodic>>(m, "DemagFieldPeriodic")
        .def(py::init<const StructuredGrid&, int>(),
             py::arg("grid"), py::arg("n_rep") = 2,
             "Periodic-BC demag field (no zero-padding). "
             "Uniform m → H=0 (k=0 zeroed). n_rep: image cells per side.")
        .def("accumulate", &DemagFieldPeriodic::accumulate)
        .def("energy",     &DemagFieldPeriodic::energy)
        .def_property_readonly("name", &DemagFieldPeriodic::name);

    py::class_<ZeemanFieldSpatial, IEffectiveField,
               std::shared_ptr<ZeemanFieldSpatial>>(m, "ZeemanFieldSpatial")
        .def(py::init<const VectorField3D&>(), py::arg("H_field"),
             py::keep_alive<1, 2>(),
             "Spatially varying Zeeman field. Update H_field in-place between steps.")
        .def("accumulate",  &ZeemanFieldSpatial::accumulate)
        .def("energy",      &ZeemanFieldSpatial::energy)
        .def("set_field",   &ZeemanFieldSpatial::set_field, py::arg("H_field"),
             py::keep_alive<1, 2>())
        .def_property_readonly("name", &ZeemanFieldSpatial::name);

    m.def("make_gaussian_field",
          [](const StructuredGrid& grid,
             double cx, double cy,      // centre of Gaussian [m]
             double sigma,              // width [m]
             double Hz_Am) -> VectorField3D {
              VectorField3D H(grid);
              for (Index iz = 0; iz < grid.nz(); ++iz)
              for (Index iy = 0; iy < grid.ny(); ++iy)
              for (Index ix = 0; ix < grid.nx(); ++ix) {
                  auto pos = grid.cell_center(ix, iy, iz);
                  double dx = pos.x - cx, dy = pos.y - cy;
                  double amp = Hz_Am * std::exp(-(dx*dx+dy*dy)/(2*sigma*sigma));
                  H.at(ix, iy, iz) = {0, 0, amp};
              }
              return H;
          },
          py::arg("grid"), py::arg("cx"), py::arg("cy"),
          py::arg("sigma"), py::arg("Hz_Am"),
          "Create a Gaussian-profile z-field (write-head approximation).");

    py::class_<RKKYField, IEffectiveField, std::shared_ptr<RKKYField>>(m, "RKKYField")
        .def(py::init<const VectorField3D&, Real, Real>(),
             py::arg("ref_m"), py::arg("J_RKKY"), py::arg("d_spacer"),
             py::keep_alive<1, 2>(),
             "RKKY inter-layer coupling: H = -J/(mu0*Ms*d)*m_ref.\n"
             "J_RKKY < 0 = antiferromagnetic, > 0 = ferromagnetic.")
        .def("accumulate", &RKKYField::accumulate)
        .def("energy",     &RKKYField::energy)
        .def_property("J", &RKKYField::J, &RKKYField::set_J)
        .def_property_readonly("d",    &RKKYField::d)
        .def_property_readonly("name", &RKKYField::name);

    py::class_<EffectiveFieldSum>(m, "EffectiveFieldSum")
        .def(py::init<>())
        .def("add",          &EffectiveFieldSum::add)
        .def("compute",      &EffectiveFieldSum::compute)
        .def("total_energy", &EffectiveFieldSum::total_energy)
        .def_property_readonly("terms",     &EffectiveFieldSum::terms)
        .def_property_readonly("num_terms", &EffectiveFieldSum::num_terms);

    // ------------------------------------------------------------------
    // Phase 1c: LLG integrator
    // ------------------------------------------------------------------

    m.attr("gamma_0") = constants::gamma_0;

    m.def("llg_torque", &llg_torque,
          py::arg("m"), py::arg("H"), py::arg("alpha"));

    py::class_<RK4Integrator>(m, "RK4Integrator")
        .def(py::init<Real>(), py::arg("dt") = Real{1e-13})
        .def("step",
             [](RK4Integrator& self, VectorField3D& mv, const Material& mat,
                const EffectiveFieldSum& heff, SpinTorqueSum* stt) {
                 self.step(mv, mat, heff, stt);
             },
             py::arg("m"), py::arg("mat"), py::arg("heff"),
             py::arg("stt") = nullptr)
        .def_property("dt", &RK4Integrator::dt, &RK4Integrator::set_dt);

    // RK45 Options struct
    py::class_<RK45Integrator::Options>(m, "RK45Options")
        .def(py::init<>())
        .def_readwrite("rtol",    &RK45Integrator::Options::rtol)
        .def_readwrite("atol",    &RK45Integrator::Options::atol)
        .def_readwrite("dt_init", &RK45Integrator::Options::dt_init)
        .def_readwrite("dt_min",  &RK45Integrator::Options::dt_min)
        .def_readwrite("dt_max",  &RK45Integrator::Options::dt_max)
        .def_readwrite("fac_min", &RK45Integrator::Options::fac_min)
        .def_readwrite("fac_max", &RK45Integrator::Options::fac_max);

    py::class_<RK45Integrator>(m, "RK45Integrator")
        .def(py::init<>())
        .def(py::init<RK45Integrator::Options>(), py::arg("opts"))
        .def("step",
             [](RK45Integrator& self, VectorField3D& mv, const Material& mat,
                const EffectiveFieldSum& heff, SpinTorqueSum* stt) {
                 return self.step(mv, mat, heff, stt);
             },
             py::arg("m"), py::arg("mat"), py::arg("heff"),
             py::arg("stt") = nullptr)
        .def_property_readonly("dt_current", &RK45Integrator::dt_current);

    // ------------------------------------------------------------------
    // Phase 1d: Spin Transfer Torque + Spin-Orbit Torque
    // ------------------------------------------------------------------

    py::class_<ISpinTorque, std::shared_ptr<ISpinTorque>>(m, "ISpinTorque")
        .def("accumulate", &ISpinTorque::accumulate)
        .def_property_readonly("name", &ISpinTorque::name);

    py::class_<SlonczewskiSTT, ISpinTorque, std::shared_ptr<SlonczewskiSTT>>(
            m, "SlonczewskiSTT")
        .def(py::init<Real, Real, Real, Vec3, Real>(),
             py::arg("J"), py::arg("P"), py::arg("d"), py::arg("p"),
             py::arg("beta") = Real{0.0})
        .def("a_J",        &SlonczewskiSTT::a_J)
        .def_property("J",    &SlonczewskiSTT::J,    &SlonczewskiSTT::set_J)
        .def_property("P",    &SlonczewskiSTT::P,    &SlonczewskiSTT::set_P)
        .def_property("beta", &SlonczewskiSTT::beta, &SlonczewskiSTT::set_beta)
        .def_property_readonly("d", &SlonczewskiSTT::d)
        .def_property_readonly("p", &SlonczewskiSTT::p);

    py::class_<SpinOrbitTorque, ISpinTorque, std::shared_ptr<SpinOrbitTorque>>(
            m, "SpinOrbitTorque")
        .def(py::init<Real, Real, Real, Vec3, Real, Real>(),
             py::arg("J_c"), py::arg("theta_SH"), py::arg("d_fm"),
             py::arg("sigma"),
             py::arg("eta_DL") = Real{1.0},
             py::arg("eta_FL") = Real{0.0})
        .def("a_SOT",      &SpinOrbitTorque::a_SOT)
        .def_property("J_c",      &SpinOrbitTorque::J_c,      &SpinOrbitTorque::set_J_c)
        .def_property("theta_SH", &SpinOrbitTorque::theta_SH, &SpinOrbitTorque::set_theta_SH)
        .def_property("eta_DL",   &SpinOrbitTorque::eta_DL,   &SpinOrbitTorque::set_eta_DL)
        .def_property("eta_FL",   &SpinOrbitTorque::eta_FL,   &SpinOrbitTorque::set_eta_FL)
        .def_property_readonly("d_fm",  &SpinOrbitTorque::d_fm)
        .def_property_readonly("sigma", &SpinOrbitTorque::sigma);

    py::class_<SpinTorqueSum>(m, "SpinTorqueSum")
        .def(py::init<>())
        .def("add",   &SpinTorqueSum::add)
        .def_property_readonly("terms",     &SpinTorqueSum::terms)
        .def_property_readonly("num_terms", &SpinTorqueSum::num_terms);

    // ------------------------------------------------------------------
    // Phase T: Stochastic LLG — HeunIntegrator + ThermalField
    // ------------------------------------------------------------------

    py::class_<ThermalField, IEffectiveField, std::shared_ptr<ThermalField>>(m, "ThermalField")
        .def(py::init<const StructuredGrid&, Real, Real, unsigned>(),
             py::arg("grid"), py::arg("T_K"), py::arg("dt"),
             py::arg("seed") = 42u)
        .def("set_temperature", &ThermalField::set_temperature, py::arg("T_K"))
        .def("set_dt",          &ThermalField::set_dt,          py::arg("dt"))
        .def_property_readonly("temperature", &ThermalField::temperature)
        .def_property_readonly("dt",          &ThermalField::dt)
        .def_property_readonly("name",        &ThermalField::name);

    py::class_<HeunIntegrator>(m, "HeunIntegrator")
        .def(py::init<Real>(), py::arg("dt") = Real{1e-13})
        .def("step",
             [](HeunIntegrator& self, VectorField3D& mv, const Material& mat,
                const EffectiveFieldSum& heff,
                ThermalField* thermal,
                SpinTorqueSum* stt) {
                 self.step(mv, mat, heff, thermal, stt);
             },
             py::arg("m"), py::arg("mat"), py::arg("heff"),
             py::arg("thermal") = nullptr,
             py::arg("stt")     = nullptr)
        .def_property("dt", &HeunIntegrator::dt, &HeunIntegrator::set_dt);

    // ------------------------------------------------------------------
    // Numpy bridge — VectorField3D ↔ numpy array (shape: nz×ny×nx×3)
    //
    //   m_np = micromag.to_numpy(m)          # copy out
    //   micromag.from_numpy(m, m_np)         # copy in
    //   micromag.mean_magnetization(m)       # returns (mx, my, mz) tuple
    // ------------------------------------------------------------------

    m.def("to_numpy",
          [](const VectorField3D& f) -> py::array_t<double> {
              const auto& g = f.grid();
              const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
              py::array_t<double> arr({(Py_ssize_t)nz, (Py_ssize_t)ny,
                                       (Py_ssize_t)nx, (Py_ssize_t)3});
              auto buf = arr.mutable_unchecked<4>();
              for (Index iz = 0; iz < nz; ++iz)
              for (Index iy = 0; iy < ny; ++iy)
              for (Index ix = 0; ix < nx; ++ix) {
                  // Linear index: x-fastest
                  const Vec3& v = f[static_cast<Index>(ix + nx*(iy + ny*iz))];
                  buf(iz, iy, ix, 0) = v.x;
                  buf(iz, iy, ix, 1) = v.y;
                  buf(iz, iy, ix, 2) = v.z;
              }
              return arr;
          },
          py::arg("field"),
          "Copy VectorField3D into a (nz, ny, nx, 3) float64 numpy array.");

    m.def("from_numpy",
          [](VectorField3D& f,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
              const auto& g = f.grid();
              const Index nx = g.nx(), ny = g.ny(), nz = g.nz();
              auto buf = arr.unchecked<4>();
              if (buf.shape(0) != nz || buf.shape(1) != ny ||
                  buf.shape(2) != nx || buf.shape(3) != 3)
                  throw std::runtime_error(
                      "from_numpy: array shape must be (nz, ny, nx, 3)");
              for (Index iz = 0; iz < nz; ++iz)
              for (Index iy = 0; iy < ny; ++iy)
              for (Index ix = 0; ix < nx; ++ix) {
                  f[static_cast<Index>(ix + nx*(iy + ny*iz))] =
                      Vec3{buf(iz, iy, ix, 0), buf(iz, iy, ix, 1), buf(iz, iy, ix, 2)};
              }
          },
          py::arg("field"), py::arg("array"),
          "Copy a (nz, ny, nx, 3) float64 numpy array into VectorField3D.");

    m.def("mean_magnetization",
          [](const VectorField3D& f) {
              double mx = 0, my = 0, mz = 0;
              const Index N = f.size();
              for (Index i = 0; i < N; ++i) {
                  mx += f[i].x; my += f[i].y; mz += f[i].z;
              }
              return py::make_tuple(mx/N, my/N, mz/N);
          },
          py::arg("field"),
          "Return (mean_mx, mean_my, mean_mz) averaged over all cells.");

    // ------------------------------------------------------------------
    // CUDA availability probe (always defined; returns False in CPU build)
    // ------------------------------------------------------------------
#ifdef MICROMAG_CUDA
    m.def("cuda_available", []() { return true; },
          "True when the module was compiled with CUDA support.");
#else
    m.def("cuda_available", []() { return false; },
          "True when the module was compiled with CUDA support.");
#endif

#ifdef MICROMAG_CUDA
    // ------------------------------------------------------------------
    // Phase G: GPU fields (IEffectiveField drop-ins, cuda preset only)
    // ------------------------------------------------------------------

    py::class_<DemagFieldGPU, IEffectiveField, std::shared_ptr<DemagFieldGPU>>(
            m, "DemagFieldGPU")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             "GPU cuFFT demag field — drop-in for DemagField, 5–20× faster.")
        .def("accumulate", &DemagFieldGPU::accumulate)
        .def("energy",     &DemagFieldGPU::energy)
        .def_property_readonly("name", &DemagFieldGPU::name);

    py::class_<ExchangeFieldGPU, IEffectiveField, std::shared_ptr<ExchangeFieldGPU>>(
            m, "ExchangeFieldGPU")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             "GPU 6-point Laplacian exchange field.")
        .def("accumulate", &ExchangeFieldGPU::accumulate)
        .def("energy",     &ExchangeFieldGPU::energy)
        .def_property_readonly("name", &ExchangeFieldGPU::name);

    py::class_<ZeemanFieldGPU, IEffectiveField, std::shared_ptr<ZeemanFieldGPU>>(
            m, "ZeemanFieldGPU")
        .def(py::init<const StructuredGrid&, Vec3>(),
             py::arg("grid"), py::arg("H_ext") = Vec3{0,0,0})
        .def("accumulate",   &ZeemanFieldGPU::accumulate)
        .def("energy",       &ZeemanFieldGPU::energy)
        .def_property("H_ext", &ZeemanFieldGPU::H_ext, &ZeemanFieldGPU::set_H_ext)
        .def_property_readonly("name", &ZeemanFieldGPU::name);

    py::class_<UniaxialAnisotropyFieldGPU, IEffectiveField,
               std::shared_ptr<UniaxialAnisotropyFieldGPU>>(
            m, "UniaxialAnisotropyFieldGPU")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             "GPU uniaxial anisotropy field (K and easy_axis from Material).")
        .def("accumulate", &UniaxialAnisotropyFieldGPU::accumulate)
        .def("energy",     &UniaxialAnisotropyFieldGPU::energy)
        .def_property_readonly("name", &UniaxialAnisotropyFieldGPU::name);

    // ------------------------------------------------------------------
    // RK4IntegratorGPU — full-GPU LLG integrator (zero PCIe per step)
    //
    // Usage:
    //   integ = mm.RK4IntegratorGPU(grid, dt)
    //   integ.upload(m)
    //   for _ in range(N):
    //       integ.step(mat, demag, exch, zeeman)       # or with aniso
    //   integ.download(m)
    //   m_np = mm.to_numpy(m)
    // ------------------------------------------------------------------
    py::class_<RK4IntegratorGPU>(m, "RK4IntegratorGPU")
        .def(py::init<const StructuredGrid&, Real>(),
             py::arg("grid"), py::arg("dt") = Real{1e-13},
             "Full-GPU fixed-step RK4 LLG integrator (zero PCIe per step).")
        .def("upload",   &RK4IntegratorGPU::upload,   py::arg("m"),
             "Upload CPU VectorField3D to GPU (once before simulation loop).")
        .def("download", &RK4IntegratorGPU::download, py::arg("m"),
             "Download GPU state into CPU VectorField3D (for monitoring).")
        .def("step",
             [](RK4IntegratorGPU& integ, const Material& mat,
                DemagFieldGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman,
                UniaxialAnisotropyFieldGPU* aniso) {
                 integ.step(mat, demag, exch, zeeman, aniso);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"),
             py::arg("aniso") = static_cast<UniaxialAnisotropyFieldGPU*>(nullptr),
             "One full RK4 step on GPU (Exchange + Demag + Zeeman [+ Aniso]).")
        .def_property("dt", &RK4IntegratorGPU::dt, &RK4IntegratorGPU::set_dt);

    // ------------------------------------------------------------------
    // RK45IntegratorGPU — adaptive DOPRI5, one D2H scalar per trial step
    //
    // Usage:
    //   integ = mm.RK45IntegratorGPU(grid)                 # default opts
    //   integ = mm.RK45IntegratorGPU(grid, mm.RK45GPUOptions())
    //   integ.upload(m)
    //   while t < t_end:
    //       dt = integ.step(mat, demag, exch, zeeman)
    //       t += dt
    //   integ.download(m)
    // ------------------------------------------------------------------
    py::class_<RK45IntegratorGPU::Options>(m, "RK45GPUOptions")
        .def(py::init<>())
        .def_readwrite("rtol",    &RK45IntegratorGPU::Options::rtol)
        .def_readwrite("atol",    &RK45IntegratorGPU::Options::atol)
        .def_readwrite("dt_init", &RK45IntegratorGPU::Options::dt_init)
        .def_readwrite("dt_min",  &RK45IntegratorGPU::Options::dt_min)
        .def_readwrite("dt_max",  &RK45IntegratorGPU::Options::dt_max)
        .def_readwrite("safety",  &RK45IntegratorGPU::Options::safety)
        .def_readwrite("fac_min", &RK45IntegratorGPU::Options::fac_min)
        .def_readwrite("fac_max", &RK45IntegratorGPU::Options::fac_max);

    py::class_<RK45IntegratorGPU>(m, "RK45IntegratorGPU")
        .def(py::init<const StructuredGrid&, RK45IntegratorGPU::Options>(),
             py::arg("grid"), py::arg("opts") = RK45IntegratorGPU::Options{},
             "Adaptive GPU DOPRI5 integrator (7-stage FSAL). "
             "One D2H scalar per trial step; zero PCIe otherwise.")
        .def("upload",   &RK45IntegratorGPU::upload,   py::arg("m"))
        .def("download", &RK45IntegratorGPU::download, py::arg("m"))
        .def("step",
             [](RK45IntegratorGPU& integ, const Material& mat,
                DemagFieldGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman,
                UniaxialAnisotropyFieldGPU* aniso) {
                 return integ.step(mat, demag, exch, zeeman, aniso);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"),
             py::arg("aniso") = static_cast<UniaxialAnisotropyFieldGPU*>(nullptr),
             "One adaptive DOPRI5 step. Returns actual dt taken.")
        .def_property_readonly("dt",         &RK45IntegratorGPU::dt_current)
        .def_property_readonly("dt_current", &RK45IntegratorGPU::dt_current)
        .def_property_readonly("n_accepted", &RK45IntegratorGPU::n_accepted)
        .def_property_readonly("n_rejected", &RK45IntegratorGPU::n_rejected);

    // ------------------------------------------------------------------
    // HeunIntegratorGPU — full-GPU Stratonovich Heun (SLLG, T > 0)
    //
    // Usage:
    //   integ = mm.HeunIntegratorGPU(grid, dt, seed=42)
    //   integ.upload(m)
    //   for _ in range(N):
    //       integ.step(mat, demag, exch, zeeman, T_K=300.0)
    //   integ.download(m)
    // ------------------------------------------------------------------
    py::class_<HeunIntegratorGPU>(m, "HeunIntegratorGPU")
        .def(py::init<const StructuredGrid&, Real, unsigned>(),
             py::arg("grid"), py::arg("dt") = Real{1e-13},
             py::arg("seed") = 42u,
             "GPU Stratonovich Heun integrator for SLLG (T > 0). "
             "T_K=0 gives deterministic Heun ODE (no cuRAND calls).")
        .def("upload",   &HeunIntegratorGPU::upload,   py::arg("m"),
             "Upload CPU VectorField3D to GPU.")
        .def("download", &HeunIntegratorGPU::download, py::arg("m"),
             "Download GPU state into CPU VectorField3D.")
        .def("step",
             [](HeunIntegratorGPU& integ, const Material& mat,
                DemagFieldGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman, Real T_K,
                UniaxialAnisotropyFieldGPU* aniso) {
                 integ.step(mat, demag, exch, zeeman, T_K, aniso);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"),
             py::arg("T_K")  = Real{0.0},
             py::arg("aniso") = static_cast<UniaxialAnisotropyFieldGPU*>(nullptr),
             "One Stratonovich Heun step on GPU. "
             "T_K=0 disables noise. "
             "Optional aniso adds uniaxial anisotropy.")
        .def_property("dt", &HeunIntegratorGPU::dt, &HeunIntegratorGPU::set_dt);
#endif  // MICROMAG_CUDA
}
