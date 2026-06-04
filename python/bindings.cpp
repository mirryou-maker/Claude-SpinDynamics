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
#include "micromag/integrator.hpp"
#include "micromag/thermal_field.hpp"
#include "micromag/spin_torque.hpp"

#ifdef MICROMAG_CUDA
#include "micromag/demag_gpu.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
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
        .def_property("boundary", &ExchangeField::boundary, &ExchangeField::set_boundary);

    py::class_<DemagField, IEffectiveField, std::shared_ptr<DemagField>>(m, "DemagField")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"))
        .def("accumulate", &DemagField::accumulate)
        .def("energy",     &DemagField::energy)
        .def_property_readonly("name", &DemagField::name);

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
#endif  // MICROMAG_CUDA
}
