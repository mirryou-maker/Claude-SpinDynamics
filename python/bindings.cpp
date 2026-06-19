#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "micromag/types.hpp"
#include "micromag/grid.hpp"
#include "micromag/field.hpp"
#include "micromag/vtk_writer.hpp"
#include "micromag/material.hpp"
#include "micromag/material_field.hpp"
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
#include "micromag/dmi.hpp"
#include "micromag/solver.hpp"
#include "micromag/ovf_io.hpp"
#include "micromag/cubic_anisotropy.hpp"
#include "micromag/surface_anisotropy.hpp"
#include "micromag/magnetoelastic.hpp"
#include "micromag/region_map.hpp"
#include "micromag/init_mag.hpp"
#include "micromag/topological_charge.hpp"
#include "micromag/skyrmion_tools.hpp"

#ifdef MICROMAG_CUDA
#include "micromag/demag_gpu_iface.hpp"
#include "micromag/demag_gpu.hpp"
#include "micromag/demag_periodic_gpu.hpp"
#include "micromag/dmi_gpu.hpp"
#include "micromag/effective_field_gpu_iface.hpp"
#include "micromag/relax_gpu.hpp"
#include "micromag/exchange_gpu.hpp"
#include "micromag/field_kernels_gpu.hpp"
#include "micromag/rk4_integrator_gpu.hpp"
#include "micromag/rk45_integrator_gpu.hpp"
#include "micromag/heun_integrator_gpu.hpp"
#include "micromag/magnetoelastic_gpu.hpp"
#include "micromag/surface_anisotropy_gpu.hpp"
#include "micromag/zeeman_spatial_gpu.hpp"
#include "micromag/rkky_gpu.hpp"
#include "micromag/spin_torque_gpu.hpp"
#endif

namespace py = pybind11;
using namespace micromag;

// ---------------------------------------------------------------------------
// Trampoline: allows Python to subclass IEffectiveField.
// Required for PythonField (custom user-defined effective-field terms).
// ---------------------------------------------------------------------------
struct PyIEffectiveField : IEffectiveField {
    using IEffectiveField::IEffectiveField;

    void accumulate(const VectorField3D& m, const Material& mat,
                    VectorField3D& H_out) const override {
        PYBIND11_OVERRIDE_PURE(void, IEffectiveField, accumulate, m, mat, H_out);
    }
    Real energy(const VectorField3D& m, const Material& mat) const override {
        PYBIND11_OVERRIDE_PURE(Real, IEffectiveField, energy, m, mat);
    }
    ScalarField3D energy_density(const VectorField3D& m,
                                  const Material& mat) const override {
        PYBIND11_OVERRIDE(ScalarField3D, IEffectiveField, energy_density, m, mat);
    }
    const char* name() const override {
        PYBIND11_OVERRIDE_PURE(const char*, IEffectiveField, name, /* no args */);
    }
};

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
        .def_readwrite("Ku2",         &Material::Ku2,
                       "Second uniaxial anisotropy constant K2 [J/m³]. "
                       "ΔE = -Ku2*(m·û)^4")
        .def_static("permalloy", &Material::permalloy)
        .def_static("cobalt",    &Material::cobalt)
        .def_static("iron",      &Material::iron);

    // Phase C1: MaterialField3D — per-cell Ms/A/K/easy_axis/alpha
    py::class_<MaterialField3D>(m, "MaterialField3D")
        .def(py::init<const StructuredGrid&, const Material&>(),
             py::arg("grid"), py::arg("uniform") = Material{},
             py::keep_alive<1, 2>(),
             "Per-cell material parameters, initialised uniformly from `uniform`.")
        .def_property_readonly("grid", &MaterialField3D::grid,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("size", &MaterialField3D::size)
        .def("set_uniform", &MaterialField3D::set_uniform, py::arg("material"),
             "Overwrite every cell with the given uniform Material.")
        .def("at", &MaterialField3D::at, py::arg("i"), py::arg("j"), py::arg("k"),
             "Assemble the Material for one cell.")
        .def("__getitem__", [](const MaterialField3D& f, Index idx) { return f[idx]; })
        .def("__setitem__",
             [](MaterialField3D& f, Index idx, const Material& mat) {
                 f.Ms_field()[idx]    = mat.Ms;
                 f.A_field()[idx]     = mat.A_exchange;
                 f.K_field()[idx]     = mat.K_uniaxial;
                 f.alpha_field()[idx] = mat.alpha;
                 f.easy_axis_field()[idx] = mat.easy_axis;
             },
             py::arg("idx"), py::arg("mat"),
             "Set one cell by linear index: matf[idx] = Material(...).")
        .def_property_readonly("Ms_field", [](MaterialField3D& f) -> ScalarField3D& {
            return f.Ms_field();
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("A_field", [](MaterialField3D& f) -> ScalarField3D& {
            return f.A_field();
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("K_field", [](MaterialField3D& f) -> ScalarField3D& {
            return f.K_field();
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("alpha_field", [](MaterialField3D& f) -> ScalarField3D& {
            return f.alpha_field();
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("easy_axis_field", [](MaterialField3D& f) -> VectorField3D& {
            return f.easy_axis_field();
        }, py::return_value_policy::reference_internal);

    m.def("voronoi_grains", &voronoi_grains,
          py::arg("grid"), py::arg("n_grains"), py::arg("base"),
          py::arg("sigma_K") = Real{0.0}, py::arg("seed") = 42u,
          "Randomized polycrystalline grain structure (mumax3 \"Voronoi "
          "Tessellation\" / random-anisotropy model). Scatters n_grains random "
          "seed points, assigns each cell to its nearest seed, then gives each "
          "grain its own randomized K_uniaxial = max(0, base.K_uniaxial + "
          "N(0, sigma_K)) and a uniformly-random easy-axis orientation. "
          "Ms/A_exchange/alpha are taken uniformly from `base`. Returns MaterialField3D.");

    // Phase E: RegionMap — integer region IDs (0-255), mumax3 DefRegion system
    py::class_<RegionMap>(m, "RegionMap")
        .def(py::init<const StructuredGrid&, uint8_t>(),
             py::arg("grid"), py::arg("default_id") = uint8_t{0},
             py::keep_alive<1, 2>(),
             "Integer region map (0-255 IDs per cell). Default all cells = default_id.")
        .def("def_region",
             [](RegionMap& rm, uint8_t id, const GeomMask& mask) {
                 rm.def_region(id, mask);
             },
             py::arg("id"), py::arg("mask"),
             "Assign region id to all cells where mask > 0.5 (last call wins).")
        .def("region_mask", &RegionMap::region_mask, py::arg("id"),
             "Return GeomMask with 1 where region==id, 0 elsewhere.")
        .def("set_magnetization",
             [](const RegionMap& rm, uint8_t id, VectorField3D& m, Vec3 val) {
                 rm.set_magnetization(id, m, val);
             },
             py::arg("id"), py::arg("m"), py::arg("val"),
             "Set m[i] = val (normalised) for all cells in region id.")
        .def("set_material",
             [](const RegionMap& rm, uint8_t id, MaterialField3D& matf, const Material& mat) {
                 rm.set_material(id, matf, mat);
             },
             py::arg("id"), py::arg("material_field"), py::arg("material"),
             "Copy material properties into MaterialField3D for all cells in region id.")
        .def("__getitem__",
             [](const RegionMap& rm, Index idx) { return rm[idx]; },
             py::arg("idx"), "Region ID at flat linear index idx.")
        .def("__setitem__",
             [](RegionMap& rm, Index idx, uint8_t id) { rm[idx] = id; },
             py::arg("idx"), py::arg("id"), "Set region ID at flat linear index idx.")
        .def_property_readonly("grid", &RegionMap::grid,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("size", &RegionMap::size);

    // Phase E: initial magnetization state factories
    m.def("uniform_mag", &uniform_mag,
          py::arg("grid"), py::arg("dir") = Vec3{0, 0, 1},
          "Uniform magnetization state. dir is normalised automatically.");
    m.def("neel_skyrmion", &neel_skyrmion,
          py::arg("grid"), py::arg("r"),
          py::arg("charge") = 1, py::arg("pol") = 1,
          py::arg("cx") = Real{0}, py::arg("cy") = Real{0},
          "Néel-type skyrmion: θ=2*atan(r/ρ), in-plane component radial.\n"
          "charge=topological charge (±1), pol=polarity (±1 sets mz sign at core).\n"
          "cx,cy: offset from box centre [m].");
    m.def("bloch_skyrmion", &bloch_skyrmion,
          py::arg("grid"), py::arg("r"),
          py::arg("charge") = 1, py::arg("pol") = 1,
          py::arg("cx") = Real{0}, py::arg("cy") = Real{0},
          "Bloch-type skyrmion: in-plane component tangential (φ_in=charge*φ+π/2).");
    m.def("two_domain", &two_domain,
          py::arg("grid"), py::arg("m1"), py::arg("m2"),
          py::arg("axis") = 'x',
          "Two-domain state split at box centre along axis ('x','y','z').\n"
          "m1 fills x<0 (or y<0 / z<0), m2 fills the other half.");
    m.def("vortex_state", &vortex_state,
          py::arg("grid"), py::arg("circ") = 1, py::arg("pol") = 1,
          "Vortex state: core at box centre, mz=pol at core, in-plane tangential.\n"
          "circ=circulation (±1), pol=polarity (±1).");
    m.def("random_mag", &random_mag,
          py::arg("grid"), py::arg("seed") = 42u,
          "Random unit-vector magnetization (uniform on sphere). seed: RNG seed.");

    // Phase F: topological charge (skyrmion number)
    m.def("topological_charge_Q", &topological_charge_Q,
          py::arg("m"),
          "Total topological charge Q = (1/4π) ∫ m·(∂m/∂x × ∂m/∂y) dA.\n"
          "Perfect skyrmion → Q ≈ ±1. Neumann BC, central differences.");
    m.def("topological_charge_density", &topological_charge_density,
          py::arg("m"),
          "Per-cell density q = m·(∂m/∂x × ∂m/∂y) (no 1/4π). Returns ScalarField3D.");
    m.def("topological_charge",
          [](const VectorField3D& m) {
              auto [Q, dens] = topological_charge(m);
              return py::make_tuple(Q, dens);
          },
          py::arg("m"),
          "Returns (Q, density_field) where Q is the total topological charge "
          "and density_field is per-cell q (ScalarField3D, no 1/4π factor).");

    // Phase G: skyrmion tracking + counting
    m.def("skyrmion_corepos",
          [](const VectorField3D& m, bool find_max) {
              auto [cx, cy] = skyrmion_corepos(m, find_max);
              return py::make_tuple(cx, cy);
          },
          py::arg("m"), py::arg("find_max") = false,
          "Skyrmion core position (box-centred x, y) [m].\n"
          "find_max=False: locate min mz (pol=+1 skyrmion, mz_core=-1).\n"
          "find_max=True:  locate max mz (pol=-1 skyrmion, mz_core=+1).\n"
          "Returns (cx, cy) in metres.");

    m.def("bubble_pos",
          [](const VectorField3D& m) {
              auto [cx, cy] = bubble_pos(m);
              return py::make_tuple(cx, cy);
          },
          py::arg("m"),
          "Topological-charge-density-weighted centroid (cx, cy) [m].\n"
          "More robust than skyrmion_corepos for asymmetric states.");

    m.def("skyrmion_count", &skyrmion_count,
          py::arg("m"), py::arg("threshold") = Real{0.5},
          "Count skyrmions via connected-component analysis of Q-density map.\n"
          "Each connected region with |Q|>=threshold counts as one skyrmion.");

    py::enum_<BoundaryCondition>(m, "BoundaryCondition")
        .value("Neumann",  BoundaryCondition::Neumann)
        .value("Periodic", BoundaryCondition::Periodic);

    py::class_<IEffectiveField, PyIEffectiveField, std::shared_ptr<IEffectiveField>>(m, "IEffectiveField")
        .def(py::init<>())
        .def("accumulate",      &IEffectiveField::accumulate)
        .def("energy",          &IEffectiveField::energy)
        .def("energy_density",  &IEffectiveField::energy_density,
             py::arg("m"), py::arg("mat"),
             "Per-cell energy density [J/m³]. Returns ScalarField3D (mumax3 Edens_*).")
        .def_property_readonly("name", &IEffectiveField::name);

    py::class_<ZeemanField, IEffectiveField, std::shared_ptr<ZeemanField>>(m, "ZeemanField")
        .def(py::init<const Vec3&>(), py::arg("H_ext") = Vec3{0, 0, 0})
        .def_property("H_ext", &ZeemanField::H_ext, &ZeemanField::set_H_ext);

    py::class_<UniaxialAnisotropyField, IEffectiveField,
               std::shared_ptr<UniaxialAnisotropyField>>(m, "UniaxialAnisotropyField")
        .def(py::init<>())
        .def("set_material_field",
             [](UniaxialAnisotropyField& f, const MaterialField3D& matf) { f.set_material_field(&matf); },
             py::arg("material_field"), py::keep_alive<1, 2>(),
             "Attach per-cell K_uniaxial/easy_axis/Ms (MaterialField3D); "
             "mumax3 \"Regions\"-style spatially-varying anisotropy.")
        .def("clear_material_field", &UniaxialAnisotropyField::clear_material_field,
             "Remove per-cell material field (use uniform Material).");

    // SurfaceAnisotropyField — interface/surface PMA (mumax3 Ks parameter)
    py::class_<SurfaceAnisotropyField, IEffectiveField,
               std::shared_ptr<SurfaceAnisotropyField>>(m, "SurfaceAnisotropyField",
        "Interface/surface anisotropy K_s [J/m²] applied to boundary cells.\n\n"
        "Equivalent to mumax3's Ks parameter.\n"
        "H_s = (2Ks / (mu0 Ms t_cell)) * (m dot n_hat) * n_hat\n\n"
        "Only surface cells — cells adjacent to vacuum (mask < 0.5) along n_hat —\n"
        "receive this field.  Interior cells are unaffected.\n\n"
        "Example (Co/Pt PMA interface):\n"
        "  sa = mm.SurfaceAnisotropyField(Ks=1.2e-3)   # z-axis default\n"
        "  sa.set_mask(disk_mask)                        # optional geometry\n"
        "  heff.add(sa_ptr)")
        .def(py::init<Real, Vec3>(),
             py::arg("Ks"), py::arg("n_hat") = Vec3{0, 0, 1},
             "Ks    : surface anisotropy constant [J/m²] (Ks > 0 = PMA easy perp.)\n"
             "n_hat : surface normal direction (default z-axis)")
        .def("set_mask",
             [](SurfaceAnisotropyField& f, const GeomMask& mask) { f.set_mask(&mask); },
             py::arg("mask"), py::keep_alive<1, 2>(),
             "Attach geometry mask; surface cells are those adjacent to vacuum (mask<0.5).")
        .def("clear_mask", &SurfaceAnisotropyField::clear_mask,
             "Remove geometry mask (outermost layers treated as surface).")
        .def_property("Ks",    &SurfaceAnisotropyField::Ks,    &SurfaceAnisotropyField::set_Ks,
             "Surface anisotropy constant [J/m²].")
        .def_property("n_hat", &SurfaceAnisotropyField::n_hat, &SurfaceAnisotropyField::set_n_hat,
             "Surface normal direction (unit vector).")
        .def("accumulate", &SurfaceAnisotropyField::accumulate)
        .def("energy",     &SurfaceAnisotropyField::energy)
        .def_property_readonly("name", &SurfaceAnisotropyField::name);

    // MagnetoelasticField — B1/B2 magnetostrictive coupling (mumax3 B1/B2/exx/...)
    py::class_<MagnetoelasticField, IEffectiveField,
               std::shared_ptr<MagnetoelasticField>>(m, "MagnetoelasticField",
        "Magnetoelastic (magnetostrictive) effective field.\n\n"
        "Cubic symmetry coupling B1, B2 [J/m3] with uniform strain tensor.\n\n"
        "Energy density:\n"
        "  e = B1*(mx2*exx + my2*eyy + mz2*ezz)\n"
        "    + 2*B2*(mx*my*exy + my*mz*eyz + mx*mz*exz)\n\n"
        "Effective field:\n"
        "  H_x = -(2/mu0Ms)[B1*mx*exx + B2*(my*exy + mz*exz)]\n"
        "  (and cyclic permutations for H_y, H_z)\n\n"
        "Example (Ni, uniaxial strain along x):\n"
        "  me = mm.MagnetoelasticField(B1=-62.4e6, B2=-27.1e6)\n"
        "  me.exx = 0.001\n"
        "  heff.add(me_ptr)")
        .def(py::init<Real, Real>(),
             py::arg("B1") = Real{0}, py::arg("B2") = Real{0},
             "B1, B2 : magnetoelastic coupling constants [J/m3]")
        .def("set_strain", &MagnetoelasticField::set_strain,
             py::arg("exx"), py::arg("eyy") = Real{0}, py::arg("ezz") = Real{0},
             py::arg("exy") = Real{0}, py::arg("exz") = Real{0}, py::arg("eyz") = Real{0},
             "Set full strain tensor (uniform, all cells).")
        .def_property("B1",  &MagnetoelasticField::B1,  &MagnetoelasticField::set_B1)
        .def_property("B2",  &MagnetoelasticField::B2,  &MagnetoelasticField::set_B2)
        .def_property("exx", &MagnetoelasticField::exx, &MagnetoelasticField::set_exx)
        .def_property("eyy", &MagnetoelasticField::eyy, &MagnetoelasticField::set_eyy)
        .def_property("ezz", &MagnetoelasticField::ezz, &MagnetoelasticField::set_ezz)
        .def_property("exy", &MagnetoelasticField::exy, &MagnetoelasticField::set_exy)
        .def_property("exz", &MagnetoelasticField::exz, &MagnetoelasticField::set_exz)
        .def_property("eyz", &MagnetoelasticField::eyz, &MagnetoelasticField::set_eyz)
        // Per-cell strain fields (Phase V: spatially varying strain)
        .def("set_exx_field",
             [](MagnetoelasticField& f, const ScalarField3D& sf) { f.set_exx_field(&sf); },
             py::arg("field"), py::keep_alive<1, 2>(),
             "Set spatially varying exx field (ScalarField3D). Overrides scalar exx per cell.")
        .def("set_eyy_field",
             [](MagnetoelasticField& f, const ScalarField3D& sf) { f.set_eyy_field(&sf); },
             py::arg("field"), py::keep_alive<1, 2>())
        .def("set_ezz_field",
             [](MagnetoelasticField& f, const ScalarField3D& sf) { f.set_ezz_field(&sf); },
             py::arg("field"), py::keep_alive<1, 2>())
        .def("set_exy_field",
             [](MagnetoelasticField& f, const ScalarField3D& sf) { f.set_exy_field(&sf); },
             py::arg("field"), py::keep_alive<1, 2>())
        .def("set_exz_field",
             [](MagnetoelasticField& f, const ScalarField3D& sf) { f.set_exz_field(&sf); },
             py::arg("field"), py::keep_alive<1, 2>())
        .def("set_eyz_field",
             [](MagnetoelasticField& f, const ScalarField3D& sf) { f.set_eyz_field(&sf); },
             py::arg("field"), py::keep_alive<1, 2>())
        .def("clear_strain_fields", &MagnetoelasticField::clear_strain_fields,
             "Remove all per-cell strain fields (revert to uniform scalars).")
        .def_property_readonly("has_spatial_strain",
             &MagnetoelasticField::has_spatial_strain,
             "True if any per-cell strain field is attached.")
        .def("accumulate", &MagnetoelasticField::accumulate)
        .def("energy",     &MagnetoelasticField::energy)
        .def_property_readonly("name", &MagnetoelasticField::name);

    // CubicAnisotropyField — Kc1/Kc2 (mumax3 parameters)
    py::class_<CubicAnisotropyField, IEffectiveField,
               std::shared_ptr<CubicAnisotropyField>>(m, "CubicAnisotropyField")
        .def(py::init<Real, Real, Vec3, Vec3>(),
             py::arg("Kc1") = Real{0}, py::arg("Kc2") = Real{0},
             py::arg("c1") = Vec3{1,0,0}, py::arg("c2") = Vec3{0,1,0},
             "Cubic magnetocrystalline anisotropy.\n"
             "e = Kc1*(a1²a2² + a2²a3² + a3²a1²) + Kc2*(a1²a2²a3²)\n"
             "where ai = m·ci.  c3 = c1×c2 computed internally.\n"
             "Default: c1={1,0,0}, c2={0,1,0} (crystal aligned with grid).\n"
             "Fe: Kc1 ≈ +48kJ/m³, Ni: Kc1 ≈ -5kJ/m³.")
        .def_property("Kc1", &CubicAnisotropyField::Kc1, &CubicAnisotropyField::set_Kc1)
        .def_property("Kc2", &CubicAnisotropyField::Kc2, &CubicAnisotropyField::set_Kc2)
        .def_property_readonly("c1", &CubicAnisotropyField::c1)
        .def_property_readonly("c2", &CubicAnisotropyField::c2)
        .def_property_readonly("c3", &CubicAnisotropyField::c3)
        .def("set_axes", &CubicAnisotropyField::set_axes,
             py::arg("c1"), py::arg("c2"), "Set cubic axes (c3=c1×c2).");

    py::class_<ExchangeField, IEffectiveField, std::shared_ptr<ExchangeField>>(m, "ExchangeField")
        .def(py::init<BoundaryCondition>(), py::arg("bc") = BoundaryCondition::Neumann)
        .def_property("boundary", &ExchangeField::boundary, &ExchangeField::set_boundary)
        .def("set_mask",
             [](ExchangeField& f, const GeomMask& mask) { f.set_mask(&mask); },
             py::arg("mask"), py::keep_alive<1, 2>(),
             "Attach geometry mask: cells with mask<0.5 become Neumann boundaries.")
        .def("clear_mask", &ExchangeField::clear_mask,
             "Remove geometry mask (no masking).")
        .def("set_material_field",
             [](ExchangeField& f, const MaterialField3D& matf) { f.set_material_field(&matf); },
             py::arg("material_field"), py::keep_alive<1, 2>(),
             "Attach per-cell Ms/A_exchange (MaterialField3D); harmonic-mean "
             "stiffness A_ij at region boundaries.")
        .def("clear_material_field", &ExchangeField::clear_material_field,
             "Remove per-cell material field (use uniform Material).")
        // Inter-region exchange coupling (mumax3 SetInterExchange analog)
        .def("set_region_map",
             [](ExchangeField& f, const RegionMap& rm) { f.set_region_map(&rm); },
             py::arg("region_map"), py::keep_alive<1, 2>(),
             "Attach RegionMap so per-region-pair exchange coupling can be applied.")
        .def("clear_region_map", &ExchangeField::clear_region_map,
             "Detach RegionMap (revert to per-cell or uniform A).")
        .def("set_inter_exchange",
             [](ExchangeField& f, int ri, int rj, Real A_IEC) {
                 f.set_inter_exchange(static_cast<uint8_t>(ri),
                                      static_cast<uint8_t>(rj), A_IEC);
             },
             py::arg("region_i"), py::arg("region_j"), py::arg("A_IEC"),
             "Set exchange coupling A_IEC [J/m] between regions region_i and region_j.\n"
             "Overrides harmonic-mean default at region boundaries.\n"
             "A_IEC=0 → no exchange at that boundary (effectively cuts exchange).\n"
             "Symmetric: set_inter_exchange(i,j,A) also sets (j,i,A).")
        .def("inter_exchange",
             [](const ExchangeField& f, int ri, int rj) {
                 return f.inter_exchange(static_cast<uint8_t>(ri),
                                         static_cast<uint8_t>(rj));
             },
             py::arg("region_i"), py::arg("region_j"),
             "Query stored inter-exchange A [J/m] between two regions (-1 = not set).")
        .def("clear_inter_exchange", &ExchangeField::clear_inter_exchange,
             "Remove all inter-exchange overrides (revert to harmonic-mean defaults).");

    py::class_<DemagField, IEffectiveField, std::shared_ptr<DemagField>>(m, "DemagField")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"))
        .def("accumulate", &DemagField::accumulate)
        .def("energy",     &DemagField::energy)
        .def_property_readonly("name", &DemagField::name)
        .def("set_material_field",
             [](DemagField& f, const MaterialField3D& matf) { f.set_material_field(&matf); },
             py::arg("material_field"), py::keep_alive<1, 2>(),
             "Attach per-cell Ms (MaterialField3D): M = Ms_i * m_i before FFT.")
        .def("clear_material_field", &DemagField::clear_material_field,
             "Remove per-cell material field (use uniform Material.Ms).");

    py::class_<DemagFieldPeriodic, IEffectiveField,
               std::shared_ptr<DemagFieldPeriodic>>(m, "DemagFieldPeriodic")
        .def(py::init<const StructuredGrid&, int>(),
             py::arg("grid"), py::arg("n_rep") = 2,
             "Periodic-BC demag field (no zero-padding). "
             "Uniform m → H=0 (k=0 zeroed). n_rep: image cells per side.")
        .def("accumulate", &DemagFieldPeriodic::accumulate)
        .def("energy",     &DemagFieldPeriodic::energy)
        .def_property_readonly("name", &DemagFieldPeriodic::name)
        .def("set_material_field",
             [](DemagFieldPeriodic& f, const MaterialField3D& matf) { f.set_material_field(&matf); },
             py::arg("material_field"), py::keep_alive<1, 2>(),
             "Attach per-cell Ms (MaterialField3D): M = Ms_i * m_i before FFT.")
        .def("clear_material_field", &DemagFieldPeriodic::clear_material_field,
             "Remove per-cell material field (use uniform Material.Ms).");

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
        .def("add",             &EffectiveFieldSum::add)
        .def("compute",         &EffectiveFieldSum::compute)
        .def("total_energy",    &EffectiveFieldSum::total_energy)
        .def("energy_density",  &EffectiveFieldSum::energy_density,
             py::arg("m"), py::arg("mat"),
             "Sum per-cell energy densities from all terms [J/m³]. Returns ScalarField3D.")
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
        .def_property("dt", &RK4Integrator::dt, &RK4Integrator::set_dt)
        .def("set_material_field",
             [](RK4Integrator& self, const MaterialField3D& matf) { self.set_material_field(&matf); },
             py::arg("material_field"), py::keep_alive<1, 2>(),
             "Attach per-cell alpha (MaterialField3D); mumax3 \"Regions\"-style damping.")
        .def("clear_material_field", &RK4Integrator::clear_material_field,
             "Remove per-cell material field (use uniform Material.alpha).");

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
        .def_property_readonly("dt_current", &RK45Integrator::dt_current)
        .def("set_material_field",
             [](RK45Integrator& self, const MaterialField3D& matf) { self.set_material_field(&matf); },
             py::arg("material_field"), py::keep_alive<1, 2>(),
             "Attach per-cell alpha (MaterialField3D); mumax3 \"Regions\"-style damping.")
        .def("clear_material_field", &RK45Integrator::clear_material_field,
             "Remove per-cell material field (use uniform Material.alpha).");

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
        .def_property("dt", &HeunIntegrator::dt, &HeunIntegrator::set_dt)
        .def("set_material_field",
             [](HeunIntegrator& self, const MaterialField3D& matf) { self.set_material_field(&matf); },
             py::arg("material_field"), py::keep_alive<1, 2>(),
             "Attach per-cell alpha (MaterialField3D); mumax3 \"Regions\"-style damping.")
        .def("clear_material_field", &HeunIntegrator::clear_material_field,
             "Remove per-cell material field (use uniform Material.alpha).");

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

    // IDemagGPU: abstract interface shared by DemagFieldGPU and DemagFieldPeriodicGPU.
    py::class_<IDemagGPU, std::shared_ptr<IDemagGPU>>(m, "IDemagGPU");

    // IEffectiveFieldGPU: abstract interface for all GPU fields usable with FieldSumGPU.
    // Must be registered before any class that inherits from it.
    py::class_<IEffectiveFieldGPU, std::shared_ptr<IEffectiveFieldGPU>>(
            m, "IEffectiveFieldGPU");

    py::class_<DemagFieldGPU, IEffectiveField, IDemagGPU,
               std::shared_ptr<DemagFieldGPU>>(
            m, "DemagFieldGPU")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             "GPU cuFFT demag field — drop-in for DemagField, 5–20× faster.")
        .def("accumulate",     &DemagFieldGPU::accumulate)
        .def("energy",         &DemagFieldGPU::energy)
        .def("energy_density", &DemagFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"),
             "Per-cell demag energy density [J/m³] (GPU accumulate + CPU reduce).")
        .def_property_readonly("name", &DemagFieldGPU::name);

    py::class_<DemagFieldPeriodicGPU, IEffectiveField, IDemagGPU,
               std::shared_ptr<DemagFieldPeriodicGPU>>(m, "DemagFieldPeriodicGPU")
        .def(py::init<const StructuredGrid&, int>(),
             py::arg("grid"), py::arg("n_rep") = 2,
             "GPU periodic-BC demag field (no zero-padding, 8x smaller FFT than "
             "DemagFieldGPU). Kernel precomputed on CPU via periodic Newell image "
             "sum then uploaded once. Uniform m -> H=0 (k=0 zeroed).")
        .def("accumulate",     &DemagFieldPeriodicGPU::accumulate)
        .def("energy",         &DemagFieldPeriodicGPU::energy)
        .def("energy_density", &DemagFieldPeriodicGPU::energy_density,
             py::arg("m"), py::arg("mat"))
        .def_property_readonly("name", &DemagFieldPeriodicGPU::name);

    py::class_<ExchangeFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<ExchangeFieldGPU>>(
            m, "ExchangeFieldGPU")
        .def(py::init<const StructuredGrid&, BoundaryCondition>(),
             py::arg("grid"),
             py::arg("bc") = BoundaryCondition::Neumann,
             "GPU 6-point Laplacian exchange field. "
             "bc=Neumann for open systems; bc=Periodic for periodic supercell "
             "(pair with DemagFieldPeriodicGPU for fully periodic GPU LLG).")
        .def("accumulate",     &ExchangeFieldGPU::accumulate)
        .def("energy",         &ExchangeFieldGPU::energy)
        .def("energy_density", &ExchangeFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"),
             "Per-cell exchange energy density [J/m³] (delegates to CPU grad²).")
        .def("set_material_field",
             [](ExchangeFieldGPU& f, const MaterialField3D& matf) { f.set_material_field(matf); },
             py::arg("matf"),
             "Upload per-cell A_exchange and Ms to GPU. Activates per-cell mode with "
             "harmonic-mean A at interfaces. Call again after MaterialField3D changes.")
        .def("clear_material_field", &ExchangeFieldGPU::clear_material_field,
             "Revert to uniform Material mode (free per-cell GPU buffers).")
        .def_property_readonly("has_material_field", &ExchangeFieldGPU::has_material_field)
        .def_property_readonly("name", &ExchangeFieldGPU::name);

    py::class_<ZeemanFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<ZeemanFieldGPU>>(
            m, "ZeemanFieldGPU")
        .def(py::init<const StructuredGrid&, Vec3>(),
             py::arg("grid"), py::arg("H_ext") = Vec3{0,0,0})
        .def("accumulate",     &ZeemanFieldGPU::accumulate)
        .def("energy",         &ZeemanFieldGPU::energy)
        .def("energy_density", &ZeemanFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"),
             "Per-cell Zeeman energy density [J/m³].")
        .def_property("H_ext", &ZeemanFieldGPU::H_ext, &ZeemanFieldGPU::set_H_ext)
        .def_property_readonly("name", &ZeemanFieldGPU::name);

    py::class_<UniaxialAnisotropyFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<UniaxialAnisotropyFieldGPU>>(
            m, "UniaxialAnisotropyFieldGPU")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             "GPU uniaxial anisotropy field (K, Ku2, and easy_axis from Material).")
        .def("accumulate",     &UniaxialAnisotropyFieldGPU::accumulate)
        .def("energy",         &UniaxialAnisotropyFieldGPU::energy)
        .def("energy_density", &UniaxialAnisotropyFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"),
             "Per-cell uniaxial anisotropy energy density [J/m³].")
        .def("set_material_field",
             [](UniaxialAnisotropyFieldGPU& f, const MaterialField3D& matf) { f.set_material_field(matf); },
             py::arg("matf"),
             "Upload per-cell K_uniaxial, easy_axis, and Ms to GPU. Activates "
             "per-cell mode for polycrystalline (voronoi_grains) simulations. "
             "Call again after MaterialField3D changes.")
        .def("clear_material_field", &UniaxialAnisotropyFieldGPU::clear_material_field,
             "Revert to uniform Material mode (free per-cell GPU buffers).")
        .def_property_readonly("has_material_field", &UniaxialAnisotropyFieldGPU::has_material_field)
        .def_property_readonly("name", &UniaxialAnisotropyFieldGPU::name);

    // Phase E: CubicAnisotropyFieldGPU
    py::class_<CubicAnisotropyFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<CubicAnisotropyFieldGPU>>(
            m, "CubicAnisotropyFieldGPU")
        .def(py::init<const StructuredGrid&, Real, Real, Vec3, Vec3>(),
             py::arg("grid"),
             py::arg("Kc1") = Real{0}, py::arg("Kc2") = Real{0},
             py::arg("c1") = Vec3{1,0,0}, py::arg("c2") = Vec3{0,1,0},
             "GPU cubic magnetocrystalline anisotropy (Fe/Ni). c3=c1×c2 auto-computed.")
        .def("accumulate",     &CubicAnisotropyFieldGPU::accumulate)
        .def("energy",         &CubicAnisotropyFieldGPU::energy)
        .def("energy_density", &CubicAnisotropyFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"))
        .def_property("Kc1", &CubicAnisotropyFieldGPU::Kc1, &CubicAnisotropyFieldGPU::set_Kc1)
        .def_property("Kc2", &CubicAnisotropyFieldGPU::Kc2, &CubicAnisotropyFieldGPU::set_Kc2)
        .def_property_readonly("c1", &CubicAnisotropyFieldGPU::c1)
        .def_property_readonly("c2", &CubicAnisotropyFieldGPU::c2)
        .def_property_readonly("c3", &CubicAnisotropyFieldGPU::c3)
        .def("set_axes", &CubicAnisotropyFieldGPU::set_axes,
             py::arg("c1"), py::arg("c2"))
        .def_property_readonly("name", &CubicAnisotropyFieldGPU::name);

    // ------------------------------------------------------------------
    // FieldSumGPU — GPU field compositor (arbitrary combination of GPU fields)
    // ------------------------------------------------------------------
    py::class_<FieldSumGPU>(m, "FieldSumGPU")
        .def(py::init<>(),
             "GPU field compositor. add() fields in evaluation order, then pass "
             "to integ.step(mat, demag, fields).")
        .def("add", [](FieldSumGPU& self, IEffectiveFieldGPU& f) { self.add(f); },
             py::arg("field"),
             "Append a GPU field (ExchangeFieldGPU, ZeemanFieldGPU, "
             "UniaxialAnisotropyFieldGPU, BulkDMIFieldGPU, etc.).")
        .def("clear", &FieldSumGPU::clear,
             "Remove all fields from this compositor.")
        .def("__len__", &FieldSumGPU::size);

    // ------------------------------------------------------------------
    // K2: BulkDMIFieldGPU + InterfacialDMIFieldGPU
    // ------------------------------------------------------------------
    py::class_<BulkDMIFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<BulkDMIFieldGPU>>(
            m, "BulkDMIFieldGPU")
        .def(py::init<const StructuredGrid&, Real>(),
             py::arg("grid"), py::arg("D") = Real{0},
             "GPU Bulk DMI (Bloch skyrmion, D>0). "
             "H = (2D/mu0/Ms) curl(m). Drop-in for BulkDMIField.")
        .def("accumulate",     &BulkDMIFieldGPU::accumulate)
        .def("energy",         &BulkDMIFieldGPU::energy)
        .def("energy_density", &BulkDMIFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"))
        .def_property("D",     &BulkDMIFieldGPU::D,     &BulkDMIFieldGPU::set_D)
        .def_property_readonly("name", &BulkDMIFieldGPU::name);

    py::class_<InterfacialDMIFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<InterfacialDMIFieldGPU>>(
            m, "InterfacialDMIFieldGPU")
        .def(py::init<const StructuredGrid&, Real>(),
             py::arg("grid"), py::arg("D") = Real{0},
             "GPU Interfacial DMI (Neel skyrmion, HM/FM interface, D>0). "
             "H = (2D/mu0/Ms)(∂mz/∂x, ∂mz/∂y, -∇xy·m). Drop-in for InterfacialDMIField.")
        .def("accumulate",     &InterfacialDMIFieldGPU::accumulate)
        .def("energy",         &InterfacialDMIFieldGPU::energy)
        .def("energy_density", &InterfacialDMIFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"))
        .def_property("D",     &InterfacialDMIFieldGPU::D,     &InterfacialDMIFieldGPU::set_D)
        .def_property_readonly("name", &InterfacialDMIFieldGPU::name);

    // ------------------------------------------------------------------
    // P4: RelaxGPU + MinimizeGPU
    // ------------------------------------------------------------------
    {
        using Opts = RelaxGPU::Options;
        py::class_<Opts>(m, "RelaxGPUOptions")
            .def(py::init<>())
            .def_readwrite("alpha_relax",   &Opts::alpha_relax)
            .def_readwrite("threshold",     &Opts::threshold)
            .def_readwrite("dt",            &Opts::dt)
            .def_readwrite("max_steps",     &Opts::max_steps)
            .def_readwrite("check_every",   &Opts::check_every)
            .def_readwrite("throw_on_max",  &Opts::throw_on_max);
    }

    py::class_<RelaxGPU>(m, "RelaxGPU")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             "GPU energy minimisation via damping-only LLG (no precession). "
             "Equivalent to mumax3 Relax(). "
             "Usage: upload(m); run(mat, demag, exch, zeeman); download(m_out).")
        .def("upload",   &RelaxGPU::upload,   py::arg("m"),
             "Upload initial magnetisation to GPU.")
        .def("download", &RelaxGPU::download, py::arg("m_out"),
             "Download current GPU magnetisation to CPU VectorField3D.")
        .def("run",
             [](RelaxGPU& self, const Material& mat,
                IDemagGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
                RelaxGPU::Options opts) {
                 return self.run(mat, demag, exch, zeeman, aniso, opts);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"), py::arg("aniso") = nullptr,
             py::arg("opts") = RelaxGPU::Options{},
             "Run GPU relax (fixed-field overload). Returns steps taken.")
        .def("run",
             [](RelaxGPU& self, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields,
                RelaxGPU::Options opts) {
                 return self.run(mat, demag, extra_fields, opts);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             py::arg("opts") = RelaxGPU::Options{},
             "Run GPU relax with FieldSumGPU (supports DMI, cubic anisotropy, etc.).")
        .def("max_torque_now",
             [](RelaxGPU& self, const Material& mat,
                IDemagGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso) {
                 return self.max_torque_now(mat, demag, exch, zeeman, aniso);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"), py::arg("aniso") = nullptr,
             "Compute max |m x H_eff| on current GPU state.")
        .def("max_torque_now",
             [](RelaxGPU& self, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields) {
                 return self.max_torque_now(mat, demag, extra_fields);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             "Compute max |m x H_eff| using FieldSumGPU.")
        .def_property_readonly("grid", &RelaxGPU::grid,
             py::return_value_policy::reference_internal);

    {
        using Opts = MinimizeGPU::Options;
        py::class_<Opts>(m, "MinimizeGPUOptions")
            .def(py::init<>())
            .def_readwrite("threshold",   &Opts::threshold)
            .def_readwrite("dt_init",     &Opts::dt_init)
            .def_readwrite("dt_max",      &Opts::dt_max)
            .def_readwrite("dt_min",      &Opts::dt_min)
            .def_readwrite("max_steps",   &Opts::max_steps)
            .def_readwrite("check_every", &Opts::check_every)
            .def_readwrite("throw_on_max",&Opts::throw_on_max);
    }

    py::class_<MinimizeGPU>(m, "MinimizeGPU")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"),
             "GPU steepest-descent minimisation with adaptive step size. "
             "Equivalent to mumax3 Minimize(). "
             "Requires one D2H energy scalar per step.")
        .def("upload",   &MinimizeGPU::upload,   py::arg("m"))
        .def("download", &MinimizeGPU::download, py::arg("m_out"))
        .def("run",
             [](MinimizeGPU& self, const Material& mat,
                IDemagGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman, UniaxialAnisotropyFieldGPU* aniso,
                MinimizeGPU::Options opts) {
                 return self.run(mat, demag, exch, zeeman, aniso, opts);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"), py::arg("aniso") = nullptr,
             py::arg("opts") = MinimizeGPU::Options{},
             "Run GPU minimize (fixed-field overload). Returns steps taken.")
        .def("run",
             [](MinimizeGPU& self, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields,
                MinimizeGPU::Options opts) {
                 return self.run(mat, demag, extra_fields, opts);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             py::arg("opts") = MinimizeGPU::Options{},
             "Run GPU minimize with FieldSumGPU. Returns steps taken.")
        .def_property_readonly("grid", &MinimizeGPU::grid,
             py::return_value_policy::reference_internal);

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
                IDemagGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman,
                UniaxialAnisotropyFieldGPU* aniso) {
                 integ.step(mat, demag, exch, zeeman, aniso);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"),
             py::arg("aniso") = static_cast<UniaxialAnisotropyFieldGPU*>(nullptr),
             "One full RK4 step on GPU (fixed-field overload).")
        .def("step",
             [](RK4IntegratorGPU& integ, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields) {
                 integ.step(mat, demag, extra_fields);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             "One full RK4 step using FieldSumGPU. Demag runs first, then extra_fields "
             "(exchange, zeeman, DMI, anisotropy, etc.) in add() order.")
        .def("step",
             [](RK4IntegratorGPU& integ, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields,
                SpinTorqueSumGPU& torques) {
                 integ.step(mat, demag, extra_fields, torques);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             py::arg("torques"),
             "One full RK4 step with FieldSumGPU + SpinTorqueSumGPU (STT/SOT). "
             "Spin torques are applied after LLG torque at each RK4 stage.")
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
                IDemagGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman,
                UniaxialAnisotropyFieldGPU* aniso) {
                 return integ.step(mat, demag, exch, zeeman, aniso);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"),
             py::arg("aniso") = static_cast<UniaxialAnisotropyFieldGPU*>(nullptr),
             "One adaptive DOPRI5 step (fixed-field overload). Returns dt taken.")
        .def("step",
             [](RK45IntegratorGPU& integ, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields) {
                 return integ.step(mat, demag, extra_fields);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             "One adaptive DOPRI5 step using FieldSumGPU. Returns dt taken.")
        .def("step",
             [](RK45IntegratorGPU& integ, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields,
                SpinTorqueSumGPU& torques) {
                 return integ.step(mat, demag, extra_fields, torques);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             py::arg("torques"),
             "One adaptive DOPRI5 step with FieldSumGPU + SpinTorqueSumGPU. Returns dt taken.")
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
                IDemagGPU& demag, ExchangeFieldGPU& exch,
                ZeemanFieldGPU& zeeman, Real T_K,
                UniaxialAnisotropyFieldGPU* aniso) {
                 integ.step(mat, demag, exch, zeeman, T_K, aniso);
             },
             py::arg("mat"), py::arg("demag"), py::arg("exch"),
             py::arg("zeeman"),
             py::arg("T_K")  = Real{0.0},
             py::arg("aniso") = static_cast<UniaxialAnisotropyFieldGPU*>(nullptr),
             "One Stratonovich Heun step on GPU. "
             "T_K=0 disables noise. demag may be DemagFieldGPU or DemagFieldPeriodicGPU.")
        .def("step",
             [](HeunIntegratorGPU& integ, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields, Real T_K) {
                 integ.step(mat, demag, extra_fields, T_K, nullptr);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             py::arg("T_K") = Real{0.0},
             "One Stratonovich Heun step with FieldSumGPU. T_K=0 disables noise.")
        .def("step",
             [](HeunIntegratorGPU& integ, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& extra_fields,
                Real T_K, SpinTorqueSumGPU& torques) {
                 integ.step(mat, demag, extra_fields, T_K, &torques);
             },
             py::arg("mat"), py::arg("demag"), py::arg("extra_fields"),
             py::arg("T_K"), py::arg("torques"),
             "One Stratonovich Heun step with FieldSumGPU + SpinTorqueSumGPU.")
        .def_property("dt", &HeunIntegratorGPU::dt, &HeunIntegratorGPU::set_dt);

    // ------------------------------------------------------------------
    // GPU Spin Torques: ISpinTorqueGPU, SpinTorqueSumGPU,
    //   SlonczewskiSTTGPU, SpinOrbitTorqueGPU, ZhangLiSTTGPU
    // ------------------------------------------------------------------

    // Base interface — registered before derived classes
    py::class_<ISpinTorqueGPU, std::shared_ptr<ISpinTorqueGPU>>(
            m, "ISpinTorqueGPU");

    py::class_<SpinTorqueSumGPU>(m, "SpinTorqueSumGPU")
        .def(py::init<>(),
             "GPU spin torque compositor. add() any ISpinTorqueGPU; "
             "pass to integ.step(mat, demag, fields, torques).")
        .def("add", [](SpinTorqueSumGPU& self, ISpinTorqueGPU& t) { self.add(t); },
             py::arg("torque"),
             "Append a spin torque term (reference held; caller owns lifetime).")
        .def("clear", &SpinTorqueSumGPU::clear, "Remove all spin torque terms.")
        .def("__len__", &SpinTorqueSumGPU::size);

    // SlonczewskiSTTGPU
    py::class_<SlonczewskiSTTGPU, ISpinTorqueGPU,
               std::shared_ptr<SlonczewskiSTTGPU>>(m, "SlonczewskiSTTGPU")
        .def(py::init<const StructuredGrid&, Real, Real, Real, Vec3, Real>(),
             py::arg("grid"), py::arg("J"), py::arg("P"), py::arg("d"),
             py::arg("p"), py::arg("beta") = Real{0.0},
             "GPU Slonczewski CPP-STT. "
             "J: current density [A/m²]; P: polarisation [0,1]; "
             "d: FL thickness [m]; p: reference polarisation direction; "
             "beta: field-like/damping-like ratio.")
        .def_property("J",    &SlonczewskiSTTGPU::J,    &SlonczewskiSTTGPU::set_J)
        .def_property("P",    &SlonczewskiSTTGPU::P,    &SlonczewskiSTTGPU::set_P)
        .def_property("beta", &SlonczewskiSTTGPU::beta, &SlonczewskiSTTGPU::set_beta)
        .def_property_readonly("d", &SlonczewskiSTTGPU::d)
        .def_property_readonly("p", &SlonczewskiSTTGPU::p);

    // SpinOrbitTorqueGPU
    py::class_<SpinOrbitTorqueGPU, ISpinTorqueGPU,
               std::shared_ptr<SpinOrbitTorqueGPU>>(m, "SpinOrbitTorqueGPU")
        .def(py::init<const StructuredGrid&, Real, Real, Real, Vec3, Real, Real>(),
             py::arg("grid"), py::arg("J_c"), py::arg("theta_SH"), py::arg("d_fm"),
             py::arg("sigma"), py::arg("eta_DL") = Real{1.0},
             py::arg("eta_FL") = Real{0.0},
             "GPU spin-orbit torque (spin Hall). "
             "J_c: charge current [A/m²]; theta_SH: spin Hall angle (signed); "
             "d_fm: FM thickness [m]; sigma: spin polarisation direction; "
             "eta_DL: damping-like efficiency; eta_FL: field-like efficiency.")
        .def_property("J_c",      &SpinOrbitTorqueGPU::J_c,      &SpinOrbitTorqueGPU::set_J_c)
        .def_property("theta_SH", &SpinOrbitTorqueGPU::theta_SH, &SpinOrbitTorqueGPU::set_theta_SH)
        .def_property("eta_DL",   &SpinOrbitTorqueGPU::eta_DL,   &SpinOrbitTorqueGPU::set_eta_DL)
        .def_property("eta_FL",   &SpinOrbitTorqueGPU::eta_FL,   &SpinOrbitTorqueGPU::set_eta_FL)
        .def_property_readonly("d_fm",  &SpinOrbitTorqueGPU::d_fm)
        .def_property_readonly("sigma", &SpinOrbitTorqueGPU::sigma);

    // ZhangLiSTTGPU
    py::class_<ZhangLiSTTGPU, ISpinTorqueGPU,
               std::shared_ptr<ZhangLiSTTGPU>>(m, "ZhangLiSTTGPU")
        .def(py::init<const StructuredGrid&, Vec3, Real, Real>(),
             py::arg("grid"), py::arg("J"), py::arg("P"), py::arg("xi"),
             "GPU Zhang-Li CIP-STT (current-driven DW motion). "
             "J: current density vector [A/m²]; P: spin polarisation; "
             "xi: non-adiabaticity parameter.")
        .def_property("J",  &ZhangLiSTTGPU::J,  &ZhangLiSTTGPU::set_J)
        .def_property("P",  &ZhangLiSTTGPU::P,  &ZhangLiSTTGPU::set_P)
        .def_property("xi", &ZhangLiSTTGPU::xi, &ZhangLiSTTGPU::set_xi);

    // Phase S+W: MagnetoelasticFieldGPU (also IEffectiveFieldGPU for FieldSumGPU)
    py::class_<MagnetoelasticFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<MagnetoelasticFieldGPU>>(m, "MagnetoelasticFieldGPU",
        "GPU magnetoelastic field (B1/B2, uniform strain, CUDA kernel).\n"
        "Drop-in GPU replacement for MagnetoelasticField.\n"
        "Per-cell strain not supported — use CPU MagnetoelasticField for that.")
        .def(py::init<Real, Real, const StructuredGrid&>(),
             py::arg("B1"), py::arg("B2"), py::arg("grid"))
        .def("set_strain", &MagnetoelasticFieldGPU::set_strain,
             py::arg("exx"), py::arg("eyy") = Real{0}, py::arg("ezz") = Real{0},
             py::arg("exy") = Real{0}, py::arg("exz") = Real{0}, py::arg("eyz") = Real{0})
        .def_property("B1",  &MagnetoelasticFieldGPU::B1,  &MagnetoelasticFieldGPU::set_B1)
        .def_property("B2",  &MagnetoelasticFieldGPU::B2,  &MagnetoelasticFieldGPU::set_B2)
        .def_property("exx", &MagnetoelasticFieldGPU::exx, &MagnetoelasticFieldGPU::set_exx)
        .def_property("eyy", &MagnetoelasticFieldGPU::eyy, &MagnetoelasticFieldGPU::set_eyy)
        .def_property("ezz", &MagnetoelasticFieldGPU::ezz, &MagnetoelasticFieldGPU::set_ezz)
        .def_property("exy", &MagnetoelasticFieldGPU::exy, &MagnetoelasticFieldGPU::set_exy)
        .def_property("exz", &MagnetoelasticFieldGPU::exz, &MagnetoelasticFieldGPU::set_exz)
        .def_property("eyz", &MagnetoelasticFieldGPU::eyz, &MagnetoelasticFieldGPU::set_eyz)
        .def("accumulate", &MagnetoelasticFieldGPU::accumulate)
        .def("energy",     &MagnetoelasticFieldGPU::energy)
        .def_property_readonly("name", &MagnetoelasticFieldGPU::name);

    // Phase S+W: SurfaceAnisotropyFieldGPU (also IEffectiveFieldGPU for FieldSumGPU)
    py::class_<SurfaceAnisotropyFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<SurfaceAnisotropyFieldGPU>>(m, "SurfaceAnisotropyFieldGPU",
        "GPU surface/interface anisotropy field (mumax3 Ks).\n"
        "Precomputes surface-cell mask at construction; CUDA kernel runs only\n"
        "on boundary cells. n_hat-change triggers mask rebuild (cheap).")
        .def(py::init<Real, const StructuredGrid&, Vec3>(),
             py::arg("Ks"), py::arg("grid"),
             py::arg("n_hat") = Vec3{0, 0, 1})
        .def("set_mask",
             [](SurfaceAnisotropyFieldGPU& f, const GeomMask& mask) { f.set_mask(mask); },
             py::arg("mask"), py::keep_alive<1, 2>())
        .def("clear_mask", &SurfaceAnisotropyFieldGPU::clear_mask)
        .def_property("Ks",    &SurfaceAnisotropyFieldGPU::Ks,    &SurfaceAnisotropyFieldGPU::set_Ks)
        .def_property("n_hat", &SurfaceAnisotropyFieldGPU::n_hat, &SurfaceAnisotropyFieldGPU::set_n_hat)
        .def("accumulate", &SurfaceAnisotropyFieldGPU::accumulate)
        .def("energy",     &SurfaceAnisotropyFieldGPU::energy)
        .def_property_readonly("name", &SurfaceAnisotropyFieldGPU::name);

    // RKKYFieldGPU — interlayer RKKY coupling GPU drop-in
    py::class_<RKKYFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<RKKYFieldGPU>>(m, "RKKYFieldGPU",
        "GPU drop-in for RKKYField (interlayer exchange coupling). "
        "H_RKKY = -J/(mu_0*Ms*d) * m_ref. "
        "Upload reference layer via set_ref(m_ref_cpu) before each step.")
        .def(py::init<const StructuredGrid&, Real, Real>(),
             py::arg("grid"), py::arg("J_RKKY"), py::arg("d_spacer"),
             py::keep_alive<1, 2>(),
             "grid: shared grid; J_RKKY [J/m²] (< 0 = AFM); d_spacer [m] spacer thickness.")
        .def("set_ref",
             [](RKKYFieldGPU& f, const VectorField3D& ref) { f.set_ref(ref); },
             py::arg("ref_m"),
             "Upload reference magnetization CPU -> GPU. "
             "Call after each step if reference layer evolves.")
        .def("accumulate",   &RKKYFieldGPU::accumulate)
        .def("energy",       &RKKYFieldGPU::energy)
        .def_property("J",   &RKKYFieldGPU::J, &RKKYFieldGPU::set_J)
        .def_property_readonly("d", &RKKYFieldGPU::d)
        .def_property_readonly("name", &RKKYFieldGPU::name);

    // ZeemanFieldSpatialGPU — per-cell spatial Zeeman field (GPU drop-in)
    py::class_<ZeemanFieldSpatialGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<ZeemanFieldSpatialGPU>>(m, "ZeemanFieldSpatialGPU",
        "Per-cell spatially-varying external field on GPU. "
        "Upload from VectorField3D via set_field(); updates propagate host->device. "
        "Drop-in for ZeemanFieldSpatial in GPU integrator pipelines.")
        .def(py::init<const StructuredGrid&>(), py::arg("grid"), py::keep_alive<1, 2>())
        .def("set_field",
             [](ZeemanFieldSpatialGPU& f, const VectorField3D& H) { f.set_field(H); },
             py::arg("H_field"),
             "Upload per-cell H field [A/m] from CPU VectorField3D to device. "
             "Call again to update (e.g. time-varying write-head field).")
        .def("accumulate",   &ZeemanFieldSpatialGPU::accumulate)
        .def("energy",       &ZeemanFieldSpatialGPU::energy)
        .def_property_readonly("name", &ZeemanFieldSpatialGPU::name);

#endif  // MICROMAG_CUDA

    // ------------------------------------------------------------------
    // DMI — Bulk and Interfacial Dzyaloshinskii-Moriya Interaction
    // ------------------------------------------------------------------

    py::class_<BulkDMIField, IEffectiveField, std::shared_ptr<BulkDMIField>>(
            m, "BulkDMIField")
        .def(py::init<Real>(), py::arg("D") = Real{0.0},
             "Bulk DMI field (mumax3 Dbulk). D in [J/m²]. "
             "H = (2D/μ₀Ms) ∇×m. Favours Bloch skyrmions.")
        .def("accumulate", &BulkDMIField::accumulate)
        .def("energy",     &BulkDMIField::energy)
        .def_property("D", &BulkDMIField::D, &BulkDMIField::set_D)
        .def_property_readonly("name", &BulkDMIField::name);

    py::class_<InterfacialDMIField, IEffectiveField,
               std::shared_ptr<InterfacialDMIField>>(m, "InterfacialDMIField")
        .def(py::init<Real>(), py::arg("D") = Real{0.0},
             "Interfacial DMI field (mumax3 Dind). D in [J/m²]. Film normal = ẑ. "
             "H_x=2D/μ₀Ms ∂mz/∂x, H_z=-2D/μ₀Ms(∂mx/∂x+∂my/∂y). "
             "Favours Néel skyrmions.")
        .def("accumulate", &InterfacialDMIField::accumulate)
        .def("energy",     &InterfacialDMIField::energy)
        .def_property("D", &InterfacialDMIField::D, &InterfacialDMIField::set_D)
        .def_property_readonly("name", &InterfacialDMIField::name);

    // ------------------------------------------------------------------
    // Zhang-Li STT — current-driven domain wall motion (CIP)
    // ------------------------------------------------------------------

    py::class_<ZhangLiSTT, ISpinTorque, std::shared_ptr<ZhangLiSTT>>(
            m, "ZhangLiSTT")
        .def(py::init<Vec3, Real, Real>(),
             py::arg("J"), py::arg("P"), py::arg("xi") = Real{0.04},
             "Zhang-Li current-in-plane STT (mumax3 J + Pol + xi).\n"
             "J: current density vector [A/m²]; P: polarisation; xi: non-adiabaticity.")
        .def("accumulate", &ZhangLiSTT::accumulate)
        .def("u",          &ZhangLiSTT::u, py::arg("Ms"),
             "Spin-drift velocity u = P*μ_B*|J|/(e*Ms) [m/s].")
        .def_property("J",  &ZhangLiSTT::J,  &ZhangLiSTT::set_J)
        .def_property("P",  &ZhangLiSTT::P,  &ZhangLiSTT::set_P)
        .def_property("xi", &ZhangLiSTT::xi, &ZhangLiSTT::set_xi)
        .def_property_readonly("name", &ZhangLiSTT::name);

    // ------------------------------------------------------------------
    // Relax / Minimize — energy minimisation
    // ------------------------------------------------------------------

    py::class_<RelaxOptions>(m, "RelaxOptions")
        .def(py::init<>())
        .def(py::init([](Real threshold, Real dt, Real alpha_relax,
                         int max_steps, bool throw_on_max_steps) {
                 RelaxOptions o;
                 o.threshold          = threshold;
                 o.dt                 = dt;
                 o.alpha_relax        = alpha_relax;
                 o.max_steps          = max_steps;
                 o.throw_on_max_steps = throw_on_max_steps;
                 return o;
             }),
             py::arg("threshold")          = Real{1.0},
             py::arg("dt")                 = Real{1e-12},
             py::arg("alpha_relax")        = Real{1.0},
             py::arg("max_steps")          = 500'000,
             py::arg("throw_on_max_steps") = false,
             "Keyword-argument constructor: RelaxOptions(threshold=1.0, dt=1e-12, …)")
        .def_readwrite("alpha_relax",        &RelaxOptions::alpha_relax)
        .def_readwrite("threshold",           &RelaxOptions::threshold)
        .def_readwrite("dt",                  &RelaxOptions::dt)
        .def_readwrite("max_steps",           &RelaxOptions::max_steps)
        .def_readwrite("throw_on_max_steps",  &RelaxOptions::throw_on_max_steps);

    m.def("max_torque",
          [](const VectorField3D& mv, const Material& mat,
             const EffectiveFieldSum& heff) {
              return max_torque(mv, mat, heff);
          },
          py::arg("m"), py::arg("mat"), py::arg("heff"),
          "Maximum |m × H_eff| over all cells [A/m]. "
          "Convergence criterion for Relax/Minimize.");

    m.def("relax",
          [](VectorField3D& mv, const Material& mat,
             const EffectiveFieldSum& heff, RelaxOptions opts,
             const MaterialField3D* matf, const SpinTorqueSum* stt) {
              return relax(mv, mat, heff, opts, matf, stt);
          },
          py::arg("m"), py::arg("mat"), py::arg("heff"),
          py::arg("opts")  = RelaxOptions{},
          py::arg("matf")  = static_cast<const MaterialField3D*>(nullptr),
          py::arg("stt")   = static_cast<const SpinTorqueSum*>(nullptr),
          "Energy minimisation via damping-only LLG (mumax3 Relax()). "
          "Runs until max|m×H_eff| < opts.threshold. Returns step count.");

    py::class_<MinimizeOptions>(m, "MinimizeOptions")
        .def(py::init<>())
        .def(py::init([](Real threshold, Real dt_init, Real dt_max, Real dt_min,
                         int max_steps, bool throw_on_max_steps) {
                 MinimizeOptions o;
                 o.threshold          = threshold;
                 o.dt_init            = dt_init;
                 o.dt_max             = dt_max;
                 o.dt_min             = dt_min;
                 o.max_steps          = max_steps;
                 o.throw_on_max_steps = throw_on_max_steps;
                 return o;
             }),
             py::arg("threshold")          = Real{1.0},
             py::arg("dt_init")            = Real{1e-12},
             py::arg("dt_max")             = Real{1e-10},
             py::arg("dt_min")             = Real{1e-17},
             py::arg("max_steps")          = 200'000,
             py::arg("throw_on_max_steps") = false,
             "Keyword-argument constructor: MinimizeOptions(threshold=1.0, dt_init=1e-12, …)")
        .def_readwrite("threshold",          &MinimizeOptions::threshold)
        .def_readwrite("dt_init",            &MinimizeOptions::dt_init)
        .def_readwrite("dt_max",             &MinimizeOptions::dt_max)
        .def_readwrite("dt_min",             &MinimizeOptions::dt_min)
        .def_readwrite("max_steps",          &MinimizeOptions::max_steps)
        .def_readwrite("throw_on_max_steps", &MinimizeOptions::throw_on_max_steps);

    m.def("minimize",
          [](VectorField3D& mv, const Material& mat,
             const EffectiveFieldSum& heff, MinimizeOptions opts,
             const MaterialField3D* matf) {
              return minimize(mv, mat, heff, opts, matf);
          },
          py::arg("m"), py::arg("mat"), py::arg("heff"),
          py::arg("opts") = MinimizeOptions{},
          py::arg("matf") = static_cast<const MaterialField3D*>(nullptr),
          "Energy minimisation via steepest descent with backtracking line search "
          "(mumax3 Minimize()). Returns step count.");

    // ------------------------------------------------------------------
    // OVF file I/O
    // ------------------------------------------------------------------

    py::enum_<OVFFormat>(m, "OVFFormat")
        .value("Text",    OVFFormat::Text,    "ASCII text — portable, larger files.")
        .value("Binary4", OVFFormat::Binary4, "IEEE 754 float (4 bytes) — half size, ~7 decimal digits.")
        .value("Binary8", OVFFormat::Binary8, "IEEE 754 double (8 bytes) — lossless, default.");

    m.def("save_ovf",
          &save_ovf,
          py::arg("filename"), py::arg("m"),
          py::arg("title")  = std::string("m"),
          py::arg("fmt")    = OVFFormat::Binary8,
          "Save VectorField3D to OVF 2.0 file (mumax3/OOMMF compatible). "
          "fmt: OVFFormat.Binary8 (default, lossless) or OVFFormat.Text.");

    m.def("load_ovf_grid",
          &load_ovf_grid,
          py::arg("filename"),
          "Read only grid dimensions from an OVF header. Returns StructuredGrid.");

    m.def("load_ovf_into",
          &load_ovf_into,
          py::arg("filename"), py::arg("m"),
          "Fill an existing VectorField3D from an OVF file (no grid allocation).");

    // load_ovf: the C++ version has a dangling StructuredGrid* bug (VectorField3D
    // stores a raw pointer to a local StructuredGrid that is destroyed on return).
    // The Python wrapper in __init__.py uses load_ovf_grid + VectorField3D() +
    // load_ovf_into to safely manage grid lifetime via keep_alive<1,2>.
    // This raw binding is kept for backward compatibility with direct _micromag usage.
    m.def("_load_ovf_raw",
          &load_ovf,
          py::arg("filename"),
          "Internal: load_ovf with dangling grid ref — use micromag.load_ovf() instead.");
}
