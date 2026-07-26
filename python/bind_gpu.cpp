// GPU bindings (compiled only with MICROMAG_CUDA) + cuda_available/gpu_float32 probes.
#include "bind_common.hpp"
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
#include "micromag/depondt_integrator_gpu.hpp"
#include "micromag/magnetoelastic_gpu.hpp"
#include "micromag/surface_anisotropy_gpu.hpp"
#include "micromag/zeeman_spatial_gpu.hpp"
#include "micromag/rkky_gpu.hpp"
#include "micromag/spin_torque_gpu.hpp"
#include "micromag/gpu_probe.hpp"
#endif

void bind_gpu(py::module_& m) {
    // ------------------------------------------------------------------
    // CUDA availability probe (always defined; returns False in CPU build)
    // ------------------------------------------------------------------
#ifdef MICROMAG_CUDA
    // Not just "compiled with CUDA": also verifies the device can RUN this
    // build's kernel images (single-arch packages on a mismatched GPU used to
    // pass this check and then crash in the first GPU constructor).
    m.def("cuda_available", []() { return gpu_probe::kernel_ok(); },
          "True when CUDA support is compiled in AND the installed GPU can "
          "execute this build's kernels (probed once, cached).");
    m.def("gpu_diagnostic", []() { return gpu_probe::diagnostic(); },
          "Human-readable GPU/build compatibility report (device, compute "
          "capability, embedded kernel archs, and why the GPU is unusable if so).");
#else
    m.def("cuda_available", []() { return false; },
          "True when CUDA support is compiled in AND the installed GPU can "
          "execute this build's kernels.");
    m.def("gpu_diagnostic", []() { return std::string("CPU-only build (no CUDA support compiled in)."); },
          "Human-readable GPU/build compatibility report.");
#endif
#ifdef MICROMAG_FLOAT32
    m.def("gpu_float32", []() { return true; },
          "True when GPU kernels use float32 (P11 MICROMAG_FLOAT32=ON).");
#else
    m.def("gpu_float32", []() { return false; },
          "True when GPU kernels use float32 (P11 MICROMAG_FLOAT32=ON).");
#endif

#ifdef MICROMAG_CUDA
    // ------------------------------------------------------------------
    // Phase G: GPU fields (IEffectiveField drop-ins, cuda preset only)
    // ------------------------------------------------------------------

    // IDemagGPU: abstract interface shared by DemagFieldGPU and DemagFieldPeriodicGPU.
    py::class_<IDemagGPU, std::shared_ptr<IDemagGPU>>(m, "IDemagGPU");
    py::class_<ZeroDemagGPU, IDemagGPU, std::shared_ptr<ZeroDemagGPU>>(
        m, "ZeroDemagGPU",
        "Null demag (no-op) for demag-disabled workflows that need an "
        "IDemagGPU argument (integrators, RelaxGPU, MinimizeGPU).")
        .def(py::init<>());

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
        .def("set_mask",
             [](ExchangeFieldGPU& f, const GeomMask& mask) { f.set_mask(mask); },
             py::arg("mask"),
             "Attach a geometry mask (mumax3 geometry). Cells with mask<0.5 are "
             "vacuum: skipped, with zero exchange flux across the interface.")
        .def("clear_mask", &ExchangeFieldGPU::clear_mask,
             "Detach the geometry mask (free GPU buffer).")
        .def_property_readonly("has_mask", &ExchangeFieldGPU::has_mask)
        .def("set_region_map",
             [](ExchangeFieldGPU& f, const RegionMap& rm) { f.set_region_map(rm); },
             py::arg("region_map"),
             "Attach a RegionMap so per-cell region IDs drive inter-region "
             "exchange coupling. Use with set_inter_exchange().")
        .def("clear_region_map", &ExchangeFieldGPU::clear_region_map,
             "Detach the region map (free GPU buffer).")
        .def_property_readonly("has_region_map", &ExchangeFieldGPU::has_region_map)
        .def("set_inter_exchange", &ExchangeFieldGPU::set_inter_exchange,
             py::arg("ri"), py::arg("rj"), py::arg("A_IEC"),
             "Set explicit exchange coupling A_IEC [J/m] across the bond between "
             "regions ri and rj (symmetric). A_IEC=0 cuts the bond. Unset pairs "
             "use the harmonic mean of the cells' A. Requires set_region_map().")
        .def("inter_exchange", &ExchangeFieldGPU::inter_exchange,
             py::arg("ri"), py::arg("rj"),
             "Return the inter-region coupling A_IEC for (ri,rj), or -1 if unset.")
        .def("clear_inter_exchange", &ExchangeFieldGPU::clear_inter_exchange,
             "Clear all inter-region coupling entries.")
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
        .def("set_Kc_field",
             [](CubicAnisotropyFieldGPU& f,
                const ScalarField3D& Kc1_f, const ScalarField3D& Kc2_f,
                const VectorField3D& c1_f,  const VectorField3D& c2_f,
                const ScalarField3D& Ms_f) {
                 f.set_Kc_field(Kc1_f, Kc2_f, c1_f, c2_f, Ms_f); },
             py::arg("Kc1_field"), py::arg("Kc2_field"),
             py::arg("c1_field"),  py::arg("c2_field"),
             py::arg("Ms_field"),
             "Upload per-cell Kc1, Kc2 [J/m³], cubic axes c1/c2 (VectorField3D), "
             "and Ms [A/m]. c3=c1×c2 computed on CPU before upload. "
             "Activates per-cell mode; use with voronoi_grains for polycrystalline Fe/Ni.")
        .def("clear_Kc_field", &CubicAnisotropyFieldGPU::clear_Kc_field,
             "Revert to uniform Kc1/Kc2 mode (frees per-cell GPU buffers).")
        .def_property_readonly("has_Kc_field", &CubicAnisotropyFieldGPU::has_Kc_field)
        .def_property_readonly("name", &CubicAnisotropyFieldGPU::name);

    // ------------------------------------------------------------------
    // FieldSumGPU — GPU field compositor (arbitrary combination of GPU fields)
    // ------------------------------------------------------------------
    py::class_<FieldSumGPU>(m, "FieldSumGPU")
        .def(py::init<>(),
             "GPU field compositor. add() fields in evaluation order, then pass "
             "to integ.step(mat, demag, fields).")
        .def("add", [](FieldSumGPU& self, IEffectiveFieldGPU& f) { self.add(f); },
             py::arg("field"), py::keep_alive<1, 2>(),
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
        .def_property("open_bc", &BulkDMIFieldGPU::open_bc, &BulkDMIFieldGPU::set_open_bc,
             "False (default): free-boundary DMI condition. True: legacy naive stencil.")
        .def("energy",         &BulkDMIFieldGPU::energy)
        .def("energy_density", &BulkDMIFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"))
        .def_property("D",     &BulkDMIFieldGPU::D,     &BulkDMIFieldGPU::set_D)
        .def("set_D_field",
             [](BulkDMIFieldGPU& f, const ScalarField3D& D_f, const ScalarField3D& Ms_f) {
                 f.set_D_field(D_f, Ms_f); },
             py::arg("D_field"), py::arg("Ms_field"),
             "Upload per-cell D [J/m²] and Ms [A/m] to GPU. Activates per-cell mode. "
             "Build D_field from voronoi_grains or manual ScalarField3D.")
        .def("clear_D_field", &BulkDMIFieldGPU::clear_D_field,
             "Revert to uniform D mode (frees per-cell GPU buffers).")
        .def_property_readonly("has_D_field", &BulkDMIFieldGPU::has_D_field)
        .def("set_material_field",
             [](BulkDMIFieldGPU& f, const MaterialField3D& matf) { f.set_material_field(matf); },
             py::arg("matf"),
             "Per-cell Ms from a MaterialField3D (uniform D kept) — parity with "
             "ExchangeFieldGPU/UniaxialAnisotropyFieldGPU so a granular material "
             "field drives DMI's 1/(mu0 Ms) prefactor per cell.")
        .def_property_readonly("name", &BulkDMIFieldGPU::name);

    py::class_<InterfacialDMIFieldGPU, IEffectiveField, IEffectiveFieldGPU,
               std::shared_ptr<InterfacialDMIFieldGPU>>(
            m, "InterfacialDMIFieldGPU")
        .def(py::init<const StructuredGrid&, Real>(),
             py::arg("grid"), py::arg("D") = Real{0},
             "GPU Interfacial DMI (Neel skyrmion, HM/FM interface, D>0). "
             "H = (2D/mu0/Ms)(∂mz/∂x, ∂mz/∂y, -∇xy·m). Drop-in for InterfacialDMIField.")
        .def("accumulate",     &InterfacialDMIFieldGPU::accumulate)
        .def_property("open_bc", &InterfacialDMIFieldGPU::open_bc, &InterfacialDMIFieldGPU::set_open_bc,
             "False (default): free-boundary DMI condition. True: legacy naive stencil.")
        .def("energy",         &InterfacialDMIFieldGPU::energy)
        .def("energy_density", &InterfacialDMIFieldGPU::energy_density,
             py::arg("m"), py::arg("mat"))
        .def_property("D",     &InterfacialDMIFieldGPU::D,     &InterfacialDMIFieldGPU::set_D)
        .def("set_D_field",
             [](InterfacialDMIFieldGPU& f, const ScalarField3D& D_f, const ScalarField3D& Ms_f) {
                 f.set_D_field(D_f, Ms_f); },
             py::arg("D_field"), py::arg("Ms_field"),
             "Upload per-cell D [J/m²] and Ms [A/m] to GPU. Activates per-cell mode.")
        .def("clear_D_field", &InterfacialDMIFieldGPU::clear_D_field,
             "Revert to uniform D mode.")
        .def_property_readonly("has_D_field", &InterfacialDMIFieldGPU::has_D_field)
        .def("set_material_field",
             [](InterfacialDMIFieldGPU& f, const MaterialField3D& matf) { f.set_material_field(matf); },
             py::arg("matf"),
             "Per-cell Ms from a MaterialField3D (uniform D kept) — parity with "
             "ExchangeFieldGPU/UniaxialAnisotropyFieldGPU.")
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
             py::keep_alive<1, 2>(),   // RelaxGPU stores const StructuredGrid* — keep grid alive
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
             py::keep_alive<1, 2>(),   // MinimizeGPU stores const StructuredGrid* — keep grid alive
             "GPU steepest-descent minimisation with adaptive step size. "
             "Equivalent to mumax3 Minimize(). "
             "Requires one D2H energy scalar per step.")
        .def("upload",   &MinimizeGPU::upload,   py::arg("m"))
        .def("download", &MinimizeGPU::download, py::arg("m_out"))
        .def("set_Ms_field", &MinimizeGPU::set_Ms_field, py::arg("Ms"),
             "Per-cell Ms weight for the line-search energy "
             "(E = -mu0/2 sum Ms_i m.H dV); use with per-cell material fields.")
        .def("clear_Ms_field", &MinimizeGPU::clear_Ms_field)
        .def_property_readonly("has_Ms_field", &MinimizeGPU::has_Ms_field)
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
        .def_property("dt", &RK4IntegratorGPU::dt, &RK4IntegratorGPU::set_dt)
        .def("max_angle_gpu", &RK4IntegratorGPU::max_angle_gpu,
             "Maximum misalignment angle between adjacent spins (degrees). "
             "Computed on GPU; only 1 double transferred D2H per call.")
        .def("invalidate_graph", &RK4IntegratorGPU::invalidate_graph,
             "P4: Force CUDA graph re-capture on the next step() call. "
             "Call after adding fields to FieldSumGPU or changing field objects.");

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
        .def_property_readonly("n_rejected", &RK45IntegratorGPU::n_rejected)
        .def("max_angle_gpu", &RK45IntegratorGPU::max_angle_gpu,
             "Maximum misalignment angle between adjacent spins (degrees). "
             "Computed on GPU; only 1 double transferred D2H per call.");

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
        .def_property("dt", &HeunIntegratorGPU::dt, &HeunIntegratorGPU::set_dt)
        .def("invalidate_graph", &HeunIntegratorGPU::invalidate_graph,
             "Force CUDA Graph re-capture on next T_K=0 step() call.")
        .def("max_angle_gpu", &HeunIntegratorGPU::max_angle_gpu,
             "Max misalignment angle between adjacent spins (°, GPU-side, no D2H transfer). "
             "Enables run_until_converged_gpu convergence check without downloading m.");

    // ------------------------------------------------------------------
    // DepondtMertensGPU — Task 1-A: rotation integrator, |m|=1 exact.
    //   integ = mm.DepondtMertensGPU(grid, dt, seed=42)
    //   integ.upload(m0); dt_used = integ.step(mat, demag, fields)  # T_K=0
    // Finite-T (T_K>0) and adaptive stepping are wired in Task 1-B/1-C.
    // ------------------------------------------------------------------
    py::class_<DepondtGPUOptions>(m, "DepondtGPUOptions")
        .def(py::init<>())
        .def_readwrite("adaptive", &DepondtGPUOptions::adaptive)
        .def_readwrite("rtol",     &DepondtGPUOptions::rtol)
        .def_readwrite("atol",     &DepondtGPUOptions::atol)
        .def_readwrite("dt_min",   &DepondtGPUOptions::dt_min)
        .def_readwrite("dt_max",   &DepondtGPUOptions::dt_max)
        .def_readwrite("safety",   &DepondtGPUOptions::safety)
        .def_readwrite("fac_min",  &DepondtGPUOptions::fac_min)
        .def_readwrite("fac_max",  &DepondtGPUOptions::fac_max);

    py::class_<DepondtMertensGPU>(m, "DepondtMertensGPU")
        .def(py::init<const StructuredGrid&, Real, unsigned>(),
             py::keep_alive<1, 2>(),
             py::arg("grid"), py::arg("dt"), py::arg("seed") = 42u)
        .def("upload",   &DepondtMertensGPU::upload,   py::arg("m"))
        .def("download", &DepondtMertensGPU::download, py::arg("m"))
        .def("step",
             [](DepondtMertensGPU& integ, const Material& mat,
                IDemagGPU& demag, FieldSumGPU& fields, Real T_K,
                SpinTorqueSumGPU* torques) {
                 return integ.step(mat, demag, fields, T_K, torques);
             },
             py::arg("mat"), py::arg("demag"), py::arg("fields"),
             py::arg("T_K") = 0.0, py::arg("torques") = nullptr,
             "One Depondt–Mertens step; returns the dt taken. T_K>0 not yet wired.")
        .def_property("dt", &DepondtMertensGPU::dt, &DepondtMertensGPU::set_dt)
        .def_property_readonly("options",
             [](DepondtMertensGPU& i) -> DepondtGPUOptions& { return i.options(); },
             py::return_value_policy::reference_internal)
        .def_static("therm_sigma", &DepondtMertensGPU::therm_sigma,
             py::arg("mat"), py::arg("dt"), py::arg("dx"), py::arg("dy"),
             py::arg("dz"), py::arg("T_K"),
             "Canonical thermal-field sigma (single 1/sqrt(dt) point, roadmap 1-D).")
        .def("max_angle_gpu", &DepondtMertensGPU::max_angle_gpu);

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
             py::arg("torque"), py::keep_alive<1, 2>(),
             "Append a spin torque term (kept alive for the compositor's lifetime).")
        .def("clear", &SpinTorqueSumGPU::clear, "Remove all spin torque terms.")
        .def("__len__", &SpinTorqueSumGPU::size);

    // SlonczewskiSTTGPU
    py::class_<SlonczewskiSTTGPU, ISpinTorqueGPU,
               std::shared_ptr<SlonczewskiSTTGPU>>(m, "SlonczewskiSTTGPU")
        .def(py::init<const StructuredGrid&, Real, Real, Real, Vec3, Real, Real>(),
             py::arg("grid"), py::arg("J"), py::arg("P"), py::arg("d"),
             py::arg("p"), py::arg("beta") = Real{0.0},
             py::arg("Lambda") = Real{1.0},
             "GPU Slonczewski CPP-STT (mumax3-compatible). "
             "J: current density [A/m²]; P: polarisation [0,1]; "
             "d: FL thickness [m]; p: reference polarisation direction; "
             "beta: field-like/damping-like ratio; "
             "Lambda: angular asymmetry, eps(m.p)=P*L^2/((L^2+1)+(L^2-1)(m.p)) "
             "(Lambda=1 -> eps=P/2 as in mumax3; Lambda->inf -> eps=P).")
        .def_property("J",      &SlonczewskiSTTGPU::J,      &SlonczewskiSTTGPU::set_J)
        .def_property("P",      &SlonczewskiSTTGPU::P,      &SlonczewskiSTTGPU::set_P)
        .def_property("beta",   &SlonczewskiSTTGPU::beta,   &SlonczewskiSTTGPU::set_beta)
        .def_property("Lambda", &SlonczewskiSTTGPU::Lambda, &SlonczewskiSTTGPU::set_Lambda)
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
        .def_property("xi", &ZhangLiSTTGPU::xi, &ZhangLiSTTGPU::set_xi)
        .def_property("thiaville_u", &ZhangLiSTTGPU::thiaville_u,
                      &ZhangLiSTTGPU::set_thiaville_u,
                      "mumax3/Thiaville convention: u scaled by 1/(1+xi^2)");

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
        .def("set_Ks_field",
             [](SurfaceAnisotropyFieldGPU& f, const ScalarField3D& Ks_f, const ScalarField3D& Ms_f) {
                 f.set_Ks_field(Ks_f, Ms_f); },
             py::arg("Ks_field"), py::arg("Ms_field"),
             "Upload per-cell Ks [J/m²] and Ms [A/m] to GPU. "
             "Only surface cells (from mask / boundary detection) receive H_s.")
        .def("clear_Ks_field", &SurfaceAnisotropyFieldGPU::clear_Ks_field,
             "Revert to uniform Ks mode.")
        .def_property_readonly("has_Ks_field", &SurfaceAnisotropyFieldGPU::has_Ks_field)
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
}
