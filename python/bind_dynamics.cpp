// Integrators, spin torques, thermal SLLG, numpy bridge, relax/minimize, OVF I/O.
#include "bind_common.hpp"

void bind_dynamics(py::module_& m) {
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
        .def(py::init<Real, Real, Real, Vec3, Real, Real>(),
             py::arg("J"), py::arg("P"), py::arg("d"), py::arg("p"),
             py::arg("beta") = Real{0.0}, py::arg("Lambda") = Real{1.0})
        .def("a_J",        &SlonczewskiSTT::a_J,
             py::arg("Ms"), py::arg("mdotp") = Real{1})
        .def_property("J",      &SlonczewskiSTT::J,      &SlonczewskiSTT::set_J)
        .def_property("P",      &SlonczewskiSTT::P,      &SlonczewskiSTT::set_P)
        .def_property("beta",   &SlonczewskiSTT::beta,   &SlonczewskiSTT::set_beta)
        .def_property("Lambda", &SlonczewskiSTT::Lambda, &SlonczewskiSTT::set_Lambda)
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
        .def("add",   &SpinTorqueSum::add, py::keep_alive<1, 2>())
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
    // DMI — Bulk and Interfacial Dzyaloshinskii-Moriya Interaction
    // ------------------------------------------------------------------

    py::class_<BulkDMIField, IEffectiveField, std::shared_ptr<BulkDMIField>>(
            m, "BulkDMIField")
        .def(py::init<Real>(), py::arg("D") = Real{0.0},
             "Bulk DMI field (mumax3 Dbulk). D in [J/m²]. "
             "H = (2D/μ₀Ms) ∇×m. Favours Bloch skyrmions.")
        .def("accumulate", &BulkDMIField::accumulate)
        .def_property("open_bc", &BulkDMIField::open_bc, &BulkDMIField::set_open_bc,
             "False (default): free-boundary DMI condition (mumax3 default). True: legacy naive stencil.")
        .def("energy",     &BulkDMIField::energy)
        .def_property("D", &BulkDMIField::D, &BulkDMIField::set_D)
        .def("set_material_field",
             [](BulkDMIField& f, const MaterialField3D& matf) { f.set_material_field(&matf); },
             py::arg("matf"), py::keep_alive<1, 2>(),
             "Per-cell Ms from a MaterialField3D (uniform D kept) — parity with "
             "the GPU DMI, exchange and anisotropy; drives DMI's 1/(mu0 Ms) "
             "prefactor per cell. Keep matf alive while attached.")
        .def("clear_material_field",
             [](BulkDMIField& f) { f.set_material_field(nullptr); })
        .def_property_readonly("has_material_field", &BulkDMIField::has_material_field)
        .def_property_readonly("name", &BulkDMIField::name);

    py::class_<InterfacialDMIField, IEffectiveField,
               std::shared_ptr<InterfacialDMIField>>(m, "InterfacialDMIField")
        .def(py::init<Real>(), py::arg("D") = Real{0.0},
             "Interfacial DMI field (mumax3 Dind). D in [J/m²]. Film normal = ẑ. "
             "H_x=2D/μ₀Ms ∂mz/∂x, H_z=-2D/μ₀Ms(∂mx/∂x+∂my/∂y). "
             "Favours Néel skyrmions.")
        .def("accumulate", &InterfacialDMIField::accumulate)
        .def_property("open_bc", &InterfacialDMIField::open_bc, &InterfacialDMIField::set_open_bc,
             "False (default): free-boundary DMI condition (mumax3 default). True: legacy naive stencil.")
        .def("energy",     &InterfacialDMIField::energy)
        .def_property("D", &InterfacialDMIField::D, &InterfacialDMIField::set_D)
        .def("set_material_field",
             [](InterfacialDMIField& f, const MaterialField3D& matf) { f.set_material_field(&matf); },
             py::arg("matf"), py::keep_alive<1, 2>(),
             "Per-cell Ms from a MaterialField3D (uniform D kept) — parity with "
             "the GPU DMI, exchange and anisotropy; drives DMI's 1/(mu0 Ms) "
             "prefactor per cell. Keep matf alive while attached.")
        .def("clear_material_field",
             [](InterfacialDMIField& f) { f.set_material_field(nullptr); })
        .def_property_readonly("has_material_field", &InterfacialDMIField::has_material_field)
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
        .def_property("thiaville_u", &ZhangLiSTT::thiaville_u,
                      &ZhangLiSTT::set_thiaville_u,
                      "mumax3/Thiaville convention: u scaled by 1/(1+xi^2)")
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
