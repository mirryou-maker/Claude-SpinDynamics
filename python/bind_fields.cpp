// Material, MaterialField3D, all CPU effective fields (+ PythonField trampoline).
#include "bind_common.hpp"
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


void bind_fields(py::module_& m) {
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
        .def("add",             &EffectiveFieldSum::add, py::keep_alive<1, 2>())
        .def("compute",         &EffectiveFieldSum::compute)
        .def("total_energy",    &EffectiveFieldSum::total_energy)
        .def("energy_density",  &EffectiveFieldSum::energy_density,
             py::arg("m"), py::arg("mat"),
             "Sum per-cell energy densities from all terms [J/m³]. Returns ScalarField3D.")
        .def_property_readonly("terms",     &EffectiveFieldSum::terms)
        .def_property_readonly("num_terms", &EffectiveFieldSum::num_terms);

}
