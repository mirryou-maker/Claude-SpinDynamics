"""Micromag: Python interface to the C++ micromagnetic core."""

import math as _math

from _micromag import (
    Vec3,
    StructuredGrid,
    VectorField3D,
    ScalarField3D,
    to_numpy,
    to_numpy_scalar,
    from_numpy,
    mean_magnetization,
    write_vtk_legacy,
    make_gaussian_field,
    # Phase B1: Geometry / Shape API
    GeomMask,
    union_,
    sub_,
    intersect_,
    ellipse,
    circle,
    rect,
    cylinder,
    translate,
    rotate,
    # Phase E: additional geometry shapes
    square,
    cuboid,
    sphere,
    ellipsoid,
    layer,
    layers,
    x_range,
    y_range,
    z_range,
    # Phase B2: MFM Imaging
    TipMode,
    MFMImage,
    # Phase 1b
    Material,
    BoundaryCondition,
    IEffectiveField,
    ZeemanField,
    ZeemanFieldSpatial,
    UniaxialAnisotropyField,
    SurfaceAnisotropyField,
    MagnetoelasticField,
    ExchangeField,
    DemagField,
    DemagFieldPeriodic,
    EffectiveFieldSum,
    RKKYField,
    # Phase E: cubic anisotropy
    CubicAnisotropyField,
    # Phase C1: per-cell material
    MaterialField3D,
    voronoi_grains,
    # Phase E: region map
    RegionMap,
    # Phase 1c
    gamma_0,
    llg_torque,
    RK4Integrator,
    RK45Integrator,
    RK45Options,
    HeunIntegrator,
    # Phase 1d
    ISpinTorque,
    SlonczewskiSTT,
    SpinOrbitTorque,
    SpinTorqueSum,
    # Thermal
    ThermalField,
    # DMI
    BulkDMIField,
    InterfacialDMIField,
    # Zhang-Li STT
    ZhangLiSTT,
    # Relax / Minimize
    RelaxOptions,
    MinimizeOptions,
    max_torque,
    relax,
    minimize,
    # OVF I/O
    OVFFormat,
    save_ovf,
    load_ovf_grid,
    load_ovf_into,
    _load_ovf_raw,
    # Phase E: initial magnetization states
    uniform_mag,
    neel_skyrmion,
    bloch_skyrmion,
    two_domain,
    vortex_state,
    random_mag,
    # Phase F: topological charge
    topological_charge_Q,
    topological_charge_density,
    topological_charge,
    # Phase G: skyrmion tracking
    skyrmion_corepos,
    bubble_pos,
    skyrmion_count,
    # CUDA availability probe
    cuda_available,
    # Phase N: MFM imaging
    MFMImage,
    TipMode,
)

# GPU classes are only present in the CUDA build — import conditionally
try:
    from _micromag import (
        IDemagGPU,                  # K1: abstract demag interface
        IEffectiveFieldGPU,         # P2: abstract GPU field interface
        FieldSumGPU,                # P2: GPU field compositor
        DemagFieldGPU,
        DemagFieldPeriodicGPU,      # J5: periodic-BC GPU demag
        BulkDMIFieldGPU,            # K2: GPU Bulk DMI (Bloch skyrmion)
        InterfacialDMIFieldGPU,     # K2: GPU Interfacial DMI (Neel skyrmion)
        RelaxGPU, RelaxGPUOptions,  # P4: GPU damping-only relax (mumax3 Relax equivalent)
        MinimizeGPU, MinimizeGPUOptions,  # P4: GPU steepest-descent minimize
        ExchangeFieldGPU,
        ZeemanFieldGPU,
        UniaxialAnisotropyFieldGPU,
        CubicAnisotropyFieldGPU,   # Phase E
        RK4IntegratorGPU,
        RK45IntegratorGPU,
        RK45GPUOptions,
        HeunIntegratorGPU,
        # P3: GPU spin torques
        ISpinTorqueGPU,
        SpinTorqueSumGPU,
        SlonczewskiSTTGPU,
        SpinOrbitTorqueGPU,
        ZhangLiSTTGPU,
        # Phase S: GPU magnetoelastic + surface anisotropy fields
        MagnetoelasticFieldGPU,
        SurfaceAnisotropyFieldGPU,
        # ZeemanFieldSpatialGPU — per-cell spatial external field GPU drop-in
        ZeemanFieldSpatialGPU,
        RKKYFieldGPU,               # interlayer RKKY coupling GPU drop-in
    )
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

__all__ = [
    # Grid / fields
    "Vec3", "StructuredGrid", "VectorField3D", "ScalarField3D",
    "to_numpy", "to_numpy_scalar", "from_numpy", "mean_magnetization",
    "write_vtk_legacy", "make_gaussian_field", "save_paraview", "save_paraview_series",
    # Geometry / Shape API (Phase B1 + E)
    "GeomMask", "union_", "sub_", "intersect_",
    "ellipse", "circle", "rect", "cylinder",
    "square", "cuboid", "sphere", "ellipsoid",
    "layer", "layers", "x_range", "y_range", "z_range",
    "translate", "rotate",
    "TipMode", "MFMImage",
    # Material / effective fields
    "Material", "BoundaryCondition", "IEffectiveField",
    "ZeemanField", "ZeemanFieldSpatial",
    "UniaxialAnisotropyField", "ExchangeField",
    "DemagField", "DemagFieldPeriodic", "EffectiveFieldSum",
    "RKKYField", "CubicAnisotropyField",
    # Per-cell material / regions
    "MaterialField3D", "voronoi_grains", "RegionMap",
    # DMI
    "BulkDMIField", "InterfacialDMIField",
    # Integrators
    "gamma_0", "llg_torque",
    "RK4Integrator", "RK45Integrator", "RK45Options",
    "HeunIntegrator", "ThermalField",
    # Spin torques
    "ISpinTorque", "SlonczewskiSTT", "SpinOrbitTorque", "SpinTorqueSum",
    "ZhangLiSTT",
    # Relax / Minimize
    "RelaxOptions", "MinimizeOptions", "max_torque", "relax", "minimize",
    # OVF I/O
    "OVFFormat", "save_ovf", "load_ovf",
    # Initial magnetization states (Phase E)
    "uniform_mag", "neel_skyrmion", "bloch_skyrmion",
    "two_domain", "vortex_state", "random_mag",
    # Run/Steps/RunWhile convenience
    "run", "steps", "run_while",
    # Topology (Phase F)
    "topological_charge_Q", "topological_charge_density", "topological_charge",
    # Skyrmion tracking (Phase G)
    "skyrmion_corepos", "bubble_pos", "skyrmion_count",
    # mumax3 Table / set_geom helpers (Phase F)
    "Table", "set_geom",
    # FMR / signal processing (Phase G)
    "sinc_pulse", "AutoSave",
    # Visualisation / output (Phase G)
    "snapshot", "cross_section_z", "cross_section_y", "cross_section_x",
    "grain_id_map", "make_scalar_gradient",
    # Phase H: inter-exchange + geometry utilities
    "snap", "invert_mask",
    # Phase I: mumax3 utility extensions
    "thermalize", "sinusoidal_field", "domain_wall_pos",
    "field_fft2d", "compute_heff",
    # Phase J: analysis + material utilities
    "save_profile", "load_profile", "normalize_field",
    "OVFReader", "OVFWriter",
    "rotate_mag", "checkerboard_regions", "zhang_li_from_current",
    # Phase L: grain boundary + OVF format + image geometry
    "adjacent_region_pairs", "set_grain_boundaries", "image_geom",
    # Phase M: torque field observable + custom field + stray field
    "get_torque_field", "max_torque_field",
    "PythonField", "stray_field",
    # Phase N: MFM, EdgeSmooth, poisson_disk_grains
    "MFMImage", "TipMode", "mfm_signal", "mfm_overlap_integral",
    "edge_smooth", "poisson_disk_grains",
    # Phase Q+S: magnetoelastic / magnetostrictive coupling (CPU + GPU)
    "MagnetoelasticField", "MagnetoelasticFieldGPU",
    # Phase P2+S: surface anisotropy GPU
    "SurfaceAnisotropyFieldGPU",
    # Phase U: hysteresis loop automation (CPU integrators)
    "hysteresis_loop",
    # Phase X: GPU convergence + GPU hysteresis loop
    "run_until_converged_gpu",
    "gpu_hysteresis_loop",
    # Phase Y: multi-layer material stack builders
    "bilayer", "trilayer", "saf_stack",
    # Phase R: convergence-based adaptive relaxation
    "run_until_converged",
    # Phase O: convergence observables + energy table + Table extensions
    "max_angle", "B_eff", "energy_table",
    # Phase P: FrozenSpins, def_region, SurfaceAnisotropyField
    "SurfaceAnisotropyField", "FrozenIntegrator", "def_region", "new_region_map",
    # Phase K: spin-wave dispersion wrapper
    "spin_wave_dispersion",
    # Phase D: dynamic write-head utility
    "moving_gaussian_field",
    # Utilities
    "cuda_available", "parameter_sweep", "multi_gpu_sweep",
    "batch_to_numpy", "save_animation",
    "skyrmion_phase_diagram_gpu", "bloch_dw_width",
    # Fast numpy-vectorized initializers (avoid Python triple-loop overhead)
    "neel_skyrmion_np", "bloch_dw_np",
    # SP#2 / grid-sizing utilities (pure Python)
    "exchange_length", "optimal_dx", "sp2_grid",
    # GPU classes (conditionally available — only in CUDA build)
    "IDemagGPU", "IEffectiveFieldGPU", "FieldSumGPU",
    "DemagFieldGPU", "DemagFieldPeriodicGPU",
    "BulkDMIFieldGPU", "InterfacialDMIFieldGPU",
    "RelaxGPU", "RelaxGPUOptions", "MinimizeGPU", "MinimizeGPUOptions",
    "ExchangeFieldGPU", "ZeemanFieldGPU", "ZeemanFieldSpatialGPU", "RKKYFieldGPU",
    "UniaxialAnisotropyFieldGPU", "CubicAnisotropyFieldGPU",
    "RK4IntegratorGPU", "RK45IntegratorGPU", "RK45GPUOptions",
    "HeunIntegratorGPU",
    "ISpinTorqueGPU", "SpinTorqueSumGPU",
    "SlonczewskiSTTGPU", "SpinOrbitTorqueGPU", "ZhangLiSTTGPU",
    # Integrator selection helper
    "recommend_integrator",
]

__version__ = "0.1.0"



# ---------------------------------------------------------------------------
# Pure-Python utility layer, split by domain (public API unchanged; __all__
# above is the contract). Each submodule re-exports through this package.
# ---------------------------------------------------------------------------
from .sim import *          # noqa: F401,F403,E402
from .analysis import *     # noqa: F401,F403,E402
from .io_utils import *     # noqa: F401,F403,E402
from .features import *     # noqa: F401,F403,E402
from .structures import *   # noqa: F401,F403,E402
from .sweep import *        # noqa: F401,F403,E402
from .viz import *          # noqa: F401,F403,E402

# mumax3 .mx3 script runner (lazy import to avoid circular import at package init)
def run_mx3(*args, **kwargs):
    """Parse and execute a mumax3-style .mx3 script. See micromag.mx3."""
    from .mx3 import run_mx3 as _run
    return _run(*args, **kwargs)


# ParaView export (lazy import). See micromag.paraview.
def save_paraview(*args, **kwargs):
    """Write a magnetization field to a ParaView-ready .vtk (vector m + scalar
    mx/my/mz/|m|/q_topo). See micromag.paraview.save_paraview."""
    from .paraview import save_paraview as _f
    return _f(*args, **kwargs)


def save_paraview_series(*args, **kwargs):
    """Write a time series (.vtk frames + .pvd) for ParaView animation.
    See micromag.paraview.save_paraview_series."""
    from .paraview import save_paraview_series as _f
    return _f(*args, **kwargs)
