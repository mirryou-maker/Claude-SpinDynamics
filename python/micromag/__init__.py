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
    )
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

__all__ = [
    # Grid / fields
    "Vec3", "StructuredGrid", "VectorField3D", "ScalarField3D",
    "to_numpy", "to_numpy_scalar", "from_numpy", "mean_magnetization",
    "write_vtk_legacy", "make_gaussian_field",
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
    "cuda_available",
    # SP#2 / grid-sizing utilities (pure Python)
    "exchange_length", "optimal_dx", "sp2_grid",
    # GPU classes (conditionally available — only in CUDA build)
    "IDemagGPU", "IEffectiveFieldGPU", "FieldSumGPU",
    "DemagFieldGPU", "DemagFieldPeriodicGPU",
    "BulkDMIFieldGPU", "InterfacialDMIFieldGPU",
    "RelaxGPU", "RelaxGPUOptions", "MinimizeGPU", "MinimizeGPUOptions",
    "ExchangeFieldGPU", "ZeemanFieldGPU", "ZeemanFieldSpatialGPU",
    "UniaxialAnisotropyFieldGPU", "CubicAnisotropyFieldGPU",
    "RK4IntegratorGPU", "RK45IntegratorGPU", "RK45GPUOptions",
    "HeunIntegratorGPU",
    "ISpinTorqueGPU", "SpinTorqueSumGPU",
    "SlonczewskiSTTGPU", "SpinOrbitTorqueGPU", "ZhangLiSTTGPU",
]

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# A5: SP#2 grid-size utilities
# ---------------------------------------------------------------------------

_mu_0 = 4 * _math.pi * 1e-7


def exchange_length(mat) -> float:
    """Return the exchange length l_ex = sqrt(2A / mu_0 Ms^2) [m].

    This is the fundamental length scale below which exchange dominates
    over magnetostatics.  Used in mumax3 SP#2 to set grid resolution:
    dx = l_ex / factor  (factor ~ 20–30 for quantitative accuracy).

    Parameters
    ----------
    mat : Material
        Material with Ms and A_exchange set.

    Returns
    -------
    float
        Exchange length in metres.
    """
    return _math.sqrt(2.0 * mat.A_exchange / (_mu_0 * mat.Ms ** 2))


def optimal_dx(mat, cells_per_lex: float = 1.0) -> float:
    """Return recommended cell size dx = l_ex / cells_per_lex [m].

    Rule of thumb:
      cells_per_lex = 1.0  → dx = l_ex  (minimum; 1 cell per exchange length)
      cells_per_lex = 2.0  → dx = l_ex/2  (good accuracy)
      cells_per_lex = 5.0  → dx = l_ex/5  (high accuracy, slow)

    mumax3 SP#2 guidance: element size d = 30 l_ex, with dx ≈ l_ex (1:1).

    Parameters
    ----------
    mat           : Material
    cells_per_lex : float, optional
        Number of cells per exchange length (default 1 → dx = l_ex).

    Returns
    -------
    float
        Recommended cell size in metres.
    """
    return exchange_length(mat) / cells_per_lex


# ---------------------------------------------------------------------------
# Run(t) / Steps(n) — mumax3-style simulation helpers
# ---------------------------------------------------------------------------

def run(integ, m, mat, heff, t_total: float,
        stt=None, callback=None, callback_dt: float = 0.0):
    """Run the integrator for t_total seconds.

    Works with RK4Integrator (fixed dt), RK45Integrator (adaptive), and
    HeunIntegrator.  For RK4/Heun the step size is integ.dt; for RK45 the
    returned dt from step() is used.

    Parameters
    ----------
    integ      : RK4Integrator | RK45Integrator | HeunIntegrator
    m          : VectorField3D  (modified in-place)
    mat        : Material
    heff       : EffectiveFieldSum
    t_total    : float  — simulation time [s]
    stt        : SpinTorqueSum | None
    callback   : callable(t, m) | None — called periodically
    callback_dt: float — minimum interval between callback calls [s] (0 = every step)

    Returns
    -------
    float — actual simulated time
    """
    t = 0.0
    t_last_cb = -1.0
    while t < t_total:
        # Step
        result = integ.step(m, mat, heff, stt) if stt is not None else integ.step(m, mat, heff)
        # Determine dt used
        if isinstance(result, float):
            dt_used = result
        else:
            dt_used = integ.dt
        t += dt_used
        # Callback
        if callback is not None:
            if callback_dt <= 0 or (t - t_last_cb) >= callback_dt:
                callback(t, m)
                t_last_cb = t
    return t


def run_while(integ, m, mat, heff, condition, t_max: float = 10e-9,
              stt=None, callback=None, callback_dt: float = 0.0,
              check_every: int = 100):
    """Run until condition(t, m) returns False or t_max is reached.

    Mirrors mumax3's RunWhile().  The condition is checked every
    ``check_every`` steps (not every step, for performance).

    Parameters
    ----------
    integ       : RK4Integrator | RK45Integrator | HeunIntegrator
    m           : VectorField3D  (modified in-place)
    mat         : Material
    heff        : EffectiveFieldSum
    condition   : callable(t, m) -> bool  — continue while True
    t_max       : float — hard stop time [s] (default 10 ns)
    stt         : SpinTorqueSum | None
    callback    : callable(t, m) | None — called periodically
    callback_dt : float — minimum interval between callbacks [s]
    check_every : int — check condition every N steps (default 100)

    Returns
    -------
    float — simulated time at which the loop stopped

    Examples
    --------
    Run until <mx> < -0.9 (switching detection):

    >>> mm.run_while(integ, m, mat, heff,
    ...     condition=lambda t, m: mm.mean_magnetization(m)[0] > -0.9,
    ...     t_max=2e-9)

    Run until energy converges (relax-like):

    >>> prev_e = [float('inf')]
    >>> def not_converged(t, m):
    ...     e = heff.total_energy(m, mat)
    ...     conv = abs(e - prev_e[0]) / (abs(prev_e[0]) + 1e-30) < 1e-6
    ...     prev_e[0] = e
    ...     return not conv
    >>> mm.run_while(integ, m, mat, heff, not_converged, t_max=5e-9)
    """
    t = 0.0
    step_count = 0
    t_last_cb = -1.0

    while t < t_max:
        result = integ.step(m, mat, heff, stt) if stt is not None else integ.step(m, mat, heff)
        dt_used = result if isinstance(result, float) else integ.dt
        t += dt_used
        step_count += 1

        if callback is not None:
            if callback_dt <= 0 or (t - t_last_cb) >= callback_dt:
                callback(t, m)
                t_last_cb = t

        if step_count % check_every == 0 and not condition(t, m):
            break

    return t


def steps(integ, m, mat, heff, n: int, stt=None):
    """Run the integrator for exactly n steps.

    Parameters
    ----------
    integ : RK4Integrator | RK45Integrator | HeunIntegrator
    m     : VectorField3D  (modified in-place)
    mat   : Material
    heff  : EffectiveFieldSum
    n     : int — number of steps
    stt   : SpinTorqueSum | None

    Returns
    -------
    float — total simulated time (dt * n for fixed-step; sum for adaptive)
    """
    t = 0.0
    for _ in range(n):
        result = integ.step(m, mat, heff, stt) if stt is not None else integ.step(m, mat, heff)
        if isinstance(result, float):
            t += result
        else:
            t += integ.dt
    return t


def sp2_grid(mat, Lx: float, Ly: float, Lz: float,
             cells_per_lex: float = 1.0) -> "StructuredGrid":
    """Create a StructuredGrid sized by the SP#2 exchange-length criterion.

    Cell size = l_ex / cells_per_lex.

    Parameters
    ----------
    mat           : Material
    Lx, Ly, Lz   : float — element dimensions [m]
    cells_per_lex : float — cells per exchange length (default 1 → dx = l_ex)

    Returns
    -------
    StructuredGrid

    Example (SP#4 element, l_ex resolution)
    ----------------------------------------
    >>> mat = Material.permalloy()
    >>> g = sp2_grid(mat, 500e-9, 125e-9, 3e-9)
    >>> print(g.nx, f'{g.dx*1e9:.1f} nm')   # 88 cells, 5.7 nm
    >>> g2 = sp2_grid(mat, 500e-9, 125e-9, 3e-9, cells_per_lex=2)
    >>> print(g2.nx, f'{g2.dx*1e9:.1f} nm')  # 176 cells, 2.8 nm (more accurate)
    """
    dx = optimal_dx(mat, cells_per_lex)
    nx = max(1, round(Lx / dx))
    ny = max(1, round(Ly / dx))
    nz = max(1, round(Lz / dx))
    return StructuredGrid(nx, ny, nz,
                          Lx / nx, Ly / ny, Lz / nz)


# ---------------------------------------------------------------------------
# Table — mumax3-style data table (analogous to TableAdd / TableSave)
# ---------------------------------------------------------------------------

class Table:
    """Accumulate simulation data rows and save as CSV.

    Mirrors mumax3's TableAdd / TableSave workflow.

    Usage
    -----
    >>> tbl = mm.Table()
    >>> tbl.add_row(t, m, mat=mat, heff=heff)      # per-step callback
    >>> tbl.save("output/table.csv")

    Custom columns (mumax3 TableAdd analog)
    ----------------------------------------
    >>> tbl.add_column("Q", lambda t, m: mm.topological_charge_Q(m))
    >>> tbl.add_column("max_angle", lambda t, m: mm.max_angle(m))
    >>> # Now add_row() automatically evaluates and stores these columns.
    >>> tbl.add_row(t, m, mat=mat, heff=heff)

    Custom quantities via extra dict (legacy, still supported)
    -----------------------------------------------------------
    >>> tbl.add_row(t, m, extra={"Q": mm.topological_charge_Q(m)})
    """

    def __init__(self):
        self._header = None
        self._rows = []
        self._custom_cols = {}   # name -> callable(t, m)

    def add_column(self, name: str, fn):
        """Register a custom column evaluated on each add_row() call.

        Parameters
        ----------
        name : str   -- column label in the CSV header
        fn   : callable(t: float, m: VectorField3D) -> float

        Example
        -------
        >>> tbl.add_column("Q", lambda t, m: mm.topological_charge_Q(m))
        >>> tbl.add_column("max_angle", lambda t, m: mm.max_angle(m))
        """
        self._custom_cols[name] = fn

    def add_row(self, t: float, m, mat=None, heff=None, extra: dict = None):
        mx, my, mz = mean_magnetization(m)
        row = {"t": t, "mx": mx, "my": my, "mz": mz}
        if mat is not None and heff is not None:
            row["E_total"] = heff.total_energy(m, mat)
        # Evaluate registered custom columns
        for col_name, fn in self._custom_cols.items():
            try:
                row[col_name] = float(fn(t, m))
            except Exception:
                row[col_name] = float("nan")
        if extra:
            row.update(extra)
        if self._header is None:
            self._header = list(row.keys())
        self._rows.append([row.get(k, float("nan")) for k in self._header])

    def save(self, filename: str):
        with open(filename, "w") as f:
            f.write(",".join(self._header) + "\n")
            for row in self._rows:
                f.write(",".join(f"{v:.12g}" for v in row) + "\n")

    def to_numpy(self):
        """Return table data as numpy array, shape (n_rows, n_cols)."""
        import numpy as _np2
        return _np2.array(self._rows, dtype=float)

    def __len__(self):
        return len(self._rows)

    @property
    def columns(self):
        return list(self._header) if self._header else []


# ---------------------------------------------------------------------------
# set_geom — mumax3-style geometry setter
# ---------------------------------------------------------------------------

def set_geom(mask, m, exch=None):
    """Apply geometry mask to magnetization (and optionally Exchange field).

    Mirrors mumax3's SetGeom(shape): zeros m outside the shape and, if exch
    is supplied, sets the Neumann-BC mask on the ExchangeField so exchange
    stiffness is correctly blocked at the boundary.

    Parameters
    ----------
    mask : GeomMask   — 1 inside geometry, 0 outside
    m    : VectorField3D  — modified in-place (m→0 where mask<0.5)
    exch : ExchangeField | None — if given, calls exch.set_mask(mask)
    """
    m.apply_mask(mask)
    if exch is not None:
        exch.set_mask(mask)


# ===========================================================================
# Phase G — FMR / signal processing, visualisation, analysis utilities
# ===========================================================================

# ---------------------------------------------------------------------------
# AutoSave — mumax3-style periodic OVF snapshot saver
# ---------------------------------------------------------------------------

class AutoSave:
    """Periodic OVF snapshot saver (mumax3 AutoSave analog).

    Pass an instance as the ``callback`` argument to :func:`run`:

    >>> saver = mm.AutoSave('output/m', dt_save=10e-12)
    >>> mm.run(integ, m, mat, heff, t_total=1e-9, callback=saver)

    Snapshots are written as ``<prefix>_000000.ovf``, ``<prefix>_000001.ovf``, …

    Parameters
    ----------
    prefix   : str   — output filename prefix (directories must exist)
    dt_save  : float — minimum time interval between saves [s]
    fmt      : OVFFormat | None — file format (None → OVFFormat.Binary8)
    field_name : str — OVF field label (default 'm')
    """

    def __init__(self, prefix: str, dt_save: float, fmt=None, field_name: str = "m"):
        self._prefix = prefix
        self._dt = dt_save
        self._fmt = fmt
        self._field_name = field_name
        self._idx = 0
        self._t_last = -1e100

    def __call__(self, t: float, m):
        if t - self._t_last < self._dt - 1e-20:
            return
        self._t_last = t
        fmt = self._fmt if self._fmt is not None else OVFFormat.Binary8
        fname = f"{self._prefix}_{self._idx:06d}.ovf"
        save_ovf(fname, m, self._field_name, fmt)
        self._idx += 1

    @property
    def count(self) -> int:
        """Number of files saved so far."""
        return self._idx


# ---------------------------------------------------------------------------
# sinc_pulse — broadband sinc excitation for FMR spectroscopy
# ---------------------------------------------------------------------------

def sinc_pulse(t: float, H0, f_max: float):
    """Sinc field for broadband FMR excitation.

    H(t) = H0 · sinc(2π f_max t)

    Flat power spectrum [0, f_max]; use as the applied field in a ZeemanField
    to excite all spin-wave modes simultaneously.

    Parameters
    ----------
    t     : float — current time [s]
    H0    : Vec3  — peak field amplitude [A/m]
    f_max : float — maximum excited frequency [Hz]

    Returns
    -------
    Vec3  — field at time t

    Example
    -------
    >>> zeeman = mm.ZeemanField(mm.Vec3(0, 0, 0))
    >>> def cb(t, m):
    ...     zeeman.set_H_ext(mm.sinc_pulse(t, mm.Vec3(1e3, 0, 0), 50e9))
    """
    x = 2.0 * _math.pi * f_max * t
    scale = (_math.sin(x) / x) if abs(x) > 1e-10 else 1.0
    return Vec3(H0.x * scale, H0.y * scale, H0.z * scale)


# ---------------------------------------------------------------------------
# make_scalar_gradient — linear gradient ScalarField3D
# ---------------------------------------------------------------------------

import numpy as _np


def make_scalar_gradient(grid, v0: float, v1: float, axis: str = "x"):
    """Create a ScalarField3D with a linear gradient from v0 to v1 along axis.

    Useful for spatially varying material parameters (e.g., Ms gradient, K gradient).

    Parameters
    ----------
    grid : StructuredGrid
    v0   : float — value at the low edge along ``axis``
    v1   : float — value at the high edge along ``axis``
    axis : str   — 'x', 'y', or 'z'

    Returns
    -------
    ScalarField3D
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    arr = _np.zeros((nz, ny, nx), dtype=_np.float64)
    if axis == "x":
        t = (_np.arange(nx) + 0.5) / nx          # shape (nx,)
        arr[:, :, :] = v0 + t * (v1 - v0)        # broadcast
    elif axis == "y":
        t = (_np.arange(ny) + 0.5) / ny          # shape (ny,)
        arr[:, :, :] = (v0 + t * (v1 - v0))[_np.newaxis, :, _np.newaxis]
    else:
        t = (_np.arange(nz) + 0.5) / nz          # shape (nz,)
        arr[:, :, :] = (v0 + t * (v1 - v0))[:, _np.newaxis, _np.newaxis]
    return from_numpy(grid, arr.reshape(nz * ny * nx))


# ---------------------------------------------------------------------------
# Cross-section extractors
# ---------------------------------------------------------------------------

def cross_section_z(m, iz: int = 0):
    """Extract xy-plane at layer iz.

    Returns
    -------
    numpy array, shape (ny, nx, 3)
    """
    import numpy as np
    arr = to_numpy(m)          # shape: (nz, ny, nx, 3) in C-order from to_numpy
    return arr[iz, :, :, :]


def cross_section_y(m, iy: int = 0):
    """Extract xz-plane at row iy.

    Returns
    -------
    numpy array, shape (nz, nx, 3)
    """
    arr = to_numpy(m)
    return arr[:, iy, :, :]


def cross_section_x(m, ix: int = 0):
    """Extract yz-plane at column ix.

    Returns
    -------
    numpy array, shape (nz, ny, 3)
    """
    arr = to_numpy(m)
    return arr[:, :, ix, :]


# ---------------------------------------------------------------------------
# snapshot — quick matplotlib visualisation
# ---------------------------------------------------------------------------

def snapshot(m, filename: str, component: str = "z",
             colormap: str = "RdBu", vmin: float = -1.0, vmax: float = 1.0,
             title: str = None, iz: int = 0, dpi: int = 150):
    """Save a quick matplotlib image of m_component at layer iz.

    Parameters
    ----------
    m         : VectorField3D
    filename  : str  — output path (.png, .pdf, …)
    component : str  — 'x', 'y', or 'z'
    colormap  : str  — matplotlib colormap name
    vmin, vmax: float — colour range
    title     : str | None — axes title (auto-generated if None)
    iz        : int  — z-layer to visualise
    dpi       : int  — output resolution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = to_numpy(m)                              # (nz, ny, nx, 3)
    comp_idx = {"x": 0, "y": 1, "z": 2}[component]
    data = arr[iz, :, :, comp_idx]                # (ny, nx)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, origin="lower", cmap=colormap, vmin=vmin, vmax=vmax,
                   aspect="equal")
    plt.colorbar(im, ax=ax, label=f"m{component}")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title(title or f"m{component}  (iz={iz})")
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# grain_id_map — visualise RegionMap as numpy uint8 array
# ---------------------------------------------------------------------------

def grain_id_map(region_map):
    """Convert a RegionMap to a (nz, ny, nx) uint8 numpy array.

    Each element contains the region ID (0–255) of that cell, suitable for
    imshow or export.

    Parameters
    ----------
    region_map : RegionMap

    Returns
    -------
    numpy array, shape (nz, ny, nx), dtype uint8
    """
    import numpy as np
    g = region_map.grid
    N = g.size
    ids = np.array([region_map[i] for i in range(N)], dtype=np.uint8)
    return ids.reshape(g.nz, g.ny, g.nx)


# ---------------------------------------------------------------------------
# snap — numbered OVF snapshot (mumax3 Snap analog)
# ---------------------------------------------------------------------------

class _SnapCounter:
    """Per-prefix sequential counter for snap()."""
    _counts: dict = {}

    @classmethod
    def next(cls, prefix: str) -> int:
        n = cls._counts.get(prefix, 0)
        cls._counts[prefix] = n + 1
        return n


def snap(m, prefix: str, field_name: str = "m", fmt=None) -> str:
    """Save a numbered OVF snapshot: ``<prefix>_000000.ovf``, ``…_000001.ovf``, …

    Each call increments the counter for the given ``prefix`` independently.
    Counters persist for the duration of the Python session.

    Parameters
    ----------
    m          : VectorField3D
    prefix     : str  — output filename prefix (directories must already exist)
    field_name : str  — OVF field label (default 'm')
    fmt        : OVFFormat | None — None → OVFFormat.Binary8

    Returns
    -------
    str — the filename that was written

    Example
    -------
    >>> for _ in range(100):
    ...     integ.step(m, mat, heff)
    ...     mm.snap(m, 'output/relax')   # writes relax_000000.ovf, …
    """
    fmt_ = fmt if fmt is not None else OVFFormat.Binary8
    idx = _SnapCounter.next(prefix)
    fname = f"{prefix}_{idx:06d}.ovf"
    save_ovf(fname, m, field_name, fmt_)
    return fname


def invert_mask(mask):
    """Return a new GeomMask with inverted occupancy (1 - v for each cell).

    Equivalent to ``~mask`` (Python bitwise NOT operator).

    Parameters
    ----------
    mask : GeomMask

    Returns
    -------
    GeomMask — new mask, does not modify the input
    """
    return ~mask


# ===========================================================================
# Phase I — mumax3 utility extensions
# ===========================================================================

def thermalize(m, mat, heff, T_K: float, t_therm: float = 0.5e-9,
               dt: float = 1e-13, seed: int = 42) -> float:
    """Run Stochastic LLG (Heun) to reach thermal equilibrium at T_K.

    Creates a temporary HeunIntegrator + ThermalField, runs SLLG for
    ``t_therm`` seconds.  ``m`` is modified in-place.

    Parameters
    ----------
    m       : VectorField3D  — modified in-place
    mat     : Material
    heff    : EffectiveFieldSum — deterministic fields (exchange, demag, …)
    T_K     : float — temperature [K]
    t_therm : float — equilibration time [s] (default 0.5 ns)
    dt      : float — Heun timestep [s] (default 0.1 ps)
    seed    : int   — random seed (default 42)

    Returns
    -------
    float — actual simulated time

    Example
    -------
    >>> mm.relax(m, mat, heff)          # find energy minimum first
    >>> mm.thermalize(m, mat, heff, T_K=300.0)  # add thermal fluctuations
    >>> mm.run(integ, m, mat, heff, 2e-9)       # dynamics at 300 K
    """
    heun    = HeunIntegrator(dt)
    thermal = ThermalField(m.grid, T_K, dt, seed)
    t = 0.0
    while t < t_therm:
        heun.step(m, mat, heff, thermal)
        t += dt
    return t


def sinusoidal_field(t: float, H0, freq: float):
    """Sinusoidal applied field for FMR / spin-wave ring-down excitation.

    H(t) = H0 · sin(2π·freq·t)

    Parameters
    ----------
    t    : float — current time [s]
    H0   : Vec3  — peak amplitude [A/m]
    freq : float — frequency [Hz]

    Returns
    -------
    Vec3 — field at time t

    Example
    -------
    >>> z_ac = mm.ZeemanField(mm.Vec3(0,0,0))
    >>> def cb(t, m): z_ac.H_ext = mm.sinusoidal_field(t, mm.Vec3(1e3,0,0), 2.8e9)
    >>> mm.run(integ, m, mat, heff, 5e-9, callback=cb)
    """
    phase = 2.0 * _math.pi * freq * t
    s = _math.sin(phase)
    return Vec3(H0.x * s, H0.y * s, H0.z * s)


def domain_wall_pos(m, component: int = 2, threshold: float = 0.0,
                    axis: int = 0) -> float:
    """Find domain wall position by linear interpolation along ``axis``.

    Averages ``m[component]`` over transverse dimensions, then locates the
    first zero crossing of (profile − threshold).

    Parameters
    ----------
    m         : VectorField3D
    component : int   — 0=mx, 1=my, 2=mz (default 2)
    threshold : float — crossing level (default 0.0 → 50 %-contour)
    axis      : int   — scan axis: 0=x, 1=y, 2=z

    Returns
    -------
    float — wall position in box-centred coords [m], or nan if no crossing
    """
    arr   = _np.array(to_numpy(m))          # (nz, ny, nx, 3)
    g     = m.grid
    sizes = (g.nx, g.ny, g.nz)
    steps = (g.dx, g.dy, g.dz)
    n  = sizes[axis]
    ds = steps[axis]

    if axis == 0:
        profile = arr[:, :, :, component].mean(axis=(0, 1))
    elif axis == 1:
        profile = arr[:, :, :, component].mean(axis=(0, 2))
    else:
        profile = arr[:, :, :, component].mean(axis=(1, 2))

    for i in range(n - 1):
        v0 = profile[i]     - threshold
        v1 = profile[i + 1] - threshold
        if v0 * v1 < 0:
            frac = -v0 / (v1 - v0)
            return (i + frac + 0.5) * ds - 0.5 * n * ds
    return float("nan")


def field_fft2d(data_xt, dt: float, dx: float):
    """2D FFT of m(x,t) data → spin-wave dispersion S(k, f).

    Parameters
    ----------
    data_xt : array (nt, nx) — time-series of one magnetization component
    dt      : float — time between frames [s]
    dx      : float — spatial cell size [m]

    Returns
    -------
    kvals : array (nx,) — wavenumber [rad/m], centred at k=0
    freqs : array (nt,) — frequency [Hz]
    S     : array (nt, nx) — |FFT|² power, k-axis fftshifted

    Example
    -------
    >>> kvals, freqs, S = mm.field_fft2d(my_xt, dt=10e-12, dx=20e-9)
    >>> plt.pcolormesh(kvals, freqs[:nt//2]/1e9, S[:nt//2], norm=LogNorm())
    """
    nt, nx = data_xt.shape
    F = _np.fft.fft2(data_xt)
    S = _np.fft.fftshift(_np.abs(F) ** 2, axes=1)
    freqs = _np.fft.fftfreq(nt, d=dt)
    kvals = _np.fft.fftshift(_np.fft.fftfreq(nx, d=dx / (2 * _math.pi)))
    return kvals, freqs, S


def compute_heff(m, mat, heff):
    """Compute H_eff and return as a new VectorField3D [A/m].

    Useful for visualising the effective field or computing derived
    quantities (torque, energy density) outside the integrator.

    Parameters
    ----------
    m    : VectorField3D
    mat  : Material
    heff : EffectiveFieldSum

    Returns
    -------
    VectorField3D — H_eff [A/m]
    """
    H = VectorField3D(m.grid)
    heff.accumulate(m, mat, H)
    return H


def get_torque_field(m, mat, heff, stt=None):
    """Compute per-cell LLG torque dm/dt [rad/s] as a VectorField3D.

    Evaluates H_eff, computes the Landau-Lifshitz torque per cell, and
    optionally adds spin-torque contributions.  Equivalent to mumax3's
    ``Torque`` quantity.

    The LLG torque (Landau-Lifshitz form):
      dm/dt = -γ'μ₀(m×H) - γ'αμ₀ m×(m×H)   (H in A/m, γ' = γ₀/(1+α²))

    Parameters
    ----------
    m    : VectorField3D
    mat  : Material
    heff : EffectiveFieldSum
    stt  : SpinTorqueSum | None — spin-torque contributions (STT, SOT, etc.)

    Returns
    -------
    VectorField3D — torque dm/dt [rad/s] at each cell

    Example
    -------
    >>> tau = mm.get_torque_field(m, mat, heff)
    >>> tau_np = mm.to_numpy(tau)          # shape (nz, ny, nx, 3)
    >>> # Max torque magnitude:
    >>> print(_np.sqrt((tau_np**2).sum(-1)).max())
    >>> # Compare with max_torque():
    >>> print(mm.max_torque(m, mat, heff))
    """
    g = m.grid
    # Compute H_eff (EffectiveFieldSum.compute zeros then accumulates all terms)
    H = VectorField3D(g)
    heff.compute(m, mat, H)

    # LLG torque per cell
    m_arr = _np.asarray(to_numpy(m))   # (nz, ny, nx, 3)
    H_arr = _np.asarray(to_numpy(H))   # (nz, ny, nx, 3)

    mu0   = 4e-7 * _math.pi
    alpha = mat.alpha
    gp    = gamma_0 * mu0 / (1.0 + alpha * alpha)

    mxH   = _np.cross(m_arr, H_arr)        # (nz, ny, nx, 3)
    mxmxH = _np.cross(m_arr, mxH)
    tau_arr = -(mxH + alpha * mxmxH) * gp  # (nz, ny, nx, 3)

    # Add spin torques if present
    if stt is not None:
        tau_stt = VectorField3D(g)
        stt.accumulate(m, mat, tau_stt)
        tau_arr = tau_arr + _np.asarray(to_numpy(tau_stt))

    result = VectorField3D(g)
    from_numpy(result, tau_arr)
    return result


def max_torque_field(m, mat, heff, stt=None):
    """Return the maximum |dm/dt| over all cells [rad/s].

    Equivalent to mm.max_torque() but computed via get_torque_field,
    allowing spin-torque contributions to be included.

    Parameters
    ----------
    m    : VectorField3D
    mat  : Material
    heff : EffectiveFieldSum
    stt  : SpinTorqueSum | None

    Returns
    -------
    float — maximum torque magnitude [rad/s]
    """
    tau = get_torque_field(m, mat, heff, stt)
    tau_arr = _np.asarray(to_numpy(tau))
    return float(_np.sqrt((tau_arr ** 2).sum(-1)).max())


# ===========================================================================
# Phase J — mumax3 analysis + material utilities
# ===========================================================================

def save_profile(m, fname: str, component: str = "z", axis: int = 0,
                 scale: float = 1.0) -> str:
    """Save a 1D averaged magnetization profile to CSV (mumax3 SaveProfile analog).

    Averages m[component] over the two transverse dimensions, then writes
    (position [m], value) rows.

    Parameters
    ----------
    m         : VectorField3D
    fname     : str   — output CSV filename
    component : str   — 'x', 'y', or 'z'
    axis      : int   — scan axis: 0=x, 1=y, 2=z
    scale     : float — multiply values before writing (e.g. for Ms·m)

    Returns
    -------
    str — the filename written
    """
    import csv
    arr   = _np.asarray(to_numpy(m))
    g     = m.grid
    sizes = (g.nx, g.ny, g.nz)
    steps = (g.dx, g.dy, g.dz)
    comp  = {"x": 0, "y": 1, "z": 2}[component]
    n, ds = sizes[axis], steps[axis]

    if axis == 0:
        profile = arr[:, :, :, comp].mean(axis=(0, 1))
    elif axis == 1:
        profile = arr[:, :, :, comp].mean(axis=(0, 2))
    else:
        profile = arr[:, :, :, comp].mean(axis=(1, 2))

    pos = (_np.arange(n) + 0.5) * ds - 0.5 * n * ds

    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"pos_{'xyz'[axis]}_m", f"m{component}"])
        for p, v in zip(pos, profile * scale):
            w.writerow([f"{p:.6e}", f"{v:.9g}"])
    return fname


def rotate_mag(m, theta: float, axis=(0.0, 0.0, 1.0)):
    """Rotate all magnetization vectors in-place by theta [rad] around axis.

    Uses Rodrigues' formula:
      m' = m cos θ + (k × m) sin θ + k(k·m)(1 − cos θ)

    Parameters
    ----------
    m     : VectorField3D  — modified in-place
    theta : float  — rotation angle [rad]
    axis  : Vec3 | tuple  — rotation axis (does not need to be normalised)

    Example
    -------
    >>> mm.rotate_mag(m, _math.pi/4, axis=mm.Vec3(0, 0, 1))   # 45° around z
    """
    arr = _np.asarray(to_numpy(m))
    if hasattr(axis, "x"):
        k = _np.array([axis.x, axis.y, axis.z], dtype=float)
    else:
        k = _np.asarray(axis, dtype=float)
    k = k / _np.linalg.norm(k)
    ct, st = _math.cos(theta), _math.sin(theta)
    kxm   = _np.cross(k, arr)
    kdotm = _np.einsum("...i,i->...", arr, k)[..., _np.newaxis]
    rot   = arr * ct + kxm * st + k * kdotm * (1 - ct)
    from_numpy(m, rot)


def load_ovf(filename: str) -> "VectorField3D":
    """Load an OVF 1.0/2.0 file into a new VectorField3D.

    Parameters
    ----------
    filename : str — OVF file path

    Returns
    -------
    VectorField3D with grid reconstructed from the OVF header.
    """
    g = load_ovf_grid(filename)   # Python-owned StructuredGrid
    m = VectorField3D(g)          # keep_alive<1,2> in binding keeps g alive
    load_ovf_into(filename, m)    # fill data (no new grid allocation)
    return m


def normalize_field(m):
    """Normalize all vectors in a VectorField3D to unit length (in-place).

    Equivalent to ``m.normalize()`` but available as a standalone function for
    consistency with other field-level helpers.

    Parameters
    ----------
    m : VectorField3D — modified in-place
    """
    m.normalize()


def load_profile(fname: str):
    """Load a 1D magnetization profile CSV written by save_profile().

    Parameters
    ----------
    fname : str — CSV file path (two columns: position [m], value)

    Returns
    -------
    tuple (positions, values) — both numpy float64 arrays of length N
    """
    import csv
    positions, values = [], []
    with open(fname, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            positions.append(float(row[0]))
            values.append(float(row[1]))
    return _np.array(positions), _np.array(values)


class OVFWriter:
    """Context-manager / callable OVF writer (class-based interface for save_ovf).

    Usage::

        writer = mm.OVFWriter("snapshot.ovf", field_name="m",
                              fmt=mm.OVFFormat.Binary8)
        writer.write(m)          # write once

        with mm.OVFWriter("state.ovf") as w:
            w.write(m)           # also writes on __exit__

    Parameters
    ----------
    path       : str — output filename
    field_name : str — OVF field label (default 'm')
    fmt        : OVFFormat | None — None -> OVFFormat.Binary8
    """
    def __init__(self, path: str, field_name: str = "m", fmt=None):
        self._path = path
        self._field_name = field_name
        self._fmt = fmt

    def write(self, m):
        """Write VectorField3D ``m`` to the configured file."""
        fmt_ = self._fmt if self._fmt is not None else OVFFormat.Binary8
        save_ovf(self._path, m, self._field_name, fmt_)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class OVFReader:
    """Context-manager / callable OVF reader (class-based interface for load_ovf).

    Usage::

        reader = mm.OVFReader("snapshot.ovf")
        m = reader.read(grid)   # allocates and fills VectorField3D

        with mm.OVFReader("state.ovf") as r:
            m = r.read(grid)

    Parameters
    ----------
    path : str — OVF file to read
    """
    def __init__(self, path: str):
        self._path = path

    def read(self, grid=None):
        """Load OVF file into a new VectorField3D.

        ``grid`` is accepted for API symmetry but ignored — the grid is
        reconstructed from the OVF header by load_ovf().
        """
        return load_ovf(self._path)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def checkerboard_regions(grid) -> "RegionMap":
    """Create a RegionMap with an alternating 0/1 checkerboard pattern.

    Region ID = (ix + iy + iz) % 2.  Useful for antiferromagnetic coupling:

    >>> rmap = mm.checkerboard_regions(g)
    >>> exch.set_region_map(rmap)
    >>> exch.set_inter_exchange(0, 1, -abs_A)   # antiferromagnetic

    Returns
    -------
    RegionMap (region 0 and region 1 in 3D checkerboard)
    """
    rm = RegionMap(grid, 0)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                rm[ix + nx * (iy + ny * iz)] = (ix + iy + iz) % 2
    return rm


def adjacent_region_pairs(region_map):
    """Find all pairs of adjacent region IDs in a RegionMap (grain-boundary finder).

    Scans all nearest-neighbour pairs (x+1, y+1, z+1) and collects unique
    (id_A, id_B) pairs where id_A != id_B.  Useful for automatically setting
    inter-exchange coupling across grain boundaries.

    Parameters
    ----------
    region_map : RegionMap

    Returns
    -------
    set of (int, int) — each pair appears once, with id_A < id_B

    Example
    -------
    >>> rmap = mm.voronoi_grains(g, n_grains=50)
    >>> pairs = mm.adjacent_region_pairs(rmap)
    >>> for (ri, rj) in pairs:
    ...     exch.set_inter_exchange(ri, rj, A_gb)  # grain-boundary A
    """
    g = region_map.grid
    nx, ny, nz = g.nx, g.ny, g.nz
    pairs = set()
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                lin = ix + nx * (iy + ny * iz)
                ri = int(region_map[lin])
                # Check +x neighbour
                if ix + 1 < nx:
                    rj = int(region_map[(ix+1) + nx * (iy + ny * iz)])
                    if ri != rj:
                        pairs.add((min(ri, rj), max(ri, rj)))
                # Check +y neighbour
                if iy + 1 < ny:
                    rj = int(region_map[ix + nx * ((iy+1) + ny * iz)])
                    if ri != rj:
                        pairs.add((min(ri, rj), max(ri, rj)))
                # Check +z neighbour
                if iz + 1 < nz:
                    rj = int(region_map[ix + nx * (iy + ny * (iz+1))])
                    if ri != rj:
                        pairs.add((min(ri, rj), max(ri, rj)))
    return pairs


def set_grain_boundaries(exch, region_map, A_gb: float):
    """Set uniform grain-boundary exchange across all adjacent region pairs.

    Convenience wrapper combining adjacent_region_pairs() and
    ExchangeField.set_inter_exchange().  Call after voronoi_grains() to
    apply a reduced exchange stiffness at grain boundaries.

    Parameters
    ----------
    exch       : ExchangeField
    region_map : RegionMap
    A_gb       : float — exchange constant at grain boundaries [J/m]
                         (typically 0.0 for decoupled grains, or
                         a fraction of material.A_exchange for partial coupling)

    Example
    -------
    >>> rmap = mm.voronoi_grains(g, n_grains=50)
    >>> exch.set_region_map(rmap)
    >>> mm.set_grain_boundaries(exch, rmap, A_gb=0.0)  # decouple grains
    """
    for ri, rj in adjacent_region_pairs(region_map):
        exch.set_inter_exchange(ri, rj, A_gb)


def zhang_li_from_current(j_amp: float, direction,
                           Ms: float, P: float = 0.5, xi: float = 0.04):
    """Create ZhangLiSTT from scalar current density and direction.

    Convenience wrapper for ZhangLiSTT(J_vec, P, xi) where J_vec is
    computed from a scalar magnitude and a direction vector.

    Parameters
    ----------
    j_amp     : float — current density magnitude [A/m²]
    direction : Vec3 | tuple — current direction (auto-normalised)
    Ms        : float — saturation magnetisation [A/m]  (used to report u)
    P         : float — spin polarisation (default 0.5)
    xi        : float — non-adiabaticity β (default 0.04)

    Returns
    -------
    ZhangLiSTT
    """
    if hasattr(direction, "x"):
        d = _np.array([direction.x, direction.y, direction.z], dtype=float)
    else:
        d = _np.asarray(direction, dtype=float)
    d = d / _np.linalg.norm(d)
    J_vec = Vec3(float(d[0] * j_amp), float(d[1] * j_amp), float(d[2] * j_amp))
    stt   = ZhangLiSTT(J_vec, P, xi)
    return stt


# ===========================================================================
# Phase K — spin-wave dispersion high-level wrapper
# ===========================================================================

def spin_wave_dispersion(grid, mat, B_bias: float, component: str = "y",
                         axis: int = 0,
                         t_sim: float = 2e-9, dt_sim: float = 0.5e-12,
                         dt_save: float = 10e-12,
                         H_pulse: float = 500.0, f_max: float = 20e9,
                         seed: int = 7):
    """Compute spin-wave dispersion S(k, f) for a 1D strip.

    Runs a sinc-pulse broadband spin-wave spectroscopy simulation and returns
    (kvals, freqs, S) ready for plotting. Equivalent to mumax3's dyn-matrix
    approach but via time-domain simulation + 2D FFT.

    Equilibrium: m along x, bias B_bias along x.
    Perturbation: sinc-pulse H_pulse along `component` axis.

    Parameters
    ----------
    grid      : StructuredGrid — 1D strip (ny=nz=1 recommended)
    mat       : Material
    B_bias    : float — bias field amplitude [T] (along x)
    component : str — magnetization component to record ('x','y','z')
    axis      : int — spatial axis to Fourier-transform (0=x,1=y,2=z)
    t_sim     : float — total simulation time [s]
    dt_sim    : float — integration time step [s]
    dt_save   : float — recording interval [s]
    H_pulse   : float — sinc-pulse amplitude [A/m]
    f_max     : float — sinc-pulse bandwidth [Hz]
    seed      : int   — RNG seed for initial noise

    Returns
    -------
    kvals : ndarray (nx,) — wavenumber [rad/m], centred
    freqs : ndarray (nt,) — frequency [Hz]
    S     : ndarray (nt, nx) — |FFT|² power spectrum

    Example
    -------
    >>> g   = mm.StructuredGrid(200, 1, 1, 20e-9, 20e-9, 20e-9)
    >>> mat = mm.Material.permalloy()
    >>> k, f, S = mm.spin_wave_dispersion(g, mat, B_bias=0.1)
    >>> plt.pcolormesh(k/1e6, f[:len(f)//2]/1e9, S[:len(f)//2], ...)
    """
    import numpy as _np2

    mu_0   = 4 * _math.pi * 1e-7
    H_bias = B_bias / mu_0

    # Fields
    exch        = ExchangeField(BoundaryCondition.Neumann)
    zeeman_bias = ZeemanField(Vec3(H_bias, 0, 0))
    zeeman_ac   = ZeemanField(Vec3(0, 0, 0))
    heff = EffectiveFieldSum()
    heff.add(zeeman_bias)
    heff.add(zeeman_ac)
    heff.add(exch)

    # Initial state
    m = uniform_mag(grid, Vec3(1, 0, 0))
    arr = _np2.asarray(to_numpy(m))
    rng = _np2.random.default_rng(seed)
    perp1, perp2 = (1, 2) if component == "x" else (0, 2) if component == "y" else (0, 1)
    arr[..., perp1] += rng.normal(0, 1e-3, arr[..., 0].shape)
    arr[..., perp2] += rng.normal(0, 1e-3, arr[..., 0].shape)
    norms = _np2.linalg.norm(arr, axis=-1, keepdims=True)
    arr /= norms
    from_numpy(m, arr)

    # Sizes
    g = grid
    sizes = (g.nx, g.ny, g.nz)
    n_axis = sizes[axis]
    dx_axis = (g.dx, g.dy, g.dz)[axis]
    comp_idx = {"x": 0, "y": 1, "z": 2}[component]

    n_frames = int(t_sim / dt_save)
    m_xt = _np2.zeros((n_frames, n_axis))

    integ = RK4Integrator(dt_sim)
    t = 0.0
    frame = 0
    t_last_save = -dt_save

    while t < t_sim and frame < n_frames:
        H_sinc = sinc_pulse(t, Vec3(0, H_pulse, 0) if component != "x"
                              else Vec3(0, 0, H_pulse), f_max)
        zeeman_ac.H_ext = H_sinc
        integ.step(m, mat, heff)
        t += dt_sim
        if t - t_last_save >= dt_save - 1e-15:
            arr_t = _np2.asarray(to_numpy(m))
            if axis == 0:
                m_xt[frame, :] = arr_t[0, 0, :, comp_idx]
            elif axis == 1:
                m_xt[frame, :] = arr_t[0, :, 0, comp_idx]
            else:
                m_xt[frame, :] = arr_t[:, 0, 0, comp_idx]
            frame += 1
            t_last_save = t

    kvals, freqs, S = field_fft2d(m_xt[:frame], dt=dt_save, dx=dx_axis)
    return kvals, freqs, S


# ===========================================================================
# Phase L — Image geometry, grain-boundary utilities
# ===========================================================================

def image_geom(grid, filename: str, threshold: float = 128.0,
               channel: str = "gray", invert: bool = False,
               layer_mode: str = "extrude"):
    """Create a GeomMask from an image file (PNG, BMP, TIFF, JPEG, ...).

    The image is resized to match the grid's (nx, ny) dimensions using
    nearest-neighbour resampling, then thresholded to produce a binary mask.
    Requires PIL/Pillow: ``pip install Pillow``.

    Parameters
    ----------
    grid      : StructuredGrid
    filename  : str   — image file path (PNG, BMP, TIFF, JPEG, …)
    threshold : float — pixel intensity in [0, 255]; pixels >= threshold → inside
    channel   : str   — 'gray' (luminance), 'r', 'g', 'b', or 'alpha'
    invert    : bool  — if True, swap inside/outside (black=inside by default)
    layer_mode: str   — 'extrude' (same mask all z), 'single' (only iz=0)

    Returns
    -------
    GeomMask — binary 0/1 mask

    Notes
    -----
    Coordinate convention: image row 0 → iy=ny-1 (y-up, matching mumax3).

    Example
    -------
    >>> mask = mm.image_geom(grid, 'sample.png', threshold=128)
    >>> mm.set_geom(mask, m, exch)
    """
    try:
        from PIL import Image as _Image
    except ImportError:
        raise ImportError("image_geom requires Pillow: pip install Pillow")

    g = grid
    img = _Image.open(filename)

    # Select channel
    if channel == "gray":
        img = img.convert("L")
        arr_img = _np.array(img, dtype=_np.float32)   # (H, W)
    elif channel in ("r", "g", "b"):
        img = img.convert("RGB")
        cidx = {"r": 0, "g": 1, "b": 2}[channel]
        arr_img = _np.array(img, dtype=_np.float32)[:, :, cidx]
    elif channel == "alpha":
        img = img.convert("RGBA")
        arr_img = _np.array(img, dtype=_np.float32)[:, :, 3]
    else:
        raise ValueError(f"channel must be 'gray','r','g','b','alpha'; got '{channel}'")

    # Resize to (ny, nx) with nearest-neighbour
    pil_resized = _Image.fromarray(arr_img).resize((g.nx, g.ny), _Image.NEAREST)
    arr = _np.array(pil_resized, dtype=_np.float32)   # shape (ny, nx)

    # Flip rows: image top (row 0) maps to iy=ny-1 (y-up convention)
    arr = arr[::-1, :].copy()

    # Threshold → 0/1 float
    inside = (arr >= threshold).astype(_np.float64)
    if invert:
        inside = 1.0 - inside

    # Build result mask: linear index = ix + nx*(iy + ny*iz)
    result = GeomMask(g)
    nz_fill = 1 if layer_mode == "single" else g.nz
    for iz in range(nz_fill):
        base = g.nx * g.ny * iz
        for iy in range(g.ny):
            for ix in range(g.nx):
                result[base + ix + g.nx * iy] = inside[iy, ix]
    return result


# ===========================================================================
# Phase D — Dynamic geometry / Moving write head utilities
# ===========================================================================

def moving_gaussian_field(grid, H_amp, sigma: float, polarity: float = 1.0,
                           axis: int = 0, direction: int = 2):
    """Create a Gaussian write-head spatial field at a given x-position.

    Returns a VectorField3D with H[direction] = polarity*H_amp * exp(-0.5*(r-x0)²/σ²)
    for all cells. Rebuild or call again with updated x0 each time the head moves.

    Parameters
    ----------
    grid      : StructuredGrid
    H_amp     : float — peak field amplitude [A/m]
    sigma     : float — Gaussian width [m]
    polarity  : float — +1 or -1 (bit polarity)
    axis      : int   — sweep axis (0=x, 1=y, 2=z)
    direction : int   — field direction component (0=x, 1=y, 2=z)

    Returns
    -------
    Callable[[float], VectorField3D] — call with head_position [m] to get field

    Example
    -------
    >>> head_fn = mm.moving_gaussian_field(g, H_write=5e5, sigma=15e-9)
    >>> for t in time_steps:
    ...     x_head = v_head * t
    ...     H_field = head_fn(x_head, polarity=+1)
    ...     zeeman_spatial.H_field = H_field
    """
    g = grid
    nx, ny, nz = g.nx, g.ny, g.nz
    sizes = (nx, ny, nz)
    steps = (g.dx, g.dy, g.dz)
    n_axis = sizes[axis]
    ds = steps[axis]

    # Cell-centre positions along sweep axis
    pos = (_np.arange(n_axis) + 0.5) * ds

    def head_field(x0: float, pol: float = polarity) -> "VectorField3D":
        """Return H field at head position x0 [m] with polarity pol."""
        H_arr = _np.zeros((nz, ny, nx, 3))
        profile = pol * H_amp * _np.exp(-0.5 * ((pos - x0) / sigma) ** 2)
        if axis == 0:
            H_arr[..., direction] = profile[_np.newaxis, _np.newaxis, :]
        elif axis == 1:
            H_arr[..., direction] = profile[_np.newaxis, :, _np.newaxis]
        else:
            H_arr[..., direction] = profile[:, _np.newaxis, _np.newaxis]
        field = VectorField3D(grid)
        from_numpy(field, H_arr)
        return field

    return head_field


# ===========================================================================
# Phase M — Custom field + TorqueField + StrayField
# ===========================================================================

class PythonField(IEffectiveField):
    """User-defined effective field term — mumax3 CustomField analog.

    Subclass IEffectiveField with a Python callable.  The callable receives
    the current magnetization (as numpy array) and returns the field
    contribution H (numpy array, same shape), which is *added* to H_eff.

    Parameters
    ----------
    fn       : callable(m_arr: ndarray) -> ndarray
                Both arrays have shape (nz, ny, nx, 3) in [A/m].
                The function should return a new array (not modify in-place).
    name_str : str — label for this field (default "PythonField")
    energy_fn: callable(m_arr) -> float | None
                Optional energy [J] callback.  If None, returns 0.0.

    Example: spatially-varying Zeeman field
    ----------------------------------------
    >>> import numpy as np
    >>> def my_H(m_arr):
    ...     H = np.zeros_like(m_arr)
    ...     H[..., 2] = 1e4  # H_z = 10 kA/m everywhere
    ...     return H
    >>> pf = mm.PythonField(my_H, name_str="MyZeeman")
    >>> heff.add(pf)

    Example: H proportional to mz (effective anisotropy)
    -----------------------------------------------------
    >>> def my_K(m_arr):
    ...     H = np.zeros_like(m_arr)
    ...     H[..., 2] = 2e3 * m_arr[..., 2]   # like UniaxialAnisotropy K/Ms
    ...     return H
    >>> heff.add(mm.PythonField(my_K))
    """

    def __init__(self, fn, name_str: str = "PythonField", energy_fn=None):
        super().__init__()
        self._fn       = fn
        self._name_str = name_str
        self._energy_fn = energy_fn

    def accumulate(self, m, mat, H_out):
        m_arr  = _np.asarray(to_numpy(m))          # (nz, ny, nx, 3)
        H_add  = _np.asarray(self._fn(m_arr))      # (nz, ny, nx, 3)
        # Read current H_out, add our contribution, write back
        H_arr  = _np.asarray(to_numpy(H_out))
        from_numpy(H_out, H_arr + H_add)

    def energy(self, m, mat):
        if self._energy_fn is not None:
            m_arr = _np.asarray(to_numpy(m))
            return float(self._energy_fn(m_arr))
        return 0.0

    def name(self):
        return self._name_str


# ---------------------------------------------------------------------------
# StrayField — static dipole stray field from an external source magnet
# ---------------------------------------------------------------------------

def stray_field(grid, Ms_ext: float, volume_ext: float,
                position, moment_dir=(0.0, 0.0, 1.0)):
    """Compute the stray field from a single magnetic dipole [A/m].

    Models an external magnet (or MFM tip) as a point magnetic dipole
    with moment m_ext = Ms_ext * volume_ext * moment_dir.

    Returns a VectorField3D [A/m] containing the stray field at each
    cell of ``grid``.  Use as the spatial field in a ZeemanFieldSpatial:

    >>> H_stray = mm.stray_field(grid, Ms=860e3, volume=1e-24, position=(0,0,50e-9))
    >>> zee = mm.ZeemanFieldSpatial(grid)
    >>> zee.H_field = H_stray
    >>> heff.add(zee)

    Parameters
    ----------
    grid        : StructuredGrid
    Ms_ext      : float — saturation magnetization of the dipole source [A/m]
    volume_ext  : float — volume of the source magnet [m³]
    position    : (x, y, z) — dipole centre position in absolute (non-centred)
                  coordinates [m] (default origin = box corner)
    moment_dir  : (mx, my, mz) — unit vector of dipole moment (auto-normalised)

    Returns
    -------
    VectorField3D — stray field [A/m] at each grid cell

    Physics
    -------
    H_dip(r) = (1/4π) * [3(m·r̂)r̂ − m] / |r|³   (SI, SI units: A/m)
    m_ext = Ms_ext * volume_ext * m̂
    """
    mu0_over_4pi = 1e-7   # μ₀/(4π) in SI

    g = grid
    nx, ny, nz = g.nx, g.ny, g.nz

    # Cell-centre positions (absolute, not box-centred)
    xs = (_np.arange(nx) + 0.5) * g.dx
    ys = (_np.arange(ny) + 0.5) * g.dy
    zs = (_np.arange(nz) + 0.5) * g.dz
    ZZ, YY, XX = _np.meshgrid(zs, ys, xs, indexing='ij')  # (nz, ny, nx)

    # Moment direction (unit vector)
    md = _np.asarray(moment_dir, dtype=float)
    md = md / _np.linalg.norm(md)
    m_ext = Ms_ext * volume_ext * md      # [A m²]

    # Displacement vectors from dipole to each cell
    pos = _np.asarray(position, dtype=float)
    Rx = XX - pos[0]   # (nz, ny, nx)
    Ry = YY - pos[1]
    Rz = ZZ - pos[2]
    R2 = Rx**2 + Ry**2 + Rz**2           # |r|²
    R  = _np.sqrt(R2)                    # |r|

    # Avoid singularity at r=0
    R  = _np.where(R < 1e-30, 1e-30, R)
    R3 = R**3
    R5 = R**5

    # m·r per cell
    m_dot_r = m_ext[0]*Rx + m_ext[1]*Ry + m_ext[2]*Rz   # (nz, ny, nx)

    # Dipole field: H = (1/4π) * [3(m·r̂)r/|r|³ − m/|r|³]
    #             = (1/4π) * [3(m·r)r/|r|⁵ − m/|r|³]
    prefac = mu0_over_4pi / mu0_over_4pi  # = 1 (H field, not B)
    # Actually H_dip = (1/(4π)) * [3(m·r)r/|r|^5 - m/|r|^3]
    H_arr = _np.zeros((nz, ny, nx, 3), dtype=float)
    H_arr[..., 0] = (3.0 * m_dot_r * Rx / R5 - m_ext[0] / R3) / (4 * _math.pi)
    H_arr[..., 1] = (3.0 * m_dot_r * Ry / R5 - m_ext[1] / R3) / (4 * _math.pi)
    H_arr[..., 2] = (3.0 * m_dot_r * Rz / R5 - m_ext[2] / R3) / (4 * _math.pi)

    result = VectorField3D(g)
    from_numpy(result, H_arr)
    return result


# ===========================================================================
# Phase O — Convergence observables + energy breakdown + B_eff
# ===========================================================================

def max_angle(m) -> float:
    """Maximum angle [degrees] between adjacent cells in the magnetization field.

    Equivalent to mumax3's MaxAngle quantity.  Used as a convergence criterion:
    a fully relaxed state typically has MaxAngle < 1 degree.

    Checks all 6 nearest-neighbour pairs (+/-x, +/-y, +/-z).

    Parameters
    ----------
    m : VectorField3D

    Returns
    -------
    float -- max inter-cell angle in degrees (0 for single-cell grids)

    Example
    -------
    >>> mm.relax(m, mat, heff)
    >>> print(f"MaxAngle = {mm.max_angle(m):.2f} deg")
    """
    m_np = _np.asarray(to_numpy(m))   # (nz, ny, nx, 3)
    nz, ny, nx = m_np.shape[:3]
    min_dot = 1.0
    if nx > 1:
        d = (m_np[:, :, :-1, :] * m_np[:, :, 1:, :]).sum(-1)
        v = float(d.min())
        if v < min_dot:
            min_dot = v
    if ny > 1:
        d = (m_np[:, :-1, :, :] * m_np[:, 1:, :, :]).sum(-1)
        v = float(d.min())
        if v < min_dot:
            min_dot = v
    if nz > 1:
        d = (m_np[:-1, :, :, :] * m_np[1:, :, :, :]).sum(-1)
        v = float(d.min())
        if v < min_dot:
            min_dot = v
    min_dot = max(-1.0, min(1.0, min_dot))
    import math as _m2
    return float(_m2.degrees(_m2.acos(min_dot)))


def B_eff(m, mat, heff):
    """Effective magnetic flux density B_eff = mu0 * H_eff [T].

    Equivalent to mumax3's B_eff quantity.  Returns the total effective
    field scaled by mu0, useful for plotting field distributions in Tesla.

    Parameters
    ----------
    m    : VectorField3D
    mat  : Material
    heff : EffectiveFieldSum

    Returns
    -------
    VectorField3D -- B_eff [T] at each cell

    Example
    -------
    >>> B = mm.B_eff(m, mat, heff)
    >>> B_np = mm.to_numpy(B)    # (nz, ny, nx, 3) in Tesla
    """
    mu0 = 4e-7 * _math.pi
    H = VectorField3D(m.grid)
    heff.compute(m, mat, H)
    H_np = _np.asarray(to_numpy(H))
    result = VectorField3D(m.grid)
    from_numpy(result, H_np * mu0)
    return result


def energy_table(m, mat, heff) -> dict:
    """Per-term energy breakdown [J] for all fields in an EffectiveFieldSum.

    Equivalent to querying mumax3's E_Zeeman, E_exch, E_demag, E_anis, ...
    individually.  Returns a dict with each term's name as key and energy [J]
    as value, plus the key 'total' for the sum.

    Parameters
    ----------
    m    : VectorField3D
    mat  : Material
    heff : EffectiveFieldSum

    Returns
    -------
    dict[str, float] -- {term_name: energy_J, ..., 'total': total_J}

    Example
    -------
    >>> E = mm.energy_table(m, mat, heff)
    >>> for k, v in E.items():
    ...     print(f"  {k:20s} = {v:.4e} J")
    """
    result = {}
    total = 0.0
    for term in heff.terms:
        name = term.name
        # handle duplicate names (e.g., two Zeeman fields)
        base = name
        idx = 1
        while name in result:
            name = f"{base}_{idx}"
            idx += 1
        e = float(term.energy(m, mat))
        result[name] = e
        total += e
    result["total"] = total
    return result


# ===========================================================================
# Phase P — FrozenSpins, DefRegion, (SurfaceAnisotropyField via C++ below)
# ===========================================================================

def def_region(region_map, region_id: int, geom_mask):
    """Assign region_id to all cells where geom_mask > 0.5.

    Equivalent to mumax3's DefRegion(id, shape).  Cells outside the shape
    are left unchanged (non-destructive: only writes into cells where mask > 0.5).

    Parameters
    ----------
    region_map : RegionMap  -- modified in place
    region_id  : int        -- region ID to assign (1..254; 0 = unassigned)
    geom_mask  : GeomMask   -- shape (cells > 0.5 get region_id)

    Example
    -------
    >>> rm = mm.RegionMap(grid)
    >>> mm.def_region(rm, 1, mm.circle(grid, 100e-9))
    >>> mm.def_region(rm, 2, mm.x_range(grid, 50e-9, 200e-9))
    """
    g = region_map.grid
    n = g.nx * g.ny * g.nz
    rid = min(max(int(region_id), 0), 255)
    for i in range(n):
        if geom_mask[i] > 0.5:
            region_map[i] = rid


def new_region_map(grid, *region_specs):
    """Build a RegionMap from (region_id, geom_mask) pairs.

    Convenience wrapper around def_region() for building a complete
    RegionMap from multiple geometry shapes in one call.

    Parameters
    ----------
    grid          : StructuredGrid
    *region_specs : (int, GeomMask) pairs -- assigned in order

    Returns
    -------
    RegionMap -- cells not covered by any spec have ID 0

    Example
    -------
    >>> rm = mm.new_region_map(
    ...     grid,
    ...     (1, mm.circle(grid, 100e-9)),
    ...     (2, mm.y_range(grid, -50e-9, 0)),
    ... )
    """
    rm = RegionMap(grid)
    for rid, geom in region_specs:
        def_region(rm, rid, geom)
    return rm


class FrozenIntegrator:
    """Integrator wrapper that holds selected cells fixed (pinned spins).

    Equivalent to mumax3's FreezeSpins(region) — after each LLG step the
    magnetization of frozen cells is restored to their initial values,
    effectively removing them from the dynamics.

    Works with any CPU integrator (RK4, RK45, Heun).  GPU integrators are
    not supported (the step executes entirely on-device; use a GeomMask on
    the effective field instead for GPU pinning).

    Parameters
    ----------
    integ       : RK4Integrator | RK45Integrator | HeunIntegrator
    freeze_mask : GeomMask -- cells with mask > 0.5 are frozen
    m_init      : VectorField3D -- initial (pinned) magnetization values

    Example
    -------
    >>> # Pin the left quarter of the strip
    >>> pin = mm.x_range(grid, -Lx/2, -Lx/4)
    >>> frozen = mm.FrozenIntegrator(integ, pin, m)
    >>> mm.run(frozen, m, mat, heff, t_total=1e-9)
    >>> # Cells outside pin evolve; cells inside pin stay fixed.
    """

    def __init__(self, integ, freeze_mask, m_init):
        self._integ = integ
        g = freeze_mask.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        n = nx * ny * nz
        # Snapshot frozen cells as numpy array for fast restore.
        # Store (linear_idx, (x, y, z)) tuples.
        m_np = _np.asarray(to_numpy(m_init))   # (nz, ny, nx, 3)
        self._frozen = []   # list of (lin_idx, x, y, z)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    lin = ix + nx*(iy + ny*iz)
                    if freeze_mask[lin] > 0.5:
                        v = m_np[iz, iy, ix, :]
                        self._frozen.append((iz, iy, ix, float(v[0]), float(v[1]), float(v[2])))

    # ------------------------------------------------------------------
    # Forward dt / set_dt / dt property to wrapped integrator
    # ------------------------------------------------------------------
    @property
    def dt(self):
        return self._integ.dt

    def set_dt(self, dt):
        self._integ.set_dt(dt)

    # ------------------------------------------------------------------
    # step — performs the LLG step then restores frozen cells
    # ------------------------------------------------------------------
    def step(self, m, mat, heff, stt=None):
        if stt is not None:
            result = self._integ.step(m, mat, heff, stt)
        else:
            result = self._integ.step(m, mat, heff)
        if self._frozen:
            # Read current state, overwrite frozen cells, write back
            m_np = _np.asarray(to_numpy(m)).copy()
            for (iz, iy, ix, vx, vy, vz) in self._frozen:
                m_np[iz, iy, ix, 0] = vx
                m_np[iz, iy, ix, 1] = vy
                m_np[iz, iy, ix, 2] = vz
            from_numpy(m, m_np)
        return result

    # Expose the underlying integrator for inspection
    @property
    def integrator(self):
        return self._integ


# ===========================================================================
# Phase U — Hysteresis loop automation
# ===========================================================================

def hysteresis_loop(m, mat, heff, integ, H_list, zee,
                    axis: str = 'x',
                    tol_deg: float = 1.0,
                    max_steps: int = 500_000,
                    check_interval: int = 200,
                    reset_m=None,
                    verbose: bool = False):
    """Sweep a ZeemanField through H_list and relax at each point.

    Equivalent to mumax3's ``for H in H_list { SetB_ext(...); RunWhile(...) }``
    pattern.  At each field value the ZeemanField ``zee`` is updated and the
    system is relaxed via ``run_until_converged``.

    Parameters
    ----------
    m              : VectorField3D -- modified in place
    mat            : Material
    heff           : EffectiveFieldSum -- must already contain ``zee``
    integ          : integrator (RK4, RK45, Heun, or FrozenIntegrator)
    H_list         : 1-D iterable of floats [A/m] OR shape-(N,3) array of Vec3
                     If 1-D, ``axis`` controls which component is set.
    zee            : ZeemanField -- its H_ext is modified at each step
    axis           : 'x', 'y', or 'z' — applied axis for scalar H_list
    tol_deg        : convergence threshold [degrees] passed to run_until_converged
    max_steps      : max LLG steps per field point
    check_interval : convergence check interval
    reset_m        : VectorField3D (optional) — if given, m is reset to this
                     state before relaxing at each field value (useful for
                     major loop starting from saturation)
    verbose        : print progress (field value + mean magnetization)

    Returns
    -------
    dict with numpy arrays, all shape (N,) where N = len(H_list):
      "H"       : applied field magnitude [A/m]  (signed, along ``axis``)
      "Hvec"    : applied field Vec3 [A/m], shape (N, 3)
      "mx", "my", "mz" : mean magnetization components
      "E_total" : total energy [J]

    Example — SP#3-style hysteresis loop
    -------------------------------------
    >>> zee = mm.ZeemanField()
    >>> heff.add(zee)
    >>> mu0 = 4e-7 * math.pi
    >>> H_list = np.linspace(0.1/mu0, -0.1/mu0, 201)  # ±100 mT
    >>> res = mm.hysteresis_loop(m, mat, heff, integ, H_list, zee,
    ...                          axis='x', tol_deg=1.0)
    >>> plt.plot(res["H"] * mu0 * 1e3, res["mx"])  # mT vs mx
    """
    ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis.lower(), 0)

    H_arr  = _np.asarray(H_list, dtype=float)
    is_vec = (H_arr.ndim == 2 and H_arr.shape[1] == 3)
    N      = len(H_arr) if not is_vec else H_arr.shape[0]

    H_mag_out = _np.zeros(N)
    Hvec_out  = _np.zeros((N, 3))
    mx_out    = _np.zeros(N)
    my_out    = _np.zeros(N)
    mz_out    = _np.zeros(N)
    E_out     = _np.zeros(N)

    # Snapshot reset state once (before any modifications)
    if reset_m is not None:
        reset_np = _np.asarray(to_numpy(reset_m)).copy()

    for i in range(N):
        if is_vec:
            hx, hy, hz = float(H_arr[i, 0]), float(H_arr[i, 1]), float(H_arr[i, 2])
        else:
            hx = hy = hz = 0.0
            if ax_idx == 0:   hx = float(H_arr[i])
            elif ax_idx == 1: hy = float(H_arr[i])
            else:             hz = float(H_arr[i])

        zee.H_ext = Vec3(hx, hy, hz)

        if reset_m is not None:
            from_numpy(m, reset_np)

        # Always take at least check_interval steps so a uniform-m state can
        # react to the new H before the convergence criterion is evaluated.
        for _ in range(check_interval):
            integ.step(m, mat, heff)

        run_until_converged(integ, m, mat, heff,
                            tol_deg=tol_deg,
                            max_steps=max(0, max_steps - check_interval),
                            check_interval=check_interval)

        mx_out[i], my_out[i], mz_out[i] = mean_magnetization(m)
        E_out[i]     = heff.total_energy(m, mat)
        H_mag_out[i] = float(H_arr[i]) if not is_vec else float(
            _np.sqrt(hx*hx + hy*hy + hz*hz))
        Hvec_out[i]  = [hx, hy, hz]

        if verbose:
            print(f"  [{i+1:3d}/{N}] H={H_mag_out[i]:+.3e} A/m"
                  f"  mx={mx_out[i]:+.4f}  my={my_out[i]:+.4f}"
                  f"  mz={mz_out[i]:+.4f}")

    return {
        "H":       H_mag_out,
        "Hvec":    Hvec_out,
        "mx":      mx_out,
        "my":      my_out,
        "mz":      mz_out,
        "E_total": E_out,
    }


# ===========================================================================
# Phase X — GPU convergence + GPU hysteresis loop
# ===========================================================================

def run_until_converged_gpu(integ, mat, demag, fsum, m_cpu,
                             tol_deg: float = 1.0,
                             max_steps: int = 500_000,
                             check_interval: int = 200,
                             torques=None,
                             verbose: bool = False):
    """Run LLG on GPU until max_angle(m) drops below *tol_deg*, then stop.

    The GPU integrator state is advanced in batches of *check_interval* steps.
    Convergence is checked by downloading m to CPU and calling max_angle().

    Parameters
    ----------
    integ          : RK4IntegratorGPU or RK45IntegratorGPU (already uploaded)
    mat            : Material
    demag          : IDemagGPU (DemagFieldGPU or DemagFieldPeriodicGPU)
    fsum           : FieldSumGPU (exchange, zeeman, anisotropy, etc.)
    m_cpu          : VectorField3D — scratch buffer for downloads (modified)
    tol_deg        : convergence threshold [degrees] (default 1.0)
    max_steps      : hard step limit (default 500 000)
    check_interval : steps per convergence check (default 200)
    torques        : SpinTorqueSumGPU or None
    verbose        : print progress each check

    Returns
    -------
    dict with keys: "converged", "steps", "max_angle", "t_sim"
    """
    step_count = 0
    t_sim = 0.0

    # Always run at least check_interval steps first so a uniform-m state
    # can react to the current H before the convergence criterion fires.
    n_warmup = min(check_interval, max_steps)
    for _ in range(n_warmup):
        if torques is not None:
            integ.step(mat, demag, fsum, torques)
        else:
            integ.step(mat, demag, fsum)
    step_count = n_warmup
    try:
        t_sim = step_count * integ.dt
    except AttributeError:
        t_sim = float("nan")

    integ.download(m_cpu)
    angle = max_angle(m_cpu)
    converged = angle < tol_deg

    while not converged and step_count < max_steps:
        n_batch = min(check_interval, max_steps - step_count)
        for _ in range(n_batch):
            if torques is not None:
                integ.step(mat, demag, fsum, torques)
            else:
                integ.step(mat, demag, fsum)
        step_count += n_batch
        try:
            t_sim = step_count * integ.dt
        except AttributeError:
            t_sim = float("nan")

        integ.download(m_cpu)
        angle = max_angle(m_cpu)
        converged = angle < tol_deg

        if verbose:
            print(f"  step {step_count:7d}  max_angle = {angle:.4f} deg"
                  f"  (tol = {tol_deg:.4f} deg)")

    return {
        "converged":  converged,
        "steps":      step_count,
        "max_angle":  angle,
        "t_sim":      t_sim,
    }


def gpu_hysteresis_loop(integ, mat, demag, fsum, zee_gpu, H_list,
                         m_cpu,
                         axis: str = 'x',
                         tol_deg: float = 1.0,
                         max_steps: int = 500_000,
                         check_interval: int = 200,
                         reset_m=None,
                         torques=None,
                         verbose: bool = False):
    """GPU-accelerated hysteresis loop: sweep ZeemanFieldGPU through H_list.

    Equivalent to ``hysteresis_loop`` but uses GPU integrators for all LLG
    stepping. The GPU integrator must already have been initialised via
    ``integ.upload(m0)`` before calling this function.

    *zee_gpu must already be added to fsum* so its H_ext update takes effect
    at each field point.

    Parameters
    ----------
    integ          : RK4IntegratorGPU or RK45IntegratorGPU (already uploaded)
    mat            : Material
    demag          : IDemagGPU
    fsum           : FieldSumGPU containing exchange, zeeman, anisotropy, etc.
    zee_gpu        : ZeemanFieldGPU — H_ext is updated at each H point
    H_list         : 1D array of scalar [A/m], OR shape-(N,3) Vec3 array
    m_cpu          : VectorField3D — download scratch (modified in-place)
    axis           : 'x' | 'y' | 'z' — axis for 1D H_list (default 'x')
    tol_deg        : convergence tolerance [deg] (default 1.0)
    max_steps      : hard step limit per H point (default 500 000)
    check_interval : steps between convergence checks (default 200)
    reset_m        : VectorField3D snapshot to reset GPU state before each point
    torques        : SpinTorqueSumGPU or None
    verbose        : print per-point progress

    Returns
    -------
    dict with keys: "H", "Hvec", "mx", "my", "mz"
    """
    import numpy as _np
    import math as _m

    H_arr  = _np.asarray(H_list, dtype=float)
    is_vec = H_arr.ndim == 2 and H_arr.shape[1] == 3
    N      = len(H_arr)
    ax_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

    reset_np = None
    if reset_m is not None:
        reset_np = _np.asarray(to_numpy(reset_m)).copy()

    mx_out   = _np.zeros(N)
    my_out   = _np.zeros(N)
    mz_out   = _np.zeros(N)
    H_out    = _np.zeros(N)
    Hvec_out = _np.zeros((N, 3))

    for i in range(N):
        if is_vec:
            hx, hy, hz = float(H_arr[i, 0]), float(H_arr[i, 1]), float(H_arr[i, 2])
        else:
            hx = hy = hz = 0.0
            if ax_idx == 0:   hx = float(H_arr[i])
            elif ax_idx == 1: hy = float(H_arr[i])
            else:             hz = float(H_arr[i])

        zee_gpu.H_ext = Vec3(hx, hy, hz)

        if reset_m is not None:
            from_numpy(m_cpu, reset_np)
            integ.upload(m_cpu)

        run_until_converged_gpu(integ, mat, demag, fsum, m_cpu,
                                tol_deg=tol_deg,
                                max_steps=max_steps,
                                check_interval=check_interval,
                                torques=torques)

        integ.download(m_cpu)
        mx_out[i], my_out[i], mz_out[i] = mean_magnetization(m_cpu)
        H_out[i]    = float(H_arr[i]) if not is_vec else float(
            _m.sqrt(hx*hx + hy*hy + hz*hz))
        Hvec_out[i] = [hx, hy, hz]

        if verbose:
            print(f"  [{i+1:3d}/{N}] H={H_out[i]:+.3e} A/m"
                  f"  mx={mx_out[i]:+.4f}  my={my_out[i]:+.4f}"
                  f"  mz={mz_out[i]:+.4f}")

    return {
        "H":    H_out,
        "Hvec": Hvec_out,
        "mx":   mx_out,
        "my":   my_out,
        "mz":   mz_out,
    }


# ===========================================================================
# Phase Y — Multi-layer material stack builders
# ===========================================================================

def _matf_set_cell(matf, idx, mat):
    """Set scalar parameters of one MaterialField3D cell from a Material."""
    matf.Ms_field[idx]    = mat.Ms
    matf.A_field[idx]     = mat.A_exchange
    matf.K_field[idx]     = mat.K_uniaxial
    matf.alpha_field[idx] = mat.alpha
    # easy_axis is a VectorField3D: set per-cell via its component ScalarFields
    # VectorField3D does not expose __setitem__; use the component trick if needed.
    # For most multilayer stacks the easy axis is identical across layers, so
    # MaterialField3D.set_uniform sets it correctly; here we skip it.


def bilayer(grid, mat_top, mat_bot, t_top, t_bot=None):
    """Create a MaterialField3D for a two-layer thin-film stack (z-stacked).

    Top layer occupies the highest z cells; bottom layer the rest.

    Parameters
    ----------
    grid    : StructuredGrid
    mat_top : Material — top layer (highest z)
    mat_bot : Material — bottom layer
    t_top   : float — top layer thickness [m]
    t_bot   : ignored (reserved)

    Returns
    -------
    MaterialField3D
    """
    from _micromag import MaterialField3D as _MF3D
    matf = _MF3D(grid, mat_bot)
    dz   = grid.dz
    nz   = grid.nz
    total_z = nz * dz
    z_top_start = total_z - t_top
    for iz in range(nz):
        z_centre = (iz + 0.5) * dz
        if z_centre >= z_top_start:
            for iy in range(grid.ny):
                for ix in range(grid.nx):
                    idx = ix + grid.nx * (iy + grid.ny * iz)
                    _matf_set_cell(matf, idx, mat_top)
    return matf


def trilayer(grid, mat_top, mat_mid, mat_bot, t_top, t_mid, t_bot=None):
    """Create a MaterialField3D for a three-layer thin-film stack.

    Layers are stacked along z (top=highest z). t_bot is implicit
    (remaining cells after top+mid).

    Parameters
    ----------
    grid    : StructuredGrid
    mat_top, mat_mid, mat_bot : Material
    t_top, t_mid : float [m]
    t_bot   : ignored (reserved)

    Returns
    -------
    MaterialField3D
    """
    from _micromag import MaterialField3D as _MF3D
    matf = _MF3D(grid, mat_bot)
    dz   = grid.dz
    nz   = grid.nz
    total_z = nz * dz
    z_mid_start = total_z - t_top - t_mid
    z_top_start = total_z - t_top
    for iz in range(nz):
        z_centre = (iz + 0.5) * dz
        if z_centre >= z_top_start:
            m_cell = mat_top
        elif z_centre >= z_mid_start:
            m_cell = mat_mid
        else:
            m_cell = mat_bot
        for iy in range(grid.ny):
            for ix in range(grid.nx):
                idx = ix + grid.nx * (iy + grid.ny * iz)
                _matf_set_cell(matf, idx, m_cell)
    return matf


def saf_stack(grid, mat_fl, mat_rl, t_fl, t_rl, t_spacer=None):
    """Create a MaterialField3D for a Synthetic Antiferromagnet (SAF) stack.

    Structure (bottom to top): RL | non-magnetic spacer | FL

    The spacer is modelled as Ms=0 cells. The spacer thickness is
    ``total_z - t_fl - t_rl`` unless *t_spacer* is given (in which case
    any remaining cells outside the three defined regions are also spacer).

    Parameters
    ----------
    grid     : StructuredGrid — total z height must >= t_fl + t_rl
    mat_fl   : Material — free layer (top)
    mat_rl   : Material — reference layer (bottom)
    t_fl     : float [m]
    t_rl     : float [m]
    t_spacer : float or None — spacer thickness (default: remainder)

    Returns
    -------
    MaterialField3D
    """
    from _micromag import MaterialField3D as _MF3D
    mat_spacer           = Material()
    mat_spacer.Ms        = 0.0
    mat_spacer.A_exchange = 0.0
    mat_spacer.alpha     = 0.0
    mat_spacer.K_uniaxial = 0.0
    matf = _MF3D(grid, mat_spacer)

    dz      = grid.dz
    nz      = grid.nz
    total_z = nz * dz
    z_rl_end   = t_rl
    z_fl_start = total_z - t_fl

    for iz in range(nz):
        z_centre = (iz + 0.5) * dz
        if z_centre < z_rl_end:
            m_cell = mat_rl
        elif z_centre >= z_fl_start:
            m_cell = mat_fl
        else:
            m_cell = mat_spacer
        for iy in range(grid.ny):
            for ix in range(grid.nx):
                idx = ix + grid.nx * (iy + grid.ny * iz)
                _matf_set_cell(matf, idx, m_cell)
    return matf


# ===========================================================================
# Phase R — Convergence-based adaptive relaxation
# ===========================================================================

def run_until_converged(integ, m, mat, heff, tol_deg: float = 1.0,
                         max_steps: int = 1_000_000,
                         check_interval: int = 100,
                         stt=None,
                         verbose: bool = False):
    """Run LLG until max_angle(m) drops below tol_deg, then stop.

    Equivalent to mumax3's ``RunWhile(MaxAngle.Get() > tol*pi/180)``.
    Checks convergence every ``check_interval`` steps to keep overhead low.

    Parameters
    ----------
    integ          : integrator (RK4, RK45, Heun, or FrozenIntegrator)
    m              : VectorField3D -- magnetization, modified in-place
    mat            : Material
    heff           : EffectiveFieldSum
    tol_deg        : float -- convergence threshold [degrees] (default 1.0)
    max_steps      : int   -- hard step limit (default 1 000 000)
    check_interval : int   -- check every N steps (default 100)
    stt            : optional spin-torque term
    verbose        : bool  -- print progress every check_interval (default False)

    Returns
    -------
    dict with keys:
      "converged"    : bool  -- True if tol_deg was reached
      "steps"        : int   -- total LLG steps taken
      "max_angle"    : float -- final MaxAngle [degrees]
      "t_sim"        : float -- total simulated time [s]

    Example (mumax3 style):
    >>> mm.run_until_converged(integ, m, mat, heff, tol_deg=0.5)
    >>> # Equivalent to mumax3: RunWhile(MaxAngle.Get() > 0.5*pi/180)
    """
    import math as _mc

    step_count = 0
    t_sim = 0.0
    angle = max_angle(m)
    converged = angle < tol_deg

    while not converged and step_count < max_steps:
        # Take check_interval steps before checking convergence
        n_batch = min(check_interval, max_steps - step_count)
        for _ in range(n_batch):
            if stt is not None:
                integ.step(m, mat, heff, stt)
            else:
                integ.step(m, mat, heff)
        step_count += n_batch
        try:
            t_sim = step_count * integ.dt
        except AttributeError:
            t_sim = float("nan")

        angle = max_angle(m)
        converged = angle < tol_deg

        if verbose:
            print(f"  step {step_count:7d}  max_angle = {angle:.4f} deg"
                  f"  (tol = {tol_deg:.4f} deg)")

    return {
        "converged":  converged,
        "steps":      step_count,
        "max_angle":  angle,
        "t_sim":      t_sim,
    }


# ---------------------------------------------------------------------------
# Phase N + Z: MFM, EdgeSmooth, Poisson-disk grains, overlap integral
# ---------------------------------------------------------------------------
from micromag._phase_n import (mfm_signal, edge_smooth,  # noqa: E402
                               poisson_disk_grains,
                               mfm_overlap_integral)
