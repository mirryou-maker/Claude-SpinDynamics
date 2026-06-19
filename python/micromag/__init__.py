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
    load_ovf,
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
)

# GPU classes are only present in the CUDA build — import conditionally
try:
    from _micromag import (
        DemagFieldGPU,
        ExchangeFieldGPU,
        ZeemanFieldGPU,
        UniaxialAnisotropyFieldGPU,
        CubicAnisotropyFieldGPU,   # Phase E
        RK4IntegratorGPU,
        RK45IntegratorGPU,
        RK45GPUOptions,
        HeunIntegratorGPU,
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
    "square", "cuboid", "sphere",
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
    # Run/Steps convenience
    "run", "steps",
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
    # Utilities
    "cuda_available",
    # SP#2 / grid-sizing utilities (pure Python)
    "exchange_length", "optimal_dx", "sp2_grid",
    # GPU classes (conditionally available — only in CUDA build)
    "DemagFieldGPU", "ExchangeFieldGPU", "ZeemanFieldGPU",
    "UniaxialAnisotropyFieldGPU", "CubicAnisotropyFieldGPU",
    "RK4IntegratorGPU", "RK45IntegratorGPU", "RK45GPUOptions",
    "HeunIntegratorGPU",
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

    Custom quantities
    -----------------
    >>> tbl.add_row(t, m, extra={"Q": mm.topological_charge_Q(m)})
    """

    def __init__(self):
        self._header = None
        self._rows = []

    def add_row(self, t: float, m, mat=None, heff=None, extra: dict = None):
        mx, my, mz = mean_magnetization(m)
        row = {"t": t, "mx": mx, "my": my, "mz": mz}
        if mat is not None and heff is not None:
            row["E_total"] = heff.total_energy(m, mat)
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
