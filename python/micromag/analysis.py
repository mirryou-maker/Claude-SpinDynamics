"""FMR / signal processing / visualisation / analysis utilities (Phases G, I)

Split from the former monolithic micromag/__init__.py; public names are
re-exported by the package __init__, so `import micromag as mm` is unchanged.
"""
import math as _math
import numpy as _np

from _micromag import *          # C++ core names  # noqa: F401,F403


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
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
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
    nz = m.grid.nz
    if not (0 <= iz < nz):
        raise IndexError(f"iz={iz} out of bounds [0, {nz})")
    arr = to_numpy(m)          # shape: (nz, ny, nx, 3) in C-order from to_numpy
    return arr[iz, :, :, :]


def cross_section_y(m, iy: int = 0):
    """Extract xz-plane at row iy.

    Returns
    -------
    numpy array, shape (nz, nx, 3)
    """
    ny = m.grid.ny
    if not (0 <= iy < ny):
        raise IndexError(f"iy={iy} out of bounds [0, {ny})")
    arr = to_numpy(m)
    return arr[:, iy, :, :]


def cross_section_x(m, ix: int = 0):
    """Extract yz-plane at column ix.

    Returns
    -------
    numpy array, shape (nz, ny, 3)
    """
    nx = m.grid.nx
    if not (0 <= ix < nx):
        raise IndexError(f"ix={ix} out of bounds [0, {nx})")
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
    import numpy as _np2
    data_xt = _np2.asarray(data_xt)
    if data_xt.ndim != 2:
        raise ValueError(f"data_xt must be 2D (nt, nx); got shape {data_xt.shape}")
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

