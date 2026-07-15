"""Spin-wave wrapper, image geometry, write-head, custom fields, stray field (Phases K, L, D, M)

Split from the former monolithic micromag/__init__.py; public names are
re-exported by the package __init__, so `import micromag as mm` is unchanged.
"""
import math as _math
import numpy as _np

from _micromag import *          # C++ core names  # noqa: F401,F403

from .analysis import field_fft2d, sinc_pulse  # noqa: F401


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
    >>> head_fn = mm.moving_gaussian_field(g, H_amp=5e5, sigma=15e-9)
    >>> for t in time_steps:
    ...     x_head = v_head * t
    ...     H_field = head_fn(x_head, pol=+1)
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

    >>> H_stray = mm.stray_field(grid, Ms_ext=860e3, volume_ext=1e-24, position=(0,0,50e-9))
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

