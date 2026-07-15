"""File I/O + field-manipulation + material utilities (Phase J)

Split from the former monolithic micromag/__init__.py; public names are
re-exported by the package __init__, so `import micromag as mm` is unchanged.
"""
import math as _math
import numpy as _np

from _micromag import *          # C++ core names  # noqa: F401,F403


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

