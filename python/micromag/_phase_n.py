"""Phase N utilities — MFM, EdgeSmooth, poisson_disk_grains.
Imported at the end of __init__.py.
"""
import math as _math
import numpy as _np


def mfm_signal(m, mat, lift_m: float, tip: str = "dipole"):
    """Compute MFM (Magnetic Force Microscopy) signal as a 2-D numpy array.

    Fourier-propagation algorithm: sums mz over layers, propagates stray
    field to lift height, applies tip kernel.

    Parameters
    ----------
    m      : VectorField3D
    mat    : Material
    lift_m : float -- tip lift height above sample top surface [m]
    tip    : "dipole" (default, dHz/dz) or "monopole" (Hz)

    Returns
    -------
    numpy.ndarray shape (ny, nx) -- MFM signal (relative units)
    """
    from _micromag import MFMImage as _MFMImage, TipMode as _TipMode
    tip_mode = _TipMode.Dipole if tip.lower().startswith("d") else _TipMode.Monopole
    MFMImage = _MFMImage
    mfm = MFMImage(m.grid, lift_m, tip_mode)
    flat = mfm.compute(m, mat)
    g = m.grid
    return _np.array(flat, dtype=float).reshape(g.ny, g.nx)


def edge_smooth(mask, n_sub: int = 8):
    """Anti-alias a GeomMask by sub-cell averaging at boundary cells.

    Each boundary cell is divided into n_sub x n_sub sub-cells; the fraction
    of sub-cells that fall inside the geometry sets the mask value (0.0-1.0).
    Interior/exterior cells remain exactly 1 and 0.  Equivalent to mumax3's
    EdgeSmooth parameter.

    Parameters
    ----------
    mask  : GeomMask -- binary input (0 or 1)
    n_sub : int -- sub-division per cell edge (default 8; mumax3 default)

    Returns
    -------
    GeomMask -- smoothed mask with fractional boundary values

    Example
    -------
    >>> disk   = mm.circle(grid, 100e-9)
    >>> smooth = mm.edge_smooth(disk, n_sub=8)
    >>> mm.set_geom(smooth, m, exch)
    """
    from _micromag import GeomMask
    g = mask.grid
    nx, ny, nz = g.nx, g.ny, g.nz

    def _lin(ix, iy, iz):
        return ix + nx * (iy + ny * iz)

    # Read original mask into numpy (nz, ny, nx) using linear indexing
    orig = _np.array([mask[_lin(ix, iy, iz)]
                      for iz in range(nz)
                      for iy in range(ny)
                      for ix in range(nx)], dtype=float).reshape(nz, ny, nx)

    result = GeomMask(g)
    offsets = (_np.arange(n_sub) + 0.5) / n_sub - 0.5  # [-0.5, 0.5)

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                v = orig[iz, iy, ix]
                lin = _lin(ix, iy, iz)
                if v >= 1.0:
                    result[lin] = 1.0
                    continue
                # Check neighbours to detect boundary cells
                has_inside_nb = False
                for dix, diy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx2, ny2 = ix + dix, iy + diy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny:
                        if orig[iz, ny2, nx2] > 0.5:
                            has_inside_nb = True
                            break
                if not has_inside_nb:
                    result[lin] = 0.0
                    continue
                # Boundary cell: sub-sample via bilinear interpolation
                count = 0.0
                for dix in offsets:
                    for diy in offsets:
                        sx = ix + float(dix)
                        sy = iy + float(diy)
                        ix0 = int(_math.floor(sx))
                        iy0 = int(_math.floor(sy))
                        tx = sx - ix0
                        ty = sy - iy0

                        def samp(ii, jj, _iz=iz):
                            if 0 <= ii < nx and 0 <= jj < ny:
                                return orig[_iz, jj, ii]
                            return 0.0

                        val = ((1 - tx) * (1 - ty) * samp(ix0,     iy0) +
                               tx       * (1 - ty) * samp(ix0 + 1, iy0) +
                               (1 - tx) * ty       * samp(ix0,     iy0 + 1) +
                               tx       * ty       * samp(ix0 + 1, iy0 + 1))
                        if val > 0.5:
                            count += 1.0
                result[lin] = count / (n_sub * n_sub)

    return result


def poisson_disk_grains(grid, avg_radius: float, min_dist_factor: float = 1.4,
                        seed: int = 42, max_tries: int = 30):
    """Generate a RegionMap with Poisson-disk-sampled grain centres.

    Unlike voronoi_grains (fixed grain count), poisson_disk_grains specifies
    the *average grain radius* and distributes seed points with a minimum
    inter-seed distance (Bridson 2007 algorithm).  This gives a more uniform
    grain-size distribution than random Voronoi seeds.

    Equivalent to mumax3 Ext_makegrains(diameter, ...).

    Parameters
    ----------
    grid             : StructuredGrid
    avg_radius       : float -- target average grain radius [m]
    min_dist_factor  : float -- minimum centre-to-centre distance as a
                               multiple of avg_radius (default 1.4 ~ sqrt(2))
    seed             : int   -- random seed
    max_tries        : int   -- rejection-sampling attempts per candidate

    Returns
    -------
    RegionMap -- region IDs 1..N (columnar: same 2-D pattern for all z-layers)

    Example
    -------
    >>> regions = mm.poisson_disk_grains(grid, avg_radius=50e-9)
    >>> mat_f = mm.MaterialField3D(grid)
    >>> for r in range(1, 256):
    ...     mat_f.set_region(regions, r, mm.Material.permalloy())
    """
    from _micromag import RegionMap
    rng = _np.random.default_rng(seed)
    g = grid
    nx, ny, nz = g.nx, g.ny, g.nz
    Lx = nx * g.dx
    Ly = ny * g.dy
    min_d = min_dist_factor * avg_radius

    # Bridson Poisson-disk sampling in 2-D
    cell_size = min_d / _math.sqrt(2.0)
    gx = int(_math.ceil(Lx / cell_size)) + 1
    gy = int(_math.ceil(Ly / cell_size)) + 1

    bg_grid = [[-1] * gy for _ in range(gx)]

    def grid_idx(x, y):
        return int(x / cell_size), int(y / cell_size)

    def in_bounds(x, y):
        return 0.0 <= x < Lx and 0.0 <= y < Ly

    def ok(x, y, pts):
        i0, j0 = grid_idx(x, y)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ii, jj = i0 + di, j0 + dj
                if 0 <= ii < gx and 0 <= jj < gy and bg_grid[ii][jj] >= 0:
                    px, py = pts[bg_grid[ii][jj]]
                    if (x - px) ** 2 + (y - py) ** 2 < min_d ** 2:
                        return False
        return True

    pts = []
    active = []

    x0, y0 = float(rng.uniform(0, Lx)), float(rng.uniform(0, Ly))
    pts.append((x0, y0))
    active.append(0)
    i0, j0 = grid_idx(x0, y0)
    bg_grid[i0][j0] = 0

    while active:
        idx = int(rng.integers(0, len(active)))
        xi, yi = pts[active[idx]]
        placed = False
        for _ in range(max_tries):
            r = float(rng.uniform(min_d, 2.0 * min_d))
            ang = float(rng.uniform(0, 2.0 * _math.pi))
            xn = xi + r * _math.cos(ang)
            yn = yi + r * _math.sin(ang)
            if in_bounds(xn, yn) and ok(xn, yn, pts):
                new_id = len(pts)
                pts.append((xn, yn))
                active.append(new_id)
                ii, jj = grid_idx(xn, yn)
                bg_grid[ii][jj] = new_id
                placed = True
                break
        if not placed:
            active.pop(idx)

    # Assign each cell to nearest seed (Voronoi partition)
    seed_xs = _np.array([p[0] for p in pts])
    seed_ys = _np.array([p[1] for p in pts])

    region_map = RegionMap(g)
    for iy in range(ny):
        cy = (iy + 0.5) * g.dy
        for ix in range(nx):
            cx = (ix + 0.5) * g.dx
            d2 = (seed_xs - cx) ** 2 + (seed_ys - cy) ** 2
            nearest = int(d2.argmin())
            rid = min(nearest + 1, 255)   # 1-indexed, cap at 255
            lin_base = ix + nx * iy
            for iz in range(nz):
                region_map[lin_base + nx * ny * iz] = rid

    return region_map
