"""ParaView export for Claude-SD magnetization fields.

Writes legacy-VTK STRUCTURED_POINTS files that ParaView opens directly, carrying
the magnetization as a vector **and** its components / magnitude / topological
charge density as scalars — so in ParaView you can immediately:

  * colour by ``mz`` (or mx/my) with a blue–white–red diverging map,
  * **Glyph** the ``m`` vector (arrows/cones) to see the spin texture,
  * **Contour** ``m_norm`` or ``q_topo`` to isolate cores / walls,
  * **Slice** a 3-D body.

Pure Python (NumPy only) — no VTK/ParaView Python bindings required.

    import micromag as mm
    mm.save_paraview(m, "state.vtk")                 # m is a VectorField3D
    mm.save_paraview(arr, "state.vtk", spacing=dx)   # or a (nz,ny,nx,3) array
    mm.save_paraview_series(frames, "run", dt=5e-12) # time series -> run.pvd
"""
from __future__ import annotations
import pathlib
import numpy as np


def _as_array(field):
    """Return an (nz, ny, nx, 3) float array from a VectorField3D or ndarray."""
    if isinstance(field, np.ndarray):
        return np.ascontiguousarray(field, dtype=np.float64)
    import micromag as mm
    return np.ascontiguousarray(mm.to_numpy(field), dtype=np.float64)


def _spacing(field, spacing):
    if spacing is not None:
        s = np.atleast_1d(spacing).astype(float)
        return (s[0], s[0], s[0]) if s.size == 1 else tuple(s[:3])
    g = getattr(field, "grid", None)
    if callable(g):
        g = g()
    if g is not None:
        return (g.dx, g.dy, g.dz)
    return (1.0, 1.0, 1.0)


def topological_charge_density(arr):
    """q(x,y) = (1/4π) m · (∂x m × ∂y m), summed over z. Sum ≈ skyrmion number."""
    m = arr.mean(axis=0)                       # (ny, nx, 3) — average over z
    dmx = np.gradient(m, axis=1)               # ∂/∂x  (x is axis 1)
    dmy = np.gradient(m, axis=0)               # ∂/∂y
    cross = np.cross(dmx, dmy)                  # (ny, nx, 3)
    q = np.sum(m * cross, axis=2) / (4.0 * np.pi)
    return q                                    # (ny, nx)


def save_paraview(field, path, name="m", spacing=None, topo=True):
    """Write a magnetization field to a ParaView-ready legacy .vtk file.

    field   : VectorField3D or (nz, ny, nx, 3) ndarray
    path    : output .vtk path
    spacing : cell size (scalar or (dx,dy,dz)); taken from the grid if omitted
    topo    : also write the 2-D topological-charge density scalar (per z-column)
    """
    arr = _as_array(field)
    nz, ny, nx, _ = arr.shape
    dx, dy, dz = _spacing(field, spacing)
    n = nx * ny * nz
    pts = arr.reshape(n, 3)                     # C-order == VTK x-fastest order

    path = pathlib.Path(path)
    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Claude-SD magnetization (ParaView export)\n")
        f.write("ASCII\nDATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {dx*0.5} {dy*0.5} {dz*0.5}\n")
        f.write(f"SPACING {dx} {dy} {dz}\n")
        f.write(f"POINT_DATA {n}\n")
        # vector m
        f.write(f"VECTORS {name} double\n")
        np.savetxt(f, pts, fmt="%.6g")
        # scalar components + magnitude
        comp = {"mx": pts[:, 0], "my": pts[:, 1], "mz": pts[:, 2],
                "m_norm": np.linalg.norm(pts, axis=1)}
        for cname, cval in comp.items():
            f.write(f"SCALARS {cname} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, cval, fmt="%.6g")
        # topological charge density (broadcast the 2-D map to every z)
        if topo:
            q2d = topological_charge_density(arr)          # (ny, nx)
            qcol = np.tile(q2d.reshape(1, ny, nx), (nz, 1, 1)).reshape(n)
            f.write("SCALARS q_topo double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, qcol, fmt="%.6g")
    return str(path)


def save_paraview_series(frames, basepath, name="m", spacing=None, dt=None, topo=True):
    """Write a time series: <basepath>_NNNN.vtk frames + a <basepath>.pvd collection
    that ParaView opens as one animatable dataset.

    frames : list of VectorField3D / ndarray (one per time step)
    dt     : time step [s] used for the .pvd timestamps (index if None)
    """
    base = pathlib.Path(basepath)
    stem = base.with_suffix("")
    entries = []
    for i, fr in enumerate(frames):
        fp = pathlib.Path(f"{stem}_{i:04d}.vtk")
        save_paraview(fr, fp, name=name, spacing=spacing, topo=topo)
        t = (i * dt) if dt is not None else float(i)
        entries.append(f'    <DataSet timestep="{t}" file="{fp.name}"/>')
    pvd = pathlib.Path(f"{stem}.pvd")
    pvd.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n" + "\n".join(entries) + "\n  </Collection>\n</VTKFile>\n")
    return str(pvd)
